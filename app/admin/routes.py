"""Admin dashboard API -- customer/doctor management + business analytics.
Mounted under /admin/* in app/main.py behind require_admin_token (app/admin/auth.py).
Deactivate/delete reuse the same domain logic the WhatsApp agent uses
(cancel_session, cancel_subscription) rather than raw table writes, so an
admin action leaves the exact same trail (cancelled calendar events,
notified parties) a customer-initiated one would."""

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile

from app.admin.intent_rating import rate_customer_intent
from app.agent.tools.booking import cancel_session
from app.deps import AppContext
from app.integrations import razorpay_client
from app.integrations.supabase_client import escape_or_filter_value, sign_storage_url, upload_to_storage

DOCTOR_DOCUMENTS_BUCKET = "doctor-documents"

logger = logging.getLogger(__name__)

router = APIRouter()

ACTIVE_SESSION_STATUSES = ["pending", "negotiating", "accepted"]
RECENT_ACTIVITY_LIMIT = 60


async def _purge_blocking_references(ctx: AppContext, profile_id: str) -> None:
    """A hard delete of a profile hits real FK constraints that don't
    cascade automatically (confirmed live: deleting a customer with any
    session history 500'd with a doctor_sessions FK violation). Most
    tables referencing profiles.id are ON DELETE CASCADE/SET NULL, but
    four are NO ACTION and must be cleared explicitly first, in any order
    (none of them reference each other)."""
    client = ctx.supabase
    client.table("doctor_sessions").delete().eq("profile_id", profile_id).execute()
    client.table("new_parent_followups").delete().eq("profile_id", profile_id).execute()
    client.table("new_parent_guides").delete().eq("profile_id", profile_id).execute()
    client.table("pet_members").delete().eq("added_by", profile_id).execute()


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


# profiles.gender has a check constraint accepting only "Male"/"Female"/"Other"
# (confirmed live -- lowercase or "Unknown" both 400). Document extraction
# produces free-text/lowercase values, so it must be normalized before insert.
_PROFILE_GENDER_VALUES = {"male": "Male", "female": "Female", "other": "Other"}


def _normalize_profile_gender(value: str | None) -> str | None:
    if not value:
        return None
    return _PROFILE_GENDER_VALUES.get(value.strip().lower())


async def _cancel_pending_sessions(ctx: AppContext, *, profile_id: str | None = None, doctor_phone: str | None = None, role: str) -> int:
    query = ctx.supabase.table("doctor_sessions").select("id").in_("status", ACTIVE_SESSION_STATUSES)
    if profile_id:
        query = query.eq("profile_id", profile_id)
    if doctor_phone:
        query = query.eq("doctor_phone", doctor_phone)
    sessions = query.execute().data or []
    agent_ctx = SimpleNamespace(role=role)
    for session in sessions:
        await cancel_session(ctx, agent_ctx, session["id"])
    return len(sessions)


async def _cancel_active_subscription(ctx: AppContext, profile_id: str) -> bool:
    rows = (
        ctx.supabase.table("subscriptions").select("*").eq("profile_id", profile_id)
        .in_("status", ["trial", "active"]).execute().data or []
    )
    if not rows:
        return False
    sub = rows[0]
    if sub.get("provider_subscription_id"):
        try:
            await razorpay_client.cancel_subscription(ctx.settings, sub["provider_subscription_id"])
        except Exception:
            logger.exception("Failed to cancel Razorpay subscription %s", sub["provider_subscription_id"])
    ctx.supabase.table("subscriptions").update(
        {"status": "cancelled", "cancelled_at": datetime.now(tz=None).isoformat()}
    ).eq("id", sub["id"]).execute()
    return True


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------

def _subscription_category(profile: dict[str, Any], subscription: dict[str, Any] | None) -> str:
    """Legacy plan category from before subscriptions were removed as a
    product concept (every feature except paid doctor consultations is now
    free for every customer) -- kept only so historical subscription rows
    still display sensibly; every customer going forward is "Free"."""
    if not subscription:
        return "Free"
    if subscription.get("status") == "active":
        return "Founding" if profile.get("is_founding_member") else "Subscriber"
    if subscription.get("status") == "trial":
        return "Trial"
    return "Free"


REGISTRATION_STEP_LABELS = {
    "awaiting_customer_name": "entering their name",
    "awaiting_pet_name": "entering their pet's name",
    "awaiting_city": "entering their city",
}


def _relative_time(iso_ts: str | None) -> str | None:
    if not iso_ts:
        return None
    then = datetime.fromisoformat(iso_ts)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - then
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return "just now"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)}d ago"


def _compute_customer_stage(
    profile: dict[str, Any], sessions: list[dict[str, Any]]
) -> dict[str, Any]:
    """A one-line "where is this customer right now, and do they need me to
    reach out" summary for the admin dashboard -- checked in priority order,
    most actionable first, so e.g. a customer mid-onboarding who also has a
    year-old completed session still shows as "onboarding", not "active"."""
    step = profile.get("registration_step")
    if step and step != "completed":
        return {
            "code": "onboarding",
            "label": "Onboarding incomplete",
            "detail": f"Stuck at {REGISTRATION_STEP_LABELS.get(step, step)}",
        }

    active = [s for s in sessions if s.get("status") in ACTIVE_SESSION_STATUSES]
    if active:
        # A customer can have more than one concurrent active session -- booking
        # is scoped per-pet (see request_doctor_session in app/agent/tools/booking.py),
        # so a multi-pet account can easily have one pet mid-payment and another
        # just starting to pick a vet at the same time. Picking whichever session
        # was merely created most recently (the old behavior) surfaces the
        # newer-but-less-urgent one and buries a stalled payment/prescription
        # behind it -- rank by actionability first (payment blocked > prescription
        # owed > negotiating > booking > already-confirmed-upcoming) and only
        # fall back to recency to break ties within the same rank.
        def _active_priority(sess: dict[str, Any]) -> int:
            if sess["status"] == "accepted":
                if sess.get("payment_status") == "awaiting":
                    return 0
                if sess.get("awaiting_from") == "doctor_prescription":
                    return 1
                return 4  # upcoming, already confirmed -- least urgent of the active states
            if sess["status"] == "negotiating":
                return 2
            return 3  # pending (booking)

        # Sort newest-first, then pick the lowest-priority-number session via a
        # stable min() -- ties within the same priority tier naturally resolve
        # to whichever was created most recently, since that's earliest in this
        # pre-sorted iteration order.
        active_by_recency = sorted(active, key=lambda sess: sess.get("created_at") or "", reverse=True)
        s = min(active_by_recency, key=_active_priority)
        if s["status"] == "pending":
            if s.get("doctor_phone") == "pending_doctor_choice":
                return {"code": "booking", "label": "Choosing a vet", "detail": "Picking a doctor from the list"}
            return {"code": "booking", "label": "Waiting on doctor", "detail": "Vet chosen, awaiting their response"}
        if s["status"] == "negotiating":
            waiting_on = "the doctor" if s.get("awaiting_from") == "doctor_time_input" else "the customer"
            return {"code": "negotiating", "label": "Negotiating appointment time", "detail": f"Waiting on {waiting_on}"}
        if s["status"] == "accepted":
            if s.get("payment_status") == "awaiting":
                return {"code": "payment", "label": "Awaiting payment", "detail": f"Booked for {s.get('preferred_time') or 'TBD'}"}
            if s.get("awaiting_from") == "doctor_prescription":
                return {"code": "prescription", "label": "Awaiting prescription", "detail": "Consultation done, doctor hasn't sent it yet"}
            return {"code": "upcoming", "label": "Appointment upcoming", "detail": f"Booked for {s.get('preferred_time') or 'TBD'}"}

    followups = [s for s in sessions if s.get("status") == "completed" and s.get("follow_up_required")]
    if followups:
        followups.sort(key=lambda s: s.get("follow_up_date") or "")
        return {"code": "followup", "label": "Follow-up needed", "detail": f"Due {followups[0].get('follow_up_date') or 'soon'}"}

    last_active = profile.get("last_active_at")
    if last_active:
        then = datetime.fromisoformat(last_active)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - then).total_seconds() / 3600
        if hours < 48:
            return {"code": "active", "label": "Active", "detail": f"Last messaged {_relative_time(last_active)}"}
        return {"code": "inactive", "label": "Gone quiet", "detail": f"No message in {int(hours // 24)}d"}

    return {"code": "new", "label": "New / no activity yet", "detail": "Hasn't messaged the bot since onboarding"}


@router.get("/customers")
async def list_customers(
    request: Request,
    search: str = "",
    date_from: str = "",
    date_to: str = "",
    tier: str = "",
    breed: str = "",
    status: str = "",
    stage: str = "",
    intent: str = "",
    active_from: str = "",
    active_to: str = "",
    active: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    ctx = _ctx(request)
    query = ctx.supabase.table("profiles").select("*").eq("role", "customer")
    if search:
        pattern = escape_or_filter_value(f"%{search}%")
        query = query.or_(f"full_name.ilike.{pattern},phone_number.ilike.{pattern},email.ilike.{pattern}")
    if status:
        query = query.eq("is_active", status.lower() == "active")
    if intent:
        # A real column (unlike stage, which is computed), so filterable
        # directly at the DB level -- "unrated" means the background job
        # hasn't gotten to them yet or they have no messages at all.
        query = query.is_("intent_rating", "null") if intent.lower() == "unrated" else query.eq("intent_rating", intent)
    if active.lower() == "never":
        query = query.is_("last_active_at", "null")
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        # Inclusive of the whole "to" day -- a bare date string sorts before
        # any timestamp on that same day, so a plain lte would exclude it.
        query = query.lte("created_at", f"{date_to}T23:59:59")
    if active_from:
        query = query.gte("last_active_at", active_from)
    if active_to:
        query = query.lte("last_active_at", f"{active_to}T23:59:59")
    rows = query.order("created_at", desc=True).execute().data or []

    profile_ids = [r["id"] for r in rows]
    subscriptions_by_profile: dict[str, dict[str, Any]] = {}
    if profile_ids:
        sub_rows = (
            ctx.supabase.table("subscriptions").select("*").in_("profile_id", profile_ids)
            .order("created_at", desc=True).execute().data or []
        )
        for sub in sub_rows:
            # Most recent per profile only (sub_rows is already newest-first) --
            # a customer's category reflects their current subscription, not history.
            subscriptions_by_profile.setdefault(sub["profile_id"], sub)
    for row in rows:
        row["subscription"] = subscriptions_by_profile.get(row["id"])
        row["subscription_category"] = _subscription_category(row, row["subscription"])

    if tier:
        rows = [r for r in rows if r["subscription_category"].lower() == tier.lower()]

    # Pets are fetched before the breed filter/pagination split -- "has a pet of
    # this breed" is a property of the full result set, not just the current page.
    profile_ids = [r["id"] for r in rows]
    pets_by_profile: dict[str, list[dict[str, Any]]] = {}
    if profile_ids:
        pet_rows = ctx.supabase.table("pets").select("*").in_("profile_id", profile_ids).order("created_at").execute().data or []
        for pet in pet_rows:
            pets_by_profile.setdefault(pet["profile_id"], []).append(pet)
    for row in rows:
        row["pets"] = pets_by_profile.get(row["id"], [])

    if breed:
        needle = breed.lower()
        rows = [r for r in rows if any(needle in (p.get("breed") or "").lower() for p in r["pets"])]

    # Stage is computed for the full filtered set, before pagination -- like
    # tier/breed, it needs to be filterable, which means it has to exist
    # before the page slice discards rows.
    filtered_ids = [r["id"] for r in rows]
    sessions_by_profile: dict[str, list[dict[str, Any]]] = {}
    if filtered_ids:
        session_rows = (
            ctx.supabase.table("doctor_sessions")
            .select("status,doctor_phone,payment_status,awaiting_from,preferred_time,created_at,follow_up_required,follow_up_date,profile_id")
            .in_("profile_id", filtered_ids).execute().data or []
        )
        for s in session_rows:
            sessions_by_profile.setdefault(s["profile_id"], []).append(s)
    for row in rows:
        stage_info = _compute_customer_stage(row, sessions_by_profile.get(row["id"], []))
        row["stage"] = stage_info["label"]
        row["stage_detail"] = stage_info["detail"]
        row["stage_code"] = stage_info["code"]

    if stage:
        rows = [r for r in rows if r["stage_code"] == stage]

    total_count = len(rows)
    rows = rows[offset:offset + limit] if offset else rows[:limit]

    return {"customers": rows, "count": total_count}


@router.get("/customers/breeds")
async def list_customer_breeds(request: Request) -> dict[str, Any]:
    """Distinct pet breeds on file, for the breed filter dropdown -- a plain
    text filter would work too, but a dropdown avoids typos/mismatches
    against what's actually in the data."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("pets").select("breed").execute().data or []
    breeds = sorted({r["breed"] for r in rows if r.get("breed")})
    return {"breeds": breeds}


@router.get("/customers/{profile_id}")
async def get_customer(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("*").eq("id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")
    profile = rows[0]

    pets = ctx.supabase.table("pets").select("*").eq("profile_id", profile_id).execute().data or []
    subscription_rows = (
        ctx.supabase.table("subscriptions").select("*").eq("profile_id", profile_id)
        .order("created_at", desc=True).limit(1).execute().data or []
    )
    sessions = (
        ctx.supabase.table("doctor_sessions")
        .select("id,status,doctor_phone,payment_status,awaiting_from,preferred_time,created_at,follow_up_required,follow_up_date")
        .eq("profile_id", profile_id).order("created_at", desc=True).limit(10).execute().data or []
    )
    pet_ids = [p["id"] for p in pets]
    document_count = len(ctx.supabase.table("documents").select("id").in_("pet_id", pet_ids).execute().data or []) if pet_ids else 0
    stage = _compute_customer_stage(profile, sessions)

    # Full activity picture for an admin trying to understand "what's going
    # on with this customer" without digging through raw DB tables --
    # fetched newest-first (cheapest way to cap it at RECENT_ACTIVITY_LIMIT)
    # then reversed so the transcript reads chronologically, oldest first,
    # like an actual conversation.
    recent_messages = (
        ctx.supabase.table("messages").select("sender_type,content,message_type,created_at")
        .eq("profile_id", profile_id).order("created_at", desc=True).limit(RECENT_ACTIVITY_LIMIT).execute().data or []
    )
    recent_messages.reverse()
    health_logs = (
        ctx.supabase.table("health_logs").select("pet_id,ai_risk_score,ai_observation,symptoms,notes,created_at")
        .eq("profile_id", profile_id).order("created_at", desc=True).limit(RECENT_ACTIVITY_LIMIT).execute().data or []
    ) if pet_ids else []
    memory_facts = ctx.supabase.table("memory").select("*").eq("profile_id", profile_id).execute().data or []

    return {
        "profile": profile,
        "pets": pets,
        "subscription": subscription_rows[0] if subscription_rows else None,
        "recent_sessions": sessions,
        "document_count": document_count,
        "recent_messages": recent_messages,
        "health_logs": health_logs,
        "memory_facts": memory_facts,
        "stage": stage["label"],
        "stage_detail": stage["detail"],
        "stage_code": stage["code"],
    }


@router.get("/customers/{profile_id}/export-chat")
async def export_customer_chat(profile_id: str, request: Request) -> Response:
    """Full chat history (not capped at RECENT_ACTIVITY_LIMIT like the
    Activity tab -- an export is explicitly the "give me everything" case)
    as a plain-text transcript, one line per message, oldest first."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id,full_name,phone_number").eq("id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")
    profile = rows[0]

    messages = (
        ctx.supabase.table("messages").select("sender_type,content,message_type,created_at")
        .eq("profile_id", profile_id).order("created_at").execute().data or []
    )

    sender_labels = {"user": "Customer", "assistant": "Bot", "vet": "Vet"}
    lines = [
        f"Chat export -- {profile.get('full_name') or 'Unnamed'} ({profile['phone_number']})",
        f"Exported {datetime.now(timezone.utc).isoformat()}",
        f"{len(messages)} messages",
        "-" * 60,
        "",
    ]
    for m in messages:
        sender = sender_labels.get(m.get("sender_type"), m.get("sender_type") or "Unknown")
        content = m.get("content") or f"[{m.get('message_type') or 'non-text'} message, no text content]"
        lines.append(f"[{m.get('created_at', '')}] {sender}: {content}")
    text = "\n".join(lines) + "\n"

    safe_phone = "".join(c for c in profile["phone_number"] if c.isalnum()) or profile_id
    filename = f"chat_export_{safe_phone}.txt"
    return Response(
        content=text,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/customers/{profile_id}/rate-intent")
async def rate_customer_intent_endpoint(profile_id: str, request: Request) -> dict[str, Any]:
    """On-demand version of the same LLM classification the
    rate_customer_intents scheduled job runs in bulk -- lets an admin force
    a fresh rating for one customer right after a new conversation, rather
    than waiting for the next scheduled pass."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).eq("role", "customer").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")

    messages = (
        ctx.supabase.table("messages").select("sender_type,content").eq("profile_id", profile_id)
        .order("created_at", desc=True).limit(RECENT_ACTIVITY_LIMIT).execute().data or []
    )
    messages.reverse()
    result = await rate_customer_intent(ctx.openai, ctx.settings, messages)
    updated = (
        ctx.supabase.table("profiles")
        .update(
            {
                "intent_rating": result["rating"],
                "intent_rating_reason": result["reason"],
                "intent_rating_updated_at": datetime.now(timezone.utc).isoformat(),
                "intent_rating_message_count": len(messages),
            }
        )
        .eq("id", profile_id).execute().data[0]
    )
    return {"success": True, "profile": updated}


@router.post("/customers/{profile_id}/activate")
async def activate_customer(profile_id: str, request: Request) -> dict[str, Any]:
    """Reverses deactivate -- just the flag. Doesn't restore the cancelled
    subscription/sessions from a prior deactivate (those are gone for real,
    same as any other cancellation); this is for un-suspending an account,
    not undoing history."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")

    ctx.supabase.table("profiles").update({"is_active": True}).eq("id", profile_id).execute()
    return {"success": True}


@router.post("/customers/{profile_id}/deactivate")
async def deactivate_customer(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")

    cancelled_sessions = await _cancel_pending_sessions(ctx, profile_id=profile_id, role="customer")
    subscription_cancelled = await _cancel_active_subscription(ctx, profile_id)
    ctx.supabase.table("profiles").update({"is_active": False}).eq("id", profile_id).execute()

    return {"success": True, "cancelled_sessions": cancelled_sessions, "subscription_cancelled": subscription_cancelled}


@router.delete("/customers/{profile_id}")
async def delete_customer(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Customer not found")

    await _cancel_pending_sessions(ctx, profile_id=profile_id, role="customer")
    await _cancel_active_subscription(ctx, profile_id)
    await _purge_blocking_references(ctx, profile_id)
    ctx.supabase.table("profiles").delete().eq("id", profile_id).execute()

    return {"success": True}


# ---------------------------------------------------------------------------
# Doctors
# ---------------------------------------------------------------------------

DOCTOR_EDITABLE_FIELDS = [
    "full_name", "qualification", "registration_number", "specialization", "experience_years",
    "consultation_fee", "area", "city", "hospital_name", "treatments", "opening_time", "closing_time",
]

DOCTOR_WELCOME_MESSAGE = (
    "Welcome to PetPulse, {name}! 🩺 You've been onboarded as a Veterinary Doctor on our platform. "
    "You'll receive session requests, appointment updates, and prescription reminders right here on WhatsApp. "
    "If you have any questions, just message us anytime."
)


async def _create_doctor_profile(ctx: AppContext, fields: dict[str, Any]) -> dict[str, Any]:
    """Shared by manual onboarding (onboard_doctor) and the Drive-drafts
    approval flow (approve_doctor_draft) so both paths create an identical
    real profile row and both send the same welcome message — a doctor
    shouldn't get a different experience depending on which route an admin
    used to onboard them."""
    row = (
        ctx.supabase.table("profiles")
        .insert({"role": "vet", "is_active": True, **fields})
        .execute()
        .data[0]
    )
    try:
        await ctx.whatsapp.send_text(row["phone_number"], DOCTOR_WELCOME_MESSAGE.format(name=row.get("full_name") or "Doctor"))
    except Exception:
        # The account is already created at this point -- a delivery hiccup
        # on the welcome message is a notification problem, not a reason to
        # fail the onboarding itself.
        logger.exception("Failed to send welcome message to newly onboarded doctor profile=%s", row["id"])
    return row


@router.get("/doctors")
async def list_doctors(
    request: Request,
    search: str = "",
    area: str = "",
    city: str = "",
    hospital: str = "",
    treatments: str = "",
    status: str = "",
) -> dict[str, Any]:
    ctx = _ctx(request)
    query = ctx.supabase.table("profiles").select("*").eq("role", "vet")
    if search:
        pattern = escape_or_filter_value(f"%{search}%")
        query = query.or_(f"full_name.ilike.{pattern},phone_number.ilike.{pattern},specialization.ilike.{pattern}")
    if area:
        query = query.ilike("area", f"%{area}%")
    if city:
        query = query.ilike("city", f"%{city}%")
    if hospital:
        query = query.ilike("hospital_name", f"%{hospital}%")
    if treatments:
        query = query.ilike("treatments", f"%{treatments}%")
    if status:
        query = query.eq("is_active", status.lower() == "active")
    rows = query.order("created_at", desc=True).execute().data or []
    return {"doctors": rows, "count": len(rows)}


def _distinct_doctor_values(ctx: AppContext, column: str) -> list[str]:
    rows = ctx.supabase.table("profiles").select(column).eq("role", "vet").execute().data or []
    return sorted({r[column] for r in rows if r.get(column)})


@router.get("/doctors/areas")
async def list_doctor_areas(request: Request) -> dict[str, Any]:
    return {"areas": _distinct_doctor_values(_ctx(request), "area")}


@router.get("/doctors/cities")
async def list_doctor_cities(request: Request) -> dict[str, Any]:
    return {"cities": _distinct_doctor_values(_ctx(request), "city")}


@router.get("/doctors/hospitals")
async def list_doctor_hospitals(request: Request) -> dict[str, Any]:
    return {"hospitals": _distinct_doctor_values(_ctx(request), "hospital_name")}


@router.get("/doctors/treatments")
async def list_doctor_treatments(request: Request) -> dict[str, Any]:
    """Splits each doctor's comma-separated treatments field and dedupes --
    treatments is free text (like specialization), not an enum, so the
    filter dropdown reflects whatever's actually been typed in so far."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("treatments").eq("role", "vet").execute().data or []
    treatments: set[str] = set()
    for row in rows:
        raw = row.get("treatments") or ""
        treatments.update(t.strip() for t in raw.split(",") if t.strip())
    return {"treatments": sorted(treatments)}


@router.post("/doctors")
async def onboard_doctor(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _ctx(request)
    full_name = (payload.get("full_name") or "").strip()
    phone_number = (payload.get("phone_number") or "").strip()
    if not full_name or not phone_number:
        raise HTTPException(status_code=422, detail="full_name and phone_number are required")

    existing = ctx.supabase.table("profiles").select("id").eq("phone_number", phone_number).limit(1).execute().data
    if existing:
        raise HTTPException(status_code=409, detail="A profile with this phone number already exists")

    row = await _create_doctor_profile(
        ctx,
        {
            "full_name": full_name,
            "phone_number": phone_number,
            "qualification": payload.get("qualification"),
            "registration_number": payload.get("registration_number"),
            "specialization": payload.get("specialization"),
            "experience_years": payload.get("experience_years"),
            "consultation_fee": payload.get("consultation_fee"),
            "area": payload.get("area"),
            "city": payload.get("city"),
            "hospital_name": payload.get("hospital_name"),
            "treatments": payload.get("treatments"),
            "opening_time": payload.get("opening_time"),
            "closing_time": payload.get("closing_time"),
        },
    )
    return {"success": True, "doctor": row}


@router.patch("/doctors/{profile_id}")
async def update_doctor(profile_id: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Partial update -- only DOCTOR_EDITABLE_FIELDS keys actually present in
    the payload change. The first way to edit a doctor after onboarding at
    all; without this, adding a filterable field (area/hospital/treatments/
    hours) would be pointless for every doctor onboarded before it existed."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")

    updates = {k: v for k, v in payload.items() if k in DOCTOR_EDITABLE_FIELDS}
    if not updates:
        raise HTTPException(status_code=422, detail="No editable fields in payload")

    row = ctx.supabase.table("profiles").update(updates).eq("id", profile_id).execute().data[0]
    return {"success": True, "doctor": row}


@router.get("/doctors/{profile_id}")
async def get_doctor(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("*").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")
    profile = rows[0]

    sessions = (
        ctx.supabase.table("doctor_sessions").select("id, status, preferred_time, profile_id")
        .eq("doctor_phone", profile["phone_number"]).order("created_at", desc=True).limit(20).execute().data or []
    )
    documents = (
        ctx.supabase.table("doctor_documents").select("*").eq("profile_id", profile_id)
        .order("uploaded_at", desc=True).execute().data or []
    )
    return {"profile": profile, "recent_sessions": sessions, "documents": documents}


@router.post("/doctors/{profile_id}/documents")
async def upload_doctor_document(profile_id: str, request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Doctor-only attachments (licenses, degrees, etc.) -- a separate table/
    bucket from the customer pet-record vault (documents.pet_id is NOT NULL
    there, so a doctor attachment doesn't fit it)."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")

    data = await file.read()
    ext = (file.filename or "").rsplit(".", 1)[-1] if "." in (file.filename or "") else "bin"
    object_path = f"{profile_id}/{uuid.uuid4().hex}.{ext}"
    upload_to_storage(ctx.supabase, DOCTOR_DOCUMENTS_BUCKET, object_path, data, file.content_type or "application/octet-stream")

    row = (
        ctx.supabase.table("doctor_documents")
        .insert(
            {
                "profile_id": profile_id,
                "document_name": file.filename or "document",
                "storage_path": object_path,
                "mime_type": file.content_type,
            }
        )
        .execute()
        .data[0]
    )
    return {"success": True, "document": row}


@router.get("/doctors/{profile_id}/documents")
async def list_doctor_documents(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = (
        ctx.supabase.table("doctor_documents").select("*").eq("profile_id", profile_id)
        .order("uploaded_at", desc=True).execute().data or []
    )
    for row in rows:
        row["url"] = sign_storage_url(ctx.supabase, DOCTOR_DOCUMENTS_BUCKET, row["storage_path"])
    return {"documents": rows}


@router.delete("/doctors/{profile_id}/documents/{document_id}")
async def delete_doctor_document(profile_id: str, document_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("doctor_documents").select("*").eq("id", document_id).eq("profile_id", profile_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        ctx.supabase.storage.from_(DOCTOR_DOCUMENTS_BUCKET).remove([rows[0]["storage_path"]])
    except Exception:
        logger.exception("Failed to remove storage object for doctor_document %s", document_id)
    ctx.supabase.table("doctor_documents").delete().eq("id", document_id).execute()
    return {"success": True}


@router.post("/doctors/{profile_id}/activate")
async def activate_doctor(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("id").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")

    ctx.supabase.table("profiles").update({"is_active": True}).eq("id", profile_id).execute()
    return {"success": True}


@router.post("/doctors/{profile_id}/deactivate")
async def deactivate_doctor(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("*").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")

    cancelled_sessions = await _cancel_pending_sessions(ctx, doctor_phone=rows[0]["phone_number"], role="vet")
    ctx.supabase.table("profiles").update({"is_active": False}).eq("id", profile_id).execute()

    return {"success": True, "cancelled_sessions": cancelled_sessions}


@router.delete("/doctors/{profile_id}")
async def delete_doctor(profile_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("*").eq("id", profile_id).eq("role", "vet").limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Doctor not found")

    await _cancel_pending_sessions(ctx, doctor_phone=rows[0]["phone_number"], role="vet")
    await _purge_blocking_references(ctx, profile_id)
    ctx.supabase.table("profiles").delete().eq("id", profile_id).execute()

    return {"success": True}


# ---------------------------------------------------------------------------
# Doctor onboarding drafts (Google Drive auto-sync — see
# app/scheduler/jobs.py sync_doctor_onboarding_drafts). Never creates a real
# doctor account directly; approve_doctor_draft is the only path from a
# draft to a real profiles row, so every auto-extracted account still gets
# a human review first.
# ---------------------------------------------------------------------------

DRAFT_EDITABLE_FIELDS = [
    "extracted_full_name", "extracted_phone_number", "extracted_email", "extracted_qualification",
    "extracted_registration_number", "extracted_specialization", "extracted_gender",
    "extracted_date_of_birth", "extracted_city", "extracted_area",
]


@router.get("/doctor-drafts")
async def list_doctor_drafts(request: Request, status: str = "pending_review") -> dict[str, Any]:
    ctx = _ctx(request)
    query = ctx.supabase.table("doctor_onboarding_drafts").select("*")
    if status:
        query = query.eq("status", status)
    rows = query.order("created_at", desc=True).execute().data or []
    return {"drafts": rows, "count": len(rows)}


@router.get("/doctor-drafts/{draft_id}")
async def get_doctor_draft(draft_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("doctor_onboarding_drafts").select("*").eq("id", draft_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Draft not found")
    return {"draft": rows[0]}


@router.patch("/doctor-drafts/{draft_id}")
async def update_doctor_draft(draft_id: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    """Lets an admin correct/fill in extracted fields before approving --
    auto-extraction is best-effort (see app/media_pipeline/doctor_documents.py),
    and fields like phone_number often aren't even printed on the source
    documents at all."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("doctor_onboarding_drafts").select("id, status").eq("id", draft_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Draft not found")
    if rows[0]["status"] != "pending_review":
        raise HTTPException(status_code=409, detail=f"Draft is already {rows[0]['status']}, cannot edit")

    updates = {k: v for k, v in payload.items() if k in DRAFT_EDITABLE_FIELDS}
    if not updates:
        raise HTTPException(status_code=422, detail="No editable fields in payload")

    row = ctx.supabase.table("doctor_onboarding_drafts").update(updates).eq("id", draft_id).execute().data[0]
    return {"success": True, "draft": row}


@router.post("/doctor-drafts/{draft_id}/approve")
async def approve_doctor_draft(draft_id: str, request: Request) -> dict[str, Any]:
    """Creates the real, active doctor profile from this draft's (possibly
    admin-corrected) fields, sends the welcome message, and marks the draft
    approved -- final, sync_doctor_onboarding_drafts never touches an
    approved/rejected draft again even if the Drive folder changes later."""
    ctx = _ctx(request)
    rows = ctx.supabase.table("doctor_onboarding_drafts").select("*").eq("id", draft_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Draft not found")
    draft = rows[0]
    if draft["status"] != "pending_review":
        raise HTTPException(status_code=409, detail=f"Draft is already {draft['status']}")

    full_name = (draft.get("extracted_full_name") or "").strip()
    phone_number = (draft.get("extracted_phone_number") or "").strip()
    if not full_name or not phone_number:
        raise HTTPException(status_code=422, detail="full_name and phone_number must be filled in (via PATCH) before approving")

    existing = ctx.supabase.table("profiles").select("*").eq("phone_number", phone_number).limit(1).execute().data
    replaced_customer: dict[str, Any] | None = None
    if existing:
        other = existing[0]
        if other.get("role") != "customer":
            # A genuine duplicate (e.g. already a vet) -- not something to
            # silently resolve, needs an admin decision.
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Phone {phone_number} already belongs to an existing {other.get('role', 'profile')} profile "
                    f"({other.get('full_name') or other['id']}). Correct the phone number on this draft first, "
                    f"or resolve the conflicting profile, before approving."
                ),
            )
        # Same phone number under a customer profile just means this person
        # signed up/explored the product as a pet owner before being
        # onboarded as a vet -- keep the doctor identity, drop the customer
        # one (same cleanup as the standalone delete-customer endpoint, since
        # a hard delete hits FK constraints that don't cascade automatically).
        # profiles.phone_number is unique, so the old row must go before the
        # new one can be inserted -- keep a full snapshot so a failure in
        # doctor-profile creation (any DB constraint, not just the gender one
        # already normalized above) can restore it rather than leaving this
        # person with no profile at all, which happened once in production.
        await _cancel_pending_sessions(ctx, profile_id=other["id"], role="customer")
        await _cancel_active_subscription(ctx, other["id"])
        await _purge_blocking_references(ctx, other["id"])
        ctx.supabase.table("profiles").delete().eq("id", other["id"]).execute()
        replaced_customer = other

    try:
        doctor = await _create_doctor_profile(
            ctx,
            {
                "full_name": full_name,
                "phone_number": phone_number,
                "email": draft.get("extracted_email"),
                "qualification": draft.get("extracted_qualification"),
                "registration_number": draft.get("extracted_registration_number"),
                "specialization": draft.get("extracted_specialization"),
                "gender": _normalize_profile_gender(draft.get("extracted_gender")),
                "date_of_birth": draft.get("extracted_date_of_birth"),
                "city": draft.get("extracted_city"),
                "area": draft.get("extracted_area"),
            },
        )
    except Exception:
        if replaced_customer is not None:
            # Best-effort restore -- re-insert the exact row we deleted so a
            # constraint failure here doesn't leave this person with no
            # profile at all (the purged session/followup/guide rows aren't
            # recovered, but those were empty in the one case this hit).
            ctx.supabase.table("profiles").insert(replaced_customer).execute()
        raise

    ctx.supabase.table("doctor_onboarding_drafts").update(
        {"status": "approved", "created_profile_id": doctor["id"], "reviewed_at": datetime.utcnow().isoformat()}
    ).eq("id", draft_id).execute()

    return {"success": True, "doctor": doctor}


@router.post("/doctor-drafts/{draft_id}/reject")
async def reject_doctor_draft(draft_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("doctor_onboarding_drafts").select("id, status").eq("id", draft_id).limit(1).execute().data
    if not rows:
        raise HTTPException(status_code=404, detail="Draft not found")
    if rows[0]["status"] != "pending_review":
        raise HTTPException(status_code=409, detail=f"Draft is already {rows[0]['status']}")

    row = (
        ctx.supabase.table("doctor_onboarding_drafts")
        .update({"status": "rejected", "reviewed_at": datetime.utcnow().isoformat()})
        .eq("id", draft_id)
        .execute()
        .data[0]
    )
    return {"success": True, "draft": row}


# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------

# pets.age is a rounded-integer year count (see app/agent/tools/onboarding.py),
# not a DOB -- these buckets bin directly on it rather than computing from date_of_birth.
AGE_GROUPS: list[tuple[str, Any]] = [
    ("Puppy/Kitten (<1yr)", lambda age: age < 1),
    ("Young (1-2yrs)", lambda age: 1 <= age <= 2),
    ("Adult (3-6yrs)", lambda age: 3 <= age <= 6),
    ("Senior (7+yrs)", lambda age: age >= 7),
]


def _age_group(age: int | None) -> str:
    if age is None:
        return "Unknown"
    for label, matcher in AGE_GROUPS:
        if matcher(age):
            return label
    return "Unknown"


def _prescription_status(session: dict[str, Any]) -> str:
    """Derived from existing doctor_sessions state -- there's no dedicated
    delivery-status column (see file_prescription/deliver_prescription in
    app/agent/tools/booking.py)."""
    if session.get("status") != "completed":
        return "N/A"
    if not session.get("pending_medications"):
        return "Not filed"
    if session.get("awaiting_from") == "prescription_format_choice":
        return "Awaiting delivery choice"
    return "Delivered"


@router.get("/appointments/cities")
async def list_appointment_cities(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("profiles").select("city").eq("role", "customer").execute().data or []
    return {"cities": sorted({r["city"] for r in rows if r.get("city")})}


@router.get("/appointments")
async def list_appointments(
    request: Request,
    date_from: str = "",
    date_to: str = "",
    breed: str = "",
    age_group: str = "",
    issue: str = "",
    doctor: str = "",
    city: str = "",
    status: str = "",
    follow_up_required: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase

    query = client.table("doctor_sessions").select("*")
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        # Inclusive of the whole "to" day, same reasoning as list_customers.
        query = query.lte("created_at", f"{date_to}T23:59:59")
    if status:
        query = query.eq("status", status)
    if issue:
        query = query.ilike("case_summary", f"%{issue}%")
    if follow_up_required:
        query = query.eq("follow_up_required", follow_up_required.lower() == "true")
    rows = query.order("created_at", desc=True).execute().data or []

    profile_ids = list({r["profile_id"] for r in rows if r.get("profile_id")})
    pet_ids = list({r["pet_id"] for r in rows if r.get("pet_id")})
    doctor_phones = list(
        {r["doctor_phone"] for r in rows if r.get("doctor_phone") and r["doctor_phone"] != "pending_doctor_choice"}
    )

    profiles_by_id: dict[str, dict[str, Any]] = {}
    if profile_ids:
        for p in client.table("profiles").select("*").in_("id", profile_ids).execute().data or []:
            profiles_by_id[p["id"]] = p

    pets_by_id: dict[str, dict[str, Any]] = {}
    if pet_ids:
        for p in client.table("pets").select("*").in_("id", pet_ids).execute().data or []:
            pets_by_id[p["id"]] = p

    doctors_by_phone: dict[str, dict[str, Any]] = {}
    if doctor_phones:
        for d in client.table("profiles").select("*").in_("phone_number", doctor_phones).eq("role", "vet").execute().data or []:
            doctors_by_phone[d["phone_number"]] = d

    appointments = []
    for row in rows:
        customer = profiles_by_id.get(row.get("profile_id"))
        pet = pets_by_id.get(row.get("pet_id"))
        doctor_profile = doctors_by_phone.get(row.get("doctor_phone"))
        appointments.append(
            {
                "session_id": row["id"],
                "customer_id": row.get("profile_id"),
                "customer_name": customer.get("full_name") if customer else None,
                "city": customer.get("city") if customer else None,
                "pet_name": pet.get("name") if pet else None,
                "breed": pet.get("breed") if pet else None,
                "age": pet.get("age") if pet else None,
                "age_group": _age_group(pet.get("age") if pet else None),
                "doctor_name": doctor_profile.get("full_name") if doctor_profile else None,
                "booked_at": row.get("created_at"),
                "completed_at": row.get("completed_at"),
                "status": row.get("status"),
                "case_summary": row.get("case_summary"),
                "prescription_status": _prescription_status(row),
                "follow_up_required": bool(row.get("follow_up_required")),
                "follow_up_date": row.get("follow_up_date"),
            }
        )

    if breed:
        needle = breed.lower()
        appointments = [a for a in appointments if needle in (a.get("breed") or "").lower()]
    if age_group:
        appointments = [a for a in appointments if a["age_group"].lower() == age_group.lower()]
    if city:
        needle = city.lower()
        appointments = [a for a in appointments if needle in (a.get("city") or "").lower()]
    if doctor:
        needle = doctor.lower()
        appointments = [a for a in appointments if needle in (a.get("doctor_name") or "").lower()]

    total_count = len(appointments)
    appointments = appointments[offset:offset + limit] if offset else appointments[:limit]

    return {"appointments": appointments, "count": total_count}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _month_start_iso() -> str:
    return datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


def _resolve_range(date_from: str, date_to: str) -> tuple[str, str]:
    """Both analytics endpoints share the same "default to the current
    calendar month, else use whatever explicit range was given" rule, so a
    bare page load still shows something meaningful before a user ever
    touches the date filter."""
    start = date_from or _month_start_iso()[:10]
    end = date_to or date.today().isoformat()
    return start, end


@router.get("/analytics/overview")
async def analytics_overview(request: Request, date_from: str = "", date_to: str = "") -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    range_start, range_end = _resolve_range(date_from, date_to)
    range_end_ts = f"{range_end}T23:59:59"

    total_customers = len(client.table("profiles").select("id").eq("role", "customer").execute().data or [])
    # Same 48h window that drives the "Active" customer-stage label
    # (_compute_customer_stage) -- kept in sync so this stat and the
    # Customers list agree on what "actively chatting" means.
    active_chat_cutoff = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    active_chatting_customers = len(
        client.table("profiles").select("id").eq("role", "customer")
        .gte("last_active_at", active_chat_cutoff).execute().data or []
    )

    new_signups = len(
        client.table("profiles").select("id").eq("role", "customer")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    # "Completed" here means status=='completed' -- consults_this_month used
    # to count status=='accepted' (upcoming/confirmed, not actually done),
    # which under-reported nothing but mislabeled what it was counting.
    sessions_in_range = (
        client.table("doctor_sessions").select("id, status, payment_status")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    completed_consultations = sum(1 for s in sessions_in_range if s.get("status") == "completed")
    pending_consultations = sum(1 for s in sessions_in_range if s.get("status") in ("pending", "negotiating", "accepted"))
    cancelled_consultations = sum(1 for s in sessions_in_range if s.get("status") in ("declined", "cancelled"))
    paid_consultations = sum(1 for s in sessions_in_range if s.get("payment_status") == "paid")
    consultation_revenue = paid_consultations * ctx.settings.razorpay_consult_fee_inr
    symptom_checks = len(
        client.table("health_logs").select("id").gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    documents_uploaded = len(
        client.table("documents").select("id").gte("uploaded_at", range_start).lte("uploaded_at", range_end_ts).execute().data or []
    )

    return {
        "date_from": range_start,
        "date_to": range_end,
        "total_customers": total_customers,
        "active_chatting_customers": active_chatting_customers,
        "new_signups": new_signups,
        "completed_consultations": completed_consultations,
        "pending_consultations": pending_consultations,
        "cancelled_consultations": cancelled_consultations,
        "paid_consultations": paid_consultations,
        "consultation_revenue": consultation_revenue,
        "symptom_checks": symptom_checks,
        "documents_uploaded": documents_uploaded,
    }


@router.get("/analytics/timeseries")
async def analytics_timeseries(request: Request, date_from: str = "", date_to: str = "", days: int = 30) -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    if date_from or date_to:
        since, until = _resolve_range(date_from, date_to)
    else:
        since = (date.today() - timedelta(days=days)).isoformat()
        until = date.today().isoformat()
    until_ts = f"{until}T23:59:59"

    signups = (
        client.table("profiles").select("created_at").eq("role", "customer")
        .gte("created_at", since).lte("created_at", until_ts).execute().data or []
    )
    paid_sessions = (
        client.table("doctor_sessions").select("created_at")
        .eq("payment_status", "paid").gte("created_at", since).lte("created_at", until_ts).execute().data or []
    )
    # Grouped by created_at (booked date), not completed_at -- completed_at
    # only exists for sessions completed after that column was added, so
    # grouping by it would show nothing for older data.
    completed_sessions = (
        client.table("doctor_sessions").select("created_at, status")
        .eq("status", "completed").gte("created_at", since).lte("created_at", until_ts).execute().data or []
    )

    signups_by_day: dict[str, int] = {}
    for row in signups:
        day = (row.get("created_at") or "")[:10]
        signups_by_day[day] = signups_by_day.get(day, 0) + 1

    revenue_by_day: dict[str, float] = {}
    for row in paid_sessions:
        day = (row.get("created_at") or "")[:10]
        revenue_by_day[day] = revenue_by_day.get(day, 0) + ctx.settings.razorpay_consult_fee_inr

    consultations_by_day: dict[str, int] = {}
    for row in completed_sessions:
        day = (row.get("created_at") or "")[:10]
        consultations_by_day[day] = consultations_by_day.get(day, 0) + 1

    return {
        "signups": [{"date": d, "count": c} for d, c in sorted(signups_by_day.items())],
        "revenue": [{"date": d, "amount": a} for d, a in sorted(revenue_by_day.items())],
        "consultations": [{"date": d, "count": c} for d, c in sorted(consultations_by_day.items())],
    }


# Canonical display order for the customer funnel -- roughly the order a
# customer moves through the journey, not alphabetical/count-sorted, so the
# funnel reads top-to-bottom the way the business actually flows.
STAGE_DISPLAY_ORDER = [
    "onboarding", "new", "booking", "negotiating", "payment",
    "upcoming", "prescription", "followup", "active", "inactive",
]
STAGE_LABELS = {
    "onboarding": "Onboarding incomplete", "new": "New / no activity yet", "booking": "Choosing/waiting on a vet",
    "negotiating": "Negotiating appointment time", "payment": "Awaiting payment", "upcoming": "Appointment upcoming",
    "prescription": "Awaiting prescription", "followup": "Follow-up needed", "active": "Active",
    "inactive": "Gone quiet",
}


@router.get("/analytics/reports")
async def analytics_reports(request: Request, date_from: str = "", date_to: str = "") -> dict[str, Any]:
    """Deeper cross-cutting reports beyond the top-line overview/timeseries:
    where customers are stuck (funnel), how bookings convert through the
    session state machine, revenue/churn by plan, and per-doctor throughput.
    Funnel and doctor-performance are point-in-time (current state, not
    date-ranged) -- "how many customers are stuck right now" isn't a range
    concept, matching how total_customers/active_subscribers already work
    in analytics_overview above."""
    ctx = _ctx(request)
    client = ctx.supabase
    range_start, range_end = _resolve_range(date_from, date_to)
    range_end_ts = f"{range_end}T23:59:59"

    # --- Customer funnel: current stage of every customer, reusing the same
    # stage logic as the Customers list/detail pages (see
    # _compute_customer_stage) so the two views can never disagree.
    all_customers = client.table("profiles").select("id,registration_step,last_active_at").eq("role", "customer").execute().data or []
    customer_ids = [c["id"] for c in all_customers]
    all_sessions_by_profile: dict[str, list[dict[str, Any]]] = {}
    if customer_ids:
        session_rows = (
            client.table("doctor_sessions")
            .select("status,doctor_phone,payment_status,awaiting_from,preferred_time,created_at,follow_up_required,follow_up_date,profile_id")
            .in_("profile_id", customer_ids).execute().data or []
        )
        for s in session_rows:
            all_sessions_by_profile.setdefault(s["profile_id"], []).append(s)
    funnel_counts: dict[str, int] = {code: 0 for code in STAGE_DISPLAY_ORDER}
    for c in all_customers:
        stage = _compute_customer_stage(c, all_sessions_by_profile.get(c["id"], []))
        funnel_counts[stage["code"]] = funnel_counts.get(stage["code"], 0) + 1
    customer_funnel = [
        {"code": code, "label": STAGE_LABELS[code], "count": funnel_counts[code]} for code in STAGE_DISPLAY_ORDER
    ]

    # --- Booking funnel: how sessions created in-range moved through the
    # state machine (see app/agent/tools/booking.py's docstring for the
    # full status list).
    sessions_in_range = (
        client.table("doctor_sessions").select("status,payment_status")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    booking_total = len(sessions_in_range)
    booking_by_status: dict[str, int] = {}
    for s in sessions_in_range:
        st = s.get("status") or "unknown"
        booking_by_status[st] = booking_by_status.get(st, 0) + 1
    completed_n = booking_by_status.get("completed", 0)
    declined_or_cancelled_n = booking_by_status.get("declined", 0) + booking_by_status.get("cancelled", 0)
    booking_funnel = {
        "total": booking_total,
        "by_status": [{"status": st, "count": n} for st, n in sorted(booking_by_status.items(), key=lambda kv: -kv[1])],
        "completed_rate": round(completed_n / booking_total, 3) if booking_total else 0,
        "declined_or_cancelled_rate": round(declined_or_cancelled_n / booking_total, 3) if booking_total else 0,
    }

    # --- Revenue: every booked session is a ₹399 consultation fee, no more
    # recurring plans -- revenue in range is just paid_count * the flat fee.
    payment_status_counts: dict[str, int] = {}
    for s in sessions_in_range:
        st = s.get("payment_status") or "none"
        payment_status_counts[st] = payment_status_counts.get(st, 0) + 1
    paid_in_range = payment_status_counts.get("paid", 0)
    revenue = {
        "paid_consultations": paid_in_range,
        "consultation_revenue_inr": paid_in_range * ctx.settings.razorpay_consult_fee_inr,
        "by_payment_status": [
            {"status": st, "count": n} for st, n in sorted(payment_status_counts.items(), key=lambda kv: -kv[1])
        ],
    }

    # --- Doctor performance: throughput per active vet, all-time (a new
    # doctor's one session shouldn't show a misleadingly perfect/empty rate
    # just because the date filter happens to exclude it).
    doctors = client.table("profiles").select("id,full_name,phone_number").eq("role", "vet").execute().data or []
    doctor_sessions_all = client.table("doctor_sessions").select("doctor_phone,status").execute().data or []
    sessions_by_doctor_phone: dict[str, list[dict[str, Any]]] = {}
    for s in doctor_sessions_all:
        phone = s.get("doctor_phone")
        if phone and phone not in ("broadcast", "pending_doctor_choice"):
            sessions_by_doctor_phone.setdefault(phone, []).append(s)
    doctor_performance = []
    for d in doctors:
        sessions = sessions_by_doctor_phone.get(d["phone_number"], [])
        total = len(sessions)
        completed = sum(1 for s in sessions if s.get("status") == "completed")
        active_now = sum(1 for s in sessions if s.get("status") in ACTIVE_SESSION_STATUSES)
        doctor_performance.append({
            "id": d["id"],
            "full_name": d.get("full_name") or "Unnamed",
            "total_sessions": total,
            "completed_sessions": completed,
            "completion_rate": round(completed / total, 3) if total else 0,
            "active_now": active_now,
        })
    doctor_performance.sort(key=lambda d: -d["total_sessions"])

    return {
        "date_from": range_start,
        "date_to": range_end,
        "customer_funnel": customer_funnel,
        "booking_funnel": booking_funnel,
        "revenue": revenue,
        "doctor_performance": doctor_performance,
    }


@router.get("/onboarding-funnel")
async def onboarding_funnel(request: Request) -> dict[str, Any]:
    """Dedicated onboarding-step funnel dashboard (2026-08 root-cause item
    #2) -- distinct from analytics_reports' customer_funnel (current stage
    of every customer across the WHOLE lifecycle, including post-signup
    booking stages). This one is specifically about the 3-question wizard:
    how many profiles are sitting at each step right now, how often each
    step's validator accepts vs. rejects a reply (from onboarding_events),
    and how long profiles that DID move on typically took to advance past
    each step (from registration_step_history) -- so a stuck cohort and a
    slow-but-not-stuck cohort don't look the same."""
    ctx = _ctx(request)
    client = ctx.supabase

    profiles = client.table("profiles").select("registration_step").eq("role", "customer").execute().data or []
    step_counts: dict[str, int] = {}
    for p in profiles:
        step = p.get("registration_step") or "completed"
        step_counts[step] = step_counts.get(step, 0) + 1

    events = client.table("onboarding_events").select("registration_step,validator_result").execute().data or []
    validator_stats: dict[str, dict[str, int]] = {}
    for e in events:
        step = e["registration_step"]
        bucket = validator_stats.setdefault(step, {"accepted": 0, "rejected": 0})
        bucket[e["validator_result"]] = bucket.get(e["validator_result"], 0) + 1

    history = (
        client.table("registration_step_history").select("profile_id,from_step,to_step,created_at")
        .order("created_at").execute().data or []
    )
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_profile.setdefault(row["profile_id"], []).append(row)

    # Time-in-step: for each profile's ordered transitions, the gap between
    # entering `from_step` and leaving it (the next row's created_at minus
    # this row's created_at) attributes that duration to `from_step`.
    durations_by_step: dict[str, list[float]] = {}
    for rows in by_profile.values():
        for i in range(len(rows) - 1):
            step = rows[i]["to_step"]  # the step they were IN during this gap
            if not step:
                continue
            try:
                started = datetime.fromisoformat(rows[i]["created_at"].replace("Z", "+00:00"))
                ended = datetime.fromisoformat(rows[i + 1]["created_at"].replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            durations_by_step.setdefault(step, []).append((ended - started).total_seconds())

    all_steps = sorted(set(step_counts) | set(validator_stats) | set(durations_by_step) - {"completed"})
    funnel = []
    for step in all_steps:
        durations = durations_by_step.get(step, [])
        stats = validator_stats.get(step, {"accepted": 0, "rejected": 0})
        funnel.append(
            {
                "step": step,
                "profiles_currently_here": step_counts.get(step, 0),
                "accepted_replies": stats.get("accepted", 0),
                "rejected_replies": stats.get("rejected", 0),
                "avg_seconds_in_step": round(sum(durations) / len(durations), 1) if durations else None,
                "profiles_advanced_past": len(durations),
            }
        )

    return {
        "funnel": funnel,
        "completed_count": step_counts.get("completed", 0),
        "total_customers": len(profiles),
    }


# ---------------------------------------------------------------------------
# Platform costs -- manual-entry spend/credits tracking per external
# platform this project runs on (OpenAI, Railway, Supabase, Vercel, ...).
# None of these expose a usage/billing API we're authenticated against yet
# (each needs its own separate admin-scoped token), so this is admin-entered
# from each platform's own dashboard rather than live-fetched -- upsert by
# platform_name means adding a new platform (e.g. Lovable, Meta/WhatsApp
# billing, later) needs no code change, just a new row through this same
# endpoint.
# ---------------------------------------------------------------------------

@router.get("/platform-costs")
async def list_platform_costs(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    rows = ctx.supabase.table("platform_costs").select("*").execute().data or []
    rows.sort(key=lambda r: r["platform_name"])
    return {"platforms": rows}


@router.put("/platform-costs/{platform_name}")
async def upsert_platform_cost(platform_name: str, request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    ctx = _ctx(request)
    fields = {k: v for k, v in payload.items() if k in ("amount_spent", "credits_remaining", "currency", "notes")}
    row = (
        ctx.supabase.table("platform_costs")
        .upsert(
            {"platform_name": platform_name, **fields, "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="platform_name",
        )
        .execute()
        .data[0]
    )
    return {"success": True, "platform": row}


@router.delete("/platform-costs/{platform_id}")
async def delete_platform_cost(platform_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    ctx.supabase.table("platform_costs").delete().eq("id", platform_id).execute()
    return {"success": True}


# ---------------------------------------------------------------------------
# P&L -- revenue (paid ₹399 consultation fees, booked in INR like the rest
# of this dashboard) vs. platform costs (platform_costs' running totals,
# mostly billed in USD). These aren't the same currency, so a single "Net"
# figure needs a conversion -- USD_TO_INR_APPROX is a fixed indicative
# rate, not a live FX lookup, and every combined figure is explicitly
# suffixed "_approx" so nothing here is mistaken for accounting-grade
# output. The by-currency breakdown is always returned too, so an admin
# who doesn't trust the approximation can just read the real numbers
# directly.
# ---------------------------------------------------------------------------

USD_TO_INR_APPROX = 83.0


@router.get("/analytics/pnl")
async def analytics_pnl(request: Request, date_from: str = "", date_to: str = "") -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    range_start, range_end = _resolve_range(date_from, date_to)
    range_end_ts = f"{range_end}T23:59:59"

    # Revenue booked in the selected period -- every consultation that was
    # actually paid for in-range, at the flat ₹399 fee.
    paid_sessions_in_range = (
        client.table("doctor_sessions").select("id")
        .eq("payment_status", "paid")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    revenue_in_range_inr = len(paid_sessions_in_range) * ctx.settings.razorpay_consult_fee_inr

    # platform_costs.amount_spent is a running total the admin updates from
    # each platform's own dashboard -- NOT scoped to date_from/date_to (that
    # table has no period concept), so this compares a period's revenue
    # against an all-time cost snapshot, not a strict apples-to-apples
    # period P&L. Labeled as such in the response and the UI.
    platform_rows = client.table("platform_costs").select("platform_name,amount_spent,currency").execute().data or []
    costs_by_currency: dict[str, float] = {}
    for p in platform_rows:
        cur = p.get("currency") or "USD"
        costs_by_currency[cur] = costs_by_currency.get(cur, 0) + (p.get("amount_spent") or 0)

    total_costs_inr_approx = sum(
        (amount * USD_TO_INR_APPROX if cur == "USD" else amount) for cur, amount in costs_by_currency.items()
    )

    return {
        "date_from": range_start,
        "date_to": range_end,
        "revenue_in_range_inr": revenue_in_range_inr,
        "costs_by_currency": [{"currency": c, "amount": a} for c, a in sorted(costs_by_currency.items())],
        "total_costs_inr_approx": round(total_costs_inr_approx, 2),
        "net_in_range_inr_approx": round(revenue_in_range_inr - total_costs_inr_approx, 2),
        "fx_rate_used": USD_TO_INR_APPROX,
    }


# ---------------------------------------------------------------------------
# Onboarding instrumentation (2026-08-04 re-engagement workstream, item 2) --
# per-step accept/reject counts from onboarding_events, so we can see
# whether a specific question is failing users repeatedly (high rejection
# rate) vs. them just going silent (few events logged relative to how many
# profiles are parked at that step).
# ---------------------------------------------------------------------------

@router.get("/onboarding-events/summary")
async def onboarding_event_summary(request: Request, date_from: str = "", date_to: str = "") -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    range_start, range_end = _resolve_range(date_from, date_to)
    range_end_ts = f"{range_end}T23:59:59"

    events = (
        client.table("onboarding_events").select("registration_step,validator_result,rejection_reason")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )

    by_step: dict[str, dict[str, Any]] = {}
    for e in events:
        step = e.get("registration_step") or "unknown"
        entry = by_step.setdefault(step, {"registration_step": step, "accepted": 0, "rejected": 0, "rejection_reasons": {}})
        if e.get("validator_result") == "rejected":
            entry["rejected"] += 1
            reason = e.get("rejection_reason") or "unknown"
            entry["rejection_reasons"][reason] = entry["rejection_reasons"].get(reason, 0) + 1
        else:
            entry["accepted"] += 1

    rows = []
    for entry in by_step.values():
        total = entry["accepted"] + entry["rejected"]
        rows.append(
            {
                **entry,
                "total_events": total,
                "rejection_rate": round(entry["rejected"] / total, 3) if total else 0,
                "rejection_reasons": [{"reason": r, "count": c} for r, c in sorted(entry["rejection_reasons"].items(), key=lambda kv: -kv[1])],
            }
        )
    rows.sort(key=lambda r: -r["rejection_rate"])

    return {"date_from": range_start, "date_to": range_end, "by_step": rows}


# ---------------------------------------------------------------------------
# Emergency-escalation human check-in queue (2026-08-04 re-engagement
# workstream, item 4) -- health_logs rows flagged by
# app.scheduler.jobs.flag_emergency_checkins because a RED/emergency
# check_symptoms result got no customer follow-up within the review window.
# Deliberately a manual queue, not another automated send.
# ---------------------------------------------------------------------------

@router.get("/human-checkin-queue")
async def human_checkin_queue(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase

    rows = (
        client.table("health_logs")
        .select("id,pet_id,profile_id,ai_risk_score,ai_observation,symptoms,created_at,human_checkin_flagged_at")
        .eq("needs_human_checkin", True)
        .is_("human_checkin_resolved_at", "null")
        .order("human_checkin_flagged_at", desc=False)
        .execute()
        .data
        or []
    )

    pet_ids = [r["pet_id"] for r in rows if r.get("pet_id")]
    profile_ids = [r["profile_id"] for r in rows if r.get("profile_id")]
    pets_by_id = {
        p["id"]: p for p in (client.table("pets").select("id,name,species").in_("id", pet_ids).execute().data or [])
    } if pet_ids else {}
    profiles_by_id = {
        p["id"]: p for p in (client.table("profiles").select("id,full_name,phone_number").in_("id", profile_ids).execute().data or [])
    } if profile_ids else {}

    queue = []
    for r in rows:
        pet = pets_by_id.get(r.get("pet_id"), {})
        profile = profiles_by_id.get(r.get("profile_id"), {})
        queue.append(
            {
                "health_log_id": r["id"],
                "pet_name": pet.get("name"),
                "species": pet.get("species"),
                "customer_name": profile.get("full_name"),
                "phone_number": profile.get("phone_number"),
                "symptom_summary": r.get("symptoms"),
                "ai_observation": r.get("ai_observation"),
                "ai_risk_score": r.get("ai_risk_score"),
                "escalation_at": r["created_at"],
                "flagged_at": r.get("human_checkin_flagged_at"),
            }
        )

    return {"count": len(queue), "queue": queue}


@router.post("/human-checkin-queue/{health_log_id}/resolve")
async def resolve_human_checkin(health_log_id: str, request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    ctx.supabase.table("health_logs").update(
        {"human_checkin_resolved_at": datetime.now(tz=timezone.utc).isoformat()}
    ).eq("id", health_log_id).execute()
    return {"success": True}
