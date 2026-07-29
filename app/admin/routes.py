"""Admin dashboard API -- customer/doctor management + business analytics.
Mounted under /admin/* in app/main.py behind require_admin_token (app/admin/auth.py).
Deactivate/delete reuse the same domain logic the WhatsApp agent uses
(cancel_session, cancel_subscription) rather than raw table writes, so an
admin action leaves the exact same trail (cancelled calendar events,
notified parties) a customer-initiated one would."""

import logging
import uuid
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.agent.tools.booking import cancel_session
from app.deps import AppContext
from app.integrations import razorpay_client
from app.integrations.supabase_client import escape_or_filter_value, sign_storage_url, upload_to_storage

DOCTOR_DOCUMENTS_BUCKET = "doctor-documents"

logger = logging.getLogger(__name__)

router = APIRouter()

ACTIVE_SESSION_STATUSES = ["pending", "negotiating", "accepted"]


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
    """Categorizes a customer for the admin list/filter -- mirrors the same
    tiers the WhatsApp bot itself gates on (is_active_subscriber,
    is_founding_member), so "Subscriber" here means the exact same thing
    it means to the bot, not a separate admin-only notion of status."""
    if not subscription:
        return "Free"
    if subscription.get("status") == "active":
        return "Founding" if profile.get("is_founding_member") else "Subscriber"
    if subscription.get("status") == "trial":
        return "Trial"
    return "Free"


@router.get("/customers")
async def list_customers(
    request: Request,
    search: str = "",
    date_from: str = "",
    date_to: str = "",
    tier: str = "",
    breed: str = "",
    status: str = "",
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
    if date_from:
        query = query.gte("created_at", date_from)
    if date_to:
        # Inclusive of the whole "to" day -- a bare date string sorts before
        # any timestamp on that same day, so a plain lte would exclude it.
        query = query.lte("created_at", f"{date_to}T23:59:59")
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
        ctx.supabase.table("doctor_sessions").select("id, status, preferred_time").eq("profile_id", profile_id)
        .order("created_at", desc=True).limit(10).execute().data or []
    )
    pet_ids = [p["id"] for p in pets]
    document_count = len(ctx.supabase.table("documents").select("id").in_("pet_id", pet_ids).execute().data or []) if pet_ids else 0

    return {
        "profile": profile,
        "pets": pets,
        "subscription": subscription_rows[0] if subscription_rows else None,
        "recent_sessions": sessions,
        "document_count": document_count,
    }


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

    existing = ctx.supabase.table("profiles").select("id").eq("phone_number", phone_number).limit(1).execute().data
    if existing:
        raise HTTPException(status_code=409, detail="A profile with this phone number already exists")

    doctor = await _create_doctor_profile(
        ctx,
        {
            "full_name": full_name,
            "phone_number": phone_number,
            "email": draft.get("extracted_email"),
            "qualification": draft.get("extracted_qualification"),
            "registration_number": draft.get("extracted_registration_number"),
            "specialization": draft.get("extracted_specialization"),
            "gender": draft.get("extracted_gender"),
            "date_of_birth": draft.get("extracted_date_of_birth"),
            "city": draft.get("extracted_city"),
            "area": draft.get("extracted_area"),
        },
    )

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
    active_subs = client.table("subscriptions").select("amount").eq("status", "active").execute().data or []
    founding_count = len(client.table("profiles").select("id").eq("is_founding_member", True).execute().data or [])

    new_signups = len(
        client.table("profiles").select("id").eq("role", "customer")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    # "Completed" here means status=='completed' -- consults_this_month used
    # to count status=='accepted' (upcoming/confirmed, not actually done),
    # which under-reported nothing but mislabeled what it was counting.
    sessions_in_range = (
        client.table("doctor_sessions").select("id, status")
        .gte("created_at", range_start).lte("created_at", range_end_ts).execute().data or []
    )
    completed_consultations = sum(1 for s in sessions_in_range if s.get("status") == "completed")
    pending_consultations = sum(1 for s in sessions_in_range if s.get("status") in ("pending", "negotiating", "accepted"))
    cancelled_consultations = sum(1 for s in sessions_in_range if s.get("status") in ("declined", "cancelled"))
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
        "active_subscribers": len(active_subs),
        "founding_members": founding_count,
        "standard_subscribers": len(active_subs) - founding_count,
        "estimated_mrr": sum(s.get("amount") or 0 for s in active_subs),
        "new_signups": new_signups,
        "completed_consultations": completed_consultations,
        "pending_consultations": pending_consultations,
        "cancelled_consultations": cancelled_consultations,
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
    subscriptions = (
        client.table("subscriptions").select("created_at, amount")
        .gte("created_at", since).lte("created_at", until_ts).execute().data or []
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
    for row in subscriptions:
        day = (row.get("created_at") or "")[:10]
        revenue_by_day[day] = revenue_by_day.get(day, 0) + (row.get("amount") or 0)

    consultations_by_day: dict[str, int] = {}
    for row in completed_sessions:
        day = (row.get("created_at") or "")[:10]
        consultations_by_day[day] = consultations_by_day.get(day, 0) + 1

    return {
        "signups": [{"date": d, "count": c} for d, c in sorted(signups_by_day.items())],
        "revenue": [{"date": d, "amount": a} for d, a in sorted(revenue_by_day.items())],
        "consultations": [{"date": d, "count": c} for d, c in sorted(consultations_by_day.items())],
    }
