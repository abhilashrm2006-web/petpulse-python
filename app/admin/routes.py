"""Admin dashboard API -- customer/doctor management + business analytics.
Mounted under /admin/* in app/main.py behind require_admin_token (app/admin/auth.py).
Deactivate/delete reuse the same domain logic the WhatsApp agent uses
(cancel_session, cancel_subscription) rather than raw table writes, so an
admin action leaves the exact same trail (cancelled calendar events,
notified parties) a customer-initiated one would."""

import logging
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.agent.tools.booking import cancel_session
from app.deps import AppContext
from app.integrations import razorpay_client

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
        pattern = f"%{search}%"
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

@router.get("/doctors")
async def list_doctors(request: Request, search: str = "") -> dict[str, Any]:
    ctx = _ctx(request)
    query = ctx.supabase.table("profiles").select("*").eq("role", "vet")
    if search:
        pattern = f"%{search}%"
        query = query.or_(f"full_name.ilike.{pattern},phone_number.ilike.{pattern},specialization.ilike.{pattern}")
    rows = query.order("created_at", desc=True).execute().data or []
    return {"doctors": rows, "count": len(rows)}


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

    row = (
        ctx.supabase.table("profiles")
        .insert(
            {
                "full_name": full_name,
                "phone_number": phone_number,
                "role": "vet",
                "is_active": True,
                "qualification": payload.get("qualification"),
                "registration_number": payload.get("registration_number"),
                "specialization": payload.get("specialization"),
                "experience_years": payload.get("experience_years"),
                "consultation_fee": payload.get("consultation_fee"),
            }
        )
        .execute()
        .data[0]
    )
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
    return {"profile": profile, "recent_sessions": sessions}


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
# Analytics
# ---------------------------------------------------------------------------

def _month_start_iso() -> str:
    return datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()


@router.get("/analytics/overview")
async def analytics_overview(request: Request) -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    month_start = _month_start_iso()

    total_customers = len(client.table("profiles").select("id").eq("role", "customer").execute().data or [])
    active_subs = client.table("subscriptions").select("amount").eq("status", "active").execute().data or []
    founding_count = len(client.table("profiles").select("id").eq("is_founding_member", True).execute().data or [])

    new_signups = len(
        client.table("profiles").select("id").eq("role", "customer").gte("created_at", month_start).execute().data or []
    )
    consults_this_month = len(
        client.table("doctor_sessions").select("id").eq("status", "accepted").gte("created_at", month_start).execute().data or []
    )
    symptom_checks_this_month = len(
        client.table("health_logs").select("id").gte("created_at", month_start).execute().data or []
    )
    documents_this_month = len(
        client.table("documents").select("id").gte("uploaded_at", month_start).execute().data or []
    )

    return {
        "total_customers": total_customers,
        "active_subscribers": len(active_subs),
        "founding_members": founding_count,
        "standard_subscribers": len(active_subs) - founding_count,
        "estimated_mrr": sum(s.get("amount") or 0 for s in active_subs),
        "new_signups_this_month": new_signups,
        "consults_this_month": consults_this_month,
        "symptom_checks_this_month": symptom_checks_this_month,
        "documents_this_month": documents_this_month,
    }


@router.get("/analytics/timeseries")
async def analytics_timeseries(request: Request, days: int = 30) -> dict[str, Any]:
    ctx = _ctx(request)
    client = ctx.supabase
    since = (date.today() - timedelta(days=days)).isoformat()

    signups = client.table("profiles").select("created_at").eq("role", "customer").gte("created_at", since).execute().data or []
    subscriptions = client.table("subscriptions").select("created_at, amount").gte("created_at", since).execute().data or []

    signups_by_day: dict[str, int] = {}
    for row in signups:
        day = (row.get("created_at") or "")[:10]
        signups_by_day[day] = signups_by_day.get(day, 0) + 1

    revenue_by_day: dict[str, float] = {}
    for row in subscriptions:
        day = (row.get("created_at") or "")[:10]
        revenue_by_day[day] = revenue_by_day.get(day, 0) + (row.get("amount") or 0)

    return {
        "signups": [{"date": d, "count": c} for d, c in sorted(signups_by_day.items())],
        "revenue": [{"date": d, "amount": a} for d, a in sorted(revenue_by_day.items())],
    }
