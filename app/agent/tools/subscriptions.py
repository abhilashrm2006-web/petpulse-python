"""Shared Subscriber-signup logic, used both by the deterministic
registration wizard (app/ingestion/registration.py, tier choice at the end
of first-contact intake) and by start_subscription (an ordinary agent
tool, for an already-registered Free customer upgrading mid-conversation).
Both paths must produce the exact same subscriptions row and the exact
same Razorpay Subscription -- this is the one place that logic lives."""

import logging
from datetime import date
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.integrations import razorpay_client

logger = logging.getLogger(__name__)

# Founding Member cohort: the first FOUNDING_MEMBER_CAP customers to subscribe
# get grandfathered at ₹99/month for full Subscriber access, forever -- capped
# by signup count (not a date cutoff), tracked via profiles.is_founding_member
# so it can't silently keep growing.
FOUNDING_MEMBER_CAP = 500
FOUNDING_MEMBER_AMOUNT = 99
STANDARD_AMOUNT = 399


def _founding_slots_available(client) -> bool:
    rows = client.table("profiles").select("id").eq("is_founding_member", True).execute().data or []
    return len(rows) < FOUNDING_MEMBER_CAP


async def create_and_send_subscription(ctx: AppContext, profile: dict[str, Any]) -> dict[str, Any]:
    """Best-effort: a Razorpay/storage failure here must never crash the
    caller (the registration wizard or the agent tool-calling loop) --
    log and tell the customer plainly, don't leave them without an
    explanation."""
    phone = profile["phone_number"]
    is_founding = _founding_slots_available(ctx.supabase)
    plan_id = ctx.settings.razorpay_founding_plan_id if is_founding else None
    amount = FOUNDING_MEMBER_AMOUNT if is_founding else STANDARD_AMOUNT

    try:
        subscription = await razorpay_client.create_subscription(
            ctx.settings,
            customer_name=profile.get("full_name") or "",
            customer_phone=phone,
            reference_id=profile["id"],
            plan_id=plan_id,
        )
    except Exception:
        logger.exception("Failed to create Razorpay subscription for profile %s", profile["id"])
        await ctx.whatsapp.send_text(phone, "Sorry, something went wrong setting up your subscription. Please try again in a moment.")
        return {"success": False, "error": "subscription_creation_failed"}

    ctx.supabase.table("subscriptions").insert(
        {
            "profile_id": profile["id"],
            # plan_name/billing_cycle/status are CHECK-constrained on the live
            # table to these exact values (confirmed live): plan_name in
            # Free/Basic/Premium/Family/Enterprise, billing_cycle in
            # Monthly/Quarterly/Yearly, status in
            # trial/active/expired/cancelled/paused. "trial" here means
            # "signup started, not yet confirmed" -- is_active_subscriber
            # only ever treats status="active" (set by the
            # subscription.activated/charged webhook) as a real Subscriber.
            # Founding Members reuse plan_name="Premium" too (same feature
            # access, see is_active_subscriber) -- only amount/plan_id differ.
            "plan_name": "Premium",
            "billing_cycle": "Monthly",
            "amount": amount,
            "currency": "INR",
            "status": "trial",
            "payment_provider": "razorpay",
            "provider_subscription_id": subscription.get("id"),
            "start_date": date.today().isoformat(),
        }
    ).execute()

    if is_founding:
        ctx.supabase.table("profiles").update({"is_founding_member": True}).eq("id", profile["id"]).execute()

    short_url = subscription.get("short_url", "")
    price_line = f"₹{FOUNDING_MEMBER_AMOUNT}/month Founding Member" if is_founding else f"₹{STANDARD_AMOUNT}/month"
    await ctx.whatsapp.send_text(
        phone,
        f"Almost there! Complete your PetPulse Subscriber signup ({price_line}) here:\n"
        f"{short_url}\n\nYou're all set to chat in the meantime — ask me anything about your pet!",
    )
    return {
        "success": True,
        "mode": "subscription_started",
        "provider_subscription_id": subscription.get("id"),
        "is_founding_member": is_founding,
    }


async def start_subscription(ctx: AppContext, agent_ctx: AgentContext) -> dict[str, Any]:
    """Agent tool: lets an already-registered Free customer upgrade mid-
    conversation, not just during the first-contact wizard."""
    return await create_and_send_subscription(ctx, agent_ctx.profile)
