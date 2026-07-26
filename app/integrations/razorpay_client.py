"""Thin Razorpay client for gating a booking on payment. The consult fee is
fixed, so rather than fighting the browser-only Payment Button embed (a
`<script>`/`<form>` pair that only works on a webpage, not inside a WhatsApp
chat), we use the Payment Links API to generate a per-session, tappable
checkout URL and send that as a plain WhatsApp message. Confirmation of
payment arrives later, asynchronously, via Razorpay's webhook — never assume
a link was paid just because it was sent."""

import hashlib
import hmac
from typing import Any

import httpx

from app.config import Settings

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"


async def create_payment_link(
    settings: Settings,
    *,
    amount_inr: int,
    reference_id: str,
    description: str,
    customer_name: str,
    customer_phone: str,
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0, auth=(settings.razorpay_key_id, settings.razorpay_key_secret)) as client:
        resp = await client.post(
            f"{RAZORPAY_API_BASE}/payment_links",
            json={
                "amount": amount_inr * 100,
                "currency": "INR",
                "reference_id": reference_id,
                "description": description,
                "customer": {"name": customer_name, "contact": customer_phone},
                "notify": {"sms": False, "email": False},
                "notes": {"session_id": reference_id},
            },
        )
        resp.raise_for_status()
        return resp.json()


async def create_subscription(
    settings: Settings,
    *,
    customer_name: str,
    customer_phone: str,
    reference_id: str,
) -> dict[str, Any]:
    """Creates a Razorpay Subscription against the pre-created monthly Plan
    (Settings.razorpay_subscription_plan_id) -- a different API from
    create_payment_link's one-off links, since this bills on a recurring
    cadence. Razorpay has no "no end date" option; total_count=100 (~8
    years of monthly billing) stands in for "until cancelled" -- the actual
    end is whatever subscription.cancelled reports via the webhook, not
    this count running out."""
    async with httpx.AsyncClient(timeout=30.0, auth=(settings.razorpay_key_id, settings.razorpay_key_secret)) as client:
        resp = await client.post(
            f"{RAZORPAY_API_BASE}/subscriptions",
            json={
                "plan_id": settings.razorpay_subscription_plan_id,
                "customer_notify": 1,
                "total_count": 100,
                "notify_info": {"notify_phone": customer_phone},
                "notes": {"reference_id": reference_id, "customer_name": customer_name},
            },
        )
        resp.raise_for_status()
        return resp.json()


def extract_subscription_event(event_body: dict[str, Any]) -> tuple[str, str, str | None] | None:
    """Returns (event_type, subscription_id, reference_id) for any
    subscription.* webhook event (activated/charged/cancelled/etc.), or
    None if this event isn't one -- reference_id is whatever profile
    create_subscription tagged it with in `notes`."""
    event = event_body.get("event", "")
    if not event.startswith("subscription."):
        return None
    try:
        entity = event_body["payload"]["subscription"]["entity"]
        return event, entity["id"], (entity.get("notes") or {}).get("reference_id")
    except (KeyError, TypeError):
        return None


def verify_webhook_signature(settings: Settings, raw_body: bytes, signature_header: str | None) -> bool:
    if not signature_header:
        return False
    expected = hmac.new(settings.razorpay_webhook_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def extract_paid_session_id(event_body: dict[str, Any]) -> str | None:
    """`reference_id` on the payment_link IS the session_id — we set it that
    way in create_payment_link, so no separate correlation table is needed."""
    try:
        if event_body.get("event") != "payment_link.paid":
            return None
        return event_body["payload"]["payment_link"]["entity"]["reference_id"]
    except (KeyError, TypeError):
        return None
