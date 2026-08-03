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


async def cancel_subscription(settings: Settings, provider_subscription_id: str) -> dict[str, Any]:
    """Cancels immediately (cancel_at_cycle_end=0) -- used by the admin
    dashboard's customer-deactivate action (app/admin/routes.py), where the
    intent is "stop this account's access now," not "let them finish out
    the billing cycle they already paid for.\""""
    async with httpx.AsyncClient(timeout=30.0, auth=(settings.razorpay_key_id, settings.razorpay_key_secret)) as client:
        resp = await client.post(
            f"{RAZORPAY_API_BASE}/subscriptions/{provider_subscription_id}/cancel",
            json={"cancel_at_cycle_end": 0},
        )
        resp.raise_for_status()
        return resp.json()


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
