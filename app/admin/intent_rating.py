"""LLM classification of a customer's purchase/engagement intent from
their WhatsApp chat history, for the admin dashboard's Customers page
(rate + filter). Kept out of app/agent/ on purpose -- this is an
admin-only lens on existing conversations, not something the bot itself
consults when deciding how to respond."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.integrations.openai_client import json_completion

logger = logging.getLogger(__name__)

INTENT_RATINGS = ("High", "Medium", "Low")

INTENT_SYSTEM_PROMPT = """You are classifying a pet owner's purchase/engagement intent from their WhatsApp \
conversation with a veterinary bot, for an admin trying to prioritize who to follow up with. Rate their intent \
as exactly one of "High", "Medium", or "Low":

- High: actively tried to book a vet consultation, asked about pricing/subscription plans or how to become a \
paying member, or described urgent/serious symptoms that need a vet now.
- Medium: multiple back-and-forth exchanges about their pet's health or care, follow-up questions, real \
engagement -- but no clear signal yet of wanting to book or pay.
- Low: a one-off or very short interaction (e.g. just onboarding, a single greeting, or one casual question) \
with no sign of wanting a paid consultation or continued engagement.

Return strict JSON: {"rating": "High"|"Medium"|"Low", "reason": "one sentence explaining why, referencing what \
they actually said"}"""

FALLBACK_REASON = "Could not complete a full assessment -- treated as Low pending a retry."


async def rate_customer_intent(
    openai_client: AsyncOpenAI, settings: Settings, messages: list[dict[str, Any]]
) -> dict[str, str]:
    if not messages:
        return {"rating": "Low", "reason": "No conversation history yet."}

    transcript = "\n".join(
        f"{m.get('sender_type', 'user')}: {m['content']}" for m in messages if m.get("content")
    )
    if not transcript.strip():
        return {"rating": "Low", "reason": "No text content in conversation history."}

    try:
        raw = await json_completion(openai_client, settings, INTENT_SYSTEM_PROMPT, transcript, reasoning_effort="low")
        data = json.loads(raw)
        rating = data.get("rating")
        if rating not in INTENT_RATINGS:
            raise ValueError(f"unexpected rating {rating!r}")
        return {"rating": rating, "reason": (data.get("reason") or "").strip()}
    except Exception:
        logger.exception("Failed to rate customer intent")
        return {"rating": "Low", "reason": FALLBACK_REASON}
