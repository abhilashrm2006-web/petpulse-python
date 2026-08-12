"""Deterministic first-contact registration wizard: greeting, then three
questions (owner name, pet name, city) -- identical for every customer,
no new-vs-existing branching. Runs entirely outside the LLM agent loop
(see app/main.py, which calls handle_registration before
build_context/run_agent_turn) so a brand-new customer's registration can
never be skipped, reordered, or misinterpreted by the model. Once
profiles.registration_step is "completed" (or the profile predates this
wizard, or isn't role=customer -- e.g. a vet), this module returns False
immediately and the existing agent pipeline runs completely unchanged.
"""

import logging
from typing import Any

from app.agent.conversation_log import log_conversation_turn
from app.deps import AppContext
from app.ingestion.webhook import ExtractedMessage
from app.integrations.supabase_client import get_profile_by_phone, is_unique_violation

logger = logging.getLogger(__name__)


def _log_step_transition(client, profile_id: str, from_step: str | None, to_step: str | None) -> None:
    """Best-effort audit trail of every registration_step change -- unlike
    profiles.registration_step (overwritten in place), this is append-only,
    so time-to-abandon per step can be measured directly instead of
    inferred from last_active_at being blank (2026-08 root-cause item #2)."""
    try:
        client.table("registration_step_history").insert(
            {"profile_id": profile_id, "from_step": from_step, "to_step": to_step}
        ).execute()
    except Exception:
        logger.exception("Failed to log registration_step_history for profile=%s", profile_id)


def _log_turn(client, profile: dict[str, Any], extracted: ExtractedMessage, outbound_texts: list[str]) -> None:
    log_conversation_turn(
        client,
        profile_id=profile["id"],
        pet_id=None,
        sender_type="user",
        inbound_text=extracted.text,
        inbound_message_type=extracted.message_type,
        inbound_wamid=extracted.message_id,
        outbound_texts_with_wamid=[(None, text) for text in outbound_texts],
    )


def _log_onboarding_event(
    client, profile_id: str, step: str, raw_input: str, accepted: bool, reason: str | None = None
) -> None:
    """Best-effort instrumentation for the two steps that previously had no
    message-level visibility at all (see 2026-08-04 root-cause CSV): lets us
    finally tell "replied and got rejected by the validator" apart from
    "never replied," and surfaces a per-step rejection-rate metric (see
    app/admin/routes.py onboarding_event_summary). Wrapped in try/except so
    a missing onboarding_events table (e.g. before the migration has been
    applied) never breaks the actual registration flow -- this is
    observability, not a load-bearing part of the wizard."""
    try:
        client.table("onboarding_events").insert(
            {
                "profile_id": profile_id,
                "registration_step": step,
                "raw_input": raw_input,
                "validator_result": "accepted" if accepted else "rejected",
                "rejection_reason": reason,
            }
        ).execute()
    except Exception:
        logger.exception("Failed to log onboarding_event for profile=%s step=%s", profile_id, step)


def _looks_like_name(text: str) -> bool:
    """Real names are short and don't read like a sentence. Confirmed live
    (2026-07-31): customers who type a complaint/question/off-topic remark at
    the name/pet-name step ("So many we have 12 dogs", "Sorry im a
    veterinarian", "No Hindi language?") were being saved verbatim as the
    name and silently advancing the wizard -- this is a cheap guard against
    that, not real NLP. False positives (a genuinely long name) just get a
    one-time re-prompt, not a hard block."""
    text = text.strip()
    if not text or len(text) > 40:
        return False
    if "?" in text:
        return False
    if len(text.split()) > 3:
        return False
    return True


async def handle_registration(ctx: AppContext, extracted: ExtractedMessage) -> bool:
    """Returns True if this message was consumed by the registration wizard
    (caller must return 200 and do nothing else). False means the profile
    is already fully registered, or isn't a customer (e.g. a vet) -- proceed
    to build_context/run_agent_turn exactly as before this module existed."""
    client = ctx.supabase
    phone = extracted.phone_number

    profile = get_profile_by_phone(client, phone)

    if profile is None:
        profile = (
            client.table("profiles")
            .insert(
                {
                    "phone_number": phone,
                    "full_name": extracted.sender_name or "",
                    "role": "customer",
                    "registration_step": "awaiting_customer_name",
                }
            )
            .execute()
            .data[0]
        )
        greeting = (
            "Hi this is Pulsy, welcome to PetPulse \U0001F43E\n"
            "नमस्ते! मैं Pulsy हूं, PetPulse में आपका स्वागत है \U0001F43E"
        )
        name_prompt = "What's your full name?"
        await ctx.whatsapp.send_text(phone, greeting)
        await ctx.whatsapp.send_text(phone, name_prompt)
        _log_step_transition(client, profile["id"], None, "awaiting_customer_name")
        _log_turn(client, profile, extracted, [greeting, name_prompt])
        return True

    step = profile.get("registration_step")
    if not step or step == "completed" or profile.get("role") != "customer":
        return False

    if step == "awaiting_customer_name":
        return await _handle_customer_name(ctx, profile, extracted)
    if step == "awaiting_pet_name":
        return await _handle_pet_name(ctx, profile, extracted)
    if step == "awaiting_city":
        return await _handle_city(ctx, profile, extracted)

    logger.warning("Unknown registration_step=%r for profile=%s -- clearing it", step, profile["id"])
    client.table("profiles").update({"registration_step": None}).eq("id", profile["id"]).execute()
    return False


async def _handle_customer_name(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    client = ctx.supabase
    phone = profile["phone_number"]
    name = (extracted.text or "").strip()
    if not _looks_like_name(name):
        _log_onboarding_event(client, profile["id"], "awaiting_customer_name", name, accepted=False, reason="not_name_like")
        reply = "Sorry, I didn't quite get that -- please just type your full name (e.g. Priya Sharma)."
        await ctx.whatsapp.send_text(phone, reply)
        _log_turn(client, profile, extracted, [reply])
        return True
    _log_onboarding_event(client, profile["id"], "awaiting_customer_name", name, accepted=True)
    client.table("profiles").update({"full_name": name, "registration_step": "awaiting_pet_name"}).eq("id", profile["id"]).execute()
    _log_step_transition(client, profile["id"], "awaiting_customer_name", "awaiting_pet_name")
    reply = f"Nice to meet you, {name}! What's your pet's name?"
    await ctx.whatsapp.send_text(phone, reply)
    _log_turn(client, profile, extracted, [reply])
    return True


async def _handle_pet_name(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    client = ctx.supabase
    phone = profile["phone_number"]
    name = (extracted.text or "").strip()
    if not _looks_like_name(name):
        _log_onboarding_event(client, profile["id"], "awaiting_pet_name", name, accepted=False, reason="not_name_like")
        reply = "Sorry, I didn't quite get that -- please just type your pet's name (e.g. Bruno)."
        await ctx.whatsapp.send_text(phone, reply)
        _log_turn(client, profile, extracted, [reply])
        return True
    _log_onboarding_event(client, profile["id"], "awaiting_pet_name", name, accepted=True)
    pet = client.table("pets").insert({"profile_id": profile["id"], "name": name, "species": "Other"}).execute().data[0]
    try:
        client.table("pet_members").insert(
            {"pet_id": pet["id"], "profile_id": profile["id"], "role": "owner", "is_primary": True, "added_by": profile["id"]}
        ).execute()
    except Exception as exc:
        # A DB trigger on this project already auto-creates the owner pet_members
        # row when a pet is inserted -- treat "already exists" as the desired
        # end state, not a failure (same pattern as onboarding.py).
        if not is_unique_violation(exc):
            raise
    client.table("profiles").update({"registration_step": "awaiting_city"}).eq("id", profile["id"]).execute()
    _log_step_transition(client, profile["id"], "awaiting_pet_name", "awaiting_city")
    reply = "Last question — what city are you in?"
    await ctx.whatsapp.send_text(phone, reply)
    _log_turn(client, profile, extracted, [reply])
    return True


async def _handle_city(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    client = ctx.supabase
    phone = profile["phone_number"]
    city = (extracted.text or "").strip()
    if not city:
        _log_onboarding_event(client, profile["id"], "awaiting_city", city, accepted=False, reason="empty")
        reply = "Please type your city."
        await ctx.whatsapp.send_text(phone, reply)
        _log_turn(client, profile, extracted, [reply])
        return True
    _log_onboarding_event(client, profile["id"], "awaiting_city", city, accepted=True)
    client.table("profiles").update({"city": city, "registration_step": "completed"}).eq("id", profile["id"]).execute()
    _log_step_transition(client, profile["id"], "awaiting_city", "completed")

    pets = (
        client.table("pets").select("name").eq("profile_id", profile["id"])
        .order("created_at", desc=True).limit(1).execute().data
    )
    pet_name = pets[0]["name"] if pets else "your pet"
    owner_name = (profile.get("full_name") or "").split(" ")[0] or "there"

    reply = (
        f"\U0001F389 You're all set, {owner_name}!\n\n"
        "Everything on PetPulse is free, always:\n"
        "🩺 Unlimited AI health chats and symptom checks\n"
        "🚨 Full emergency triage and first-aid guidance\n"
        f"🗂️ Unlimited document vault for {pet_name}'s records\n"
        "🐾 Add as many pets as you like, all on one account\n"
        "📍 Nearby vet finder with open-now filters\n"
        "🌐 Chat in English or Hindi\n\n"
        "The only paid part: booking a doctor consultation costs ₹399 per visit.\n\n"
        f"Ask me anything about {pet_name}'s health, or say \"book a vet\" to schedule a consultation."
    )
    await ctx.whatsapp.send_text(phone, reply)
    _log_turn(client, profile, extracted, [reply])
    return True
