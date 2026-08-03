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

from app.deps import AppContext
from app.ingestion.webhook import ExtractedMessage
from app.integrations.supabase_client import get_profile_by_phone, is_unique_violation

logger = logging.getLogger(__name__)


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
        await ctx.whatsapp.send_text(
            phone,
            "Hi this is Pulsy, welcome to PetPulse \U0001F43E\n"
            "नमस्ते! मैं Pulsy हूं, PetPulse में आपका स्वागत है \U0001F43E",
        )
        await ctx.whatsapp.send_text(phone, "What's your full name?")
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
    phone = profile["phone_number"]
    name = (extracted.text or "").strip()
    if not _looks_like_name(name):
        await ctx.whatsapp.send_text(phone, "Sorry, I didn't quite get that -- please just type your full name (e.g. Priya Sharma).")
        return True
    ctx.supabase.table("profiles").update({"full_name": name, "registration_step": "awaiting_pet_name"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, f"Nice to meet you, {name}! What's your pet's name?")
    return True


async def _handle_pet_name(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    name = (extracted.text or "").strip()
    if not _looks_like_name(name):
        await ctx.whatsapp.send_text(phone, "Sorry, I didn't quite get that -- please just type your pet's name (e.g. Bruno).")
        return True
    client = ctx.supabase
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
    await ctx.whatsapp.send_text(phone, "Last question — what city are you in?")
    return True


async def _handle_city(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    city = (extracted.text or "").strip()
    if not city:
        await ctx.whatsapp.send_text(phone, "Please type your city.")
        return True
    ctx.supabase.table("profiles").update({"city": city, "registration_step": "completed"}).eq("id", profile["id"]).execute()

    pets = (
        ctx.supabase.table("pets").select("name").eq("profile_id", profile["id"])
        .order("created_at", desc=True).limit(1).execute().data
    )
    pet_name = pets[0]["name"] if pets else "your pet"
    owner_name = (profile.get("full_name") or "").split(" ")[0] or "there"

    await ctx.whatsapp.send_text(
        phone,
        f"\U0001F389 You're all set, {owner_name}!\n\n"
        "Everything on PetPulse is free, always:\n"
        "🩺 Unlimited AI health chats and symptom checks\n"
        "🚨 Full emergency triage and first-aid guidance\n"
        f"🗂️ Unlimited document vault for {pet_name}'s records\n"
        "🐾 Add as many pets as you like, all on one account\n"
        "📍 Nearby vet finder with open-now filters\n"
        "🌐 Chat in English or Hindi\n\n"
        "The only paid part: booking a doctor consultation costs ₹399 per visit.\n\n"
        f"Ask me anything about {pet_name}'s health, or say \"book a vet\" to schedule a consultation.",
    )
    return True
