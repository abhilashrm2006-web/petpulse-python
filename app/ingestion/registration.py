"""Deterministic first-contact registration wizard: greeting, New/Existing
Member choice, structured intake, and membership tier selection. Runs
entirely outside the LLM agent loop (see app/main.py, which calls
handle_registration before build_context/run_agent_turn) so a brand-new
customer's registration can never be skipped, reordered, or
misinterpreted by the model. Once profiles.registration_step is
"completed" (or the profile predates this wizard, or isn't role=customer
-- e.g. a vet), this module returns False immediately and the existing
agent pipeline runs completely unchanged.
"""

import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any

from app.agent.tools.subscriptions import create_and_send_subscription
from app.deps import AppContext
from app.ingestion.webhook import ExtractedMessage
from app.integrations import razorpay_client
from app.integrations.supabase_client import get_profile_by_phone, is_unique_violation
from app.utils.phone import normalize_phone

logger = logging.getLogger(__name__)

MEMBER_TYPE_BUTTONS = [
    {"id": "member_type|new", "title": "New Member"},
    {"id": "member_type|existing", "title": "Existing Member"},
]
TIER_BUTTONS = [
    {"id": "tier_choice|free", "title": "Free Member"},
    {"id": "tier_choice|subscriber", "title": "Subscriber"},
]
FREE_SUBCHOICE_BUTTONS = [
    {"id": "free_subchoice|continue", "title": "Continue"},
    {"id": "free_subchoice|subscribe", "title": "Subscribe"},
]


def _yes_no_buttons(prefix: str) -> list[dict[str, str]]:
    return [{"id": f"{prefix}|yes", "title": "Yes"}, {"id": f"{prefix}|no", "title": "No"}]


def _reply_choice(extracted: ExtractedMessage) -> str:
    """A button tap's id is "<prefix>|<choice>" -- take the choice. A plain
    typed reply (e.g. just "yes", "new") works the same way, matching the
    button-or-text pattern already used for recording consent/prescription
    format elsewhere in this codebase."""
    if extracted.button_reply_id:
        return extracted.button_reply_id.rsplit("|", 1)[-1].lower()
    return (extracted.text or "").strip().lower()


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


def _get_active_pet(client, profile_id: str) -> dict[str, Any] | None:
    rows = (
        client.table("pets").select("*").eq("profile_id", profile_id)
        .order("created_at", desc=True).limit(1).execute().data
    )
    return rows[0] if rows else None


def _parse_dob(value: str) -> str | None:
    value = (value or "").strip()
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        pass
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", value)
    if match:
        day, month, year = match.groups()
        try:
            return date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            return None
    if re.match(r"^\d{4}$", value):
        return f"{value}-01-01"
    return None


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
                    "registration_step": "awaiting_member_type",
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
        await ctx.whatsapp.send_interactive_buttons(
            phone,
            "Are you a new or existing PetPulse member?\n"
            "क्या आप एक नए या मौजूदा सदस्य हैं? नीचे विकल्प चुनें।",
            MEMBER_TYPE_BUTTONS,
        )
        return True

    step = profile.get("registration_step")
    if not step or step == "completed" or profile.get("role") != "customer":
        return False

    reply = _reply_choice(extracted)

    if step == "awaiting_member_type":
        return await _handle_member_type(ctx, profile, reply)
    if step == "awaiting_customer_name":
        return await _handle_customer_name(ctx, profile, extracted)
    if step == "awaiting_pet_name":
        return await _handle_pet_name(ctx, profile, extracted)
    if step == "awaiting_pet_dob":
        return await _handle_pet_dob(ctx, profile, extracted)
    if step == "awaiting_pet_age":
        return await _handle_pet_age(ctx, profile, extracted)
    if step == "awaiting_pet_weight":
        return await _handle_pet_weight(ctx, profile, extracted)
    if step == "awaiting_kci_status":
        return await _handle_kci_status(ctx, profile, reply)
    if step == "awaiting_vaccination_status":
        return await _handle_vaccination_status(ctx, profile, reply)
    if step == "awaiting_microchip_status":
        return await _handle_microchip_status(ctx, profile, reply)
    if step == "awaiting_microchip_number":
        return await _handle_microchip_number(ctx, profile, extracted)
    if step == "awaiting_city":
        return await _handle_city(ctx, profile, extracted)
    if step == "awaiting_tier_choice":
        return await _handle_tier_choice(ctx, profile, reply)
    if step == "awaiting_free_subchoice":
        return await _handle_free_subchoice(ctx, profile, reply)
    if step == "awaiting_existing_phone":
        return await _handle_existing_phone(ctx, profile, extracted)
    if step == "awaiting_existing_verify":
        return await _handle_existing_verify(ctx, profile, extracted)

    logger.warning("Unknown registration_step=%r for profile=%s -- clearing it", step, profile["id"])
    client.table("profiles").update({"registration_step": None}).eq("id", profile["id"]).execute()
    return False


async def _handle_member_type(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    phone = profile["phone_number"]
    client = ctx.supabase
    if reply == "new":
        client.table("profiles").update({"registration_step": "awaiting_customer_name"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_text(phone, "Great! What's your full name?")
        return True
    if reply == "existing":
        client.table("profiles").update({"registration_step": "awaiting_existing_phone"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_text(phone, "Please enter the phone number you originally registered with (with country code).")
        return True
    await ctx.whatsapp.send_interactive_buttons(
        phone,
        "Sorry, I didn't understand that -- please tap one of the options below.\n"
        "माफ़ कीजिए, कृपया नीचे दिए गए विकल्पों में से एक चुनें।",
        MEMBER_TYPE_BUTTONS,
    )
    return True


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
    client.table("profiles").update({"registration_step": "awaiting_pet_dob"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, f"What's {name}'s date of birth? (DD/MM/YYYY, or just the year if you're not sure)")
    return True


async def _handle_pet_dob(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    dob = _parse_dob(extracted.text or "")
    if not dob:
        await ctx.whatsapp.send_text(phone, "Please share the date as DD/MM/YYYY, or just the year if you're not sure.")
        return True
    pet = _get_active_pet(ctx.supabase, profile["id"])
    if pet:
        ctx.supabase.table("pets").update({"date_of_birth": dob}).eq("id", pet["id"]).execute()
    ctx.supabase.table("profiles").update({"registration_step": "awaiting_pet_age"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "Got it! How old is your pet, in years?")
    return True


async def _handle_pet_age(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    match = re.search(r"[\d.]+", extracted.text or "")
    if not match:
        await ctx.whatsapp.send_text(phone, "Please share your pet's age as a number, e.g. 2.")
        return True
    age = math.floor(float(match.group()) + 0.5)
    pet = _get_active_pet(ctx.supabase, profile["id"])
    if pet:
        ctx.supabase.table("pets").update({"age": age}).eq("id", pet["id"]).execute()
    ctx.supabase.table("profiles").update({"registration_step": "awaiting_pet_weight"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "And what's your pet's weight, in kg?")
    return True


async def _handle_pet_weight(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    text = extracted.text or ""
    match = re.search(r"[\d.]+", text)
    if not match:
        await ctx.whatsapp.send_text(phone, "Please share your pet's weight as a number, e.g. 12 (in kg).")
        return True
    weight = float(match.group())
    if re.search(r"lb|pound", text, re.IGNORECASE):
        weight *= 0.4536
    pet = _get_active_pet(ctx.supabase, profile["id"])
    if pet:
        ctx.supabase.table("pets").update({"weight": round(weight, 1)}).eq("id", pet["id"]).execute()
    ctx.supabase.table("profiles").update({"registration_step": "awaiting_kci_status"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_interactive_buttons(
        phone, "Does your pet have a KCI (Kennel Club of India) certificate?", _yes_no_buttons("kci_status")
    )
    return True


async def _handle_yes_no_pet_field(
    ctx: AppContext,
    profile: dict[str, Any],
    reply: str,
    *,
    column: str,
    current_prompt: str,
    current_buttons: list[dict[str, str]],
    next_step: str,
    next_prompt: str,
    next_buttons: list[dict[str, str]],
) -> bool:
    phone = profile["phone_number"]
    if reply not in ("yes", "no"):
        await ctx.whatsapp.send_interactive_buttons(phone, current_prompt, current_buttons)
        return True
    pet = _get_active_pet(ctx.supabase, profile["id"])
    if pet:
        ctx.supabase.table("pets").update({column: reply == "yes"}).eq("id", pet["id"]).execute()
    ctx.supabase.table("profiles").update({"registration_step": next_step}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_interactive_buttons(phone, next_prompt, next_buttons)
    return True


async def _handle_kci_status(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    return await _handle_yes_no_pet_field(
        ctx, profile, reply,
        column="has_kci_certificate",
        current_prompt="Does your pet have a KCI (Kennel Club of India) certificate?",
        current_buttons=_yes_no_buttons("kci_status"),
        next_step="awaiting_vaccination_status",
        next_prompt="Is your pet up to date on vaccinations?",
        next_buttons=_yes_no_buttons("vaccination_status"),
    )


async def _handle_vaccination_status(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    return await _handle_yes_no_pet_field(
        ctx, profile, reply,
        column="vaccination_confirmed",
        current_prompt="Is your pet up to date on vaccinations?",
        current_buttons=_yes_no_buttons("vaccination_status"),
        next_step="awaiting_microchip_status",
        next_prompt="Does your pet have a microchip?",
        next_buttons=_yes_no_buttons("microchip_status"),
    )


async def _handle_microchip_status(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    phone = profile["phone_number"]
    if reply not in ("yes", "no"):
        await ctx.whatsapp.send_interactive_buttons(phone, "Does your pet have a microchip?", _yes_no_buttons("microchip_status"))
        return True
    if reply == "yes":
        ctx.supabase.table("profiles").update({"registration_step": "awaiting_microchip_number"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_text(phone, "What's the microchip number?")
        return True
    ctx.supabase.table("profiles").update({"registration_step": "awaiting_city"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "Last question — what city are you in?")
    return True


async def _handle_microchip_number(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    number = (extracted.text or "").strip()
    if not number:
        await ctx.whatsapp.send_text(phone, "Please type the microchip number.")
        return True
    pet = _get_active_pet(ctx.supabase, profile["id"])
    if pet:
        ctx.supabase.table("pets").update({"microchip_number": number}).eq("id", pet["id"]).execute()
    ctx.supabase.table("profiles").update({"registration_step": "awaiting_city"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "Last question — what city are you in?")
    return True


async def _handle_city(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    city = (extracted.text or "").strip()
    if not city:
        await ctx.whatsapp.send_text(phone, "Please type your city.")
        return True
    ctx.supabase.table("profiles").update({"city": city, "registration_step": "awaiting_tier_choice"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "\U0001F389 Welcome to PetPulse AI World!")
    # Feature order here matches the product's positioning: PetPulse is "the system that
    # manages your pet's health between vet visits," not "AI vet in your pocket" -- so the
    # passport/vault/multi-pet anchors lead, the flashier AI chat/triage features trail.
    await ctx.whatsapp.send_text(
        phone,
        "As a Subscriber (₹399/month) you get:\n"
        "🩺 Full vaccination passport — shareable with any vet or boarding facility\n"
        "🗂️ Unlimited records vault — store, search, and share every document\n"
        "🐾 Unlimited pets on one account\n"
        "💬 Unlimited AI health chats, in English or Hindi\n"
        "🚨 Full emergency triage with a first-aid checklist\n"
        "📍 Nearby vet finder with open-now/emergency filters\n"
        "...plus 1 free vet consultation every month.",
    )
    await ctx.whatsapp.send_interactive_buttons(phone, "Which membership would you like?", TIER_BUTTONS)
    return True


async def _handle_tier_choice(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    phone = profile["phone_number"]
    if reply == "free":
        ctx.supabase.table("profiles").update({"registration_step": "awaiting_free_subchoice"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_interactive_buttons(
            phone, "Would you like to continue on the Free plan, or subscribe?", FREE_SUBCHOICE_BUTTONS
        )
        return True
    if reply == "subscriber":
        await _complete_wizard_with_subscription(ctx, profile)
        return True
    await ctx.whatsapp.send_interactive_buttons(phone, "Please choose one of the options below.", TIER_BUTTONS)
    return True


async def _handle_free_subchoice(ctx: AppContext, profile: dict[str, Any], reply: str) -> bool:
    phone = profile["phone_number"]
    if reply == "continue":
        ctx.supabase.table("profiles").update({"registration_step": "completed"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_text(
            phone,
            "You're all set! Ask me anything about your pet's health, or say \"book a vet\" to schedule a consultation.",
        )
        return True
    if reply == "subscribe":
        await _complete_wizard_with_subscription(ctx, profile)
        return True
    await ctx.whatsapp.send_interactive_buttons(phone, "Please choose one of the options below.", FREE_SUBCHOICE_BUTTONS)
    return True


async def _complete_wizard_with_subscription(ctx: AppContext, profile: dict[str, Any]) -> None:
    """The wizard is done either way once this is called -- even if the
    Razorpay/storage call inside create_and_send_subscription fails, being
    stuck re-showing the tier-choice buttons forever is worse than being
    marked complete with a failed subscribe attempt they can retry later
    via the start_subscription agent tool."""
    await create_and_send_subscription(ctx, profile)
    ctx.supabase.table("profiles").update({"registration_step": "completed"}).eq("id", profile["id"]).execute()


async def _handle_existing_phone(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    phone = profile["phone_number"]
    typed = normalize_phone(extracted.text or "")
    if not typed:
        await ctx.whatsapp.send_text(phone, "Please enter a valid phone number with country code.")
        return True

    matched = get_profile_by_phone(ctx.supabase, typed)
    if not matched:
        ctx.supabase.table("profiles").update({"registration_step": "awaiting_customer_name"}).eq("id", profile["id"]).execute()
        await ctx.whatsapp.send_text(phone, "I couldn't find an account with that number — let's get you registered instead!")
        await ctx.whatsapp.send_text(phone, "What's your full name?")
        return True

    pet = _get_active_pet(ctx.supabase, matched["id"])
    summary_lines = ["Is this you?", f"Name: {matched.get('full_name') or 'Not on file'}"]
    if pet:
        summary_lines.append(f"Pet: {pet.get('name') or 'Not on file'}")
        summary_lines.append(f"DOB: {pet.get('date_of_birth') or 'Not on file'}")
        summary_lines.append(f"Age: {pet['age'] if pet.get('age') is not None else 'Not on file'}")
        summary_lines.append(f"Weight: {pet['weight'] if pet.get('weight') is not None else 'Not on file'} kg")
        summary_lines.append(f"KCI: {'Yes' if pet.get('has_kci_certificate') else 'No'}")
        summary_lines.append(f"Vaccination: {'Yes' if pet.get('vaccination_confirmed') else 'No'}")
        summary_lines.append(f"Microchip: {pet.get('microchip_number') or 'No'}")
    summary_lines.append(f"City: {matched.get('city') or 'Not on file'}")

    ctx.supabase.table("profiles").update({"registration_step": "awaiting_existing_verify"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_interactive_buttons(
        phone,
        "\n".join(summary_lines),
        [
            {"id": f"existing_verify|{matched['id']}|yes", "title": "Yes, that's me"},
            {"id": f"existing_verify|{matched['id']}|no", "title": "No"},
        ],
    )
    return True


async def _handle_existing_verify(ctx: AppContext, profile: dict[str, Any], extracted: ExtractedMessage) -> bool:
    """Requires an actual button tap -- the matched profile's id lives only
    in the button id (same encoding pattern as recording_consent|<id>|yes
    elsewhere), so a typed "yes"/"no" can't be resolved back to it."""
    phone = profile["phone_number"]
    parts = (extracted.button_reply_id or "").split("|")
    if len(parts) != 3 or parts[0] != "existing_verify":
        await ctx.whatsapp.send_text(phone, "Please tap \"Yes, that's me\" or \"No\" above.")
        return True

    matched_profile_id, answer = parts[1], parts[2]
    client = ctx.supabase

    if answer == "yes":
        # Delete the placeholder profile row BEFORE moving phone_number onto
        # the matched row, or the two rows would briefly share the same
        # phone_number and collide with its unique constraint.
        if profile["id"] != matched_profile_id:
            client.table("profiles").delete().eq("id", profile["id"]).execute()
        client.table("profiles").update({"phone_number": phone, "registration_step": "completed"}).eq("id", matched_profile_id).execute()
        await ctx.whatsapp.send_text(phone, "Welcome back to PetPulse AI World! How can I help you today?")
        return True

    client.table("profiles").update({"registration_step": "awaiting_member_type"}).eq("id", profile["id"]).execute()
    await ctx.whatsapp.send_text(phone, "No problem — let's start over.")
    await ctx.whatsapp.send_interactive_buttons(
        phone,
        "Are you a new or existing PetPulse member?\n"
        "क्या आप एक नए या मौजूदा सदस्य हैं? नीचे विकल्प चुनें।",
        MEMBER_TYPE_BUTTONS,
    )
    return True


async def handle_subscription_webhook(ctx: AppContext, event_body: dict[str, Any]) -> bool:
    """Razorpay's async confirmation of a Subscriber signup's lifecycle --
    mirrors handle_payment_webhook's shape in app/agent/tools/booking.py.
    Idempotent: an already-active row just gets updated to the same status
    again, never double-notified (nothing is sent to the customer here at
    all, only the subscriptions row's status changes)."""
    parsed = razorpay_client.extract_subscription_event(event_body)
    if not parsed:
        return False
    event, subscription_id, _reference_id = parsed

    client = ctx.supabase
    rows = client.table("subscriptions").select("*").eq("provider_subscription_id", subscription_id).limit(1).execute().data
    if not rows:
        logger.warning("Subscription webhook event=%s for unknown provider_subscription_id=%s", event, subscription_id)
        return False
    row = rows[0]

    if event in ("subscription.activated", "subscription.charged"):
        client.table("subscriptions").update({"status": "active"}).eq("id", row["id"]).execute()
    elif event == "subscription.cancelled":
        client.table("subscriptions").update(
            {"status": "cancelled", "cancelled_at": datetime.now(tz=timezone.utc).isoformat()}
        ).eq("id", row["id"]).execute()
    else:
        logger.info("Subscription webhook event=%s for subscription_id=%s: no state change needed", event, subscription_id)
    return True
