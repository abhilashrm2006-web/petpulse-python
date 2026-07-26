"""Ports the context-gathering half of Core Engine (spec §2): user/pet
resolution, active-pet-from-message, memory + medical context, open
booking-session state, onboarding status. This is all deterministic *data
gathering* — none of it decides what to do, it only builds the picture the
single agent reasons over (including role: vet vs customer, replacing
n8n's `Is Doctor?` hard branch with plain context)."""

import asyncio
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from supabase import Client

from app.ingestion.webhook import ExtractedMessage
from app.integrations.supabase_client import get_pets_for_profile, get_profile_by_phone, is_active_subscriber
from app.utils.pet_resolution import resolve_active_pet_from_message

ONBOARDING_REQUIRED_FIELDS = ["email", "city", "pet_name", "breed", "age", "weight", "dob"]


@dataclass
class AgentContext:
    profile: dict[str, Any]
    role: str
    is_new_profile: bool
    pets: list[dict[str, Any]]
    active_pet: dict[str, Any] | None
    active_pet_matched_from_message: bool
    memory_context: list[dict[str, Any]]
    medical_context: dict[str, Any]
    knowledge_base: list[dict[str, Any]]
    open_session: dict[str, Any] | None
    pending_negotiation: dict[str, Any] | None
    onboarding: dict[str, Any] = field(default_factory=dict)
    pending_media: Any = None  # set post-construction to this turn's MediaResult (app.ingestion.media), if any
    awaiting_prescription_session: dict[str, Any] | None = None
    is_subscriber: bool = False


async def _to_thread(fn, *args, **kwargs):
    return await asyncio.to_thread(fn, *args, **kwargs)


def _get_or_create_profile(client: Client, phone_number: str, sender_name: str) -> tuple[dict[str, Any], bool]:
    existing = get_profile_by_phone(client, phone_number)
    if existing:
        return existing, False
    resp = (
        client.table("profiles")
        .insert({"phone_number": phone_number, "full_name": sender_name, "role": "customer"})
        .execute()
    )
    return resp.data[0], True


def _load_memory(client: Client, profile_id: str, pet_id: str | None) -> list[dict[str, Any]]:
    user_rows = client.table("memory").select("*").eq("profile_id", profile_id).execute().data or []
    if not pet_id:
        return user_rows
    pet_rows = client.table("memory").select("*").eq("pet_id", pet_id).execute().data or []
    seen_ids = {r["id"] for r in user_rows}
    return user_rows + [r for r in pet_rows if r["id"] not in seen_ids]


def _bucket_by_pet(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get("pet_id")), []).append(row)
    return buckets


def _load_medical_context(client: Client, profile_id: str) -> dict[str, Any]:
    medical_records = client.table("medical_records").select("*").eq("profile_id", profile_id).execute().data or []
    vaccinations = client.table("vaccinations").select("*").eq("profile_id", profile_id).execute().data or []
    health_logs = client.table("health_logs").select("*").eq("profile_id", profile_id).execute().data or []
    documents = client.table("documents").select("*").eq("profile_id", profile_id).execute().data or []
    return {
        "medical_records_by_pet": _bucket_by_pet(medical_records),
        "vaccinations_by_pet": _bucket_by_pet(vaccinations),
        "health_logs_by_pet": _bucket_by_pet(health_logs),
        "documents_by_pet": _bucket_by_pet(documents),
    }


def _load_knowledge_base(client: Client, message_text: str) -> list[dict[str, Any]]:
    if not message_text:
        return []
    pattern = f"%{message_text[:60]}%"
    resp = (
        client.table("knowledge_base")
        .select("*")
        .ilike("category", pattern)
        .ilike("content", pattern)
        .ilike("title", pattern)
        .limit(5)
        .execute()
    )
    return resp.data or []


def _load_open_session(client: Client, profile_id: str) -> dict[str, Any] | None:
    resp = (
        client.table("doctor_sessions")
        .select("*")
        .eq("profile_id", profile_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _load_pending_negotiation(client: Client, profile_id: str) -> dict[str, Any] | None:
    resp = (
        client.table("doctor_sessions")
        .select("*")
        .eq("profile_id", profile_id)
        .eq("status", "negotiating")
        .eq("awaiting_from", "customer_time_input")
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _load_doctor_open_session(client: Client, doctor_phone: str) -> dict[str, Any] | None:
    resp = (
        client.table("doctor_sessions")
        .select("*")
        .eq("doctor_phone", doctor_phone)
        .in_("status", ["pending", "negotiating", "accepted"])
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _load_awaiting_prescription_session(client: Client, doctor_phone: str) -> dict[str, Any] | None:
    """A completed session still awaiting_from="doctor_prescription" isn't
    picked up by _load_doctor_open_session (its status query only covers
    pending/negotiating/accepted), so without this the vet's follow-up
    message has no deterministic way to know which session a document they
    send belongs to — surfacing it here, not relying on chat-history recall,
    is what lets file_prescription_document be called reliably."""
    resp = (
        client.table("doctor_sessions")
        .select("*")
        .eq("doctor_phone", doctor_phone)
        .eq("status", "completed")
        .eq("awaiting_from", "doctor_prescription")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def compute_onboarding_status(profile: dict[str, Any], pets: list[dict[str, Any]]) -> dict[str, Any]:
    primary_pet = pets[0] if pets else {}
    field_values = {
        "email": profile.get("email"),
        "city": profile.get("city"),
        "pet_name": primary_pet.get("name"),
        "breed": primary_pet.get("breed"),
        "age": primary_pet.get("age"),
        "weight": primary_pet.get("weight"),
        "dob": primary_pet.get("date_of_birth"),
    }
    missing = [f for f in ONBOARDING_REQUIRED_FIELDS if not field_values.get(f)]
    return {
        "missing_fields": missing,
        "all_pets": pets,
        "complete": not missing,
    }


async def build_context(client: Client, extracted: ExtractedMessage) -> AgentContext:
    profile, is_new = await _to_thread(_get_or_create_profile, client, extracted.phone_number, extracted.sender_name)
    role = profile.get("role", "customer")

    pets = await _to_thread(get_pets_for_profile, client, profile["id"])
    active_pet, active_pet_matched = resolve_active_pet_from_message(pets, extracted.text)

    if role == "vet":
        open_session, awaiting_prescription_session = await asyncio.gather(
            _to_thread(_load_doctor_open_session, client, extracted.phone_number),
            _to_thread(_load_awaiting_prescription_session, client, extracted.phone_number),
        )
        pending_negotiation = None
        memory_context: list[dict[str, Any]] = []
        medical_context: dict[str, Any] = {}
        knowledge_base: list[dict[str, Any]] = []
    else:
        awaiting_prescription_session = None
        memory_context, medical_context, knowledge_base, open_session, pending_negotiation = await asyncio.gather(
            _to_thread(_load_memory, client, profile["id"], active_pet["id"] if active_pet else None),
            _to_thread(_load_medical_context, client, profile["id"]),
            _to_thread(_load_knowledge_base, client, extracted.text),
            _to_thread(_load_open_session, client, profile["id"]),
            _to_thread(_load_pending_negotiation, client, profile["id"]),
        )

    onboarding = compute_onboarding_status(profile, pets) if role != "vet" else {}
    is_subscriber = await _to_thread(is_active_subscriber, client, profile["id"]) if role != "vet" else False

    return AgentContext(
        profile=profile,
        role=role,
        is_new_profile=is_new,
        pets=pets,
        active_pet=active_pet,
        active_pet_matched_from_message=active_pet_matched,
        memory_context=memory_context,
        medical_context=medical_context,
        knowledge_base=knowledge_base,
        open_session=open_session,
        pending_negotiation=pending_negotiation,
        onboarding=onboarding,
        awaiting_prescription_session=awaiting_prescription_session,
        is_subscriber=is_subscriber,
    )


def mark_onboarding_complete_if_needed(client: Client, profile: dict[str, Any], onboarding: dict[str, Any]) -> None:
    if onboarding.get("complete") and not profile.get("onboarding_completed"):
        client.table("profiles").update({"onboarding_completed": True}).eq("id", profile["id"]).execute()


def today_iso() -> str:
    return date.today().isoformat()
