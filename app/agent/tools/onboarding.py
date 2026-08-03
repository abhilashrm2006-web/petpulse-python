"""Ports `Va6xq5ZLATsTqrXK` — Onboarding Field Saver / `save_onboarding_field`
(spec §3.1)."""

import math
import re
from datetime import date, datetime
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.integrations.supabase_client import is_unique_violation
from app.utils.pet_resolution import AMBIGUOUS_PET, resolve_pet

SPECIES_SYNONYMS = {
    "puppy": "Dog", "indie": "Dog", "street dog": "Dog", "dog": "Dog",
    "kitten": "Cat", "feline": "Cat", "cat": "Cat",
    "bunny": "Rabbit", "rabbit": "Rabbit",
    "goldfish": "Fish", "betta": "Fish", "fish": "Fish",
    "parrot": "Bird", "budgie": "Bird", "hen": "Bird", "bird": "Bird",
    "cavy": "Guinea Pig", "guinea pig": "Guinea Pig",
    "tortoise": "Turtle", "turtle": "Turtle",
    "calf": "Cow", "buffalo": "Cow", "cow": "Cow",
    "hamster": "Hamster", "goat": "Goat",
}
CANONICAL_SPECIES = ["Dog", "Cat", "Rabbit", "Fish", "Bird", "Hamster", "Guinea Pig", "Turtle", "Cow", "Goat", "Other"]

GENDER_SYNONYMS = {
    "male": "Male", "m": "Male", "boy": "Male", "he": "Male", "him": "Male",
    "female": "Female", "f": "Female", "girl": "Female", "she": "Female", "her": "Female",
}

PROFILE_FIELDS = {"email", "city", "full_name"}
PET_FIELD_TO_COLUMN = {
    "name": "name", "pet_name": "name", "breed": "breed", "age": "age",
    "weight": "weight", "dob": "date_of_birth", "date_of_birth": "date_of_birth", "species": "species",
    "gender": "gender",
}


def _normalize_value(field: str, value: str) -> Any:
    value = (value or "").strip()
    if field == "email":
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value):
            return None
        return value.lower()
    if field == "age":
        match = re.search(r"[\d.]+", value)
        # round() uses round-half-to-even (round(2.5) == 2 but round(1.5) == 2
        # and round(3.5) == 4) -- surprising, inconsistent behavior for a
        # user-facing age. floor(x + 0.5) always rounds .5 up.
        return math.floor(float(match.group()) + 0.5) if match else None
    if field == "weight":
        match = re.search(r"[\d.]+", value)
        if not match:
            return None
        weight = float(match.group())
        if re.search(r"lb|pound", value, re.IGNORECASE):
            weight *= 0.4536
        return round(weight, 1)
    if field in ("dob", "date_of_birth"):
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except ValueError:
            match = re.match(r"^(\d{1,2})/(\d{4})$", value)
            if match:
                month, year = match.groups()
                return f"{year}-{int(month):02d}-01"
            return None
    if field == "species":
        return SPECIES_SYNONYMS.get(value.lower(), value.title() if value.title() in CANONICAL_SPECIES else "Other")
    if field == "gender":
        return GENDER_SYNONYMS.get(value.lower(), "Unknown")
    return re.sub(r"\s+", " ", value)


async def save_onboarding_field(
    ctx: AppContext, agent_ctx: AgentContext, field: str, value: str, pet_name: str = ""
) -> dict[str, Any]:
    client = ctx.supabase
    normalized = _normalize_value(field, value)
    if normalized is None:
        return {"success": False, "field": field, "error": "invalid_value"}

    if field in PROFILE_FIELDS:
        client.table("profiles").update({field: normalized}).eq("id", agent_ctx.profile["id"]).execute()
        return {"success": True, "field": field, "savedValue": normalized}

    column = PET_FIELD_TO_COLUMN.get(field)
    if not column:
        return {"success": False, "field": field, "error": "unknown_field"}

    if field in ("name", "pet_name"):
        resolution = resolve_pet(agent_ctx.pets, pet_name=pet_name, auto_resolve_single=False)
        if resolution.pet:
            client.table("pets").update({"name": normalized}).eq("id", resolution.pet["id"]).execute()
            resolution.pet["name"] = normalized  # keep agent_ctx.pets in sync for later calls this same turn
            return {"success": True, "field": field, "savedValue": normalized}
        created = (
            client.table("pets")
            .insert({"profile_id": agent_ctx.profile["id"], "name": normalized, "species": "Other"})
            .execute()
        )
        pet = created.data[0]
        try:
            client.table("pet_members").insert(
                {"pet_id": pet["id"], "profile_id": agent_ctx.profile["id"], "role": "owner", "is_primary": True, "added_by": agent_ctx.profile["id"]}
            ).execute()
        except Exception as exc:
            # A DB trigger on this project already auto-creates the owner pet_members
            # row when a pet is inserted (confirmed live) — our own insert then always
            # conflicts. Treat "already exists" as the desired end state, not a failure;
            # anything else should still surface.
            if not is_unique_violation(exc):
                raise
        # Registering a pet is typically several save_onboarding_field calls in ONE turn
        # (name, then species/breed/age as the owner mentions them) — agent_ctx.pets was
        # loaded once at turn start, so without this, every call after this one can't find
        # the pet it just created and silently fails with no_pet_on_file.
        agent_ctx.pets.append(pet)
        return {"success": True, "field": field, "savedValue": normalized, "created_pet": True}

    resolution = resolve_pet(agent_ctx.pets, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return {"success": False, "error": "ambiguous_pet", "message": "Which pet is this for?"}
    if not resolution.pet:
        return {"success": False, "field": field, "error": "no_pet_on_file"}

    client.table("pets").update({column: normalized}).eq("id", resolution.pet["id"]).execute()
    resolution.pet[column] = normalized
    return {"success": True, "field": field, "savedValue": normalized}
