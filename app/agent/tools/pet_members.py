"""Ports `PA2bDT8OlmHQvdg6` — Add Pet Member / `add_pet_member` (spec §3.8)."""

from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext
from app.utils.phone import normalize_phone
from app.utils.pet_resolution import resolve_pet

VALID_ROLES = {"owner", "family", "caregiver", "vet"}


async def add_pet_member(
    ctx: AppContext,
    agent_ctx: AgentContext,
    pet_id: str = "",
    pet_name: str = "",
    member_name: str = "",
    member_phone: str = "",
    role: str = "family",
) -> dict[str, Any]:
    normalized_phone = normalize_phone(member_phone)
    if not normalized_phone or not member_name:
        return {"success": False, "error": "invalid_input", "message": "I need the invitee's name and phone number (with country code)."}

    role = role if role in VALID_ROLES else "family"

    # Security gate: only pets the REQUESTER is already a member of are eligible —
    # agent_ctx.pets is already scoped to the requester via pet_members join.
    resolution = resolve_pet(agent_ctx.pets, pet_id=pet_id, pet_name=pet_name, auto_resolve_single=True)
    if resolution.ambiguous:
        return {"success": False, "error": "ambiguous_pet", "message": "Which pet is this for?"}
    if not resolution.pet:
        return {"success": False, "error": "requester_not_a_member", "message": "You're not registered as a member of that pet."}

    pet = resolution.pet
    client = ctx.supabase

    invitee = client.table("profiles").select("*").eq("phone_number", normalized_phone).limit(1).execute().data
    is_new_user = False
    if invitee:
        invitee_profile = invitee[0]
    else:
        # A brand-new phone number invited with role="vet" must actually become a
        # vet in profiles.role, not just a pet_members row -- otherwise every
        # vet-gated tool (accept_session, file_prescription, etc.) stays
        # unavailable to them the first time they message the bot.
        new_profile = {"phone_number": normalized_phone, "full_name": member_name}
        if role == "vet":
            new_profile["role"] = "vet"
        created = client.table("profiles").insert(new_profile).execute()
        invitee_profile = created.data[0]
        is_new_user = True

    existing_membership = (
        client.table("pet_members").select("*").eq("pet_id", pet["id"]).eq("profile_id", invitee_profile["id"]).execute().data
    )
    if existing_membership:
        return {"success": False, "error": "already_member", "message": f"{member_name} is already a member of {pet['name']}'s care team."}

    client.table("pet_members").insert(
        {
            "pet_id": pet["id"],
            "profile_id": invitee_profile["id"],
            "role": role,
            "is_primary": False,
            "added_by": agent_ctx.profile["id"],
        }
    ).execute()

    requester_name = agent_ctx.profile.get("full_name", "Someone")
    await ctx.whatsapp.send_text(
        normalized_phone,
        f"Hi {member_name}! {requester_name} added you as a '{role}' on {pet['name']}'s care team on PetPulse. "
        f"Message this number any time for updates on {pet['name']}'s health.",
    )

    return {
        "success": True,
        "mode": "member_added",
        "pet_name": pet["name"],
        "member_name": member_name,
        "role": role,
        "invitee_is_new_user": is_new_user,
    }
