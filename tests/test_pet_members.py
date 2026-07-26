"""Reproduces and verifies the fix for a real audit finding: inviting a
phone number that has never messaged the bot before, with role="vet", left
that profile's own `role` column unset (defaulting to "customer" elsewhere
in this codebase) -- so the invited vet never actually got access to any
vet-gated tool (accept_session, file_prescription, etc.) despite being a
valid pet_members role."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.pet_members import add_pet_member
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase=None):
    whatsapp = SimpleNamespace(send_text=AsyncMock())
    return SimpleNamespace(supabase=supabase or FakeSupabaseClient(), whatsapp=whatsapp, settings=object())


def _make_agent_ctx(pets):
    return SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}, pets=pets)


@pytest.mark.asyncio
async def test_add_pet_member_grants_vet_role_to_a_brand_new_invitee():
    supabase = FakeSupabaseClient(initial={"pets": [{"id": "pet-a", "name": "Max"}]})
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[{"id": "pet-a", "name": "Max"}])

    result = await add_pet_member(
        ctx, agent_ctx, pet_id="pet-a", member_name="Dr. Rao", member_phone="919000000001", role="vet"
    )

    assert result["success"] is True
    assert result["invitee_is_new_user"] is True
    profile = next(p for p in supabase.rows("profiles") if p["phone_number"] == "919000000001")
    assert profile["role"] == "vet"


@pytest.mark.asyncio
async def test_add_pet_member_does_not_force_a_role_for_a_family_invitee():
    """Only role="vet" should set profiles.role -- family/caregiver
    invitees are ordinary customers and must not be force-set to something
    else."""
    supabase = FakeSupabaseClient(initial={"pets": [{"id": "pet-a", "name": "Max"}]})
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[{"id": "pet-a", "name": "Max"}])

    result = await add_pet_member(
        ctx, agent_ctx, pet_id="pet-a", member_name="John", member_phone="919111111111", role="family"
    )

    assert result["success"] is True
    profile = next(p for p in supabase.rows("profiles") if p["phone_number"] == "919111111111")
    assert "role" not in profile


@pytest.mark.asyncio
async def test_add_pet_member_does_not_change_role_for_an_existing_profile():
    """An invitee who already has a profile (e.g. an existing customer) must
    keep whatever role they already have -- this fix only covers profiles
    created fresh by this call."""
    supabase = FakeSupabaseClient(
        initial={
            "pets": [{"id": "pet-a", "name": "Max"}],
            "profiles": [{"id": "existing-1", "phone_number": "919222222222", "full_name": "Existing Vet", "role": "customer"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[{"id": "pet-a", "name": "Max"}])

    result = await add_pet_member(
        ctx, agent_ctx, pet_id="pet-a", member_name="Existing Vet", member_phone="919222222222", role="vet"
    )

    assert result["success"] is True
    assert result["invitee_is_new_user"] is False
    profile = next(p for p in supabase.rows("profiles") if p["phone_number"] == "919222222222")
    assert profile["role"] == "customer"
