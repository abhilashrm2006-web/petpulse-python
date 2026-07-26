"""Reproduces and verifies the fix for a real reported bug: registering a
new pet via several save_onboarding_field calls in one turn (name, then
species/breed/age — exactly what the system prompt tells the agent to do)
silently dropped every field after the first, because agent_ctx.pets was
loaded once at turn start and never saw the pet just created moments
earlier in the same turn."""

from types import SimpleNamespace

import pytest

from app.agent.tools.onboarding import _normalize_value, save_onboarding_field
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx():
    return SimpleNamespace(supabase=FakeSupabaseClient())


def _make_agent_ctx(pets=None):
    return SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=pets if pets is not None else [])


@pytest.mark.asyncio
async def test_naming_a_new_pet_then_setting_species_in_the_same_turn_succeeds():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx(pets=[])

    name_result = await save_onboarding_field(ctx, agent_ctx, field="pet_name", value="Max")
    assert name_result["success"] is True
    assert name_result.get("created_pet") is True

    # Same agent_ctx, same turn — species call must now be able to find "Max"
    # without a fresh build_context() reload.
    species_result = await save_onboarding_field(ctx, agent_ctx, field="species", value="dog", pet_name="Max")
    assert species_result["success"] is True
    assert species_result["savedValue"] == "Dog"

    breed_result = await save_onboarding_field(ctx, agent_ctx, field="breed", value="Labrador", pet_name="Max")
    assert breed_result["success"] is True

    stored_pet = ctx.supabase.rows("pets")[0]
    assert stored_pet["name"] == "Max"
    assert stored_pet["species"] == "Dog"
    assert stored_pet["breed"] == "Labrador"


@pytest.mark.asyncio
async def test_created_pet_is_appended_to_agent_ctx_pets_in_place():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx(pets=[])

    await save_onboarding_field(ctx, agent_ctx, field="pet_name", value="Rex")

    assert len(agent_ctx.pets) == 1
    assert agent_ctx.pets[0]["name"] == "Rex"


@pytest.mark.asyncio
async def test_renaming_existing_pet_updates_agent_ctx_pets_in_place_too():
    ctx = _make_ctx()
    existing_pet = {"id": "pet-1", "name": "Buddy", "species": "Dog"}
    agent_ctx = _make_agent_ctx(pets=[existing_pet])
    ctx.supabase._store["pets"] = [dict(existing_pet)]

    result = await save_onboarding_field(ctx, agent_ctx, field="breed", value="Beagle", pet_name="Buddy")

    assert result["success"] is True
    assert agent_ctx.pets[0]["breed"] == "Beagle"


@pytest.mark.asyncio
async def test_new_pet_creation_survives_a_db_trigger_already_creating_pet_members():
    """Reproduces a real bug found in live testing: this Supabase project has
    a trigger that auto-creates the owner pet_members row when a pet is
    inserted, so our own explicit insert always conflicted (23505) and
    crashed the whole turn — every single new-pet registration failed."""
    ctx = _make_ctx()
    ctx.supabase.force_conflict_on_insert("pet_members")
    agent_ctx = _make_agent_ctx(pets=[])

    result = await save_onboarding_field(ctx, agent_ctx, field="pet_name", value="Bella")

    assert result["success"] is True
    assert result["created_pet"] is True
    assert ctx.supabase.rows("pets")[0]["name"] == "Bella"


@pytest.mark.asyncio
async def test_gender_field_is_supported_and_normalized():
    ctx = _make_ctx()
    existing_pet = {"id": "pet-1", "name": "Bella", "species": "Cat"}
    agent_ctx = _make_agent_ctx(pets=[existing_pet])
    ctx.supabase._store["pets"] = [dict(existing_pet)]

    result = await save_onboarding_field(ctx, agent_ctx, field="gender", value="female", pet_name="Bella")

    assert result["success"] is True
    assert result["savedValue"] == "Female"
    assert agent_ctx.pets[0]["gender"] == "Female"


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("1.5 years", 2),
        ("2.5 years", 3),
        ("3.5 years", 4),
        ("2 years", 2),
    ],
)
def test_age_rounds_half_up_consistently(spoken, expected):
    """Audit bug: plain round() uses round-half-to-even, so "2.5" used to
    round DOWN to 2 while "1.5"/"3.5" rounded up -- inconsistent behavior a
    user would never expect for their own pet's age."""
    assert _normalize_value("age", spoken) == expected
