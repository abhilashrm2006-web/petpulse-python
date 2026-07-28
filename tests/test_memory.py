"""extract_and_update_memory must scope the `memory` row it reads/writes by
pet_id, not just profile_id -- a real cross-pet data-contamination bug found
via audit: a multi-pet Subscriber account was reading/writing a single
profile_id-only row regardless of which pet the turn was about, so the
second pet discussed silently overwrote the first pet's durable facts."""

import json

import pytest

from app.agent.memory import extract_and_update_memory
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    from types import SimpleNamespace

    return SimpleNamespace(supabase=supabase)


@pytest.mark.asyncio
async def test_two_pets_on_one_account_get_separate_memory_rows(monkeypatch):
    supabase = FakeSupabaseClient()

    async def fake_json_completion(client, settings, system_prompt, user_prompt, reasoning_effort="low"):
        if "Rex" in user_prompt:
            return json.dumps({"pet_name": "Rex", "species": "Dog"})
        return json.dumps({"pet_name": "Whiskers", "species": "Cat"})

    monkeypatch.setattr("app.agent.memory.json_completion", fake_json_completion)

    await extract_and_update_memory(None, None, supabase, "profile-1", "Rex is a dog", "Noted!", pet_id="pet-rex")
    await extract_and_update_memory(None, None, supabase, "profile-1", "Whiskers is a cat", "Noted!", pet_id="pet-whiskers")

    rows = supabase.rows("memory")
    assert len(rows) == 2
    rex_row = next(r for r in rows if r["pet_id"] == "pet-rex")
    whiskers_row = next(r for r in rows if r["pet_id"] == "pet-whiskers")
    assert rex_row["pet_name"] == "Rex"
    assert whiskers_row["pet_name"] == "Whiskers"


@pytest.mark.asyncio
async def test_same_pet_updates_its_own_row_in_place(monkeypatch):
    supabase = FakeSupabaseClient()
    calls = iter([json.dumps({"pet_name": "Rex", "species": "Dog"}), json.dumps({"breed": "Labrador"})])

    async def fake_json_completion(*args, **kwargs):
        return next(calls)

    monkeypatch.setattr("app.agent.memory.json_completion", fake_json_completion)

    await extract_and_update_memory(None, None, supabase, "profile-1", "Rex is a dog", "Noted!", pet_id="pet-rex")
    await extract_and_update_memory(None, None, supabase, "profile-1", "He's a Labrador", "Noted!", pet_id="pet-rex")

    rows = supabase.rows("memory")
    assert len(rows) == 1
    assert rows[0]["pet_name"] == "Rex"
    assert rows[0]["breed"] == "Labrador"


@pytest.mark.asyncio
async def test_no_active_pet_uses_an_account_level_row(monkeypatch):
    supabase = FakeSupabaseClient()

    async def fake_json_completion(*args, **kwargs):
        return json.dumps({"communication_style": "casual"})

    monkeypatch.setattr("app.agent.memory.json_completion", fake_json_completion)

    await extract_and_update_memory(None, None, supabase, "profile-1", "keep it casual please", "Sure!", pet_id=None)

    rows = supabase.rows("memory")
    assert len(rows) == 1
    assert rows[0]["pet_id"] is None
