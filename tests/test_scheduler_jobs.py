"""Covers send_vaccination_reminders. Also the site of a real regression
found via live testing: it used to build the pet-members-with-phone-number
join as a raw `pet_members.select("profile_id, profiles(phone_number)")`
call, which crashed on the real schema (PGRST201, "Could not embed because
more than one relationship was found for 'pet_members' and 'profiles'")
since pet_members has two FKs into profiles (profile_id and added_by).
FakeSupabaseClient doesn't model that ambiguity, so this suite covers the
tier-gating behavior; the join fix itself was confirmed against the real
DB by reusing get_pet_member_contacts (already disambiguated there).

Per the product spec, Free customers now DO get vaccination reminders --
just the basic due-date ping, not the fuller Subscriber version (extra
manufacturer/batch detail when on file). Reminders are a content upgrade
by tier, not a send gate."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_vaccination_reminders
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


@pytest.mark.asyncio
async def test_free_and_subscriber_pet_members_both_get_a_reminder_different_detail():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
                    "manufacturer": "Zoetis", "batch_number": "LOT-1",
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [
                {"pet_id": "pet-1", "profile_id": "free-profile", "role": "owner"},
                {"pet_id": "pet-1", "profile_id": "sub-profile", "role": "family"},
            ],
            "profiles": [
                {"id": "free-profile", "phone_number": "919000000001", "full_name": "Free Owner"},
                {"id": "sub-profile", "phone_number": "919000000002", "full_name": "Sub Owner"},
            ],
            "subscriptions": [{"id": "sub-1", "profile_id": "sub-profile", "status": "active"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)

    sent = {call.args[0]: call.args[1] for call in ctx.whatsapp.send_text.await_args_list}
    assert set(sent) == {"919000000001", "919000000002"}
    assert "Zoetis" not in sent["919000000001"]
    assert "Zoetis" in sent["919000000002"]


@pytest.mark.asyncio
async def test_reminder_marked_sent_even_with_no_pet_members():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [],
            "profiles": [],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()
    assert supabase.rows("vaccinations")[0]["reminder_sent"] is True
