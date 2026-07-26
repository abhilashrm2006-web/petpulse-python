"""Covers send_vaccination_reminders. Also the site of a real regression
found via live testing: it used to build the pet-members-with-phone-number
join as a raw `pet_members.select("profile_id, profiles(phone_number)")`
call, which crashed on the real schema (PGRST201, "Could not embed because
more than one relationship was found for 'pet_members' and 'profiles'")
since pet_members has two FKs into profiles (profile_id and added_by).
FakeSupabaseClient doesn't model that ambiguity, so this suite covers the
tier-gating behavior; the join fix itself was confirmed against the real
DB by reusing get_pet_member_contacts (already disambiguated there)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_vaccination_reminders
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


@pytest.mark.asyncio
async def test_only_active_subscriber_pet_members_get_a_reminder():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
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

    sent_numbers = {call.args[0] for call in ctx.whatsapp.send_text.await_args_list}
    assert sent_numbers == {"919000000002"}


@pytest.mark.asyncio
async def test_no_active_subscribers_sends_nothing_but_still_marks_reminder_sent():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "free-profile", "role": "owner"}],
            "profiles": [{"id": "free-profile", "phone_number": "919000000001", "full_name": "Free Owner"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)

    ctx.whatsapp.send_text.assert_not_awaited()
    assert supabase.rows("vaccinations")[0]["reminder_sent"] is True
