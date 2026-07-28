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

from app.scheduler.jobs import send_new_parent_followups, send_vaccination_reminders
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


@pytest.mark.asyncio
async def test_reminder_not_marked_sent_when_every_send_fails():
    """Real bug found via audit: reminder_sent was previously set
    unconditionally after the send loop, so a transient WhatsApp outage (or
    every member's 24h messaging window being closed) at the moment the cron
    fires silently and permanently excluded that vaccination from every
    future run (the query filters on `reminder_sent != True`), with no retry
    and no visible failure."""
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"}],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_vaccination_reminders(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()
    assert supabase.rows("vaccinations")[0]["reminder_sent"] is False


@pytest.mark.asyncio
async def test_reminder_marked_sent_when_at_least_one_send_succeeds():
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
                {"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"},
                {"pet_id": "pet-1", "profile_id": "profile-2", "role": "family"},
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"},
                {"id": "profile-2", "phone_number": "919000000002", "full_name": "Family"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    async def flaky_send(phone, text):
        if phone == "919000000001":
            raise RuntimeError("expired 24h window")

    ctx.whatsapp.send_text = AsyncMock(side_effect=flaky_send)

    await send_vaccination_reminders(ctx)

    assert supabase.rows("vaccinations")[0]["reminder_sent"] is True


@pytest.mark.asyncio
async def test_running_the_job_twice_never_double_sends_the_same_reminder():
    """The scheduler is in-process (app/scheduler/runner.py) and would run
    once per worker process if this service is ever deployed with more than
    one (see Dockerfile's --workers), plus once per replica if ever scaled
    horizontally -- without an atomic claim, two runs racing on the same
    "due" row would both send the customer the same reminder. This proves
    the end-to-end effect: running the job twice back-to-back for the same
    row sends it only once, because the first run's claim (an atomic
    conditional UPDATE, not a plain read-then-write) makes the row no longer
    "due" for the second run."""
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": "2026-08-01", "status": "scheduled", "reminder_sent": False,
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"}],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)
    await send_vaccination_reminders(ctx)

    assert ctx.whatsapp.send_text.await_count == 1


@pytest.mark.asyncio
async def test_new_parent_followup_running_twice_never_double_sends():
    supabase = FakeSupabaseClient(
        initial={
            "new_parent_followups": [
                {"id": "f1", "profile_id": "profile-1", "status": "pending", "due_at": "2020-01-01T00:00:00", "message_text": "Welcome!"}
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_new_parent_followups(ctx)
    await send_new_parent_followups(ctx)

    assert ctx.whatsapp.send_text.await_count == 1
    assert supabase.rows("new_parent_followups")[0]["status"] == "sent"


@pytest.mark.asyncio
async def test_new_parent_followup_claim_is_reverted_on_send_failure():
    supabase = FakeSupabaseClient(
        initial={
            "new_parent_followups": [
                {"id": "f1", "profile_id": "profile-1", "status": "pending", "due_at": "2020-01-01T00:00:00", "message_text": "Welcome!"}
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_new_parent_followups(ctx)

    assert supabase.rows("new_parent_followups")[0]["status"] == "pending"
