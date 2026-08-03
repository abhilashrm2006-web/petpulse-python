"""Covers send_vaccination_reminders (both the T-3..T-0 daily countdown and
the separate one-shot overdue ping) and send_new_parent_followups. Also the
site of a real regression found via live testing: it used to build the
pet-members-with-phone-number join as a raw
`pet_members.select("profile_id, profiles(phone_number)")` call, which
crashed on the real schema (PGRST201, "Could not embed because more than one
relationship was found for 'pet_members' and 'profiles'") since pet_members
has two FKs into profiles (profile_id and added_by). FakeSupabaseClient
doesn't model that ambiguity; the join fix itself was confirmed against the
real DB by reusing get_pet_member_contacts (already disambiguated there).

Every pet member gets the full reminder detail (manufacturer/batch when on
file) -- there's no more tier distinction to gate content on.

Dates are computed relative to date.today() throughout (not hardcoded) so
these tests stay valid regardless of when they're run -- the T-3..T-0
countdown window is itself relative to "today"."""

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.scheduler.jobs import send_new_parent_followups, send_vaccination_reminders
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase):
    return SimpleNamespace(supabase=supabase, whatsapp=SimpleNamespace(send_text=AsyncMock()))


def _due_in(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


@pytest.mark.asyncio
async def test_every_pet_member_gets_a_countdown_reminder_with_full_detail():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {
                    "id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies",
                    "next_due_date": _due_in(3), "status": "scheduled",
                    "manufacturer": "Zoetis", "batch_number": "LOT-1",
                }
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [
                {"pet_id": "pet-1", "profile_id": "owner-profile", "role": "owner"},
                {"pet_id": "pet-1", "profile_id": "family-profile", "role": "family"},
            ],
            "profiles": [
                {"id": "owner-profile", "phone_number": "919000000001", "full_name": "Owner"},
                {"id": "family-profile", "phone_number": "919000000002", "full_name": "Family Member"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)

    sent = {call.args[0]: call.args[1] for call in ctx.whatsapp.send_text.await_args_list}
    assert set(sent) == {"919000000001", "919000000002"}
    assert "Zoetis" in sent["919000000001"]
    assert "Zoetis" in sent["919000000002"]
    assert "3 days" in sent["919000000001"]


@pytest.mark.asyncio
@pytest.mark.parametrize("offset,phrase", [(3, "3 days"), (2, "2 days"), (1, "tomorrow"), (0, "today")])
async def test_countdown_sends_the_right_phrase_for_each_day_offset(offset, phrase):
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(offset), "status": "scheduled"}
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"}],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)

    ctx.whatsapp.send_text.assert_awaited_once()
    assert phrase in ctx.whatsapp.send_text.await_args.args[1]
    assert supabase.rows("vaccinations")[0]["last_reminder_offset_sent"] == offset


@pytest.mark.asyncio
async def test_countdown_progresses_through_each_day_but_never_goes_backwards():
    """The whole point of the countdown: a customer gets a ping on T-3, T-2,
    T-1, AND T-0 -- four separate reminders leading up to the due date, not
    just one. Simulates each day passing by re-pointing next_due_date closer
    (equivalent to date.today() advancing by one from the row's perspective)
    and confirms each new, smaller offset sends -- while a same-or-larger
    offset (the row already has a smaller one on file) never re-sends,
    which is what actually prevents same-day double-sends across workers."""
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(3), "status": "scheduled"}
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"}],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"}],
        }
    )
    ctx = _make_ctx(supabase)

    await send_vaccination_reminders(ctx)  # T-3
    assert ctx.whatsapp.send_text.await_count == 1
    assert supabase.rows("vaccinations")[0]["last_reminder_offset_sent"] == 3

    await send_vaccination_reminders(ctx)  # same day again -- must not re-send
    assert ctx.whatsapp.send_text.await_count == 1

    supabase.rows("vaccinations")[0]["next_due_date"] = _due_in(2)  # a day passes
    await send_vaccination_reminders(ctx)  # T-2
    assert ctx.whatsapp.send_text.await_count == 2
    assert supabase.rows("vaccinations")[0]["last_reminder_offset_sent"] == 2

    supabase.rows("vaccinations")[0]["next_due_date"] = _due_in(0)  # two more days pass
    await send_vaccination_reminders(ctx)  # T-0
    assert ctx.whatsapp.send_text.await_count == 3
    assert supabase.rows("vaccinations")[0]["last_reminder_offset_sent"] == 0


@pytest.mark.asyncio
async def test_countdown_claim_is_reverted_on_total_send_failure():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(2), "status": "scheduled"}
            ],
            "pets": [{"id": "pet-1", "name": "Rex"}],
            "pet_members": [{"pet_id": "pet-1", "profile_id": "profile-1", "role": "owner"}],
            "profiles": [{"id": "profile-1", "phone_number": "919000000001", "full_name": "Owner"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_text = AsyncMock(side_effect=RuntimeError("WhatsApp outage"))

    await send_vaccination_reminders(ctx)

    assert supabase.rows("vaccinations")[0]["last_reminder_offset_sent"] is None


@pytest.mark.asyncio
async def test_overdue_reminder_marked_sent_even_with_no_pet_members():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(-1), "status": "scheduled", "reminder_sent": False}
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
async def test_overdue_reminder_not_marked_sent_when_every_send_fails():
    """Real bug found via audit: reminder_sent was previously set
    unconditionally after the send loop, so a transient WhatsApp outage (or
    every member's 24h messaging window being closed) at the moment the cron
    fires silently and permanently excluded that vaccination from every
    future run, with no retry and no visible failure."""
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(-1), "status": "scheduled", "reminder_sent": False}
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
async def test_overdue_reminder_marked_sent_when_at_least_one_send_succeeds():
    supabase = FakeSupabaseClient(
        initial={
            "vaccinations": [
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(-1), "status": "scheduled", "reminder_sent": False}
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
    assert supabase.rows("vaccinations")[0]["status"] == "overdue"


@pytest.mark.asyncio
async def test_overdue_reminder_running_twice_never_double_sends():
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
                {"id": "vax-1", "pet_id": "pet-1", "vaccine_name": "Rabies", "next_due_date": _due_in(-1), "status": "scheduled", "reminder_sent": False}
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
