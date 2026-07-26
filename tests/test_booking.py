"""Reproduces and verifies the fix for a real reported bug: a customer with
an open vet-choice request for one pet was completely blocked from
starting a booking for a DIFFERENT pet, because the "already have an open
booking" guard only checked profile_id, never pet_id. Also covers the
naive-timestamp bug (a proposed/preferred time with no UTC offset landing
5.5 hours off once stored)."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.booking import (
    _normalize_to_ist,
    book_slot,
    cancel_session,
    deliver_prescription,
    file_prescription,
    handle_payment_webhook,
    mark_session_done,
    propose_time,
    request_doctor_session,
    reschedule_session,
    respond_to_recording_consent,
    respond_to_time_proposal,
)
from app.availability.slots import IST, Slot
from tests.fake_supabase import FakeSupabaseClient


def _make_ctx(supabase=None):
    whatsapp = SimpleNamespace(send_interactive_list=AsyncMock(), send_interactive_buttons=AsyncMock(), send_text=AsyncMock())
    return SimpleNamespace(supabase=supabase or FakeSupabaseClient(), whatsapp=whatsapp, settings=object())


def _make_agent_ctx(pets):
    return SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=pets)


@pytest.mark.asyncio
async def test_pending_request_for_one_pet_does_not_block_a_different_pet():
    pet_a = {"id": "pet-a", "name": "Max"}
    pet_b = {"id": "pet-b", "name": "Luna"}
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "pending_doctor_choice",
                    "status": "pending",
                }
            ],
            "profiles": [{"id": "vet-1", "role": "vet", "phone_number": "919000000001", "full_name": "Dr. Rao"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[pet_a, pet_b])

    result = await request_doctor_session(ctx, agent_ctx, pet_name="Luna", case_summary="routine checkup")

    assert result["success"] is True
    assert result["mode"] == "doctor_catalogue_sent"
    # A brand new session for pet_b must have been created, not blocked by pet_a's.
    sessions_for_luna = [s for s in supabase.rows("doctor_sessions") if s.get("pet_id") == "pet-b"]
    assert len(sessions_for_luna) == 1
    ctx.whatsapp.send_interactive_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_request_for_the_same_pet_resends_the_list_with_session_id():
    pet_a = {"id": "pet-a", "name": "Max"}
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "pending_doctor_choice",
                    "status": "pending",
                }
            ],
            "profiles": [{"id": "vet-1", "role": "vet", "phone_number": "919000000001", "full_name": "Dr. Rao"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[pet_a])

    result = await request_doctor_session(ctx, agent_ctx, pet_name="Max", case_summary="routine checkup")

    assert result["success"] is True
    assert result["mode"] == "doctor_catalogue_sent"
    assert result["session_id"] == "session-a"
    # No duplicate session row created for the same pet.
    sessions_for_max = [s for s in supabase.rows("doctor_sessions") if s.get("pet_id") == "pet-a"]
    assert len(sessions_for_max) == 1
    ctx.whatsapp.send_interactive_list.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_session_resolves_pet_id_instead_of_dropping_it():
    pet_a = {"id": "pet-a", "name": "Max"}
    supabase = FakeSupabaseClient(
        initial={"profiles": [{"id": "vet-1", "role": "vet", "phone_number": "919000000001", "full_name": "Dr. Rao"}]}
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[pet_a])

    result = await request_doctor_session(ctx, agent_ctx, pet_name="Max", case_summary="limping")

    assert result["success"] is True
    session = supabase.rows("doctor_sessions")[0]
    assert session["pet_id"] == "pet-a"


def test_normalize_to_ist_adds_offset_when_missing():
    normalized = _normalize_to_ist("2026-07-28T14:00:00")
    dt = datetime.fromisoformat(normalized)
    assert dt.tzinfo is not None
    assert dt.utcoffset() == IST.utcoffset(None)
    assert dt.hour == 14  # stays 14:00 IST, not reinterpreted as UTC


def test_normalize_to_ist_preserves_explicit_offset():
    normalized = _normalize_to_ist("2026-07-28T14:00:00+05:30")
    assert normalized == "2026-07-28T14:00:00+05:30"


def test_normalize_to_ist_rejects_garbage():
    assert _normalize_to_ist("not a time") is None
    assert _normalize_to_ist("") is None


@pytest.mark.asyncio
async def test_propose_time_stores_offset_aware_timestamp():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [{"id": "session-a", "profile_id": "profile-1", "doctor_phone": "919000000001"}],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await propose_time(ctx, agent_ctx, session_id="session-a", proposed_time="2026-07-28T14:00:00", proposed_by="customer")

    assert result["success"] is True
    session = supabase.rows("doctor_sessions")[0]
    assert "+05:30" in session["preferred_time"]


@pytest.mark.asyncio
async def test_mark_session_done_acknowledges_customer_immediately():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001", "status": "accepted"}
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await mark_session_done(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    ctx.whatsapp.send_text.assert_awaited_once()
    call_args = ctx.whatsapp.send_text.call_args
    assert call_args[0][0] == "919876543210"
    assert "Dr. Rao" in call_args[0][1]
    assert "Max" in call_args[0][1]
    assert "ended" in call_args[0][1]


@pytest.mark.asyncio
async def test_mark_session_done_is_idempotent_and_never_double_notifies():
    """Reproduces a real reported bug: two "your session has ended" messages
    reaching the customer. A re-tap, a duplicate inbound message, or the vet
    saying it twice must not fire the notification a second time once the
    session is already completed."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001", "status": "completed"}
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await mark_session_done(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    assert result["mode"] == "already_completed"
    ctx.whatsapp.send_text.assert_not_awaited()


def _prescription_test_session(**overrides):
    session = {
        "id": "session-a",
        "profile_id": "profile-1",
        "pet_id": "pet-a",
        "doctor_phone": "919000000001",
        "status": "completed",
        "case_summary": "Coughing for 2 days",
    }
    session.update(overrides)
    return FakeSupabaseClient(
        initial={
            "doctor_sessions": [session],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )


@pytest.mark.asyncio
async def test_file_prescription_asks_customer_for_format_choice_instead_of_sending():
    """Real behavior change: the vet's medications/treatment plan must not
    reach the customer directly — they choose text or PDF first."""
    supabase = _prescription_test_session()
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await file_prescription(
        ctx, agent_ctx, session_id="session-a", medications="Amoxicillin 250mg twice daily", treatment_plan="Rest for 5 days"
    )

    assert result["success"] is True
    assert result["mode"] == "prescription_filed_awaiting_format_choice"
    ctx.whatsapp.send_text.assert_not_awaited()
    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()
    call = ctx.whatsapp.send_interactive_buttons.call_args
    assert call.args[0] == "919876543210"
    button_ids = {b["id"] for b in call.args[2]}
    assert button_ids == {"prescription_format|session-a|text", "prescription_format|session-a|pdf"}

    record = supabase.rows("medical_records")[0]
    assert record["medications"] == "Amoxicillin 250mg twice daily"

    session = supabase.rows("doctor_sessions")[0]
    assert session["awaiting_from"] == "prescription_format_choice"
    assert session["pending_medications"] == "Amoxicillin 250mg twice daily"
    assert session["pending_treatment_plan"] == "Rest for 5 days"


@pytest.mark.asyncio
async def test_deliver_prescription_sends_text_when_chosen():
    supabase = _prescription_test_session(
        awaiting_from="prescription_format_choice",
        pending_medications="Amoxicillin 250mg twice daily",
        pending_treatment_plan="Rest for 5 days",
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="text")

    assert result["success"] is True
    assert result["format"] == "text"
    ctx.whatsapp.send_text.assert_awaited_once()
    message = ctx.whatsapp.send_text.call_args[0][1]
    assert "Max" in message
    assert "Dr. Rao" in message
    assert "Coughing for 2 days" in message
    assert "Amoxicillin 250mg twice daily" in message
    assert "Rest for 5 days" in message

    session = supabase.rows("doctor_sessions")[0]
    assert session["awaiting_from"] is None


@pytest.mark.asyncio
async def test_deliver_prescription_uses_llm_formatted_message_when_available(monkeypatch):
    """The vet's own shorthand should be reorganized into a clean prescription
    layout rather than just echoed back under bare Medications:/Treatment
    plan: labels — but the underlying clinical content must still reach the
    customer, whichever path produced it."""
    supabase = _prescription_test_session(
        awaiting_from="prescription_format_choice", pending_medications="amox 250 bid", pending_treatment_plan=""
    )
    ctx = _make_ctx(supabase)
    ctx.openai = object()
    formatted_text = "🐾 *Prescription — Max*\nVet: Dr. Rao\n\n*Medications*\nAmoxicillin 250mg — twice daily"
    monkeypatch.setattr("app.agent.tools.booking.text_completion", AsyncMock(return_value=formatted_text))
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="text")

    assert result["success"] is True
    ctx.whatsapp.send_text.assert_awaited_once_with("919876543210", formatted_text)


@pytest.mark.asyncio
async def test_deliver_prescription_falls_back_to_plain_layout_when_formatting_fails(monkeypatch):
    """A failure in the formatting call (OpenAI down, timeout, etc.) must
    never lose the customer's prescription entirely — fall back to the
    plain deterministic layout instead of erroring the whole tool call out."""
    supabase = _prescription_test_session(
        awaiting_from="prescription_format_choice",
        pending_medications="Amoxicillin 250mg twice daily",
        pending_treatment_plan="Rest for 5 days",
    )
    ctx = _make_ctx(supabase)
    ctx.openai = object()
    monkeypatch.setattr("app.agent.tools.booking.text_completion", AsyncMock(side_effect=RuntimeError("openai down")))
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="text")

    assert result["success"] is True
    message = ctx.whatsapp.send_text.call_args[0][1]
    assert "Amoxicillin 250mg twice daily" in message
    assert "Rest for 5 days" in message


@pytest.mark.asyncio
async def test_deliver_prescription_sends_pdf_when_chosen(monkeypatch):
    supabase = _prescription_test_session(
        awaiting_from="prescription_format_choice",
        pending_medications="Amoxicillin 250mg twice daily",
        pending_treatment_plan="Rest for 5 days",
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_document = AsyncMock()
    monkeypatch.setattr("app.agent.tools.booking.upload_to_storage", lambda *a, **k: None)
    monkeypatch.setattr("app.agent.tools.booking.sign_storage_url", lambda *a, **k: "https://signed.example/prescription.pdf")
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="pdf")

    assert result["success"] is True
    assert result["format"] == "pdf"
    ctx.whatsapp.send_text.assert_not_awaited()
    ctx.whatsapp.send_document.assert_awaited_once_with(
        "919876543210", "https://signed.example/prescription.pdf", "Max_prescription.pdf", "Prescription for Max from Dr. Rao"
    )
    doc = supabase.rows("documents")[0]
    assert doc["document_type"] == "Prescription"
    assert doc["pet_id"] == "pet-a"


@pytest.mark.asyncio
async def test_deliver_prescription_uses_the_treating_vets_own_profile_fields(monkeypatch):
    """Doctor name/qualification/registration on the PDF are per-consultation,
    sourced from the treating vet's own profile row — not hardcoded
    clinic-wide values."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "status": "completed",
                    "case_summary": "Coughing for 2 days",
                    "awaiting_from": "prescription_format_choice",
                    "pending_medications": "Amoxicillin 250mg twice daily",
                    "pending_treatment_plan": "Rest for 5 days",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {
                    "id": "vet-1",
                    "phone_number": "919000000001",
                    "full_name": "Dr. Rao",
                    "role": "vet",
                    "qualification": "BVSc & AH, MVSc (Surgery)",
                    "registration_number": "TNVC-2024-00123",
                },
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.whatsapp.send_document = AsyncMock()
    monkeypatch.setattr("app.agent.tools.booking.upload_to_storage", lambda *a, **k: None)
    monkeypatch.setattr("app.agent.tools.booking.sign_storage_url", lambda *a, **k: "https://signed.example/prescription.pdf")
    captured = {}

    def fake_build_prescription_pdf(**kwargs):
        captured.update(kwargs)
        return b"%PDF-1.3 fake"

    monkeypatch.setattr("app.agent.tools.booking.build_prescription_pdf", fake_build_prescription_pdf)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="pdf")

    assert result["success"] is True
    assert captured["doctor_qualification"] == "BVSc & AH, MVSc (Surgery)"
    assert captured["doctor_registration_number"] == "TNVC-2024-00123"


@pytest.mark.asyncio
async def test_deliver_prescription_succeeds_even_if_pdf_generation_fails(monkeypatch):
    """The PDF send failing must never fail the whole tool call or leave
    the customer's chosen format un-delivered without explanation."""
    supabase = _prescription_test_session(
        awaiting_from="prescription_format_choice",
        pending_medications="Amoxicillin 250mg twice daily",
        pending_treatment_plan="Rest for 5 days",
    )
    ctx = _make_ctx(supabase)
    monkeypatch.setattr(
        "app.agent.tools.booking.upload_to_storage",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("storage down")),
    )
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="pdf")

    assert result["success"] is True
    assert result["format"] == "pdf"


@pytest.mark.asyncio
async def test_deliver_prescription_is_idempotent_once_already_delivered():
    supabase = _prescription_test_session(awaiting_from=None)
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="text")

    assert result["success"] is True
    assert result["mode"] == "already_delivered"
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_deliver_prescription_rejects_invalid_format():
    supabase = _prescription_test_session(awaiting_from="prescription_format_choice")
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await deliver_prescription(ctx, agent_ctx, session_id="session-a", format="carrier pigeon")

    assert result["success"] is False
    assert result["error"] == "invalid_format"
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_rescheduling_a_confirmed_session_updates_the_existing_calendar_event(monkeypatch):
    """A session with a calendar_event_id already set means a real Meet-linked
    event already exists (confirmed booking, not just a pending request) —
    accepting a new proposed time for it must UPDATE that event, never create
    a second one that leaves the old one orphaned."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "status": "negotiating",
                    "awaiting_from": "doctor_time_input",
                    "preferred_time": "2026-07-28T14:00:00+05:30",
                    "calendar_event_id": "existing-event-123",
                    "meet_link": "https://meet.google.com/existing-link",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    create_called = AsyncMock()
    update_called = AsyncMock(return_value={"success": True, "event_id": "existing-event-123", "meet_link": "https://meet.google.com/existing-link"})
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.create_event_with_meet", create_called)
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.update_event_time", update_called)

    result = await respond_to_time_proposal(ctx, agent_ctx, session_id="session-a", decision="accept")

    assert result["success"] is True
    assert result["mode"] == "rescheduled"
    update_called.assert_awaited_once()
    create_called.assert_not_called()
    session = supabase.rows("doctor_sessions")[0]
    assert session["calendar_event_id"] == "existing-event-123"
    assert "rescheduled to" in ctx.whatsapp.send_text.call_args_list[0][0][1]


@pytest.mark.asyncio
async def test_reschedule_session_derives_proposed_by_from_role():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [{"id": "session-a", "profile_id": "profile-1", "doctor_phone": "919000000001"}],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=[], role="customer")

    result = await reschedule_session(ctx, agent_ctx, session_id="session-a", new_time="2026-07-29T15:00:00+05:30")

    assert result["success"] is True
    session = supabase.rows("doctor_sessions")[0]
    assert session["status"] == "negotiating"
    assert session["awaiting_from"] == "doctor_time_input"  # customer proposed -> awaiting the vet


@pytest.mark.asyncio
async def test_cancel_session_deletes_the_real_calendar_event(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "doctor_phone": "919000000001",
                    "calendar_event_id": "existing-event-123",
                }
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=[], role="customer")

    delete_called = AsyncMock()
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.delete_event", delete_called)

    result = await cancel_session(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    delete_called.assert_awaited_once_with(ctx.settings, "existing-event-123")
    session = supabase.rows("doctor_sessions")[0]
    assert session["status"] == "cancelled"


@pytest.mark.asyncio
async def test_cancel_session_survives_calendar_delete_failure(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "doctor_phone": "919000000001", "calendar_event_id": "gone-already"}
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=[], role="customer")

    async def boom(*args, **kwargs):
        raise RuntimeError("event not found")

    monkeypatch.setattr("app.agent.tools.booking.google_calendar.delete_event", boom)

    result = await cancel_session(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    session = supabase.rows("doctor_sessions")[0]
    assert session["status"] == "cancelled"


@pytest.mark.asyncio
async def test_session_notifications_broadcast_to_every_household_member():
    """A pet with multiple pet_members (owner + family added via
    add_pet_member) must have session notifications reach ALL of them, not
    just whichever one happens to be driving the current conversation."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001", "status": "accepted"}
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "profile-2", "phone_number": "919111111111", "full_name": "John"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
            "pet_members": [
                {"pet_id": "pet-a", "profile_id": "profile-1", "role": "owner"},
                {"pet_id": "pet-a", "profile_id": "profile-2", "role": "family"},
            ],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    result = await mark_session_done(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    notified_phones = {call.args[0] for call in ctx.whatsapp.send_text.call_args_list}
    assert notified_phones == {"919876543210", "919111111111"}


@pytest.mark.asyncio
async def test_relay_to_customer_attributes_to_vet_and_broadcasts():
    from app.agent.tools.booking import relay_to_customer

    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [{"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001"}],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "profile-2", "phone_number": "919111111111", "full_name": "John"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
            "pet_members": [
                {"pet_id": "pet-a", "profile_id": "profile-1", "role": "owner"},
                {"pet_id": "pet-a", "profile_id": "profile-2", "role": "family"},
            ],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "vet-1", "phone_number": "919000000001"}, pets=[], role="vet")

    result = await relay_to_customer(ctx, agent_ctx, session_id="session-a", message="Keep him hydrated and rest for 2 days.")

    assert result["success"] is True
    notified_phones = {call.args[0] for call in ctx.whatsapp.send_text.call_args_list}
    assert notified_phones == {"919876543210", "919111111111"}
    message = ctx.whatsapp.send_text.call_args_list[0].args[1]
    assert "Dr. Rao" in message
    assert "Max" in message
    assert "Keep him hydrated" in message


@pytest.mark.asyncio
async def test_relay_to_doctor_attributes_to_customer():
    from app.agent.tools.booking import relay_to_doctor

    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [{"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001"}],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=[], role="customer")

    result = await relay_to_doctor(ctx, agent_ctx, session_id="session-a", message="He's been vomiting again this morning.")

    assert result["success"] is True
    ctx.whatsapp.send_text.assert_awaited_once_with("919000000001", "*Message from Jane* (re: Max):\nHe's been vomiting again this morning.")


@pytest.mark.asyncio
async def test_finalize_booking_passes_attendee_emails(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "pending_doctor_choice",
                    "status": "pending",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane", "email": "jane@example.com"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet", "email": "dr.rao@example.com"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
            "pet_members": [{"pet_id": "pet-a", "profile_id": "profile-1", "role": "owner"}],
        }
    )
    ctx = _make_ctx(supabase)

    create_called = AsyncMock(return_value={"success": True, "event_id": "evt-1", "meet_link": "https://meet.google.com/xyz"})
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.create_event_with_meet", create_called)
    monkeypatch.setattr("app.availability.slots.compute_doctor_slots", AsyncMock(return_value=[]))

    from app.agent.tools.booking import _finalize_booking
    from datetime import datetime as dt

    session = supabase.rows("doctor_sessions")[0]
    start = dt.fromisoformat("2026-07-28T14:00:00+05:30")
    end = dt.fromisoformat("2026-07-28T14:30:00+05:30")

    result = await _finalize_booking(ctx, session, "919000000001", start, end)

    assert result["success"] is True
    create_called.assert_awaited_once()
    kwargs = create_called.call_args.kwargs
    assert set(kwargs["attendees"]) == {"jane@example.com", "dr.rao@example.com"}


@pytest.mark.asyncio
async def test_book_slot_requests_payment_instead_of_finalizing_immediately(monkeypatch):
    """A booking must never go straight to a Calendar event/Meet link —
    the consult fee has to be collected first. book_slot should hold the
    slot on the session and send a Razorpay payment link, not confirm."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001", "status": "pending"}
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    ctx.settings = SimpleNamespace(razorpay_consult_fee_inr=500, razorpay_key_id="x", razorpay_key_secret="y")
    agent_ctx = _make_agent_ctx(pets=[])

    slot_start = "2026-07-28T14:00:00+05:30"
    start_dt = datetime.fromisoformat(slot_start)
    monkeypatch.setattr("app.agent.tools.booking.compute_doctor_slots", AsyncMock(return_value=[Slot(start=start_dt, end=start_dt)]))
    create_link = AsyncMock(return_value={"id": "plink_123", "short_url": "https://rzp.io/i/abc123"})
    monkeypatch.setattr("app.agent.tools.booking.razorpay_client.create_payment_link", create_link)

    result = await book_slot(ctx, agent_ctx, session_id="session-a", slot_start=slot_start, doctor_phone="919000000001")

    assert result["success"] is True
    assert result["mode"] == "payment_requested"
    assert result["payment_link"] == "https://rzp.io/i/abc123"
    create_link.assert_awaited_once()
    assert create_link.call_args.kwargs["reference_id"] == "session-a"

    session = supabase.rows("doctor_sessions")[0]
    assert session["payment_status"] == "awaiting"
    assert session["payment_link_id"] == "plink_123"
    assert session["awaiting_from"] == "payment"
    assert session["preferred_time"] == slot_start

    ctx.whatsapp.send_text.assert_awaited_once()
    assert "https://rzp.io/i/abc123" in ctx.whatsapp.send_text.call_args.args[1]


@pytest.mark.asyncio
async def test_handle_payment_webhook_asks_for_recording_consent_instead_of_finalizing(monkeypatch):
    """Payment alone must NOT create the Calendar event — the recording-consent
    question has to go out first, and finalization waits for the customer's
    answer (see respond_to_recording_consent)."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "status": "pending",
                    "awaiting_from": "payment",
                    "payment_status": "awaiting",
                    "payment_link_id": "plink_123",
                    "preferred_time": "2026-07-28T14:00:00+05:30",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)

    create_called = AsyncMock(return_value={"success": True, "event_id": "evt-1", "meet_link": "https://meet.google.com/xyz"})
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.create_event_with_meet", create_called)

    event_body = {
        "event": "payment_link.paid",
        "payload": {"payment_link": {"entity": {"id": "plink_123", "reference_id": "session-a"}}},
    }

    handled = await handle_payment_webhook(ctx, event_body)

    assert handled is True
    create_called.assert_not_called()
    session = supabase.rows("doctor_sessions")[0]
    assert session["payment_status"] == "paid"
    assert session["awaiting_from"] == "recording_consent"
    assert session["recording_consent"] == "pending"
    assert session["status"] == "pending"  # not yet accepted — still waiting on consent

    ctx.whatsapp.send_interactive_buttons.assert_awaited_once()
    call = ctx.whatsapp.send_interactive_buttons.call_args
    assert call.args[0] == "919876543210"
    button_ids = {b["id"] for b in call.args[2]}
    assert button_ids == {"recording_consent|session-a|yes", "recording_consent|session-a|no"}


@pytest.mark.asyncio
@pytest.mark.parametrize("consent,expected", [(True, "given"), (False, "declined")])
async def test_respond_to_recording_consent_finalizes_booking_either_way(monkeypatch, consent, expected):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "status": "pending",
                    "awaiting_from": "recording_consent",
                    "payment_status": "paid",
                    "recording_consent": "pending",
                    "preferred_time": "2026-07-28T14:00:00+05:30",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = _make_agent_ctx(pets=[])

    create_called = AsyncMock(return_value={"success": True, "event_id": "evt-1", "meet_link": "https://meet.google.com/xyz"})
    monkeypatch.setattr("app.agent.tools.booking.google_calendar.create_event_with_meet", create_called)

    result = await respond_to_recording_consent(ctx, agent_ctx, session_id="session-a", consent=consent)

    assert result["success"] is True
    assert result["mode"] == "booked"
    create_called.assert_awaited_once()
    session = supabase.rows("doctor_sessions")[0]
    assert session["recording_consent"] == expected
    assert session["status"] == "accepted"

    doctor_call = next(c for c in ctx.whatsapp.send_text.call_args_list if c.args[0] == "919000000001")
    expected_label = "Recording consent: Given" if consent else "Recording consent: Declined — do not record"
    assert expected_label in doctor_call.args[1]


@pytest.mark.asyncio
async def test_handle_payment_webhook_is_idempotent_for_already_paid_session():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "doctor_phone": "919000000001", "payment_status": "paid"}
            ],
            "profiles": [{"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"}],
        }
    )
    ctx = _make_ctx(supabase)
    event_body = {
        "event": "payment_link.paid",
        "payload": {"payment_link": {"entity": {"id": "plink_123", "reference_id": "session-a"}}},
    }

    handled = await handle_payment_webhook(ctx, event_body)

    assert handled is False
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_payment_webhook_ignores_non_paid_events():
    ctx = _make_ctx()
    event_body = {"event": "payment_link.expired", "payload": {}}

    handled = await handle_payment_webhook(ctx, event_body)

    assert handled is False


@pytest.mark.asyncio
async def test_finalize_booking_sends_a_detailed_assignment_notice_to_the_doctor(monkeypatch):
    """The vet needs enough detail in ONE message to know who/what/when
    without digging through chat history — patient (with species/breed),
    owner name + phone, time, reason for visit, and the Meet link."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "pending_doctor_choice",
                    "status": "pending",
                    "case_summary": "Limping on left hind leg for 3 days",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max", "species": "Dog", "breed": "Labrador"}],
        }
    )
    ctx = _make_ctx(supabase)
    monkeypatch.setattr(
        "app.agent.tools.booking.google_calendar.create_event_with_meet",
        AsyncMock(return_value={"success": True, "event_id": "evt-1", "meet_link": "https://meet.google.com/xyz"}),
    )

    from app.agent.tools.booking import _finalize_booking
    from datetime import datetime as dt

    session = supabase.rows("doctor_sessions")[0]
    start = dt.fromisoformat("2026-07-28T14:00:00+05:30")
    end = dt.fromisoformat("2026-07-28T14:30:00+05:30")

    result = await _finalize_booking(ctx, session, "919000000001", start, end)

    assert result["success"] is True
    assert result["mode"] == "booked"
    doctor_call = next(c for c in ctx.whatsapp.send_text.call_args_list if c.args[0] == "919000000001")
    message = doctor_call.args[1]
    assert "New session assigned to you" in message
    assert "Max (Dog, Labrador)" in message
    assert "Jane" in message and "919876543210" in message
    assert "Limping on left hind leg for 3 days" in message
    assert "https://meet.google.com/xyz" in message


@pytest.mark.asyncio
async def test_notify_household_continues_past_one_members_send_failure():
    """Real bug found via live data: one household member's WhatsApp send
    failing (expired 24h messaging window, invalid number, rate limit) used
    to raise out of the loop and skip every member queued after it."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "profile_id": "profile-1", "pet_id": "pet-a", "doctor_phone": "919000000001", "status": "accepted"}
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "profile-2", "phone_number": "919111111111", "full_name": "John"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
            "pet_members": [
                {"pet_id": "pet-a", "profile_id": "profile-1", "role": "owner"},
                {"pet_id": "pet-a", "profile_id": "profile-2", "role": "family"},
            ],
        }
    )
    ctx = _make_ctx(supabase)

    async def flaky_send(phone, message):
        if phone == "919876543210":
            raise RuntimeError("WhatsApp send failed")

    ctx.whatsapp.send_text = AsyncMock(side_effect=flaky_send)

    from app.agent.tools.booking import _notify_household

    session = supabase.rows("doctor_sessions")[0]
    await _notify_household(ctx, session, "Your session is confirmed.")

    notified_phones = {c.args[0] for c in ctx.whatsapp.send_text.call_args_list}
    assert notified_phones == {"919876543210", "919111111111"}


@pytest.mark.asyncio
async def test_finalize_booking_still_notifies_doctor_when_household_notification_fails(monkeypatch):
    """Real bug found via live data: a WhatsApp delivery failure to the
    customer/household used to raise out of _finalize_booking before it ever
    reached the doctor-notification line right after it — so the vet
    silently never heard about a session that was, in fact, booked (calendar
    event + DB row already committed by that point)."""
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "pending_doctor_choice",
                    "status": "pending",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)

    async def flaky_send(phone, message):
        if phone == "919876543210":
            raise RuntimeError("WhatsApp 24h window expired")

    ctx.whatsapp.send_text = AsyncMock(side_effect=flaky_send)
    monkeypatch.setattr(
        "app.agent.tools.booking.google_calendar.create_event_with_meet",
        AsyncMock(return_value={"success": True, "event_id": "evt-1", "meet_link": "https://meet.google.com/xyz"}),
    )

    from app.agent.tools.booking import _finalize_booking
    from datetime import datetime as dt

    session = supabase.rows("doctor_sessions")[0]
    start = dt.fromisoformat("2026-07-28T14:00:00+05:30")
    end = dt.fromisoformat("2026-07-28T14:30:00+05:30")

    result = await _finalize_booking(ctx, session, "919000000001", start, end)

    assert result["success"] is True
    doctor_call = next(c for c in ctx.whatsapp.send_text.call_args_list if c.args[0] == "919000000001")
    assert "New session assigned to you" in doctor_call.args[1]


@pytest.mark.asyncio
async def test_reschedule_confirmation_notifies_doctor_with_rescheduled_header(monkeypatch):
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "status": "negotiating",
                    "awaiting_from": "doctor_time_input",
                    "preferred_time": "2026-07-28T14:00:00+05:30",
                    "calendar_event_id": "existing-event-123",
                    "meet_link": "https://meet.google.com/existing-link",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max"}],
        }
    )
    ctx = _make_ctx(supabase)
    monkeypatch.setattr(
        "app.agent.tools.booking.google_calendar.update_event_time",
        AsyncMock(return_value={"success": True, "event_id": "existing-event-123", "meet_link": "https://meet.google.com/existing-link"}),
    )

    result = await respond_to_time_proposal(ctx, agent_ctx=SimpleNamespace(profile={"id": "profile-1"}, pets=[]), session_id="session-a", decision="accept")

    assert result["success"] is True
    doctor_call = next(c for c in ctx.whatsapp.send_text.call_args_list if c.args[0] == "919000000001")
    assert "Session rescheduled" in doctor_call.args[1]


@pytest.mark.asyncio
async def test_cancel_session_sends_detailed_notice_to_doctor_when_customer_cancels():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {
                    "id": "session-a",
                    "profile_id": "profile-1",
                    "pet_id": "pet-a",
                    "doctor_phone": "919000000001",
                    "preferred_time": "2026-07-28T14:00:00+05:30",
                    "case_summary": "Annual checkup",
                }
            ],
            "profiles": [
                {"id": "profile-1", "phone_number": "919876543210", "full_name": "Jane"},
                {"id": "vet-1", "phone_number": "919000000001", "full_name": "Dr. Rao", "role": "vet"},
            ],
            "pets": [{"id": "pet-a", "name": "Max", "species": "Dog"}],
        }
    )
    ctx = _make_ctx(supabase)
    agent_ctx = SimpleNamespace(profile={"id": "profile-1", "phone_number": "919876543210"}, pets=[], role="customer")

    result = await cancel_session(ctx, agent_ctx, session_id="session-a")

    assert result["success"] is True
    ctx.whatsapp.send_text.assert_awaited_once()
    message = ctx.whatsapp.send_text.call_args.args[1]
    assert ctx.whatsapp.send_text.call_args.args[0] == "919000000001"
    assert "Session cancelled" in message
    assert "Max (Dog)" in message
    assert "Jane" in message and "919876543210" in message
    assert "Annual checkup" in message
    assert "Tue 28 Jul, 02:00 PM IST" in message
