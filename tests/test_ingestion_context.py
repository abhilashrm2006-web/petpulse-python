"""_load_awaiting_prescription_session must find a completed session still
awaiting_from="doctor_prescription" for a vet -- without this, a vet's
follow-up document upload has no deterministic way to know which session
it belongs to (open-session queries elsewhere only cover
pending/negotiating/accepted, not completed)."""

from app.ingestion.context import _load_awaiting_prescription_session
from tests.fake_supabase import FakeSupabaseClient


def test_finds_completed_session_awaiting_prescription():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919000000001", "status": "completed", "awaiting_from": "doctor_prescription"},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is not None
    assert result["id"] == "session-a"


def test_ignores_sessions_not_awaiting_prescription():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919000000001", "status": "accepted", "awaiting_from": None},
                {"id": "session-b", "doctor_phone": "919000000001", "status": "completed", "awaiting_from": None},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is None


def test_ignores_other_doctors_sessions():
    supabase = FakeSupabaseClient(
        initial={
            "doctor_sessions": [
                {"id": "session-a", "doctor_phone": "919111111111", "status": "completed", "awaiting_from": "doctor_prescription"},
            ],
        }
    )

    result = _load_awaiting_prescription_session(supabase, "919000000001")

    assert result is None
