"""Covers load_chat_history's session-gap annotation (2026-09 root-cause
fix): raw chat-history turns previously carried no date/time info at all,
so the model couldn't tell a 3-week-old turn from yesterday's -- a
contributing cause of old, unrelated incidents getting silently blended
into a fresh new complaint (confirmed live: an old chocolate/salt-water
mention bled into an unrelated new medication-overdose report)."""

from app.agent.memory import load_chat_history
from tests.fake_supabase import FakeSupabaseClient


_next_id = 0


def _row(session_id, role, content, created_at, pet_id=None):
    global _next_id
    _next_id += 1
    row = {
        "id": _next_id,
        "session_id": session_id,
        "message": {"type": role, "data": {"content": content, "additional_kwargs": {}, "type": role}},
        "created_at": created_at,
    }
    if pet_id:
        row["pet_id"] = pet_id
    return row


def test_no_gap_annotation_for_a_normal_back_and_forth():
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [
                _row("919876543210", "human", "my dog seems tired", "2026-09-01T10:00:00+00:00"),
                _row("919876543210", "ai", "how long has this been going on?", "2026-09-01T10:00:05+00:00"),
                _row("919876543210", "human", "since this morning", "2026-09-01T10:05:00+00:00"),
            ]
        }
    )

    history = load_chat_history(supabase, "919876543210")

    assert all("later, possibly a new/unrelated topic" not in m["content"] for m in history)


def test_gap_annotation_added_after_a_long_silence():
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [
                _row("919876543210", "human", "he ate some chocolate today", "2026-08-01T10:00:00+00:00"),
                _row("919876543210", "ai", "how much chocolate, and what size is he?", "2026-08-01T10:00:05+00:00"),
                _row("919876543210", "human", "I gave him a paracetamol tablet for pain", "2026-09-01T19:00:00+00:00"),
            ]
        }
    )

    history = load_chat_history(supabase, "919876543210")

    assert "later, possibly a new/unrelated topic" in history[-1]["content"]
    assert "I gave him a paracetamol tablet for pain" in history[-1]["content"]
    # The earlier turns are untouched
    assert history[0]["content"] == "he ate some chocolate today"


def test_missing_timestamps_never_crash_and_never_annotate():
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [
                {"id": 1, "session_id": "919876543210", "message": {"type": "human", "data": {"content": "hi"}}},
                {"id": 2, "session_id": "919876543210", "message": {"type": "ai", "data": {"content": "hello!"}}},
            ]
        }
    )

    history = load_chat_history(supabase, "919876543210")

    assert history[0]["content"] == "hi"
    assert history[1]["content"] == "hello!"


def test_gap_annotation_only_applies_to_user_turns_not_assistant_turns():
    supabase = FakeSupabaseClient(
        initial={
            "n8n_chat_history_petpulse": [
                _row("919876543210", "human", "old topic", "2026-08-01T10:00:00+00:00"),
                _row("919876543210", "ai", "old reply", "2026-09-01T10:00:00+00:00"),
            ]
        }
    )

    history = load_chat_history(supabase, "919876543210")

    assert "later" not in history[1]["content"]
