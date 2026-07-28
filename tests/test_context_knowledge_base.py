"""_load_knowledge_base must OR across category/content/title, not AND them.
Chaining three .ilike() calls on one PostgREST query builder ANDs them
(separate query params) -- a row needed the same substring in all three
columns at once to ever match, which essentially never happens. Real bug
found via audit: every customer question that should have surfaced
knowledge_base rows via a content-only or title-only match returned []."""

import pytest

from app.ingestion.context import _load_knowledge_base
from tests.fake_supabase import FakeSupabaseClient


def test_matches_on_content_alone():
    # _load_knowledge_base's ilike pattern is substring-based (message_text must
    # literally appear inside the column), matching how the .ilike() calls
    # already worked before this fix -- only the AND-vs-OR combination was buggy.
    supabase = FakeSupabaseClient(
        initial={
            "knowledge_base": [
                {"id": "kb-1", "category": "nutrition", "title": "Feeding", "content": "puppies need frequent small meals daily"}
            ]
        }
    )
    rows = _load_knowledge_base(supabase, "frequent small meals")
    assert len(rows) == 1


def test_matches_on_title_alone():
    supabase = FakeSupabaseClient(
        initial={
            "knowledge_base": [
                {"id": "kb-1", "category": "vaccines", "title": "core puppy vaccine schedule", "content": "unrelated text"}
            ]
        }
    )
    rows = _load_knowledge_base(supabase, "vaccine schedule")
    assert len(rows) == 1


def test_no_match_returns_empty():
    supabase = FakeSupabaseClient(
        initial={"knowledge_base": [{"id": "kb-1", "category": "nutrition", "title": "Feeding", "content": "meals"}]}
    )
    rows = _load_knowledge_base(supabase, "completely unrelated grooming question")
    assert rows == []


def test_empty_message_short_circuits():
    supabase = FakeSupabaseClient(initial={"knowledge_base": [{"id": "kb-1", "category": "x", "title": "y", "content": "z"}]})
    assert _load_knowledge_base(supabase, "") == []
