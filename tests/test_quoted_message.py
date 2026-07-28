"""Covers WhatsApp's "reply"/quote feature: a customer can tap reply on any
earlier bubble and ask about it specifically (context.id on their next
message, extracted as ExtractedMessage.quoted_wamid). Previously this was
parsed out of the webhook payload but never used anywhere -- the agent had
no way to know what a vague follow-up like "what about this?" was actually
referring to. _resolve_quoted_message closes that gap by looking up the
wamid against messages.metadata->>wamid, written on every row this
codebase inserts (see app/agent/orchestrator.py)."""

from app.agent.system_prompt import build_turn_context
from app.ingestion.context import _resolve_quoted_message
from app.ingestion.webhook import ExtractedMessage
from tests.fake_supabase import FakeSupabaseClient


def _extracted(**overrides):
    defaults = dict(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.new",
        timestamp="123", message_type="text", text="what about this one?",
    )
    return ExtractedMessage(**{**defaults, **overrides})


def test_resolves_quoted_wamid_to_its_original_text():
    supabase = FakeSupabaseClient(
        initial={
            "messages": [
                {"id": "m1", "content": "Rex has been vomiting since this morning", "metadata": {"wamid": "wamid.old123"}},
                {"id": "m2", "content": "unrelated message", "metadata": {"wamid": "wamid.other"}},
            ]
        }
    )
    assert _resolve_quoted_message(supabase, "wamid.old123") == "Rex has been vomiting since this morning"


def test_returns_none_when_wamid_not_found():
    supabase = FakeSupabaseClient(initial={"messages": [{"id": "m1", "content": "x", "metadata": {"wamid": "wamid.other"}}]})
    assert _resolve_quoted_message(supabase, "wamid.old123") is None


def test_returns_none_when_no_quote_at_all():
    supabase = FakeSupabaseClient()
    assert _resolve_quoted_message(supabase, None) is None


def test_turn_context_surfaces_the_quoted_text_to_the_agent():
    from types import SimpleNamespace

    agent_ctx = SimpleNamespace(
        profile={"full_name": "Jane"}, pets=[], active_pet=None, active_pet_matched_from_message=False,
        role="customer", memory_context=[], medical_context={}, knowledge_base=[], onboarding={},
        pending_negotiation=None, open_session=None, awaiting_prescription_session=None,
        quoted_message_text="Rex has been vomiting since this morning",
    )
    extracted = _extracted(quoted_wamid="wamid.old123")

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "Rex has been vomiting since this morning" in context
    assert 'tapped "reply"' in context


def test_turn_context_asks_for_clarification_when_quote_cant_be_resolved():
    from types import SimpleNamespace

    agent_ctx = SimpleNamespace(
        profile={"full_name": "Jane"}, pets=[], active_pet=None, active_pet_matched_from_message=False,
        role="customer", memory_context=[], medical_context={}, knowledge_base=[], onboarding={},
        pending_negotiation=None, open_session=None, awaiting_prescription_session=None,
        quoted_message_text=None,
    )
    extracted = _extracted(quoted_wamid="wamid.old123")

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "couldn't be looked up" in context
    assert "ask them to clarify" in context


def test_turn_context_has_no_quote_line_for_a_normal_message():
    from types import SimpleNamespace

    agent_ctx = SimpleNamespace(
        profile={"full_name": "Jane"}, pets=[], active_pet=None, active_pet_matched_from_message=False,
        role="customer", memory_context=[], medical_context={}, knowledge_base=[], onboarding={},
        pending_negotiation=None, open_session=None, awaiting_prescription_session=None,
        quoted_message_text=None,
    )
    extracted = _extracted(quoted_wamid=None)

    context = build_turn_context(agent_ctx, extracted, media_context="", document_filing_status="")

    assert "reply" not in context.lower()
