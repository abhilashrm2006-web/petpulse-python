"""Covers send_proactive_message (2026-08-27): every scheduler-fired
customer nudge used to send plain free-form text, which WhatsApp silently
accepts then fails to deliver (error 131047) once the customer has gone
quiet for over 24h -- exactly the case every one of these jobs fires in.
This picks free-form text inside the session window, the approved generic
template outside it, and falls back to free-form (today's behavior) when
no template is configured yet."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.integrations.proactive_messaging import send_proactive_message
from tests.fake_supabase import FakeSupabaseClient


def _hours_ago(hours: float) -> str:
    return (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()


def _make_ctx(supabase, settings=None):
    whatsapp = SimpleNamespace(send_text=AsyncMock(), send_template=AsyncMock())
    return SimpleNamespace(supabase=supabase, whatsapp=whatsapp, settings=settings)


@pytest.mark.asyncio
async def test_no_template_configured_always_sends_free_form():
    """Backward compatibility: every existing caller's ctx has no
    whatsapp_generic_nudge_template_name at all yet -- must behave exactly
    as before this module existed."""
    supabase = FakeSupabaseClient(initial={"messages": [{"profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(48)}]})
    ctx = _make_ctx(supabase, settings=None)

    await send_proactive_message(ctx, "p1", "919876543210", "hello")

    ctx.whatsapp.send_text.assert_awaited_once_with("919876543210", "hello")
    ctx.whatsapp.send_template.assert_not_awaited()


@pytest.mark.asyncio
async def test_within_session_window_sends_free_form_even_with_template_configured():
    supabase = FakeSupabaseClient(initial={"messages": [{"profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(2)}]})
    settings = SimpleNamespace(whatsapp_generic_nudge_template_name="petpulse_generic_nudge", whatsapp_generic_nudge_template_language="en")
    ctx = _make_ctx(supabase, settings=settings)

    await send_proactive_message(ctx, "p1", "919876543210", "hello")

    ctx.whatsapp.send_text.assert_awaited_once_with("919876543210", "hello")
    ctx.whatsapp.send_template.assert_not_awaited()


@pytest.mark.asyncio
async def test_outside_session_window_uses_the_template():
    supabase = FakeSupabaseClient(initial={"messages": [{"profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(48)}]})
    settings = SimpleNamespace(whatsapp_generic_nudge_template_name="petpulse_generic_nudge", whatsapp_generic_nudge_template_language="en")
    ctx = _make_ctx(supabase, settings=settings)

    await send_proactive_message(ctx, "p1", "919876543210", "hello")

    ctx.whatsapp.send_template.assert_awaited_once_with("919876543210", "petpulse_generic_nudge", "en", ["hello"])
    ctx.whatsapp.send_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_never_messaged_before_counts_as_outside_the_window():
    supabase = FakeSupabaseClient(initial={"messages": []})
    settings = SimpleNamespace(whatsapp_generic_nudge_template_name="petpulse_generic_nudge", whatsapp_generic_nudge_template_language="en")
    ctx = _make_ctx(supabase, settings=settings)

    await send_proactive_message(ctx, "p1", "919876543210", "hello")

    ctx.whatsapp.send_template.assert_awaited_once()


@pytest.mark.asyncio
async def test_template_send_failure_falls_back_to_free_form():
    supabase = FakeSupabaseClient(initial={"messages": [{"profile_id": "p1", "sender_type": "user", "created_at": _hours_ago(48)}]})
    settings = SimpleNamespace(whatsapp_generic_nudge_template_name="petpulse_generic_nudge", whatsapp_generic_nudge_template_language="en")
    ctx = _make_ctx(supabase, settings=settings)
    ctx.whatsapp.send_template.side_effect = Exception("template not approved yet")

    await send_proactive_message(ctx, "p1", "919876543210", "hello")

    ctx.whatsapp.send_text.assert_awaited_once_with("919876543210", "hello")
