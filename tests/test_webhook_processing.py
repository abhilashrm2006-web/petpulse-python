"""Covers the webhook route's fast-ack-then-background-process pattern,
switched from synchronous in-request processing to a background task (see
app/main.py). Root cause: WhatsApp Cloud API retries a webhook delivery it
doesn't get a fast response to, re-sending the identical payload -- with a
potentially-slow multi-tool-call agent turn running inline in the request
handler, a slow turn could cause Meta to retry, adding load and risking a
messaging-quality hit on Meta's side (the existing dedup claim already makes
a retry harmless content-wise, but avoiding it in the first place is still
better). Calls the route function directly with a fake Request (same
approach as test_main_passport.py) rather than via TestClient/lifespan."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app import main
from tests.fake_supabase import FakeSupabaseClient


def _fake_request(ctx, body: dict):
    raw = json.dumps(body).encode()

    async def _body():
        return raw

    async def _json():
        return body

    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(ctx=ctx)),
        body=_body,
        json=_json,
        headers=SimpleNamespace(get=lambda *a, **k: None),
    )


def _message_body(message_id="wamid.abc123", text="hi"):
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "contacts": [{"profile": {"name": "Jane"}}],
                            "messages": [
                                {
                                    "from": "919876543210",
                                    "id": message_id,
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": text},
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }


def _make_ctx(supabase=None):
    settings = SimpleNamespace(whatsapp_app_secret="")
    return SimpleNamespace(settings=settings, supabase=supabase or FakeSupabaseClient())


async def _flush_background_tasks():
    """Background tasks scheduled via asyncio.create_task need the event
    loop to actually get a turn to run them -- a few no-op awaits is the
    standard way to let already-scheduled tasks execute in a test."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_receive_webhook_acks_before_processing_completes(monkeypatch):
    """The route must return before the (mocked, artificially slow)
    processing function has finished -- proving processing genuinely runs in
    the background rather than being awaited inline."""
    started = asyncio.Event()
    finished = asyncio.Event()

    async def slow_process(ctx, extracted):
        started.set()
        await asyncio.sleep(0.05)
        finished.set()

    monkeypatch.setattr(main, "_process_inbound_message", slow_process)
    monkeypatch.setattr(main, "claim", lambda client, message_id: True)

    ctx = _make_ctx()
    request = _fake_request(ctx, _message_body())

    response = await main.receive_webhook(request)

    assert response.status_code == 200
    assert not finished.is_set()  # response returned before the slow work completed

    await asyncio.sleep(0.1)
    assert started.is_set()
    assert finished.is_set()


@pytest.mark.asyncio
async def test_duplicate_delivery_is_acked_without_scheduling_processing(monkeypatch):
    process_mock = AsyncMock()
    monkeypatch.setattr(main, "_process_inbound_message", process_mock)
    monkeypatch.setattr(main, "claim", lambda client, message_id: False)  # already processed

    ctx = _make_ctx()
    request = _fake_request(ctx, _message_body())

    response = await main.receive_webhook(request)
    await _flush_background_tasks()

    assert response.status_code == 200
    process_mock.assert_not_called()


@pytest.mark.asyncio
async def test_background_task_is_kept_referenced_until_done(monkeypatch):
    """A bare asyncio.create_task() result with nothing holding a reference
    to it can be garbage-collected mid-flight -- must be tracked in
    main._background_tasks and removed only once actually finished."""
    monkeypatch.setattr(main, "_process_inbound_message", AsyncMock())
    monkeypatch.setattr(main, "claim", lambda client, message_id: True)

    ctx = _make_ctx()
    request = _fake_request(ctx, _message_body())

    await main.receive_webhook(request)
    assert len(main._background_tasks) == 1

    await _flush_background_tasks()
    assert len(main._background_tasks) == 0


@pytest.mark.asyncio
async def test_process_inbound_message_skips_agent_turn_when_registration_handles_it(monkeypatch):
    from app.ingestion.webhook import ExtractedMessage

    monkeypatch.setattr(main, "handle_registration", AsyncMock(return_value=True))
    run_agent_turn_mock = AsyncMock()
    monkeypatch.setattr(main, "run_agent_turn", run_agent_turn_mock)

    ctx = _make_ctx()
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="hi",
    )

    await main._process_inbound_message(ctx, extracted)

    run_agent_turn_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_inbound_message_exceptions_are_caught_and_logged(monkeypatch, caplog):
    """Nothing downstream can catch an exception from a background task --
    it must be swallowed here (logged, never propagated) or it becomes an
    unhandled-exception-in-task warning with no other effect."""
    from app.ingestion.webhook import ExtractedMessage

    monkeypatch.setattr(main, "handle_registration", AsyncMock(side_effect=RuntimeError("boom")))

    ctx = _make_ctx()
    extracted = ExtractedMessage(
        phone_number="919876543210", sender_name="Jane", message_id="wamid.1",
        timestamp="1700000000", message_type="text", text="hi",
    )

    await main._process_inbound_message(ctx, extracted)  # must not raise
