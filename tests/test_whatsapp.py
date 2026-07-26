"""Reproduces a real reported bug: a doctor's booking-assignment notice
(case details + Meet link) went missing in production with nothing but a
caught-and-logged exception to show for it -- a transient Graph API hiccup
on that one send, gone for good. _post must retry transient failures
instead of giving up after one attempt, matching the same pattern already
used for the Calendar Bridge client."""

import httpx
import pytest

from app.config import Settings
from app.integrations.whatsapp import WhatsAppClient


class _FakeHttpClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def post(self, url, headers, json):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _ok_response(body):
    request = httpx.Request("POST", "https://graph.facebook.com/messages")
    return httpx.Response(200, json=body, request=request)


def _server_error_response():
    request = httpx.Request("POST", "https://graph.facebook.com/messages")
    return httpx.Response(500, json={"error": {"message": "internal"}}, request=request)


def _client_error_response():
    request = httpx.Request("POST", "https://graph.facebook.com/messages")
    return httpx.Response(400, json={"error": {"message": "expired 24h window"}}, request=request)


async def _no_sleep(*args):
    return None


@pytest.mark.asyncio
async def test_retries_transport_error_then_succeeds(monkeypatch):
    fake = _FakeHttpClient([httpx.ConnectError("boom"), _ok_response({"messages": [{"id": "wamid.1"}]})])
    client = WhatsAppClient(Settings(), fake)
    monkeypatch.setattr("app.integrations.whatsapp.asyncio.sleep", _no_sleep)

    result = await client.send_text("919000000001", "hello")

    assert result["messages"][0]["id"] == "wamid.1"
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_retries_5xx_then_succeeds(monkeypatch):
    fake = _FakeHttpClient([_server_error_response(), _ok_response({"messages": [{"id": "wamid.1"}]})])
    client = WhatsAppClient(Settings(), fake)
    monkeypatch.setattr("app.integrations.whatsapp.asyncio.sleep", _no_sleep)

    result = await client.send_text("919000000001", "hello")

    assert result["messages"][0]["id"] == "wamid.1"
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_does_not_retry_4xx():
    fake = _FakeHttpClient([_client_error_response()])
    client = WhatsAppClient(Settings(), fake)

    with pytest.raises(httpx.HTTPStatusError):
        await client.send_text("919000000001", "hello")

    assert fake.calls == 1


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(monkeypatch):
    fake = _FakeHttpClient([httpx.ConnectError("1"), httpx.ConnectError("2"), httpx.ConnectError("3")])
    client = WhatsAppClient(Settings(), fake)
    monkeypatch.setattr("app.integrations.whatsapp.asyncio.sleep", _no_sleep)

    from app.integrations import whatsapp

    with pytest.raises(httpx.ConnectError):
        await client.send_text("919000000001", "hello")

    assert fake.calls == whatsapp.SEND_MAX_ATTEMPTS
