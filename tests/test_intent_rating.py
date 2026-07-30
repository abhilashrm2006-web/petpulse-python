"""Covers rate_customer_intent: LLM classification of a customer's
purchase/engagement intent from their chat history, used by both the
on-demand admin endpoint and the bulk scheduled job."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.admin.intent_rating import rate_customer_intent


@pytest.mark.asyncio
async def test_returns_low_with_no_messages():
    result = await rate_customer_intent(AsyncMock(), AsyncMock(), [])
    assert result["rating"] == "Low"


@pytest.mark.asyncio
async def test_parses_valid_llm_response():
    messages = [{"sender_type": "user", "content": "I want to book a vet appointment now"}]
    with patch(
        "app.admin.intent_rating.json_completion",
        AsyncMock(return_value=json.dumps({"rating": "High", "reason": "Asked to book a vet."})),
    ):
        result = await rate_customer_intent(AsyncMock(), AsyncMock(), messages)

    assert result == {"rating": "High", "reason": "Asked to book a vet."}


@pytest.mark.asyncio
async def test_falls_back_to_low_on_malformed_json():
    messages = [{"sender_type": "user", "content": "hi"}]
    with patch("app.admin.intent_rating.json_completion", AsyncMock(return_value="not json")):
        result = await rate_customer_intent(AsyncMock(), AsyncMock(), messages)

    assert result["rating"] == "Low"


@pytest.mark.asyncio
async def test_falls_back_to_low_on_unexpected_rating_value():
    messages = [{"sender_type": "user", "content": "hi"}]
    with patch(
        "app.admin.intent_rating.json_completion",
        AsyncMock(return_value=json.dumps({"rating": "Extreme", "reason": "nonsense"})),
    ):
        result = await rate_customer_intent(AsyncMock(), AsyncMock(), messages)

    assert result["rating"] == "Low"


@pytest.mark.asyncio
async def test_falls_back_to_low_when_completion_raises():
    messages = [{"sender_type": "user", "content": "hi"}]
    with patch("app.admin.intent_rating.json_completion", AsyncMock(side_effect=RuntimeError("outage"))):
        result = await rate_customer_intent(AsyncMock(), AsyncMock(), messages)

    assert result["rating"] == "Low"
