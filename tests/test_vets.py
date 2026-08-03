"""Covers find_nearby_vets filter behavior: open to every customer, and
filters apply whenever passed. OSM data has no ratings field, so that spec
filter is deliberately not implemented (see module docstring in
app/agent/tools/vets.py)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.vets import find_nearby_vets

OSM_ELEMENTS = [
    {"lat": 12.90, "lon": 77.60, "tags": {"name": "24/7 Emergency Pet Hospital", "amenity": "veterinary", "opening_hours": "24/7"}},
    {"lat": 12.91, "lon": 77.61, "tags": {"name": "Daytime Vet Clinic", "amenity": "veterinary", "opening_hours": "Mo-Fr 09:00-18:00"}},
]


def _make_ctx():
    http = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"lat": "12.9", "lon": "77.6"}])),
        post=AsyncMock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"elements": OSM_ELEMENTS})),
    )
    return SimpleNamespace(http=http)


def _make_agent_ctx():
    return SimpleNamespace(profile={"city": "Bengaluru", "state": "", "country": ""})


@pytest.mark.asyncio
async def test_emergency_24h_filter_narrows_results():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru", emergency_24h=True)

    assert result["count"] == 1
    assert result["clinics"][0]["name"] == "24/7 Emergency Pet Hospital"


@pytest.mark.asyncio
async def test_no_filters_gets_everything():
    ctx = _make_ctx()
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["count"] == 2
