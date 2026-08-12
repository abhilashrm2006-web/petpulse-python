"""Covers find_nearby_vets filter behavior (open to every customer, filters
apply whenever passed -- OSM has no ratings field, so that spec filter is
deliberately not implemented, see module docstring in
app/agent/tools/vets.py) and its resilience path: retry-then-succeed,
retry-exhausted-then-fallback-directory, and total failure still leaving
the customer a concrete next step instead of a raw error."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.vets import find_nearby_vets
from tests.fake_supabase import FakeSupabaseClient

OSM_ELEMENTS = [
    {"lat": 12.90, "lon": 77.60, "tags": {"name": "24/7 Emergency Pet Hospital", "amenity": "veterinary", "opening_hours": "24/7"}},
    {"lat": 12.91, "lon": 77.61, "tags": {"name": "Daytime Vet Clinic", "amenity": "veterinary", "opening_hours": "Mo-Fr 09:00-18:00"}},
]


def _ok_geocode_response():
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"lat": "12.9", "lon": "77.6"}])


def _ok_overpass_response():
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"elements": OSM_ELEMENTS})


def _make_ctx(supabase=None, get=None, post=None):
    http = SimpleNamespace(
        get=get or AsyncMock(return_value=_ok_geocode_response()),
        post=post or AsyncMock(return_value=_ok_overpass_response()),
    )
    return SimpleNamespace(http=http, supabase=supabase or FakeSupabaseClient())


def _make_agent_ctx(city="Bengaluru"):
    return SimpleNamespace(
        profile={"id": "profile-1", "phone_number": "919876543210", "city": city, "state": "", "country": ""},
        active_pet=None,
        medical_context={},
    )


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


@pytest.mark.asyncio
async def test_overpass_transient_failure_then_success_retries_and_recovers():
    post = AsyncMock(side_effect=[Exception("connection reset"), _ok_overpass_response()])
    ctx = _make_ctx(post=post)
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert post.await_count == 2
    assert result["count"] == 2
    assert result["clinics"][0]["name"] in {"24/7 Emergency Pet Hospital", "Daytime Vet Clinic"}


@pytest.mark.asyncio
async def test_overpass_exhausted_falls_back_to_directory():
    post = AsyncMock(side_effect=Exception("map service is failing"))
    supabase = FakeSupabaseClient(
        initial={
            "vet_directory_fallback": [
                {"city": "Bengaluru", "name": "Trusted Fallback Vet", "address": "MG Road", "phone": "080-1234", "maps_url": None, "website": None, "opening_hours": None}
            ]
        }
    )
    ctx = _make_ctx(supabase=supabase, post=post)
    agent_ctx = _make_agent_ctx(city="Bengaluru")

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert post.await_count == 3  # exhausted all retries
    assert result["success"] is True
    assert result["count"] == 1
    assert result["clinics"][0]["name"] == "Trusted Fallback Vet"
    assert "temporarily unavailable" in result["message"]


@pytest.mark.asyncio
async def test_total_failure_gives_a_concrete_next_step_not_a_raw_error():
    post = AsyncMock(side_effect=Exception("map service is failing"))
    ctx = _make_ctx(post=post)  # empty fallback directory too
    agent_ctx = _make_agent_ctx(city="Nowhereville")

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Nowhereville")

    assert result["success"] is True
    assert result["count"] == 0
    assert "map service is failing" not in result["message"]
    assert "emergency vet" in result["message"] or "team member" in result["message"]


@pytest.mark.asyncio
async def test_no_matches_within_radius_still_checks_fallback_directory():
    post = AsyncMock(return_value=SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"elements": []}))
    supabase = FakeSupabaseClient(
        initial={
            "vet_directory_fallback": [
                {"city": "Bengaluru", "name": "Backup Clinic", "address": None, "phone": None, "maps_url": None, "website": None, "opening_hours": None}
            ]
        }
    )
    ctx = _make_ctx(supabase=supabase, post=post)
    agent_ctx = _make_agent_ctx(city="Bengaluru")

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["count"] == 1
    assert result["clinics"][0]["name"] == "Backup Clinic"
