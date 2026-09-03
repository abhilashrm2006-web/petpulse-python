"""Covers find_nearby_vets filter behavior (open to every customer, filters
apply whenever passed -- OSM has no ratings field, so that spec filter is
deliberately not implemented, see module docstring in
app/agent/tools/vets.py) and its resilience path: retry-then-succeed,
retry-exhausted-then-fallback-directory, and total failure still leaving
the customer a concrete next step instead of a raw error."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agent.tools.vets import OVERPASS_MIRROR_URL, OVERPASS_URL, find_nearby_vets
from tests.fake_supabase import FakeSupabaseClient

OSM_ELEMENTS = [
    {"lat": 12.90, "lon": 77.60, "tags": {"name": "24/7 Emergency Pet Hospital", "amenity": "veterinary", "opening_hours": "24/7"}},
    {"lat": 12.91, "lon": 77.61, "tags": {"name": "Daytime Vet Clinic", "amenity": "veterinary", "opening_hours": "Mo-Fr 09:00-18:00"}},
]


def _ok_geocode_response():
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"lat": "12.9", "lon": "77.6"}])


def _ok_overpass_response():
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"elements": OSM_ELEMENTS})


def _make_ctx(supabase=None, get=None, post=None, google_maps_api_key=""):
    http = SimpleNamespace(
        get=get or AsyncMock(return_value=_ok_geocode_response()),
        post=post or AsyncMock(return_value=_ok_overpass_response()),
    )
    settings = SimpleNamespace(google_maps_api_key=google_maps_api_key)
    return SimpleNamespace(http=http, supabase=supabase or FakeSupabaseClient(), settings=settings)


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
async def test_overpass_call_always_sends_a_user_agent():
    """Live audit bug (2026-08-12): Overpass returns a bare 406 for any
    request with no User-Agent -- this was silently true in production for
    every real call until this was caught. Regression guard against
    dropping the header again."""
    post = AsyncMock(return_value=_ok_overpass_response())
    ctx = _make_ctx(post=post)
    agent_ctx = _make_agent_ctx()

    await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    _, kwargs = post.call_args
    assert kwargs.get("headers", {}).get("User-Agent")


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
async def test_last_retry_attempt_switches_to_a_different_overpass_mirror():
    """Live-confirmed 2026-08-27: retrying the same overloaded/down public
    Overpass instance 3 times back-to-back doesn't help -- the last attempt
    must hit a genuinely different provider."""
    post = AsyncMock(side_effect=[Exception("down"), Exception("still down"), _ok_overpass_response()])
    ctx = _make_ctx(post=post)
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["count"] == 2
    urls_called = [call.args[0] for call in post.call_args_list]
    assert urls_called == [OVERPASS_URL, OVERPASS_URL, OVERPASS_MIRROR_URL]


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


PLACES_RESULTS = [
    {
        "displayName": {"text": "Premium Pet Hospital"},
        "formattedAddress": "MG Road, Bengaluru",
        "location": {"latitude": 12.91, "longitude": 77.61},
        "rating": 4.8,
        "userRatingCount": 210,
        "internationalPhoneNumber": "+91 80 1234 5678",
        "googleMapsUri": "https://maps.google.com/?cid=1",
        "currentOpeningHours": {"openNow": True},
    },
    {
        "displayName": {"text": "Corner Vet Clinic"},
        "formattedAddress": "Indiranagar, Bengaluru",
        "location": {"latitude": 12.905, "longitude": 77.605},
        "rating": 3.2,
        "userRatingCount": 15,
        "internationalPhoneNumber": "+91 80 8765 4321",
        "googleMapsUri": "https://maps.google.com/?cid=2",
        "currentOpeningHours": {"openNow": False},
    },
]


def _ok_places_response(places=None):
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"places": places if places is not None else PLACES_RESULTS})


@pytest.mark.asyncio
async def test_google_places_used_when_api_key_configured():
    post = AsyncMock(return_value=_ok_places_response())
    ctx = _make_ctx(post=post, google_maps_api_key="fake-key")
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["success"] is True
    assert result["count"] == 2
    assert result["clinics"][0]["rating"] is not None
    assert "distance and rating" in result["message"]
    # Overpass must never be hit when Places already returned results
    urls_called = [call.args[0] for call in post.call_args_list]
    assert all("overpass" not in u for u in urls_called)


@pytest.mark.asyncio
async def test_google_places_ranks_by_distance_and_rating_composite():
    """A much closer, lower-rated clinic can still rank behind a slightly
    farther, meaningfully-better-rated one -- not pure nearest-first."""
    close_but_mediocre = {
        "displayName": {"text": "Close Clinic"},
        "formattedAddress": "Right here",
        "location": {"latitude": 12.9005, "longitude": 77.6005},  # ~70m away
        "rating": 2.5,
        "userRatingCount": 40,
        "currentOpeningHours": {"openNow": True},
    }
    far_but_excellent = {
        "displayName": {"text": "Excellent Clinic"},
        "formattedAddress": "A bit further",
        "location": {"latitude": 12.93, "longitude": 77.63},  # ~4.6km away
        "rating": 4.9,
        "userRatingCount": 500,
        "currentOpeningHours": {"openNow": True},
    }
    post = AsyncMock(return_value=_ok_places_response([close_but_mediocre, far_but_excellent]))
    ctx = _make_ctx(post=post, google_maps_api_key="fake-key")
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["clinics"][0]["name"] == "Excellent Clinic"


@pytest.mark.asyncio
async def test_google_places_open_now_filter():
    post = AsyncMock(return_value=_ok_places_response())
    ctx = _make_ctx(post=post, google_maps_api_key="fake-key")
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru", open_now=True)

    assert result["count"] == 1
    assert result["clinics"][0]["name"] == "Premium Pet Hospital"


@pytest.mark.asyncio
async def test_google_places_failure_falls_back_to_osm():
    call_log = []

    async def post(url, **kwargs):
        call_log.append(url)
        if "places.googleapis.com" in url:
            raise Exception("places down")
        return _ok_overpass_response()

    ctx = _make_ctx(post=AsyncMock(side_effect=post), google_maps_api_key="fake-key")
    agent_ctx = _make_agent_ctx()

    result = await find_nearby_vets(ctx, agent_ctx, location_text="Bengaluru")

    assert result["success"] is True
    assert result["count"] == 2  # the OSM fixture results
    assert any("places.googleapis.com" in u for u in call_log)
    assert any("overpass" in u for u in call_log)


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
