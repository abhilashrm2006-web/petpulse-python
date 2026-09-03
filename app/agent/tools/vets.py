"""Ports `S50mzeVEaYXIbhk2` — Nearby Vet Finder / `find_nearby_vets` (spec
§3.4).

Primary data source (2026-09): Google Places API (New) `searchNearby`, when
Settings.google_maps_api_key is configured -- gives real ratings/review
counts and an accurate live open-now flag, sorted by a distance+rating
composite score. Falls back automatically to the original free OSM
Nominatim+Overpass path (unchanged) when the key isn't configured, or when
Places itself fails after retries -- this is additive, not a
hard dependency swap.

Resilience (2026-08 root-cause fix, still applies to the OSM fallback path):
two real transcripts showed this tool failing outright mid-conversation --
once during a kidney-emergency case, once during a surgery-referral
request -- with the raw exception text relayed straight to the customer as
"the map service is failing" and no next step offered. Both Nominatim and
Overpass calls retry with backoff; a total Overpass failure falls back to
an admin-curated `vet_directory_fallback` Supabase table (empty until ops
seeds it with verified clinics -- never fabricated data); and if even that
has nothing for the customer's city, the reply still gives a concrete next
step instead of a dead end. Every failure that reaches the fallback/
escalation path is logged with profile/pet context so an emergency-adjacent
failure can be alerted on, not discovered later in an export."""

import asyncio
import logging
import math
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext

logger = logging.getLogger(__name__)

PLACES_SEARCH_NEARBY_URL = "https://places.googleapis.com/v1/places:searchNearby"
PLACES_FIELD_MASK = (
    "places.id,places.displayName,places.formattedAddress,places.location,places.rating,"
    "places.userRatingCount,places.internationalPhoneNumber,places.googleMapsUri,"
    "places.currentOpeningHours.openNow,places.regularOpeningHours.weekdayDescriptions"
)
# Distance still matters most at a large gap (a vet 15km away is a real
# problem even at 4.9 stars), but a meaningfully better-rated clinic can
# outrank a somewhat closer one -- e.g. a 4.9 vs. a 2.5-star clinic (2.4
# stars apart) swaps ranking at roughly a 6km distance gap. Tunable if live
# feedback says distance should matter more/less relative to rating.
RATING_DISTANCE_KM_WEIGHT = 2.5
DEFAULT_RATING_WHEN_UNRATED = 3.5  # neutral -- doesn't penalize a real clinic that just has no reviews yet

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Confirmed live 2026-08-27: a real customer hit the total-failure escalation
# message during normal use, and the primary overpass-api.de instance is a
# free, best-effort public service with known intermittent overload/outages
# -- a customer report of this NOT being a rare edge case. Retrying the same
# overloaded instance 3 times back-to-back doesn't help; the last retry
# attempt now goes to a genuinely different provider instead, since
# different public Overpass mirrors go down independently of each other.
OVERPASS_MIRROR_URL = "https://overpass.kumi.systems/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RADIUS_METERS = 8000

RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.5)  # delay before attempt 2, then before attempt 3

# A health_logs row with ai_risk_score >= this is a check_symptoms severity
# 4-5 (RED) result -- see app/agent/tools/symptoms.py's SEVERITY_COLOR
# mapping (ai_risk_score = severity * 20) and the same threshold used by
# app/scheduler/jobs.py's flag_emergency_checkins. Used here only to decide
# whether a tool-failure log line should be flagged urgent, not to change
# behavior.
URGENT_RISK_SCORE_THRESHOLD = 80

SUPPORT_CONTACT_NOTE = (
    "I'm having trouble reaching the clinic directory right now, so I don't want to leave you without an "
    "option — if this is urgent, please call your nearest emergency vet directly, or reply here and a "
    "PetPulse team member will follow up with clinic options shortly."
)


async def _with_retries(call, *, what: str):
    """Runs `call(attempt)` (an async callable taking the zero-based attempt
    index) up to RETRY_ATTEMPTS times, sleeping a short backoff between
    attempts. Re-raises the last exception if every attempt fails --
    callers decide what to do next (fallback, escalation), this helper only
    owns the retry loop itself. The attempt index lets a caller vary WHAT
    it calls per attempt (e.g. switch provider on the last try), not just
    retry the identical call."""
    last_exc: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await call(attempt)
        except Exception as exc:
            last_exc = exc
            logger.warning("find_nearby_vets: %s attempt %d/%d failed: %s", what, attempt + 1, RETRY_ATTEMPTS, exc)
            if attempt < RETRY_ATTEMPTS - 1:
                await asyncio.sleep(RETRY_BACKOFF_SECONDS[attempt])
    raise last_exc


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


async def _geocode(ctx: AppContext, query: str) -> tuple[float, float] | None:
    async def _call(attempt: int):
        resp = await ctx.http.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1},
            headers={"User-Agent": "PetPulse/1.0"},
        )
        resp.raise_for_status()
        return resp.json()

    try:
        results = await _with_retries(_call, what=f"geocode({query!r})")
    except Exception:
        logger.warning("find_nearby_vets: geocode(%r) failed after %d attempts, giving up on this query", query, RETRY_ATTEMPTS)
        return None

    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"])


def _is_24h(opening_hours: str | None) -> bool:
    return bool(opening_hours) and "24/7" in opening_hours


def _matches_category(tags: dict[str, str], category: str) -> bool:
    """OSM has no dedicated clinic/hospital/pharmacy split for amenity=veterinary --
    best-effort keyword match against whatever tags/name are on file rather than a
    hard data field. Known limitation, documented in the module docstring."""
    haystack = " ".join(filter(None, [tags.get("healthcare"), tags.get("name", "")])).lower()
    return category.lower() in haystack


def _urgent_context(agent_ctx: AgentContext) -> bool:
    """Best-effort "was this conversation's pet flagged urgent recently"
    signal for failure logging -- inspects the already-loaded
    medical_context rather than making a fresh query, since this only
    matters for a log line, not customer-facing behavior."""
    pet = agent_ctx.active_pet
    if not pet:
        return False
    logs = (agent_ctx.medical_context or {}).get("health_logs_by_pet", {}).get(pet.get("id"), [])
    return any((log.get("ai_risk_score") or 0) >= URGENT_RISK_SCORE_THRESHOLD for log in logs)


def _log_tool_failure(ctx: AppContext, agent_ctx: AgentContext, stage: str, exc: Exception) -> None:
    profile = agent_ctx.profile or {}
    logger.error(
        "find_nearby_vets tool_failure stage=%s profile_id=%s phone=%s pet_id=%s urgent_context=%s error=%s",
        stage,
        profile.get("id"),
        profile.get("phone_number"),
        (agent_ctx.active_pet or {}).get("id"),
        _urgent_context(agent_ctx),
        exc,
    )


async def _fallback_clinics(ctx: AppContext, city: str | None) -> list[dict[str, Any]]:
    """Admin-curated fallback (see app/admin/routes.py) -- empty until ops
    seeds it with verified clinics, never fabricated data. Queried only
    after Overpass has already failed every retry."""
    if not city:
        return []
    try:
        rows = (
            ctx.supabase.table("vet_directory_fallback")
            .select("*")
            .ilike("city", city.strip())
            .execute()
            .data
            or []
        )
    except Exception:
        logger.exception("find_nearby_vets: fallback directory lookup failed for city=%r", city)
        return []

    return [
        {
            "name": row.get("name", "Unnamed clinic"),
            "address": row.get("address"),
            "phone": row.get("phone"),
            "website": row.get("website"),
            "opening_hours": row.get("opening_hours"),
            "maps_url": row.get("maps_url"),
            "distance_km": None,
        }
        for row in rows
    ]


def _places_is_24h(place: dict[str, Any]) -> bool:
    descriptions = (place.get("regularOpeningHours") or {}).get("weekdayDescriptions") or []
    return any("24 hours" in d.lower() for d in descriptions)


def _places_matches_category(place: dict[str, Any], category: str) -> bool:
    haystack = f"{place.get('displayName', {}).get('text', '')} {place.get('formattedAddress', '')}".lower()
    return category.lower() in haystack


def _ranking_score(distance_km: float, rating: float | None) -> float:
    """Lower is better. A clinic with no rating yet is treated as neutral
    (DEFAULT_RATING_WHEN_UNRATED), not penalized to the bottom just for
    lacking reviews."""
    effective_rating = rating if rating is not None else DEFAULT_RATING_WHEN_UNRATED
    return distance_km - (effective_rating - DEFAULT_RATING_WHEN_UNRATED) * RATING_DISTANCE_KM_WEIGHT


async def _call_google_places(ctx: AppContext, lat: float, lon: float, api_key: str) -> list[dict[str, Any]]:
    async def _call(attempt: int):
        resp = await ctx.http.post(
            PLACES_SEARCH_NEARBY_URL,
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": PLACES_FIELD_MASK,
            },
            json={
                "includedTypes": ["veterinary_care"],
                "maxResultCount": 20,
                "locationRestriction": {"circle": {"center": {"latitude": lat, "longitude": lon}, "radius": float(RADIUS_METERS)}},
            },
        )
        resp.raise_for_status()
        return resp.json().get("places", [])

    return await _with_retries(_call, what="google places searchNearby")


def _format_places_clinics(
    places: list[dict[str, Any]], lat: float, lon: float, *, open_now: bool | None, emergency_24h: bool | None, category: str
) -> list[dict[str, Any]]:
    clinics = []
    for place in places:
        location = place.get("location") or {}
        p_lat, p_lon = location.get("latitude"), location.get("longitude")
        if p_lat is None or p_lon is None:
            continue
        is_open_now = (place.get("currentOpeningHours") or {}).get("openNow")
        if open_now and not is_open_now:
            continue
        if emergency_24h and not _places_is_24h(place):
            continue
        if category and not _places_matches_category(place, category):
            continue
        distance_km = round(_haversine_km(lat, lon, p_lat, p_lon), 2)
        rating = place.get("rating")
        clinics.append(
            {
                "name": (place.get("displayName") or {}).get("text", "Unnamed clinic"),
                "address": place.get("formattedAddress"),
                "phone": place.get("internationalPhoneNumber"),
                "website": place.get("googleMapsUri"),
                "opening_hours": "Open now" if is_open_now else ("Closed now" if is_open_now is False else None),
                "maps_url": place.get("googleMapsUri"),
                "distance_km": distance_km,
                "rating": rating,
                "rating_count": place.get("userRatingCount"),
                "_score": _ranking_score(distance_km, rating),
            }
        )
    clinics.sort(key=lambda c: c["_score"])
    for c in clinics:
        del c["_score"]
    return clinics


async def find_nearby_vets(
    ctx: AppContext,
    agent_ctx: AgentContext,
    location_text: str = "",
    latitude: float | None = None,
    longitude: float | None = None,
    open_now: bool | None = None,
    emergency_24h: bool | None = None,
    category: str = "",
) -> dict[str, Any]:
    lat, lon = latitude, longitude
    city_for_fallback = (agent_ctx.profile or {}).get("city")

    if lat is None or lon is None:
        if location_text:
            coords = await _geocode(ctx, location_text)
            if coords:
                lat, lon = coords

    if lat is None or lon is None:
        profile = agent_ctx.profile
        city_query = ", ".join(filter(None, [profile.get("city"), profile.get("state"), profile.get("country")]))
        if city_query:
            coords = await _geocode(ctx, city_query)
            if coords:
                lat, lon = coords

    if lat is None or lon is None:
        return {
            "success": True,
            "count": 0,
            "clinics": [],
            "message": "I don't have a location for you yet — could you share your location or tell me your city?",
        }

    api_key = getattr(getattr(ctx, "settings", None), "google_maps_api_key", "") or ""
    if api_key:
        try:
            places = await _call_google_places(ctx, lat, lon, api_key)
            clinics = _format_places_clinics(places, lat, lon, open_now=open_now, emergency_24h=emergency_24h, category=category)
            if clinics:
                return {
                    "success": True,
                    "count": len(clinics[:5]),
                    "clinics": clinics[:5],
                    "message": "Here are the nearest vet clinics I found, sorted by distance and rating.",
                }
            # Places succeeded but genuinely found nothing (or everything got
            # filtered out) -- fall through to OSM rather than treating an
            # empty real result as a failure needing retry/fallback-directory.
        except Exception as exc:
            _log_tool_failure(ctx, agent_ctx, "google_places_exhausted", exc)
            # Falls through to the OSM path below rather than escalating --
            # Places failing doesn't mean the free path will too.

    query = f'[out:json][timeout:25];(node["amenity"="veterinary"](around:{RADIUS_METERS},{lat},{lon});way["amenity"="veterinary"](around:{RADIUS_METERS},{lat},{lon});relation["amenity"="veterinary"](around:{RADIUS_METERS},{lat},{lon}););out center tags;'

    async def _call(attempt: int):
        # Confirmed live 2026-08-12 audit: Overpass rejects any request with
        # no User-Agent with a bare 406, no body -- unlike the Nominatim call
        # above, this POST never set one, so every real production call was
        # silently falling through to retries-then-fallback/escalation and
        # NEVER actually returning live results. Overpass's own usage policy
        # asks for a contactable identifier, not just any string.
        #
        # Confirmed live 2026-08-27: a real customer hit the escalation
        # message during normal use even with the header fix -- the primary
        # instance is a free, best-effort public service that does go down/
        # overload on its own. The last attempt switches to a different
        # public mirror rather than retrying the same (possibly still-down)
        # instance a third time.
        url = OVERPASS_URL if attempt < RETRY_ATTEMPTS - 1 else OVERPASS_MIRROR_URL
        resp = await ctx.http.post(
            url, data={"data": query}, headers={"User-Agent": "PetPulse/1.0 (support@petpulse.app)"}
        )
        resp.raise_for_status()
        return resp.json().get("elements", [])

    try:
        elements = await _with_retries(_call, what="overpass query")
    except Exception as exc:
        _log_tool_failure(ctx, agent_ctx, "overpass_exhausted", exc)
        fallback = await _fallback_clinics(ctx, city_for_fallback or location_text)
        if fallback:
            return {
                "success": True,
                "count": len(fallback),
                "clinics": fallback[:5],
                "message": "The live clinic search is temporarily unavailable, so here are clinics from our own directory for your area.",
            }
        return {
            "success": True,
            "count": 0,
            "clinics": [],
            "message": SUPPORT_CONTACT_NOTE,
        }

    clinics = []
    for el in elements:
        tags = el.get("tags", {})
        el_lat = el.get("lat") or el.get("center", {}).get("lat")
        el_lon = el.get("lon") or el.get("center", {}).get("lon")
        if el_lat is None or el_lon is None:
            continue
        opening_hours = tags.get("opening_hours")
        if open_now and not _is_24h(opening_hours):
            continue  # can't reliably evaluate arbitrary opening_hours syntax against "right now" -- only 24/7 is a safe yes
        if emergency_24h and not _is_24h(opening_hours):
            continue
        if category and not _matches_category(tags, category):
            continue
        clinics.append(
            {
                "name": tags.get("name", "Unnamed clinic"),
                "address": tags.get("addr:full") or ", ".join(
                    filter(None, [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:city")])
                ),
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "website": tags.get("website") or tags.get("contact:website"),
                "opening_hours": opening_hours,
                "maps_url": f"https://www.openstreetmap.org/?mlat={el_lat}&mlon={el_lon}",
                "distance_km": round(_haversine_km(lat, lon, el_lat, el_lon), 2),
            }
        )

    clinics.sort(key=lambda c: c["distance_km"])
    top = clinics[:5]

    if top:
        return {
            "success": True,
            "count": len(top),
            "clinics": top,
            "message": "Here are the nearest vet clinics I found.",
        }

    # Overpass succeeded but genuinely found nothing within radius -- not a
    # tool failure, so no retry/fallback needed, but still worth trying the
    # admin-curated directory before telling the customer there's nothing.
    fallback = await _fallback_clinics(ctx, city_for_fallback or location_text)
    if fallback:
        return {
            "success": True,
            "count": len(fallback),
            "clinics": fallback[:5],
            "message": "I couldn't find anything in the live map data within 8km, but here are clinics from our own directory for your area.",
        }
    return {
        "success": True,
        "count": 0,
        "clinics": [],
        "message": "I couldn't find any vet clinics within 8km of that location.",
    }
