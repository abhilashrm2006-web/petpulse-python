"""Ports `S50mzeVEaYXIbhk2` — Nearby Vet Finder / `find_nearby_vets` (spec
§3.4). Live version uses Nominatim + Overpass (OSM), fixed 8km radius. Every
customer can pass open_now/emergency_24h/category to filter results,
computed from whatever OSM opening_hours/healthcare/name tags happen to be
on file for that clinic -- there is no ratings/reviews field in OSM data,
so that specific filter from the product spec has no data source without
adding a paid Google Places API dependency, and is deliberately not
implemented here.

Resilience (2026-08 root-cause fix): two real transcripts showed this tool
failing outright mid-conversation -- once during a kidney-emergency case,
once during a surgery-referral request -- with the raw exception text
relayed straight to the customer as "the map service is failing" and no
next step offered. Both Nominatim and Overpass calls now retry with
backoff; a total Overpass failure falls back to an admin-curated
`vet_directory_fallback` Supabase table (empty until ops seeds it with
verified clinics -- never fabricated data); and if even that has nothing
for the customer's city, the reply still gives a concrete next step
instead of a dead end. Every failure that reaches the fallback/escalation
path is logged with profile/pet context so an emergency-adjacent failure
can be alerted on, not discovered later in an export."""

import asyncio
import logging
import math
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext

logger = logging.getLogger(__name__)

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
