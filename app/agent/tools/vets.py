"""Ports `S50mzeVEaYXIbhk2` — Nearby Vet Finder / `find_nearby_vets` (spec
§3.4). Live version uses Nominatim + Overpass (OSM), fixed 8km radius. Every
customer can pass open_now/emergency_24h/category to filter results,
computed from whatever OSM opening_hours/healthcare/name tags happen to be
on file for that clinic -- there is no ratings/reviews field in OSM data,
so that specific filter from the product spec has no data source without
adding a paid Google Places API dependency, and is deliberately not
implemented here."""

import math
from typing import Any

from app.deps import AppContext
from app.ingestion.context import AgentContext

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RADIUS_METERS = 8000


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return r * 2 * math.asin(math.sqrt(a))


async def _geocode(ctx: AppContext, query: str) -> tuple[float, float] | None:
    resp = await ctx.http.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1},
        headers={"User-Agent": "PetPulse/1.0"},
    )
    resp.raise_for_status()
    results = resp.json()
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
    resp = await ctx.http.post(OVERPASS_URL, data={"data": query})
    resp.raise_for_status()
    elements = resp.json().get("elements", [])

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

    return {
        "success": True,
        "count": len(top),
        "clinics": top,
        "message": "Here are the nearest vet clinics I found." if top else "I couldn't find any vet clinics within 8km of that location.",
    }
