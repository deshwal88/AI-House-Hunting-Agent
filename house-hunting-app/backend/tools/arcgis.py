import os
import logging
import requests
from langchain_core.tools import tool

log = logging.getLogger("agent.arcgis")

ARCGIS_BASE     = "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1"
SEARCH_RADIUS   = 5000
METERS_PER_MILE = 1609.34

# searchText queries sent directly to the places/near-point endpoint.
# This bypasses the fragile category-ID lookup which was returning empty results.
AMENITY_SEARCHES: dict[str, str] = {
    "grocery": "grocery store",
    "school":  "elementary school",
    "gym":     "gym fitness",
    "transit": "bus stop",
}


@tool
def arcgis_amenity_distances(latitude: float, longitude: float) -> dict:
    """Return distances in miles to nearest grocery, school, gym, transit."""
    token = os.getenv("ARCGIS_API_KEY")
    if not token:
        log.warning("[ARCGIS] ✗ ARCGIS_API_KEY not set — defaulting all distances to 99mi")
        return {k: 99.0 for k in AMENITY_SEARCHES}

    log.info(f"[ARCGIS] → Looking up amenities at ({latitude:.4f}, {longitude:.4f})")
    results: dict[str, float] = {}

    for amenity, search_text in AMENITY_SEARCHES.items():
        dist = _nearest_distance_miles(latitude, longitude, search_text, token)
        results[amenity] = dist
        status = f"{dist:.2f}mi" if dist < 90 else "not found within 5km"
        log.info(f"[ARCGIS]   {amenity:10s} → {status}")

    return results


def _nearest_distance_miles(lat: float, lng: float, search_text: str, token: str) -> float:
    try:
        resp = requests.get(
            f"{ARCGIS_BASE}/places/near-point",
            params={
                "y":          lat,
                "x":          lng,
                "radius":     SEARCH_RADIUS,
                "searchText": search_text,
                "f":          "json",
                "token":      token,
            },
            timeout=10,
        )

        if not resp.ok:
            log.error(
                f"[ARCGIS] ✗ Places near-point HTTP {resp.status_code} "
                f"for '{search_text}': {resp.text[:200]}"
            )
            return 99.0

        body = resp.json()

        # ArcGIS sometimes returns HTTP 200 with an error object in the body
        if "error" in body:
            err = body["error"]
            log.error(
                f"[ARCGIS] ✗ API error for '{search_text}': "
                f"code={err.get('code')} message={err.get('message')}"
            )
            return 99.0

        places = body.get("results", [])
        if not places:
            log.info(f"[ARCGIS]   No '{search_text}' found within {SEARCH_RADIUS}m")
            return 99.0

        distance_m = places[0].get("distance", 99 * METERS_PER_MILE)
        return round(distance_m / METERS_PER_MILE, 2)

    except Exception as e:
        log.error(f"[ARCGIS] ✗ Places lookup exception for '{search_text}': {e}")
        return 99.0
