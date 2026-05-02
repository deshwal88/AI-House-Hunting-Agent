import os
import logging
import requests
from langchain_core.tools import tool

log = logging.getLogger("agent.rentcast")
RENTCAST_BASE = "https://api.rentcast.io/v1"


@tool
def rentcast_search(
    city: str,
    state: str,
    min_price: int,
    max_price: int,
    bedrooms: int = None,
    bathrooms: float = None,
    zip_code: str = None,
    radius_miles: float = 10.0,
    limit: int = 20,
) -> list:
    """Search active rental listings from RentCast."""
    api_key = os.getenv("RENTCAST_API_KEY")
    if not api_key:
        log.error("[RENTCAST] ✗ RENTCAST_API_KEY not set in environment")
        raise ValueError("RENTCAST_API_KEY not set in environment")

    params: dict = {
        "status": "Active",
        "limit": limit,
        "maxPrice": max_price,
        "minPrice": min_price,
    }
    if zip_code:
        params["zipCode"] = zip_code
        log.info(f"[RENTCAST] → Searching ZIP={zip_code}  price=${min_price:,}–${max_price:,}")
    else:
        params["city"]  = city
        params["state"] = state
        log.info(f"[RENTCAST] → Searching {city}, {state}  price=${min_price:,}–${max_price:,}  radius={radius_miles}mi")

    if bedrooms  is not None:
        params["bedrooms"]  = bedrooms
        log.info(f"[RENTCAST]   beds={bedrooms}  baths={bathrooms}")
    if bathrooms is not None:
        params["bathrooms"] = bathrooms

    try:
        resp = requests.get(
            f"{RENTCAST_BASE}/listings/rental/long-term",
            headers={"X-Api-Key": api_key},
            params=params,
            timeout=15,
        )
    except requests.exceptions.Timeout:
        log.error("[RENTCAST] ✗ Request timed out after 15s")
        raise
    except requests.exceptions.ConnectionError as e:
        log.error(f"[RENTCAST] ✗ Connection error: {e}")
        raise

    if not resp.ok:
        log.error(f"[RENTCAST] ✗ HTTP {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(f"RentCast {resp.status_code}: {resp.text[:200]}")

    raw = resp.json()
    raw_list = raw if isinstance(raw, list) else raw.get("listings", [])
    normalized = [_normalize(p) for p in raw_list]

    log.info(f"[RENTCAST] ✓ {len(normalized)} listings returned")
    for i, p in enumerate(normalized[:5], 1):
        log.info(f"[RENTCAST]   #{i} {p['address']}, {p['city']} — ${p['rent']:,}/mo  {p['bedrooms']}bd/{p['bathrooms']}ba")
    if len(normalized) > 5:
        log.info(f"[RENTCAST]   ... and {len(normalized)-5} more")

    return normalized


def _normalize(prop: dict) -> dict:
    addr  = prop.get("addressLine1", "")
    city  = prop.get("city", "")
    state = prop.get("state", "")
    return {
        "property_id":    prop.get("id") or f"{addr},{city},{state}",
        "address":        addr,
        "city":           city,
        "state":          state,
        "zip_code":       prop.get("zipCode", ""),
        "rent":           prop.get("price", 0),
        "bedrooms":       prop.get("bedrooms"),
        "bathrooms":      prop.get("bathrooms"),
        "sqft":           prop.get("squareFootage"),
        "property_type":  prop.get("propertyType", ""),
        "latitude":       prop.get("latitude"),
        "longitude":      prop.get("longitude"),
        "images":         [],
        "floor_plan_url": None,
        "amenity_distances": {},
        "raw_rentcast":   prop,
    }
