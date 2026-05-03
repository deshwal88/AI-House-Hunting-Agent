"""
Node 3 — enrich_properties

For each property fetched from RentCast:
  Calls ArcGIS Places to get distances to nearby amenities.

All properties are enriched in parallel using a thread pool so the total
wall-clock time is ~max(single property time) instead of sum(all property times).
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.state import AgentState
from backend.tools.arcgis import arcgis_amenity_distances

log = logging.getLogger("agent.enrich")
MAX_WORKERS = 5   # parallel threads — keeps API rate limits safe


def _enrich_one(prop: dict, index: int, total: int) -> dict:
    addr = f"{prop.get('address', '?')}, {prop.get('city', '?')}"
    lat  = prop.get("latitude")
    lng  = prop.get("longitude")
    log.info(f"[ENRICH]   [{index}/{total}] {addr}")

    # ── Amenity distances via ArcGIS ──────────────────────────────────────
    amenity_distances: dict[str, float] = {
        "grocery": 99.0, "school": 99.0, "gym": 99.0, "transit": 99.0,
    }
    if lat is not None and lng is not None:
        try:
            amenity_distances = arcgis_amenity_distances.invoke({
                "latitude":  lat,
                "longitude": lng,
            })
        except Exception as e:
            log.warning(f"[ENRICH]     [{index}/{total}] ✗ ArcGIS failed: {e}")
    else:
        log.warning(f"[ENRICH]     [{index}/{total}] ✗ No lat/lng — amenities default to 99mi")

    return {
        **prop,
        "images":            [],
        "floor_plan_url":    None,
        "amenity_distances": amenity_distances,
    }


ENRICH_LIMIT = 15   # enrich only the top candidates to keep Apify/ArcGIS time manageable


def _prefilter(props: list[dict], req: dict) -> list[dict]:
    """Sort by price proximity to the budget midpoint and keep the top ENRICH_LIMIT."""
    price_min = req.get("property", {}).get("price_min", 0)
    price_max = req.get("property", {}).get("price_max", 10_000)
    midpoint  = (price_min + price_max) / 2
    span      = max(price_max - price_min, 1)

    def _score(p: dict) -> float:
        rent = p.get("rent") or 0
        return abs(rent - midpoint) / span

    return sorted(props, key=_score)[:ENRICH_LIMIT]


def enrich_properties(state: AgentState) -> AgentState:
    all_props = state["properties"]
    props     = _prefilter(all_props, state["requirements"])
    total     = len(props)
    log.info(
        f"[ENRICH] → Pre-filtered {len(all_props)} → {total} candidates  "
        f"(workers={MAX_WORKERS}, Apify + ArcGIS)"
    )

    # Submit all enrichment tasks concurrently
    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_enrich_one, prop, i + 1, total): i
            for i, prop in enumerate(props)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.error(f"[ENRICH] ✗ Worker crashed for property #{idx+1}: {e}")
                results[idx] = {**props[idx],
                                "images": [], "floor_plan_url": None,
                                "amenity_distances": {"grocery": 99.0, "school": 99.0,
                                                      "gym": 99.0, "transit": 99.0}}

    # Reassemble in original order
    enriched = [results[i] for i in range(total)]
    log.info(f"[ENRICH] ✓ All {total} properties enriched")
    state["enriched_properties"] = enriched
    return state
