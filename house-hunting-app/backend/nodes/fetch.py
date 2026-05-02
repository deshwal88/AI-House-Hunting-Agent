"""
Node 2 — fetch_properties

Calls the RentCast API via the rentcast_search tool and stores
normalized property dicts in state["properties"].
"""

import logging

from backend.state import AgentState
from backend.tools.rentcast import rentcast_search

log = logging.getLogger("agent.fetch")


def fetch_properties(state: AgentState) -> AgentState:
    req  = state["requirements"]
    loc  = req.get("location", {})
    prop = req.get("property", {})

    log.info(f"[FETCH] → Calling RentCast for {loc.get('city')}, {loc.get('state')}")
    try:
        properties = rentcast_search.invoke({
            "city":         loc.get("city", ""),
            "state":        loc.get("state", ""),
            "zip_code":     loc.get("zip") or None,
            "min_price":    prop.get("price_min", 0),
            "max_price":    prop.get("price_max", 10_000),
            "bedrooms":     prop.get("bedrooms"),
            "bathrooms":    prop.get("bathrooms"),
            "radius_miles": loc.get("search_radius_miles", 10.0),
            "limit":        10,
        })
        log.info(f"[FETCH] ✓ {len(properties)} properties fetched from RentCast")
    except Exception as e:
        log.error(f"[FETCH] ✗ RentCast call failed: {e}")
        properties = []

    state["properties"] = properties
    return state
