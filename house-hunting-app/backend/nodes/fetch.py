"""
Node 2 — fetch_properties

Calls the RentCast API via the rentcast_search tool and stores
normalized property dicts in state["properties"].

Fetches up to 50 candidates so the downstream scorer has a wide pool
to choose from.  If fewer than 50 are returned, the radius is expanded
(2× then 3× the user's value) and retried.
"""

import logging

from backend.state import AgentState
from backend.tools.rentcast import rentcast_search

log = logging.getLogger("agent.fetch")

FETCH_TARGET      = 50
RADIUS_MULTIPLIERS = [1, 2, 3]   # successive attempts if target not met


def fetch_properties(state: AgentState) -> AgentState:
    req  = state["requirements"]
    loc  = req.get("location", {})
    prop = req.get("property", {})

    base_radius = loc.get("search_radius_miles", 10.0)
    properties: list[dict] = []

    for mult in RADIUS_MULTIPLIERS:
        radius = base_radius * mult
        if mult > 1:
            log.info(
                f"[FETCH]   Only {len(properties)} listings — "
                f"expanding radius to {radius:.0f}mi (×{mult})"
            )

        log.info(f"[FETCH] → RentCast: {loc.get('city')}, {loc.get('state')}  radius={radius:.0f}mi")
        try:
            properties = rentcast_search.invoke({
                "city":         loc.get("city", ""),
                "state":        loc.get("state", ""),
                "zip_code":     loc.get("zip") or None,
                "min_price":    prop.get("price_min", 0),
                "max_price":    prop.get("price_max", 10_000),
                "bedrooms":     prop.get("bedrooms"),
                "bathrooms":    prop.get("bathrooms"),
                "radius_miles": radius,
                "limit":        FETCH_TARGET,
            })
        except Exception as e:
            log.error(f"[FETCH] ✗ RentCast call failed: {e}")
            properties = []

        if len(properties) >= FETCH_TARGET:
            break

    log.info(f"[FETCH] ✓ {len(properties)} properties fetched from RentCast")
    state["properties"] = properties
    return state
