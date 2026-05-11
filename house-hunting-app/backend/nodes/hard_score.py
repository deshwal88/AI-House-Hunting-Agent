"""
Node 4 — hard_score

Fixed-weight scoring on objective criteria (price, beds, baths, sqft/dollar).
These weights never change across feedback loops.
"""

import logging

from backend.state import AgentState
from backend.tools.scorer import hard_score, normalize_sqft_per_dollar

log = logging.getLogger("agent.hard_score")


def hard_score_node(state: AgentState) -> AgentState:
    props = state["enriched_properties"]
    req   = state["requirements"]
    log.info(f"[HARD] → Scoring {len(props)} properties")

    sqft_norms = normalize_sqft_per_dollar(props)

    scores: dict[str, float] = {}
    for prop in props:
        pid = prop["property_id"]
        scores[pid] = hard_score(prop, req, sqft_norms.get(pid, 0.5))

    if scores:
        top    = max(scores.values())
        bottom = min(scores.values())
        log.info(f"[HARD] ✓ Scores range: {bottom:.3f}–{top:.3f}")
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for pid, sc in top3:
            prop = next((p for p in props if p["property_id"] == pid), {})
            log.info(f"[HARD]   {sc:.3f}  {prop.get('address', pid)}, {prop.get('city', '')}")

    state["hard_scores"] = scores
    return state
