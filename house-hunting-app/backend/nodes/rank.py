"""
Node 6 — rank_and_display

Combines hard and soft scores into a final weighted score, sorts properties,
and writes the ranked_list to state for Streamlit to display.

Final score = 0.6 * hard_score + 0.4 * soft_score

The 60/40 split keeps objective criteria dominant while giving
personalization meaningful influence.
"""

import logging

from backend.state import AgentState

log = logging.getLogger("agent.rank")

HARD_WEIGHT = 0.6
SOFT_WEIGHT = 0.4


def rank_and_display(state: AgentState) -> AgentState:
    props     = state["enriched_properties"]
    iteration = state.get("iteration", 0)
    log.info(f"[RANK] → Ranking {len(props)} properties  iteration={iteration}  (hard×{HARD_WEIGHT} + soft×{SOFT_WEIGHT})")
    ranked: list[dict] = []

    for prop in props:
        pid = prop["property_id"]
        h   = state["hard_scores"].get(pid, 0.0)
        s   = state["soft_scores"].get(pid, 0.0)
        final = round(HARD_WEIGHT * h + SOFT_WEIGHT * s, 4)

        ranked.append({
            **prop,
            "hard_score":  h,
            "soft_score":  s,
            "final_score": final,
            "rationale":   state.get("soft_rationales", {}).get(pid, ""),
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    ranked = ranked[:10]   # surface only the top 10 to the UI

    log.info(f"[RANK] ✓ Top results:")
    for i, p in enumerate(ranked[:5], 1):
        log.info(
            f"[RANK]   #{i}  final={p['final_score']:.3f} "
            f"(hard={p['hard_score']:.3f}, soft={p['soft_score']:.3f})  "
            f"{p.get('address', '?')}, {p.get('city', '?')}  ${p.get('rent', 0):,}/mo"
        )

    state["ranked_list"] = ranked
    return state
