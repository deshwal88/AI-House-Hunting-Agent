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

HARD_WEIGHT_INITIAL  = 0.6   # before any feedback
SOFT_WEIGHT_INITIAL  = 0.4
HARD_WEIGHT_FEEDBACK = 0.3   # after feedback — learned preferences dominate
SOFT_WEIGHT_FEEDBACK = 0.7


def rank_and_display(state: AgentState) -> AgentState:
    props     = state["enriched_properties"]
    iteration = state.get("iteration", 0)

    if iteration == 0:
        hard_w, soft_w = HARD_WEIGHT_INITIAL, SOFT_WEIGHT_INITIAL
    else:
        hard_w, soft_w = HARD_WEIGHT_FEEDBACK, SOFT_WEIGHT_FEEDBACK

    log.info(f"[RANK] → Ranking {len(props)} properties  iteration={iteration}  (hard×{hard_w} + soft×{soft_w})")
    ranked: list[dict] = []

    for prop in props:
        pid = prop["property_id"]
        h   = state["hard_scores"].get(pid, 0.0)
        s   = state["soft_scores"].get(pid, 0.0)
        final = round(hard_w * h + soft_w * s, 4)

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
