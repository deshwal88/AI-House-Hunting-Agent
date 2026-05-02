"""
Node 5 — soft_score

Learnable-weight scoring on preference-based criteria (amenity distances,
pet-friendly, quiet neighborhood). Weights start equal and are updated by the
LLM after each feedback round.

This node runs on EVERY feedback loop iteration with the current soft_weights.
The LLM generates short rationales only for properties that score above 0.55.
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from backend.state import AgentState
from backend.tools.scorer import soft_score as compute_soft_score

log = logging.getLogger("agent.soft_score")


def soft_score_node(state: AgentState) -> AgentState:
    props      = state["enriched_properties"]
    iteration  = state.get("iteration", 0)
    weights    = state["soft_weights"]
    log.info(f"[SOFT] → Scoring {len(props)} properties  iteration={iteration}")
    log.info(f"[SOFT]   Weights: { {k: round(v,3) for k, v in weights.items()} }")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
    )

    scores:     dict[str, float] = {}
    rationales: dict[str, str]   = {}
    rationale_count = 0

    for prop in props:
        pid   = prop["property_id"]
        score = compute_soft_score(prop, weights)
        scores[pid] = score

        if score >= 0.55:
            rationale = _generate_rationale(llm, prop, state["preference_profile"])
            rationales[pid] = rationale
            rationale_count += 1

    if scores:
        top    = max(scores.values())
        bottom = min(scores.values())
        log.info(f"[SOFT] ✓ Scores range: {bottom:.3f}–{top:.3f}  rationales generated: {rationale_count}")
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        for pid, sc in top3:
            prop = next((p for p in props if p["property_id"] == pid), {})
            log.info(f"[SOFT]   {sc:.3f}  {prop.get('address', pid)}, {prop.get('city', '')}")

    state["soft_scores"]    = scores
    state["soft_rationales"] = rationales
    return state


def _generate_rationale(llm, prop: dict, preference_profile: str) -> str:
    amenities = prop.get("amenity_distances", {})
    amenity_str = ", ".join(
        f"{k}: {v:.1f} mi" for k, v in amenities.items() if v < 90
    )

    prompt = f"""User preference profile: {preference_profile}

Property:
- Address: {prop.get("address")}, {prop.get("city")}, {prop.get("state")}
- Rent: ${prop.get("rent", "?")}/mo  |  {prop.get("bedrooms", "?")} bed  |  {prop.get("bathrooms", "?")} bath
- Nearby: {amenity_str or "data unavailable"}

In exactly 1–2 sentences, explain why this property fits this user's preferences.
Be specific. No filler phrases like "This property offers"."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return resp.content[0]['text'].strip()
    except Exception as e:
        log.warning(f"[SOFT]   ✗ Rationale LLM call failed for {prop.get('address', '?')}: {e}")
        return ""
