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

    props_by_id = {prop["property_id"]: prop for prop in props}
    for prop in props:
        scores[prop["property_id"]] = compute_soft_score(prop, weights)

    top10_ids = sorted(scores, key=scores.get, reverse=True)[:10]
    top10_props = [props_by_id[pid] for pid in top10_ids]

    for prop in top10_props:
        break
        pid = prop["property_id"]
        rationale = _generate_rationale(llm, prop, state.get("preference_profile", ""))
        rationales[pid] = rationale
        

    state["soft_scores"] = scores
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

Write a concise rationale in 30–35 words explaining why this property fits the user's preferences.
Use specific detail, avoid generic filler, and keep it self-contained and easy to read."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        rationale = resp.content[0]['text'].strip()
        words = rationale.split()
        if len(words) > 35:
            rationale = " ".join(words[:35]).rstrip(".,;:!") + "..."
        return rationale

    except Exception as e:
        log.warning(f"[SOFT]   ✗ Rationale LLM call failed for {prop.get('address', '?')}: {e}")
        return ""
