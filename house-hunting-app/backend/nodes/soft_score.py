"""
Node 5 — soft_score

Learnable-weight scoring on preference-based criteria (amenity distances,
pet-friendly, quiet neighborhood). Weights start equal and are updated by the
LLM after each feedback round.

This node runs on EVERY feedback loop iteration with the current soft_weights.
Rationales for all new properties are generated in a single batched LLM call.
"""

import json
import os
import re
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from backend.state import AgentState
from backend.tools.scorer import soft_score as compute_soft_score

log = logging.getLogger("agent.soft_score")


def soft_score_node(state: AgentState) -> AgentState:
    props     = state["enriched_properties"]
    iteration = state.get("iteration", 0)
    weights   = state["soft_weights"]
    log.info(f"[SOFT] → Scoring {len(props)} properties  iteration={iteration}")
    log.info(f"[SOFT]   Weights: { {k: round(v,3) for k, v in weights.items()} }")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
    )

    # Compute all soft scores (pure Python, instant)
    scores: dict[str, float] = {
        prop["property_id"]: compute_soft_score(prop, weights)
        for prop in props
    }

    # Carry forward rationales from previous iterations; generate missing ones in one call
    rationales: dict[str, str] = dict(state.get("soft_rationales") or {})
    qualifying = [p for p in props if p["property_id"] not in rationales]

    if qualifying:
        new_rationales = _generate_rationales_batch(llm, qualifying, state["preference_profile"])
        rationales.update(new_rationales)

    if scores:
        top, bottom = max(scores.values()), min(scores.values())
        log.info(f"[SOFT] ✓ Scores range: {bottom:.3f}–{top:.3f}  rationales generated: {len(rationales)}")
        for pid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            prop = next((p for p in props if p["property_id"] == pid), {})
            log.info(f"[SOFT]   {sc:.3f}  {prop.get('address', pid)}, {prop.get('city', '')}")

    state["soft_scores"]     = scores
    state["soft_rationales"] = rationales
    return state


def _generate_rationales_batch(llm, props: list[dict], preference_profile: str) -> dict[str, str]:
    """Single LLM call that returns a rationale for every property at once."""
    lines = []
    for prop in props:
        amenity_str = ", ".join(
            f"{k}: {v:.1f}mi"
            for k, v in (prop.get("amenity_distances") or {}).items()
            if v < 90
        )
        lines.append(
            f'  "{prop["property_id"]}": '
            f'{prop.get("address")}, {prop.get("city")}, {prop.get("state")} | '
            f'${prop.get("rent","?")}/mo | {prop.get("bedrooms","?")}bd/{prop.get("bathrooms","?")}ba | '
            f'nearby: {amenity_str or "N/A"}'
        )

    prompt = f"""User preference profile: {preference_profile}

For each property below write at most 20 words explaining why it fits the user's preferences.
Be specific. No filler phrases like "This property offers".

Properties:
{chr(10).join(lines)}

Respond ONLY with a JSON object mapping each property_id to its rationale string, like:
{{"<property_id>": "<rationale>", ...}}
No markdown, no extra keys."""

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw  = resp.content[0]["text"].strip()
        raw  = re.sub(r"^```(?:json)?\s*", "", raw)
        raw  = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        log.info(f"[SOFT]   ✓ Batch rationale call returned {len(data)} entries")
        return {pid: str(rat) for pid, rat in data.items()}
    except Exception as e:
        log.warning(f"[SOFT]   ✗ Batch rationale call failed: {e} — falling back to empty")
        return {prop["property_id"]: "" for prop in props}
