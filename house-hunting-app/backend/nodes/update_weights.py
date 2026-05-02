"""
Node 8 — update_weights

The core personalization engine. Uses the LLM to interpret the gap between
the system's ranking and the user's drag-and-drop reordering, infer what
the user values more/less, and update soft_weights + preference_profile.

Convergence is checked here under 3 conditions:
  1. User ordering matches system ranking exactly.
  2. Cosine similarity of old vs new weights exceeds 0.97.
  3. Hard cap of 5 iterations.
"""

import os
import json
import re
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from backend.state import AgentState
from backend.tools.scorer import weights_converged

log = logging.getLogger("agent.update_weights")
MAX_ITERATIONS = 5


def update_weights(state: AgentState) -> AgentState:
    system_ranking = [p["property_id"] for p in state["ranked_list"]]
    user_ranking   = state.get("user_feedback", [])
    iteration      = state.get("iteration", 0)

    log.info(f"[WEIGHTS] → Iteration {iteration+1}: processing user feedback")
    log.info(f"[WEIGHTS]   System order : {system_ranking[:5]}")
    log.info(f"[WEIGHTS]   User order   : {user_ranking[:5]}")

    # ── Convergence check 1: orderings match ─────────────────────────────
    if system_ranking == user_ranking:
        log.info("[WEIGHTS] ✓ Convergence: user order matches system ranking")
        state["converged"]          = True
        state["convergence_reason"] = "Your ranking matches our recommendation — preferences locked in!"
        return state

    # ── Build comparison for LLM ──────────────────────────────────────────
    id_to_prop = {p["property_id"]: p for p in state["ranked_list"]}
    comparison_lines: list[str] = []

    for user_rank, pid in enumerate(user_ranking, start=1):
        prop = id_to_prop.get(pid)
        if not prop:
            continue
        sys_rank  = system_ranking.index(pid) + 1 if pid in system_ranking else "?"
        amenities = prop.get("amenity_distances", {})
        am_str    = ", ".join(f"{k}: {v:.1f}mi" for k, v in amenities.items() if v < 90)
        comparison_lines.append(
            f"  User #{user_rank} (System #{sys_rank}): "
            f"{prop.get('address')}, {prop.get('city')} | "
            f"${prop.get('rent', '?')}/mo | {prop.get('bedrooms', '?')}bd/{prop.get('bathrooms', '?')}ba | "
            f"Amenities: {am_str or 'N/A'}"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.2,
    )

    prompt = f"""You are a preference learning engine for a rental recommendation system.

Current preference profile:
{state["preference_profile"]}

Current soft weights (must sum to 1.0):
{json.dumps(state["soft_weights"], indent=2)}

The system ranked properties in one order. The user dragged them into a different order.
Here is the user's preferred ranking with each property's details:

{chr(10).join(comparison_lines)}

Analyze the differences and determine:
1. Which amenities/features does the user value MORE than currently weighted?
2. Which does the user value LESS?
3. Update the preference profile to reflect what you learned.
4. Update the soft weights — they MUST sum to exactly 1.0.

Respond in this exact JSON format (no markdown, no preamble):
{{
  "updated_preference_profile": "...",
  "updated_soft_weights": {{
    "grocery_distance": 0.0,
    "school_distance": 0.0,
    "gym_distance": 0.0,
    "transit_distance": 0.0,
    "pet_friendly": 0.0,
    "quiet_neighborhood": 0.0
  }},
  "reasoning": "one sentence explaining the key weight change"
}}"""

    old_weights = dict(state["soft_weights"])
    log.info(f"[WEIGHTS]   Old weights: { {k: round(v,3) for k, v in old_weights.items()} }")
    log.info("[WEIGHTS]   Calling Gemini to infer new weights...")

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        print(f"LLM response:\n{resp.content}")
        raw  = resp.content[0]['text'].strip()
        raw  = re.sub(r"^```(?:json)?\s*", "", raw)
        raw  = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)

        new_weights = data["updated_soft_weights"]

        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        log.info(f"[WEIGHTS]   New weights: { {k: round(v,3) for k, v in new_weights.items()} }")
        log.info(f"[WEIGHTS]   Reasoning  : {data.get('reasoning', 'n/a')}")

        state["preference_profile"] = data["updated_preference_profile"]
        state["soft_weights"]       = new_weights

    except Exception as e:
        log.error(f"[WEIGHTS] ✗ LLM call failed — keeping old weights: {e}")
        new_weights = old_weights

    state["iteration"] += 1

    # ── Convergence check 2: weight stability ─────────────────────────────
    if weights_converged(old_weights, new_weights):
        log.info("[WEIGHTS] ✓ Convergence: weights are stable (cosine similarity > 0.97)")
        state["converged"]          = True
        state["convergence_reason"] = "Your preferences have stabilised — we've learned your taste!"
        return state

    # ── Convergence check 3: max iterations ───────────────────────────────
    if state["iteration"] >= MAX_ITERATIONS:
        log.info(f"[WEIGHTS] ✓ Convergence: max iterations ({MAX_ITERATIONS}) reached")
        state["converged"]          = True
        state["convergence_reason"] = f"Reached {MAX_ITERATIONS} refinement rounds — final results ready."
    else:
        log.info(f"[WEIGHTS]   Not converged — iteration {state['iteration']}/{MAX_ITERATIONS}, re-ranking")

    return state
