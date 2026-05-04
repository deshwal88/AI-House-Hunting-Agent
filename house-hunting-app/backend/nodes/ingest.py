"""
Node 1 — ingest_requirements

Reads the structured requirements dict from Streamlit, parses any free-text
additional_comments with an LLM, infers latent preferences, and builds the
initial preference_profile string.

LLM is central here: the structured form can't capture nuance like
"I work from home and have a large dog" → quiet, pet-friendly, space matters.
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from backend.state import AgentState
from backend.tools.scorer import DEFAULT_SOFT_WEIGHTS

log = logging.getLogger("agent.ingest")


def ingest_requirements(state: AgentState) -> AgentState:
    req      = state["requirements"]
    comments = req.get("additional_comments", "").strip()
    sid      = state.get("session_id", "?")

    log.info(f"[INGEST] → Session {sid}: ingesting requirements")
    loc  = req.get("location", {})
    prop = req.get("property", {})
    log.info(f"[INGEST]   Location : {loc.get('city')}, {loc.get('state')}  ZIP={loc.get('zip', 'n/a')}")
    log.info(f"[INGEST]   Property : {prop.get('bedrooms', '?')}bd/{prop.get('bathrooms', '?')}ba  "
             f"${prop.get('price_min', 0):,}–${prop.get('price_max', 0):,}/mo")
    if comments:
        log.info(f"[INGEST]   Comments : {comments[:120]}")

    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        log.error("[INGEST] ✗ GEMINI_API_KEY not set — LLM call will fail")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        google_api_key=gemini_key,
        temperature=0.2,
    )

    prompt = f"""You are a rental requirements analyst for a personalized home-finding AI.

The user completed a structured search form with these details:
{req}

{"They also added this free-text comment: " + repr(comments) if comments else "They left no additional comments."}

Your tasks:
1. Identify latent preferences implied by their inputs — for example, "I have a dog" implies pet-friendly and possibly ground-floor preference; "work from home" implies need for quiet space and fast internet.
2. Write a concise preference profile (3–5 sentences) summarizing what matters most to this user.

Return ONLY the preference profile. No preamble, no bullet points, no markdown."""

    log.info("[INGEST]   Calling Gemini to build preference profile...")
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        profile = resp.content[0]['text'].strip()
        log.info(f"[INGEST] ✓ Preference profile built: {profile[:120]}...")
    except Exception as e:
        log.error(f"[INGEST] ✗ LLM call failed: {e}")
        profile = "No preference profile available due to LLM error."

    state["preference_profile"] = profile
    state["iteration"]          = 0
    state["converged"]          = False
    state["convergence_reason"] = ""
    state["soft_weights"]       = dict(DEFAULT_SOFT_WEIGHTS)
    state["soft_rationales"]    = {}
    state["user_feedback"]      = []
    state["ranked_list"]        = []
    state["hard_scores"]        = {}

    state["soft_scores"]        = {}
    state["enriched_properties"] = []
    state["properties"]         = []

    log.info(f"[INGEST] ✓ State initialised  soft_weights={list(DEFAULT_SOFT_WEIGHTS.keys())}")
    return state
