from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    requirements: dict          # structured dict from Streamlit wizard
    session_id: str

    # ── After Node 2: fetch_properties ────────────────────────────────────
    properties: List[dict]      # normalized property dicts from RentCast

    # ── After Node 3: enrich_properties ───────────────────────────────────
    enriched_properties: List[dict]  # properties + images + amenity_distances

    # ── After Node 4: hard_score ───────────────────────────────────────────
    hard_scores: Dict[str, float]    # property_id -> 0.0–1.0

    # ── After Node 5: soft_score (updated each loop) ──────────────────────
    soft_scores: Dict[str, float]    # property_id -> 0.0–1.0
    soft_rationales: Dict[str, str]  # property_id -> LLM explanation

    # ── After Node 6: rank_and_display ────────────────────────────────────
    ranked_list: List[dict]     # sorted properties with final_score + rationale

    # ── After Node 7: collect_feedback ────────────────────────────────────
    user_feedback: List[str]    # ordered list of property_ids from drag-drop

    # ── Learned state (evolves across feedback loops) ─────────────────────
    soft_weights: Dict[str, float]   # feature_name -> weight (must sum to 1.0)
    preference_profile: str          # LLM natural-language summary of user prefs

    # ── Control flow ──────────────────────────────────────────────────────
    iteration: int              # feedback loop counter, starts at 0
    converged: bool             # termination flag
    convergence_reason: str     # why it converged (shown in UI)
