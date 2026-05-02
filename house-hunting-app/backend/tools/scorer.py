"""
Scoring functions — pure Python, no API calls.

hard_score : fixed weights, objective criteria, never changes across loops.
soft_score : learnable weights, preference-based, updated after each feedback round.
"""

from __future__ import annotations
import numpy as np


# ── Hard scoring ──────────────────────────────────────────────────────────────

HARD_WEIGHTS = {
    "price_match":    0.35,
    "bedroom_match":  0.25,
    "bathroom_match": 0.15,
    "sqft_per_dollar": 0.15,
    "within_radius":  0.10,
}

assert abs(sum(HARD_WEIGHTS.values()) - 1.0) < 1e-9, "Hard weights must sum to 1.0"


def hard_score(property: dict, requirements: dict, sqft_per_dollar_norm: float = 0.5) -> float:
    """
    Score a property on objective, fixed criteria.

    Args:
        property              : normalized property dict from fetch node
        requirements          : requirements dict from Streamlit wizard
        sqft_per_dollar_norm  : pre-computed normalized value (0-1) for this
                                property's sqft/price ratio (computed across
                                the whole batch in hard_score_node)
    Returns:
        float in [0, 1]
    """
    prop_section = requirements.get("property", {})
    score = 0.0

    # ── Price match (0.35) ────────────────────────────────────────────────
    price_min = prop_section.get("price_min", 0) or 0
    price_max = prop_section.get("price_max", float("inf")) or float("inf")
    mid = (price_min + price_max) / 2 if price_max != float("inf") else price_min
    rent = property.get("rent") or 0
    if mid > 0:
        deviation = abs(rent - mid) / mid
        score += HARD_WEIGHTS["price_match"] * max(0.0, 1.0 - deviation)
    else:
        score += HARD_WEIGHTS["price_match"] * 0.5

    # ── Bedroom match (0.25) ──────────────────────────────────────────────
    wanted_beds = prop_section.get("bedrooms")
    actual_beds = property.get("bedrooms")
    if wanted_beds is not None and actual_beds is not None:
        diff = abs(actual_beds - wanted_beds)
        score += HARD_WEIGHTS["bedroom_match"] * max(0.0, 1.0 - diff * 0.5)
    else:
        score += HARD_WEIGHTS["bedroom_match"] * 0.5  # neutral

    # ── Bathroom match (0.15) ─────────────────────────────────────────────
    wanted_baths = prop_section.get("bathrooms")
    actual_baths = property.get("bathrooms")
    if wanted_baths is not None and actual_baths is not None:
        diff = abs(actual_baths - wanted_baths)
        score += HARD_WEIGHTS["bathroom_match"] * max(0.0, 1.0 - diff * 0.5)
    else:
        score += HARD_WEIGHTS["bathroom_match"] * 0.5

    # ── Sqft per dollar (0.15) ────────────────────────────────────────────
    score += HARD_WEIGHTS["sqft_per_dollar"] * sqft_per_dollar_norm

    # ── Within radius (0.10) — RentCast already filters; assume true ──────
    score += HARD_WEIGHTS["within_radius"] * 1.0

    return round(min(score, 1.0), 4)


def normalize_sqft_per_dollar(properties: list[dict]) -> dict[str, float]:
    """
    Compute and normalize sqft/price ratios across a batch of properties.
    Returns a dict of property_id -> normalized_value (0-1).
    """
    ratios: dict[str, float] = {}
    for p in properties:
        sqft = p.get("sqft") or 0
        rent = p.get("rent") or 1
        ratios[p["property_id"]] = sqft / rent

    vals = list(ratios.values())
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi > lo else 1.0

    return {pid: round((v - lo) / span, 4) for pid, v in ratios.items()}


# ── Soft scoring ──────────────────────────────────────────────────────────────

DEFAULT_SOFT_WEIGHTS: dict[str, float] = {
    "grocery_distance":  1 / 6,
    "school_distance":   1 / 6,
    "gym_distance":      1 / 6,
    "transit_distance":  1 / 6,
    "pet_friendly":      1 / 6,
    "quiet_neighborhood": 1 / 6,
}


def soft_score(property: dict, soft_weights: dict | None = None) -> float:
    """
    Score a property on learnable, preference-based criteria.

    The weights default to equal (1/6 each) and are updated by the LLM
    in update_weights node after each feedback round.

    Returns:
        float in [0, 1]
    """
    weights = soft_weights if soft_weights else DEFAULT_SOFT_WEIGHTS
    amenities = property.get("amenity_distances", {})
    score = 0.0

    # ── Grocery (linear decay: 0 mi = 1.0, ≥2 mi = 0.0) ──────────────────
    d = amenities.get("grocery", 99.0)
    score += weights.get("grocery_distance", 1/6) * max(0.0, 1.0 - d / 2.0)

    # ── School (linear decay: 0 mi = 1.0, ≥3 mi = 0.0) ───────────────────
    d = amenities.get("school", 99.0)
    score += weights.get("school_distance", 1/6) * max(0.0, 1.0 - d / 3.0)

    # ── Gym (linear decay: 0 mi = 1.0, ≥2 mi = 0.0) ─────────────────────
    d = amenities.get("gym", 99.0)
    score += weights.get("gym_distance", 1/6) * max(0.0, 1.0 - d / 2.0)

    # ── Transit (linear decay: 0 mi = 1.0, ≥0.5 mi = 0.0) ────────────────
    d = amenities.get("transit", 99.0)
    score += weights.get("transit_distance", 1/6) * max(0.0, 1.0 - d / 0.5)

    # ── Pet friendly (binary) ─────────────────────────────────────────────
    pet_ok = 1.0 if property.get("petFriendly") or property.get("pet_friendly") else 0.0
    score += weights.get("pet_friendly", 1/6) * pet_ok

    # ── Quiet neighborhood (proxy: low transit density = quieter) ─────────
    transit_d = amenities.get("transit", 99.0)
    quiet_proxy = max(0.0, min(1.0, transit_d / 1.0))  # farther from transit = quieter
    score += weights.get("quiet_neighborhood", 1/6) * quiet_proxy

    return round(min(score, 1.0), 4)


# ── Convergence check ─────────────────────────────────────────────────────────

def weights_converged(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    threshold: float = 0.97,
) -> bool:
    """Return True if cosine similarity between old and new weight vectors exceeds threshold."""
    old = np.array([old_weights.get(k, 0) for k in DEFAULT_SOFT_WEIGHTS])
    new = np.array([new_weights.get(k, 0) for k in DEFAULT_SOFT_WEIGHTS])
    norm_product = np.linalg.norm(old) * np.linalg.norm(new)
    if norm_product == 0:
        return False
    return float(np.dot(old, new) / norm_product) > threshold
