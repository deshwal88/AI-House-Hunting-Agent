"""
rentcast_adapter.py
-------------------
Bridges the RentalRequirements object (from feature_extractor.py)
into the features dict that rentcast_search.py expects.

Usage:
    from feature_extractor import extract_features
    from rentcast_adapter import adapt_requirements
    from rentcast_search import find_top_properties

    requirements = extract_features(user_query, api_key)
    features     = adapt_requirements(requirements)
    top_10       = find_top_properties(features)
"""

from feature_extractor import RentalRequirements


def _parse_location(location: str) -> tuple[str | None, str | None, str | None]:
    """
    Splits a location string into (city, state, zip_code).

    Handles formats like:
      "Austin, TX"            → ("Austin", "TX", None)
      "New Brunswick, NJ"     → ("New Brunswick", "NJ", None)
      "Jersey City, NJ 07302" → ("Jersey City", "NJ", "07302")
      "Austin"                → ("Austin", None, None)
    """
    if not location:
        return None, None, None

    city      = None
    state     = None
    zip_code  = None

    parts = [p.strip() for p in location.split(",")]

    if len(parts) >= 1:
        city = parts[0]

    if len(parts) >= 2:
        # Second part may be "TX" or "NJ 07302"
        state_part = parts[1].strip().split()
        state = state_part[0] if state_part else None

        # If there's a zip code attached to the state
        if len(state_part) == 2 and state_part[1].isdigit():
            zip_code = state_part[1]

    return city, state, zip_code


def adapt_requirements(req: RentalRequirements) -> dict:
    """
    Converts a RentalRequirements object into the features dict
    expected by rentcast_search.py → find_top_properties().

    RentalRequirements field   →   RentCast features key
    ──────────────────────────────────────────────────────
    max_budget                 →   budget        (used as maxPrice in API)
    min_budget                 →   min_budget    (used as minPrice in API)
    location (single string)   →   city + state + zip_code (split)
    bedrooms                   →   bedrooms
    bathrooms                  →   bathrooms
    property_type (lowercase)  →   property_type (title-cased)

    Note: parking_required, pet_friendly, amenities, max_commute_minutes
    are NOT passed to RentCast (it doesn't support them). They are
    preserved in the returned dict under "extra" for your next teammate
    to use in post-processing.
    """

    # ── Location ────────────────────────────────────────────
    city, state, zip_code = _parse_location(req.location)

    # ── Property type — title-case to match RentCast's expected format ──
    # e.g. "apartment" → "Apartment", "single family" → "Single Family"
    property_type = req.property_type.title() if req.property_type else None

    # ── Core features dict (what rentcast_search.py consumes) ───────────
    features = {
        "city":          city,
        "state":         state,
        "zip_code":      zip_code,
        "bedrooms":      req.bedrooms,
        "bathrooms":     req.bathrooms,
        "budget":        req.max_budget,    # maps to maxPrice in RentCast API
        "min_budget":    req.min_budget,    # maps to minPrice in RentCast API
        "property_type": property_type,
    }

    # ── Extra fields — not used by RentCast, preserved for next steps ───
    # Your next teammate can read these from the returned dict.
    features["extra"] = {
        "parking_required":    req.parking_required,
        "pet_friendly":        req.pet_friendly,
        "amenities":           req.amenities,
        "max_commute_minutes": req.max_commute_minutes,
        "commute_destination": req.commute_destination,
    }

    return features


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
# if __name__ == "__main__":
#     import json
#     from feature_extractor import extract_features

#     api_key    = "YOUR_GEMINI_API_KEY"
#     user_query = "2 bed apartment in Austin, TX under $2500/month, pet friendly with parking"

#     print(f"Query: {user_query}\n")

#     requirements = extract_features(user_query, api_key)
#     features     = adapt_requirements(requirements)

#     print("Adapted features dict:")
#     print(json.dumps(features, indent=2))
