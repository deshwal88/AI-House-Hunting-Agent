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


def adapt_requirements(req: RentalRequirements) -> dict:
    """
    Converts a RentalRequirements object into the features dict
    expected by rentcast_search.py → find_top_properties().

    RentalRequirements field   →   RentCast features key
    ──────────────────────────────────────────────────────
    max_budget                 →   budget        (used as maxPrice in API)
    min_budget                 →   min_budget    (used as minPrice in API)
    city, state, zip_code      →   city, state, zip_code (already split)
    bedrooms                   →   bedrooms
    bathrooms                  →   bathrooms
    property_type              →   property_type (already title-cased from extractor)

    Note: parking_required, pet_friendly, amenities, max_commute_minutes,
    square_footage, lot_size, year_built are NOT passed to RentCast.
    They are preserved under "extra" for post-processing steps.
    """

    # ── Core features dict (what rentcast_search.py consumes) ───────────
    features = {
        "street":        req.street,
        "city":          req.city,
        "state":         req.state,
        "zip_code":      req.zip_code,
        "radius":        req.radius,
        "bedrooms":      req.bedrooms,
        "bathrooms":     req.bathrooms,
        "budget":        req.max_budget,    # maps to maxPrice in RentCast API
        "min_budget":    req.min_budget,    # maps to minPrice in RentCast API
        "property_type": req.property_type,
    }

    # ── Extra fields — not used by RentCast, preserved for next steps ───
    features["extra"] = {
        "parking_required":      req.parking_required,
        "pet_friendly":          req.pet_friendly,
        "amenities":             req.amenities,
        "max_commute_minutes":   req.max_commute_minutes,
        "commute_destination":   req.commute_destination,
        "square_footage":        req.square_footage,
        "square_footage_operator": req.square_footage_operator,
        "lot_size":              req.lot_size,
        "lot_size_operator":     req.lot_size_operator,
        "year_built":            req.year_built,
        "year_built_operator":   req.year_built_operator,
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