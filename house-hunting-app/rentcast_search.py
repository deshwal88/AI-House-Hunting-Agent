import requests
import streamlit as st
RENTCAST_API_KEY = st.secrets["RENTCAST_API_KEY"]
RENTCAST_BASE_URL = "https://api.rentcast.io/v1"

# ─────────────────────────────────────────────
# 1. FETCH LISTINGS FROM RENTCAST (1 API call)
# ─────────────────────────────────────────────
def fetch_listings(features: dict) -> list:
    """
    Fetches up to 100 ACTIVE rental listings from RentCast.

    We intentionally only pass area + budget to RentCast (broad filters),
    and leave bedrooms, bathrooms, and property_type for local scoring.

    Why? RentCast's limit=100 returns the 100 most recently listed matches.
    If we pass all filters, we might miss great properties listed slightly
    earlier. By keeping filters broad, we get a more representative pool
    for the local ranker to work with.
    """
    params = {}

    # ── Location (required) ──────────────────────────────────
    # Prefer zip_code if available, else city + state
    if features.get("zip_code"):
        params["zipCode"] = features["zip_code"]
    else:
        if features.get("city"):  params["city"]  = features["city"]
        if features.get("state"): params["state"] = features["state"]

    # ── Budget (passed to RentCast to avoid wildly irrelevant results) ──
    if features.get("budget")     is not None: params["maxPrice"] = features["budget"]
    if features.get("min_budget") is not None: params["minPrice"] = features["min_budget"]

    # ── bedrooms, bathrooms, property_type are intentionally NOT passed ──
    # They are handled by the local scoring function instead.

    # Fetch 100 active listings — still just 1 API call
    params["limit"]  = 100
    params["status"] = "Active"

    response = requests.get(
        f"{RENTCAST_BASE_URL}/listings/rental/long-term",
        headers={
            "X-Api-Key": RENTCAST_API_KEY,
            "Content-Type": "application/json",
        },
        params=params,
    )

    if not response.ok:
        raise Exception(f"RentCast API error: {response.status_code} — {response.text}")

    return response.json()  # List of property dicts


# ─────────────────────────────────────────────
# 2. SCORE EACH PROPERTY (local, no API call)
# ─────────────────────────────────────────────
# Scoring weights (total = 100 pts):
#   Budget fit      — 40 pts  (most important for renters)
#   Bedrooms match  — 25 pts
#   Bathrooms match — 20 pts
#   Property type   — 15 pts

def score_property(property: dict, features: dict) -> dict:
    breakdown = {}

    # ── Budget (40 pts) ──────────────────────────────────────
    if features.get("budget") and property.get("price"):
        price  = property["price"]
        budget = features["budget"]

        if price <= budget:
            # Full score if within budget. Small bonus for being well under.
            savings = (budget - price) / budget          # 0.0 → 1.0
            breakdown["budget"] = min(40, 30 + savings * 10)   # 30–40 pts
        else:
            # Partial score if slightly over budget (up to 10% over)
            overage = (price - budget) / budget
            breakdown["budget"] = 30 * (1 - overage / 0.10) if overage <= 0.10 else 0
    else:
        breakdown["budget"] = 20  # Neutral if budget not specified

    # ── Bedrooms (25 pts) ────────────────────────────────────
    if features.get("bedrooms") is not None and property.get("bedrooms") is not None:
        diff = abs(property["bedrooms"] - features["bedrooms"])
        if   diff == 0: breakdown["bedrooms"] = 25
        elif diff == 1: breakdown["bedrooms"] = 15
        elif diff == 2: breakdown["bedrooms"] = 5
        else:           breakdown["bedrooms"] = 0
    else:
        breakdown["bedrooms"] = 12  # Neutral

    # ── Bathrooms (20 pts) ───────────────────────────────────
    if features.get("bathrooms") is not None and property.get("bathrooms") is not None:
        diff = abs(property["bathrooms"] - features["bathrooms"])
        if   diff == 0:    breakdown["bathrooms"] = 20
        elif diff <= 0.5:  breakdown["bathrooms"] = 15
        elif diff <= 1.0:  breakdown["bathrooms"] = 8
        else:              breakdown["bathrooms"] = 0
    else:
        breakdown["bathrooms"] = 10  # Neutral

    # ── Property Type (15 pts) ───────────────────────────────
    if features.get("property_type") and property.get("propertyType"):
        breakdown["property_type"] = (
            15 if property["propertyType"].lower() == features["property_type"].lower()
            else 0
        )
    else:
        breakdown["property_type"] = 7  # Neutral if not specified

    total_score = sum(breakdown.values())

    return {
        **property,
        "_score": round(total_score),
        "_score_breakdown": breakdown,
    }


# ─────────────────────────────────────────────
# 3. RANK AND RETURN TOP 10 (local, no API call)
# ─────────────────────────────────────────────
def rank_properties(listings: list, features: dict) -> list:
    scored = [score_property(p, features) for p in listings]
    scored.sort(key=lambda p: p["_score"], reverse=True)
    return scored[:10]


# ─────────────────────────────────────────────
# 4. MAIN — ENTRY POINT
# ─────────────────────────────────────────────
def find_top_properties(features: dict) -> list:
    print(f"🔍 Searching RentCast with features: {features}\n")

    listings = fetch_listings(features)

    if not listings:
        print("No listings found for the given criteria.")
        return []

    print(f"📦 Fetched {len(listings)} active listings. Scoring and ranking...\n")

    top_10 = rank_properties(listings, features)

    print(f"🏆 Top {len(top_10)} Properties:\n")
    for i, prop in enumerate(top_10, start=1):
        print(f"{i}. {prop.get('addressLine1')}, {prop.get('city')}, {prop.get('state')} {prop.get('zipCode')}")
        print(f"   💰 ${prop.get('price')}/mo | 🛏 {prop.get('bedrooms')} bed | 🚿 {prop.get('bathrooms')} bath | 🏠 {prop.get('propertyType')}")
        print(f"   📍 ({prop.get('latitude')}, {prop.get('longitude')})")
        bd = prop["_score_breakdown"]
        print(f"   ⭐ Score: {prop['_score']}/100 | Budget: {bd['budget']:.1f} | Beds: {bd['bedrooms']} | Baths: {bd['bathrooms']} | Type: {bd['property_type']}")
        print()

    return top_10


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
# if __name__ == "__main__":

#     # This is what your extracted features dict looks like
#     # after processing the user's natural language input.
#     extracted_features = {
#         "city":          "Austin",
#         "state":         "TX",
#         "zip_code":      None,
#         "bedrooms":      3,
#         "bathrooms":     2,
#         "budget":        2500,
#         "property_type": "Apartment",
#     }

#     top_properties = find_top_properties(extracted_features)

    # top_properties is a list of 10 dicts, each with:
    #   - all original RentCast fields (address, price, lat, lng, etc.)
    #   - _score          (int, 0–100)
    #   - _score_breakdown (dict, per-criterion scores)
    #
    # The latitude/longitude on each property feeds directly
    # into your next ArcGIS enrichment step.
