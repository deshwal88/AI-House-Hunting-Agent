import requests
import streamlit as st
from feature_extractor import RentalRequirements

RENTCAST_API_KEY = st.secrets["RENTCAST_API_KEY"]
ARCGIS_API_KEY   = st.secrets["ARCGIS_API_KEY"]
RENTCAST_BASE_URL = "https://api.rentcast.io/v1"
SEARCH_RADIUS = 2000  # meters for nearby amenities


# ─────────────────────────────────────────────
# 0. GEOCODE A LOCATION (ArcGIS API)
# ─────────────────────────────────────────────
def geocode_location(location: str) -> tuple[float, float] | None:
    """
    Converts a location string to (latitude, longitude) using ArcGIS Geocoding API.

    Works for landmarks, addresses, universities, neighborhoods, etc.
    e.g. "Rutgers University, NJ"  → (40.500, -74.447)
         "downtown Austin TX"      → (30.267, -97.743)
         "5th Street Austin TX"    → (30.269, -97.741)

    Returns None if the location could not be geocoded.
    """
    url = "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
    params = {
        "SingleLine": location,
        "token":      ARCGIS_API_KEY,
        "f":          "json",
    }

    response = requests.get(url, params=params)

    if not response.ok:
        print(f"  ⚠️  Geocoding error: {response.status_code} — {response.text}")
        return None

    candidates = response.json().get("candidates", [])

    if not candidates:
        print(f"  ⚠️  No geocoding results found for: {location}")
        return None

    location_data = candidates[0]["location"]
    return location_data["y"], location_data["x"]  # latitude, longitude


# ─────────────────────────────────────────────
# 1. FETCH LISTINGS FROM RENTCAST (1 API call)
# ─────────────────────────────────────────────
def fetch_listings(features: dict) -> list:
    """
    Fetches ACTIVE rental listings from RentCast.

    Search strategy (in order of priority):
    - street provided    → address + radius search (limit 100)
    - near_location set  → ArcGIS geocode → lat/lng + radius search (limit 100)
    - city/state only    → broad city-wide search (limit 500)

    Broad filters strategy:
    - Location, budget, square footage, lot size, and year built are passed
      to RentCast to narrow the pool.
    - Bedrooms, bathrooms, and property_type are intentionally NOT passed —
      they are handled by the local scoring function for finer control.
    """
    params = {}
    precise_search = False  # tracks whether street or near_location is used

    # ── Location ─────────────────────────────────────────────
    if features.get("street"):
        # Street address provided → build full address + radius
        address_parts = [features["street"]]
        if features.get("city"):  address_parts.append(features["city"])
        if features.get("state"): address_parts.append(features["state"])
        if features.get("zip_code"): address_parts.append(features["zip_code"])

        params["address"] = ", ".join(address_parts)
        params["radius"]  = features.get("radius") or 3   # miles, default 3
        precise_search    = True
        print(f"  📍 Street search: {params['address']}, radius: {params['radius']} miles")

    elif features.get("near_location"):
        # Landmark/area provided → geocode to lat/lng + radius
        print(f"  📍 Geocoding: {features['near_location']}")
        coords = geocode_location(features["near_location"])

        if coords:
            params["latitude"]  = coords[0]
            params["longitude"] = coords[1]

            params["radius"]    = features.get("radius") or 3   # miles, default 3
            precise_search      = True
            print(f"  ✅ Geocoded to ({coords[0]:.4f}, {coords[1]:.4f}), radius: {params['radius']} miles")
        else:
            # Geocoding failed — fall back to city/state search
            print(f"  ⚠️  Geocoding failed, falling back to city/state search")
            if features.get("zip_code"):
                params["zipCode"] = features["zip_code"]
            else:
                if features.get("city"):  params["city"]  = features["city"]
                if features.get("state"): params["state"] = features["state"]
    else:
        # No specific location → broad city/state search
        if features.get("zip_code"):
            params["zipCode"] = features["zip_code"]
        else:
            if features.get("city"):  params["city"]  = features["city"]
            if features.get("state"): params["state"] = features["state"]

    # ── Budget ───────────────────────────────────────────────
    if features.get("budget")     is not None: params["maxPrice"] = features["budget"]
    if features.get("min_budget") is not None: params["minPrice"] = features["min_budget"]

    # ── Square Footage ───────────────────────────────────────
    # RentCast range format: "800:" = at least 800, ":1200" = at most 1200
    if features.get("square_footage") is not None:
        op = features.get("square_footage_operator", "min")
        params["squareFootage"] = f"{features['square_footage']}:" if op == "min" else f":{features['square_footage']}"

    # ── Lot Size ─────────────────────────────────────────────
    if features.get("lot_size") is not None:
        op = features.get("lot_size_operator", "min")
        params["lotSize"] = f"{features['lot_size']}:" if op == "min" else f":{features['lot_size']}"

    # ── Year Built ───────────────────────────────────────────
    if features.get("year_built") is not None:
        op = features.get("year_built_operator", "min")
        params["yearBuilt"] = f"{features['year_built']}:" if op == "min" else f":{features['year_built']}"

    # ── bedrooms, bathrooms, property_type intentionally NOT passed ──
    # These are handled by the local scoring function instead.

    # ── Fetch limit — dynamic based on search type ───────────
    # Precise search (street/near_location) → 100
    # Broad city search → 500
    params["limit"]  = 100 if precise_search else 500
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
# 1.5 PET FRIENDLY FILTER (before scoring)
# ─────────────────────────────────────────────
def filter_pet_friendly(listings: list, features: dict) -> list:
    """
    If the user requires a pet-friendly property, remove all listings
    that are explicitly marked as NOT pet friendly.

    Properties where petFriendly is None/missing are kept — we only
    remove those explicitly marked False.
    """
    if not features.get("pet_friendly"):
        return listings  # User doesn't need pet-friendly — no filtering

    before = len(listings)
    filtered = [p for p in listings if p.get("petFriendly") is not False]
    after = len(filtered)

    if before != after:
        print(f"  🐾 Pet-friendly filter: removed {before - after} listings, {after} remain")

    return filtered


# ─────────────────────────────────────────────
# 2. SCORE EACH PROPERTY (local, no API call)
# ─────────────────────────────────────────────
# Scoring weights (total = 100 pts):
#   Budget fit      — 30 pts
#   Bedrooms match  — 20 pts
#   Bathrooms match — 15 pts
#   Property type   — 10 pts
#   Square footage  — 15 pts
#   Parking         — 10 pts

def score_property(property: dict, features: dict) -> dict:
    breakdown = {}

    # ── Budget (30 pts) ──────────────────────────────────────
    if features.get("budget") and property.get("price"):
        price  = property["price"]
        budget = features["budget"]

        if price <= budget:
            # Full score if within budget. Small bonus for being well under.
            savings = (budget - price) / budget          # 0.0 → 1.0
            breakdown["budget"] = min(30, 22 + savings * 8)   # 22–30 pts
        else:
            # Partial score if slightly over budget (up to 10% over)
            overage = (price - budget) / budget
            breakdown["budget"] = 22 * (1 - overage / 0.10) if overage <= 0.10 else 0
    else:
        breakdown["budget"] = 15  # Neutral if budget not specified

    # ── Bedrooms (20 pts) ────────────────────────────────────
    if features.get("bedrooms") is not None and property.get("bedrooms") is not None:
        diff = abs(property["bedrooms"] - features["bedrooms"])
        if   diff == 0: breakdown["bedrooms"] = 20
        elif diff == 1: breakdown["bedrooms"] = 12
        elif diff == 2: breakdown["bedrooms"] = 4
        else:           breakdown["bedrooms"] = 0
    else:
        breakdown["bedrooms"] = 10  # Neutral

    # ── Bathrooms (15 pts) ───────────────────────────────────
    if features.get("bathrooms") is not None and property.get("bathrooms") is not None:
        diff = abs(property["bathrooms"] - features["bathrooms"])
        if   diff == 0:    breakdown["bathrooms"] = 15
        elif diff <= 0.5:  breakdown["bathrooms"] = 10
        elif diff <= 1.0:  breakdown["bathrooms"] = 5
        else:              breakdown["bathrooms"] = 0
    else:
        breakdown["bathrooms"] = 7  # Neutral

    # ── Property Type (10 pts) ───────────────────────────────
    if features.get("property_type") and property.get("propertyType"):
        breakdown["property_type"] = (
            10 if property["propertyType"].lower() == features["property_type"].lower()
            else 0
        )
    else:
        breakdown["property_type"] = 5  # Neutral if not specified

    # ── Square Footage (15 pts) ──────────────────────────────
    if features.get("square_footage") is not None and property.get("squareFootage") is not None:
        user_sqft     = features["square_footage"]
        prop_sqft     = property["squareFootage"]
        op            = features.get("square_footage_operator", "min")

        if op == "min":
            # User wants at least X sqft
            if prop_sqft >= user_sqft:
                breakdown["square_footage"] = 15                          # Meets or exceeds
            else:
                shortfall = (user_sqft - prop_sqft) / user_sqft          # 0.0 → 1.0
                breakdown["square_footage"] = max(0, 15 * (1 - shortfall * 2))  # Tapers quickly
        else:
            # User wants at most X sqft
            if prop_sqft <= user_sqft:
                breakdown["square_footage"] = 15                          # Within limit
            else:
                excess = (prop_sqft - user_sqft) / user_sqft
                breakdown["square_footage"] = max(0, 15 * (1 - excess * 2))
    else:
        breakdown["square_footage"] = 7  # Neutral if not specified

    # ── Parking (10 pts) ─────────────────────────────────────
    if features.get("parking_required") and property.get("parkingSpaces") is not None:
        breakdown["parking"] = 10 if property["parkingSpaces"] > 0 else 0
    else:
        breakdown["parking"] = 5  # Neutral if not specified or data missing

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
# 4. RADIUS FAILSAFE
# ─────────────────────────────────────────────
def fetch_with_radius_failsafe(features: dict) -> list:
    """
    For precise searches (street or near_location), if fewer than 20
    listings are returned, double the radius and retry — up to 2 times.

    Caps at 100 miles (RentCast maximum).

    City/state searches are not affected — they already cover the whole city.
    """
    is_precise = features.get("street") or features.get("near_location")

    if not is_precise:
        # City/state search — no failsafe needed
        return fetch_listings(features)

    MAX_DOUBLINGS = 2
    MIN_RESULTS   = 20
    MAX_RADIUS    = 100  # miles — RentCast maximum

    current_features = features.copy()
    listings         = fetch_listings(current_features)

    for attempt in range(1, MAX_DOUBLINGS + 1):
        if len(listings) >= MIN_RESULTS:
            break

        current_radius = current_features.get("radius") or 3
        new_radius     = min(current_radius * 2, MAX_RADIUS)

        print(f"  ⚠️  Only {len(listings)} listings found. Expanding radius: {current_radius} → {new_radius} miles (attempt {attempt}/{MAX_DOUBLINGS})")

        current_features          = current_features.copy()
        current_features["radius"] = new_radius
        listings                  = fetch_listings(current_features)

    print(f"  ✅ Radius failsafe complete: {len(listings)} listings found")
    return listings


# ─────────────────────────────────────────────
# 5. MAIN — ENTRY POINT
# ─────────────────────────────────────────────
def find_top_properties(features: dict) -> list:
    """
    Main entry point. Accepts a features dict from the feature extractor.

    Pipeline:
    1. Fetch listings (with radius failsafe for precise searches)
    2. Filter out non-pet-friendly if required
    3. Score and rank top 10
    """
    print(f"🔍 Searching RentCast with features: {features}\n")

    # Step 1 — Fetch with radius failsafe
    listings = fetch_with_radius_failsafe(features)

    if not listings:
        print("No listings found for the given criteria.")
        return []

    print(f"📦 Fetched {len(listings)} active listings.")

    # Step 2 — Pet friendly filter
    listings = filter_pet_friendly(listings, features)

    if not listings:
        print("No listings remaining after pet-friendly filter.")
        return []

    print(f"🏆 Scoring and ranking {len(listings)} listings...\n")

    # Step 3 — Score and rank
    top_10 = rank_properties(listings, features)
    return top_10

# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
# if __name__ == "__main__":
#     from feature_extractor import extract_features
#     import streamlit as st
#
#     api_key = st.secrets["GEMINI_API_KEY"]
#
#     # Example 1 — city/state search (whole city, limit 500)
#     req = extract_features("2 bed apartment in Austin TX under $2500", api_key)
#     top_properties = find_top_properties(req)
#
#     # Example 2 — street search (address + radius, limit 100)
#     req = extract_features("2 bed near 5th Avenue Austin TX under $2500", api_key)
#     top_properties = find_top_properties(req)
#
#     # Example 3 — near_location search (geocode + radius, limit 100)
#     req = extract_features("2 bed near Rutgers University NJ under $2000", api_key)
#     top_properties = find_top_properties(req)
