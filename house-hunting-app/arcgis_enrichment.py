import requests
import streamlit as st

ARCGIS_API_KEY = st.secrets["ARCGIS_API_KEY"]
ARCGIS_PLACES_URL = "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/places/near-point"

# ─────────────────────────────────────────────
# DYNAMICALLY FETCH CORRECT CATEGORY IDs
# ─────────────────────────────────────────────
def get_category_id(filter_term: str) -> str | None:
    """
    Queries ArcGIS /categories endpoint to find the correct category ID
    for a given search term. This avoids hardcoding IDs that may change.
    """
    url = "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/categories"
    params = {
        "filter": filter_term,
        "f":      "json",
        "token":  ARCGIS_API_KEY,
    }
    response = requests.get(url, params=params)
    if not response.ok:
        return None
    results = response.json().get("categories", [])
    return results[0]["categoryId"] if results else None


# Fetch correct category IDs at startup — never hardcode these
POI_CATEGORIES = {
    "hospitals": get_category_id("Emergency Room"),
    "schools":   get_category_id("Elementary School"),
    "transit":   get_category_id("Bus Stop"),
}

# Search radius in meters (2km is a good default for walkability)
SEARCH_RADIUS = 2000


# ─────────────────────────────────────────────
# 1. FETCH NEARBY POIs FOR ONE CATEGORY
# ─────────────────────────────────────────────
def fetch_nearby_pois(latitude: float, longitude: float, category_id: str, label: str) -> list:
    """
    Fetches nearby places of a given category around a lat/lng.
    Returns a simplified list of POIs.
    """
    params = {
        "y":           latitude,
        "x":           longitude,
        "radius":      SEARCH_RADIUS,
        "categoryIds": category_id,
        "f":           "json",
        "token":       ARCGIS_API_KEY,
    }

    response = requests.get(ARCGIS_PLACES_URL, params=params)

    if not response.ok:
        print(f"  ⚠️  ArcGIS error for {label}: {response.status_code} — {response.text}")
        return []

    data = response.json()
    results = data.get("results", [])

    # Simplify each POI to just the fields we care about
    simplified = []
    for place in results:
        simplified.append({
            "name":      place.get("name"),
            "distance":  round(place.get("distance", 0)),   # meters
            "latitude":  place.get("location", {}).get("y"),
            "longitude": place.get("location", {}).get("x"),
            "address":   place.get("address", {}).get("label"),
        })

    # Sort by distance ascending
    simplified.sort(key=lambda p: p["distance"])

    return simplified


# ─────────────────────────────────────────────
# 2. ENRICH ONE PROPERTY WITH ALL POI TYPES
# ─────────────────────────────────────────────
def enrich_property(property: dict) -> dict:
    """
    Takes a single property dict (with latitude/longitude from RentCast)
    and adds nearby hospitals, schools, and transit stops to it.
    Makes 3 ArcGIS API calls — one per POI type.
    """
    lat = property.get("latitude")
    lng = property.get("longitude")
    address = property.get("addressLine1", "Unknown address")

    print(f"  📍 Enriching: {address}")

    nearby = {}
    for label, category_id in POI_CATEGORIES.items():
        pois = fetch_nearby_pois(lat, lng, category_id, label)
        nearby[label] = pois
        print(f"     ✅ {label.capitalize()}: {len(pois)} found")

    return {
        **property,
        "nearby": nearby,
    }


# ─────────────────────────────────────────────
# 3. ENRICH ALL TOP 10 PROPERTIES
# ─────────────────────────────────────────────
def enrich_all_properties(top_properties: list) -> list:
    """
    Enriches all 10 properties with nearby POIs.
    Total ArcGIS API calls = 10 properties × 3 POI types = 30 calls.
    """
    print(f"\n🏥 Enriching {len(top_properties)} properties with ArcGIS Places...\n")

    enriched = []
    for i, property in enumerate(top_properties, start=1):
        print(f"[{i}/{len(top_properties)}]")
        enriched_property = enrich_property(property)
        enriched.append(enriched_property)

    print(f"\n✅ Enrichment complete for all {len(enriched)} properties.")
    return enriched


# ─────────────────────────────────────────────
# 4. PRETTY PRINT RESULTS
# ─────────────────────────────────────────────
def print_enriched_results(enriched_properties: list):
    print("\n" + "═" * 60)
    print("🏆 TOP PROPERTIES WITH NEARBY AMENITIES")
    print("═" * 60)

    for i, prop in enumerate(enriched_properties, start=1):
        print(f"\n{i}. {prop.get('addressLine1')}, {prop.get('city')}, {prop.get('state')}")
        print(f"   💰 ${prop.get('price')}/mo | 🛏 {prop.get('bedrooms')} bed | 🚿 {prop.get('bathrooms')} bath")
        print(f"   ⭐ Match Score: {prop.get('_score')}/100")

        nearby = prop.get("nearby", {})

        # Hospitals
        hospitals = nearby.get("hospitals", [])
        print(f"\n   🏥 Hospitals ({len(hospitals)} within {SEARCH_RADIUS}m):")
        for h in hospitals[:3]:  # Show top 3 closest
            print(f"      • {h['name']} — {h['distance']}m away")

        # Schools
        schools = nearby.get("schools", [])
        print(f"\n   🏫 Schools ({len(schools)} within {SEARCH_RADIUS}m):")
        for s in schools[:3]:
            print(f"      • {s['name']} — {s['distance']}m away")

        # Transit
        transit = nearby.get("transit", [])
        print(f"\n   🚌 Transit Stops ({len(transit)} within {SEARCH_RADIUS}m):")
        for t in transit[:3]:
            print(f"      • {t['name']} — {t['distance']}m away")

        print("\n   " + "─" * 50)


# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────
# if __name__ == "__main__":

#     # This is the output from the RentCast step (top 10 properties).
#     # In your full pipeline, you pass the result of find_top_properties() here.
#     # Using mock data below to illustrate the structure.

#     top_10_from_rentcast = [
#         {
#             "addressLine1": "123 Main St",
#             "city": "Austin",
#             "state": "TX",
#             "zipCode": "78701",
#             "price": 2400,
#             "bedrooms": 3,
#             "bathrooms": 2,
#             "propertyType": "Apartment",
#             "latitude": 30.267,
#             "longitude": -97.743,
#             "_score": 95,
#             "_score_breakdown": {"budget": 38, "bedrooms": 25, "bathrooms": 20, "property_type": 15},
#         },
#         # ... 9 more properties from RentCast
#     ]

#     # Enrich all 10 properties with nearby hospitals, schools, transit
#     enriched_properties = enrich_all_properties(top_10_from_rentcast)

#     # Print the final results
#     print_enriched_results(enriched_properties)

#     # enriched_properties is the final output of your pipeline.
#     # Each property dict now has a "nearby" key structured like:
#     #
#     # {
#     #   "nearby": {
#     #     "hospitals": [ { "name": "...", "distance": 450, "latitude": ..., "longitude": ... }, ... ],
#     #     "schools":   [ { "name": "...", "distance": 300, ... }, ... ],
#     #     "transit":   [ { "name": "...", "distance": 120, ... }, ... ],
#     #   }
#     # }