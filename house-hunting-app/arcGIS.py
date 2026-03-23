import requests
import streamlit as st

api_key = st.secrets["ARCGIS_API_KEY"]
ARCGIS_PLACES_URL = "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/places/near-point"
ARCGIS_CATEGORIES_URL = "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/categories"
SEARCH_RADIUS = 2000
categories = ["Emergency Service", "Transit Stop", "Grocery Store", "Restaurants"]

def get_category_id(filter_term: str) -> str | None:
    params = {
        "filter": filter_term,
        "f":      "json",
        "token":  api_key,
    }
    response = requests.get(ARCGIS_CATEGORIES_URL, params=params)
    if not response.ok:
        return None
    results = response.json().get("categories", [])
    return results[0]["categoryId"] if results else None

def fetch_nearby_pois(latitude: float, longitude: float, category_id: str, label: str) -> list:
    params = {
        "y":           latitude,
        "x":           longitude,
        "radius":      SEARCH_RADIUS,
        "categoryIds": category_id,
        "f":           "json",
        "token":       api_key,
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
        })

    # Sort by distance ascending
    simplified.sort(key=lambda p: p["distance"])

    return simplified

def enrich_property(property: dict, requirements: dict) -> dict:
    property["nearby"] = []
    lat = property.get("latitude")
    lng = property.get("longitude")
    
    amenity_data = categories + requirements.get("amenities", [])
    for category in amenity_data:
        category_id = get_category_id(category)
        if category_id:
            pois = fetch_nearby_pois(lat, lng, category_id, category)
            property["nearby"] = property["nearby"] + pois
        else:
            print(f"  ⚠️  Could not find category ID for '{category}'")
    return property


def enrich_all_properties(top_properties: list, requirements: dict) -> list:
    enriched = []
    for i, property in enumerate(top_properties, start=1):
        enriched_property = enrich_property(property, requirements)
        enriched.append(enriched_property)

    return enriched