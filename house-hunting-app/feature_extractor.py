"""
feature_extractor.py
--------------------
Step 2 of the AI House-Hunting Agent pipeline.

Converts a natural language rental query into a structured RentalRequirements
object using Gemini + Pydantic validation.

Example input : "2 bed apartment under $2400 near Rutgers with parking and grocery stores nearby"
Example output: RentalRequirements(bedrooms=2, budget=2400, city="New Brunswick", state="NJ", ...)
"""

import json
import re
from typing import Optional
from pydantic import BaseModel, Field
from google import genai


# ---------------------------------------------------------------------------
# Pydantic schema — this is the structured output we want from every query
# ---------------------------------------------------------------------------

class RentalRequirements(BaseModel):
    """Structured rental preferences extracted from a natural language query."""

    # Location
    street: Optional[str] = Field(None, description="Street address if mentioned (e.g. '220 Lincoln Avenue')")
    city: Optional[str] = Field(None, description="City name (e.g. 'Austin', 'New Brunswick')")
    state: Optional[str] = Field(None, description="2-letter state abbreviation (e.g. 'TX', 'NJ')")
    zip_code: Optional[str] = Field(None, description="ZIP code if mentioned (e.g. '07302')")
    near_location: Optional[str] = Field(None, description="Any landmark, store, or area the user wants to be near (e.g. 'Rutgers University NJ', 'Walmart Austin TX', 'downtown Austin')")
    radius: int = Field(3, description="Search radius in miles. Default 3 if not mentioned.")

    # Property specs
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms requested (e.g. 1, 2, 3)")
    bathrooms: Optional[int] = Field(None, description="Number of bathrooms requested")
    property_type: Optional[str] = Field(None, description="One of: Apartment, Condo, Townhouse, Single Family, Multi-Family, Manufactured, or null")
    square_footage: Optional[float] = Field(None, description="Square footage value if mentioned")
    square_footage_operator: Optional[str] = Field(None, description="'min' or 'max' — min means at least, max means at most")
    lot_size: Optional[float] = Field(None, description="Lot size value if mentioned")
    lot_size_operator: Optional[str] = Field(None, description="'min' or 'max'")
    year_built: Optional[int] = Field(None, description="Year built if mentioned")
    year_built_operator: Optional[str] = Field(None, description="'min' or 'max' — min means built after, max means built before")

    # Budget
    budget: Optional[float] = Field(None, description="Maximum monthly rent in USD")
    min_budget: Optional[float] = Field(None, description="Minimum monthly rent in USD (if specified)")

    # Extra — not sent to RentCast, passed downstream
    parking_required: Optional[bool] = Field(False, description="True if parking is explicitly required")
    pet_friendly: Optional[bool] = Field(False, description="True if pet-friendly is required")
    amenities: list[str] = Field(default_factory=list, description="Names of amenities or places mentioned (e.g. ['Walmart', 'grocery store', 'gym'])")


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are a data extraction assistant for a rental search engine.

Extract structured rental requirements from the user's query below.
Respond ONLY with a valid JSON object matching this exact schema — no markdown, no explanation, no extra keys:

{{
  "street": <string or null>,
  "city": <string or null>,
  "state": <2-letter abbreviation or null>,
  "zip_code": <string or null>,
  "near_location": <string or null>,
  "radius": <int, default 3 if not mentioned>,
  "bedrooms": <int or null>,
  "bathrooms": <int or null>,
  "property_type": <"Apartment"|"Condo"|"Townhouse"|"Single Family"|"Multi-Family"|"Manufactured"|null>,
  "square_footage": <float or null>,
  "square_footage_operator": <"min"|"max"|null>,
  "lot_size": <float or null>,
  "lot_size_operator": <"min"|"max"|null>,
  "year_built": <int or null>,
  "year_built_operator": <"min"|"max"|null>,
  "budget": <float or null>,
  "min_budget": <float or null>,
  "parking_required": <true or false>,
  "pet_friendly": <true or false>,
  "amenities": [<string>, ...]
}}

Rules:
- If a field is not mentioned, use null (or 3 for radius, [] for amenities).
- For budget: extract the number only (no $ signs). "$2,400" → 2400.0. This is the MAXIMUM budget.
- For location: extract street, city, state, zip_code separately. Infer state if possible (e.g. "near Rutgers" → city: "New Brunswick", state: "NJ")
- For near_location: if the user says "near X" and X is a landmark, store, university, or area (NOT a street address) → set near_location to "X, City State". If the user gives a specific street address → set street instead. near_location and street should NOT both be set.
- If user says "near Walmart" → set near_location: "Walmart, City State" AND add "Walmart" to amenities
- If user mentions a generic amenity like "grocery stores nearby" → add "grocery store" to amenities only (not near_location)
- For amenities: capture names of specific places or generic amenity types (e.g. ['Walmart', 'grocery store', 'gym', 'laundry', 'park'])
- For studios: set bedrooms to 0, unless the query explicitly says a number like "1 bed studio" — use that number instead
- parking_required and pet_friendly must always be true or false (never null)
- For property_type: map to the closest of Apartment, Condo, Townhouse, Single Family, Multi-Family, Manufactured. "house" → "Single Family", "studio" → "Apartment"
- For square_footage, lot_size, year_built operators:
    "at least" / "minimum" / "more than" / "over" / "after" → "min"
    "at most" / "maximum" / "less than" / "under" / "no more than" / "before" → "max"
    "around" / "approximately" → "min"
    e.g. "at least 800 sqft" → square_footage: 800, square_footage_operator: "min"
    e.g. "built after 2010"  → year_built: 2010, year_built_operator: "min"
    e.g. "under 1500 sqft"   → square_footage: 1500, square_footage_operator: "max"
- For radius: if user says "within X miles" → radius: X. If user says "within X km" → convert to miles (1 km = 0.621 miles). Default 3 if not mentioned.

User query: "{query}"
"""


def extract_features(user_query: str, api_key: str) -> RentalRequirements:
    """
    Parse a natural language rental query into a RentalRequirements object.

    Args:
        user_query: Raw text from the user, e.g. "2 bed under $2400 near Rutgers"
        api_key:    Gemini API key

    Returns:
        RentalRequirements (Pydantic model, all fields validated)

    Raises:
        ValueError: If Gemini returns unparseable JSON after retries
    """
    prompt = EXTRACTION_PROMPT.format(query=user_query)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="models/gemini-3-flash-preview", contents=prompt)
    raw = response.text.strip()

    # Strip markdown code fences if Gemini wraps the JSON anyway
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON:\n{raw}\n\nError: {e}")

    return RentalRequirements(**data)
