"""
feature_extractor.py
--------------------
Step 2 of the AI House-Hunting Agent pipeline.

Converts a natural language rental query into a structured RentalRequirements
object using Gemini + Pydantic validation.

Example input : "2 bed apartment under $2400 near Rutgers with parking and grocery stores nearby"
Example output: RentalRequirements(bedrooms=2, max_budget=2400, location="Rutgers, NJ", ...)
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

    # Core filters
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms requested (e.g. 1, 2, 3)")
    bathrooms: Optional[int] = Field(None, description="Number of bathrooms requested")
    max_budget: Optional[float] = Field(None, description="Maximum monthly rent in USD")
    min_budget: Optional[float] = Field(None, description="Minimum monthly rent in USD (if specified)")

    # Location
    location: Optional[str] = Field(None, description="Target city, neighborhood, or landmark (e.g. 'New Brunswick, NJ')")
    max_commute_minutes: Optional[int] = Field(None, description="Maximum acceptable commute time in minutes")
    commute_destination: Optional[str] = Field(None, description="Where the user commutes TO (e.g. 'Rutgers University')")

    # Amenities
    parking_required: Optional[bool] = Field(False, description="True if parking is explicitly required")
    pet_friendly: Optional[bool] = Field(False, description="True if pet-friendly is required")
    amenities: list[str] = Field(default_factory=list, description="Other amenities mentioned (e.g. ['grocery store', 'gym', 'laundry'])")

    # Property type
    property_type: Optional[str] = Field(None, description="Type of property: apartment, house, condo, studio, etc.")


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are a data extraction assistant for a rental search engine.

Extract structured rental requirements from the user's query below.
Respond ONLY with a valid JSON object matching this exact schema — no markdown, no explanation, no extra keys:

{{
  "bedrooms": <int or null>,
  "bathrooms": <int or null>,
  "max_budget": <float or null>,
  "min_budget": <float or null>,
  "location": <string or null>,
  "max_commute_minutes": <int or null>,
  "commute_destination": <string or null>,
  "parking_required": <true or false>,
  "pet_friendly": <true or false>,
  "amenities": [<string>, ...],
  "property_type": <string or null>
}}

Rules:
- If a field is not mentioned, use null (or [] for amenities).
- For budget: extract the number only (no $ signs). "$2,400" → 2400.0
- For commute: "under 20 min" → max_commute_minutes: 20
- For location: infer the full place name if possible (e.g. "near Rutgers" → "New Brunswick, NJ")
- For amenities: capture things like grocery stores, gym, laundry, transit, parks, etc.
- For studios: set bedrooms to 0 (a studio has no separate bedroom), unless the query explicitly mentions a number of bedrooms like "1 bed studio", in which case use that number
- parking_required and pet_friendly must always be true or false (never null)

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


def format_requirements(req: RentalRequirements) -> str:
    """Human-readable summary of extracted requirements (useful for debugging / UI display)."""
    lines = ["📋 **Extracted Requirements**"]

    if req.bedrooms is not None:
        lines.append(f"- 🛏  Bedrooms: {req.bedrooms}")
    if req.bathrooms is not None:
        lines.append(f"- 🚿 Bathrooms: {req.bathrooms}")
    if req.property_type:
        lines.append(f"- 🏠 Property type: {req.property_type}")
    if req.max_budget is not None:
        budget_str = f"up to ${req.max_budget:,.0f}/mo"
        if req.min_budget is not None:
            budget_str = f"${req.min_budget:,.0f} – ${req.max_budget:,.0f}/mo"
        lines.append(f"- 💰 Budget: {budget_str}")
    if req.location:
        lines.append(f"- 📍 Location: {req.location}")
    if req.commute_destination:
        commute = f"to {req.commute_destination}"
        if req.max_commute_minutes:
            commute += f" within {req.max_commute_minutes} min"
        lines.append(f"- 🚗 Commute: {commute}")
    if req.parking_required:
        lines.append("- 🅿️  Parking: required")
    if req.pet_friendly:
        lines.append("- 🐾 Pet-friendly: required")
    if req.amenities:
        lines.append(f"- ✨ Amenities: {', '.join(req.amenities)}")

    return "\n".join(lines)