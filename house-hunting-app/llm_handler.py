# import google.generativeai as genai  # deprecated
from google import genai
from feature_extractor import extract_features, format_requirements  # Step 2: added

def get_rental_response(user_prompt: str, api_key: str) -> str:
    # genai.configure(api_key=api_key)                        # deprecated
    # model = genai.GenerativeModel("gemini-3-flash-preview") # deprecated
    system_context = """You are a helpful rental house assistant. 
    Help users find rental houses based on their preferences like 
    location, budget, number of rooms, amenities, etc."""

    full_prompt = f"{system_context}\n\nUser Query: {user_prompt}"

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="models/gemini-3-flash-preview", contents=full_prompt)
    # response = model.generate_content(full_prompt)  # deprecated

    # Step 2: Extract structured features from the user query
    try:
        requirements = extract_features(user_prompt, api_key)
        summary = format_requirements(requirements)
        return f"{summary}\n\n---\n\n{response.text}"
    except Exception:
        # If extraction fails, fall back to original LLM response
        return response.text