import google.generativeai as genai

def get_rental_response(user_prompt: str, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")  # Free tier model

    system_context = """You are a helpful rental house assistant. 
    Help users find rental houses based on their preferences like 
    location, budget, number of rooms, amenities, etc."""

    full_prompt = f"{system_context}\n\nUser Query: {user_prompt}"
    
    response = model.generate_content(full_prompt)
    return response.text