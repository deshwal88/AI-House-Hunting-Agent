from google import genai
from feature_extractor import extract_features
from rentcast_search import find_top_properties
from arcGIS import enrich_all_properties
import streamlit as st
import json

def get_rental_response(user_prompt: str, api_key: str) -> str:
    try:
        requirements = extract_features(user_prompt, api_key).model_dump()
        top_properties = find_top_properties(requirements)
        enriched_properties = enrich_all_properties(top_properties, requirements)

        # Step 3: Apply soft scoring based on user personality and nearby amenities
        final_properties = soft_score_properties(user_prompt, requirements, enriched_properties, api_key)

        # Display the properties
        display_property_details(final_properties)

        return f"Found {len(final_properties)} top rental properties based on your requirements. Details displayed above."

    except Exception as e:
        print(f"Error in rental response processing: {e}")
        return "Sorry, there was an error processing your request. Please try again."


def soft_score_properties(user_prompt: str, requirements, top_properties: list, api_key: str) -> list:
    client = genai.Client(api_key=api_key)

    # Extract user personality/background from the original prompt
    personality_prompt = f"""
    Based on this user's rental search query, create a detailed personality profile of the person looking for a rental property.
    Consider their lifestyle, priorities, and preferences from their query.

    User Query: "{user_prompt}"

    Create a personality profile that includes:
    - Age range and life stage
    - Lifestyle (urban vs suburban, active vs relaxed, etc.)
    - Priorities (convenience, safety, community, etc.)
    - Deal-breakers or strong preferences (Pet firendly, parking, etc.)
    - Daily routine patterns that would affect amenity preferences

    Respond with a detailed personality description.
    """

    try:
        personality_response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=personality_prompt
        )
        user_personality = personality_response.text
    except Exception as e:
        user_personality = "A typical renter looking for a comfortable and convenient living space."

    scored_properties = []

    for i, property in enumerate(top_properties):
        # Create scoring prompt for this specific property
        property_details = {
            "address": property.get("addressLine1", "Unknown"),
            "price": property.get("price", 0),
            "bedrooms": property.get("bedrooms", 0),
            "bathrooms": property.get("bathrooms", 0),
            "type": property.get("propertyType", "Unknown"),
            "hard_score": property.get("_score", 0)
        }

        # Extract nearby amenities
        nearby_amenities = property.get("nearby", [])
        amenities_text = ""
        for place in nearby_amenities:
            amenities_text += f"{place.get('name', 'Unknown')} ({place.get('distance', 0)}m), "

        # User preferences for amenities
        user_amenities = requirements.amenities if hasattr(requirements, 'amenities') else []
        user_amenities_text = ", ".join(user_amenities) if user_amenities else "No specific amenities mentioned"

        scoring_prompt = f"""
        You are acting as a potential renter with this personality and background:

        {user_personality}

        You are considering renting this property:

        Property Details:
        - Address: {property_details['address']}
        - Rent: ${property_details['price']:,}/month
        - Bedrooms: {property_details['bedrooms']}
        - Bathrooms: {property_details['bathrooms']}
        - Type: {property_details['type']}
        - Current Match Score: {property_details['hard_score']}/100

        Nearby Amenities (within 2000m):
        {amenities_text}

        Your stated amenity preferences: {user_amenities_text}

        Rate this property from 0-100 based on how well it fits your lifestyle and preferences, considering:

        1. **Distance to Important Amenities**: How convenient are the nearby places you care about?
           - If you mentioned specific amenities (like grocery stores, gyms, transit), give high scores if they're close
           - If you didn't mention specific amenities, consider general convenience (hospitals, schools, transit)

        2. **Lifestyle Fit**: Does this location support your daily routine and preferences?
           - Urban vs suburban feel
           - Walkability and convenience
           - Community and safety factors

        3. **Overall Appeal**: How excited would you be to live here based on the amenities?

        Scoring Guidelines:
        - 90-100: Perfect fit, all your amenity needs are met with excellent convenience
        - 70-89: Very good fit, most important amenities are accessible
        - 50-69: Decent fit, adequate amenities but some compromises needed
        - 30-49: Poor fit, important amenities are inconvenient or missing
        - 0-29: Very poor fit, major amenity gaps that would affect daily life

        Respond with ONLY a JSON object in this exact format:
        {{
            "soft_score": <number 0-100>,
            "reasoning": "<brief explanation of the score>",
            "key_factors": ["factor1", "factor2", "factor3"]
        }}
        """

        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=scoring_prompt
            )

            # Parse the JSON response
            response_text = response.text.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            score_data = json.loads(response_text)
            soft_score = score_data.get("soft_score", 50)  # Default to 50 if parsing fails
            reasoning = score_data.get("reasoning", "No reasoning provided")
            key_factors = score_data.get("key_factors", [])

        except Exception as e:
            print(f"Error getting soft score for property {i+1}: {e}")
            soft_score = 50  # Neutral score if API fails
            reasoning = "Unable to assess due to technical issues"
            key_factors = ["Technical assessment error"]

        # Calculate final combined score (weighted average)
        hard_score = property.get("_score", 0)
        final_score = round((hard_score * 0.7) + (soft_score * 0.3))  # 70% hard, 30% soft

        # Add soft scoring data to property
        property["_soft_score"] = soft_score
        property["_soft_reasoning"] = reasoning
        property["_soft_factors"] = key_factors
        property["_final_score"] = final_score

        scored_properties.append(property)

    # Sort by final score (descending)
    scored_properties.sort(key=lambda x: x.get("_final_score", 0), reverse=True)

    return scored_properties


def display_property_details(properties: list):
    """
    Display the top 10 properties with a summary table and detailed views.
    Shows final scores, property details, amenities distances, and score breakdowns.
    """
    if not properties:
        st.warning("No properties found to display.")
        return

    # Take top 10 if more than 10
    top_properties = properties[:10]

    st.header("🏠 Top Rental Properties")

    # Summary Table
    st.subheader("Property Summary")

    table_data = []
    for prop in top_properties:
        table_data.append({
            "Address": prop.get("addressLine1", "Unknown"),
            "Bedrooms": prop.get("bedrooms", "N/A"),
            "Bathrooms": prop.get("bathrooms", "N/A"),
            "Area (sq ft)": prop.get("squareFootage", "N/A"),
            "Price": f"${prop.get('price', 0):,}",
            "Type": prop.get("propertyType", "Unknown"),
            "Final Score": prop.get("_final_score", 0)
        })

    st.dataframe(table_data, use_container_width=True)

    # Detailed Property Views
    st.subheader("Property Details")

    for i, prop in enumerate(top_properties, 1):
        with st.expander(f"#{i} - {prop.get('addressLine1', 'Unknown')} (Score: {prop.get('_final_score', 0)})"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Property Details:**")
                st.write(f"**Address:** {prop.get('addressLine1', 'Unknown')}")
                st.write(f"**Price:** ${prop.get('price', 0):,}/month")
                st.write(f"**Bedrooms:** {prop.get('bedrooms', 'N/A')}")
                st.write(f"**Bathrooms:** {prop.get('bathrooms', 'N/A')}")
                st.write(f"**Area:** {prop.get('squareFootage', 'N/A')} sq ft")
                st.write(f"**Type:** {prop.get('propertyType', 'Unknown')}")

            with col2:
                st.markdown("**Score Breakdown:**")
                hard_score = prop.get("_score", 0)
                soft_score = prop.get("_soft_score", 50)
                final_score = prop.get("_final_score", 0)

                st.write(f"**Hard Score:** {hard_score}/100 (70% weight)")
                st.write(f"**Soft Score:** {soft_score}/100 (30% weight)")
                st.write(f"**Final Score:** {final_score}/100")

                # Hard score breakdown
                if "_score_breakdown" in prop:
                    breakdown = prop["_score_breakdown"]
                    st.markdown("**Hard Score Details:**")
                    for category, score in breakdown.items():
                        st.write(f"- {category.replace('_', ' ').title()}: {score} pts")

                # Soft score reasoning
                if "_soft_reasoning" in prop:
                    st.markdown("**Soft Score Reasoning:**")
                    st.write(prop["_soft_reasoning"])

                if "_soft_factors" in prop and prop["_soft_factors"]:
                    st.markdown("**Key Factors:**")
                    for factor in prop["_soft_factors"]:
                        st.write(f"- {factor}")

            # Amenities Section
            nearby = prop.get("nearby", [])
            st.markdown("**Nearby Amenities (within 2000m):**")
            if nearby:
                amenity_rows = []
                for amenity in nearby:
                    distance_m = amenity.get("distance", 0)
                    amenity_rows.append({
                        "Name": amenity.get("name", "Unknown"),
                        "Distance (m)": distance_m,
                        "Distance (km)": f"{distance_m / 1000:.2f}"
                    })
                st.table(amenity_rows)
            else:
                st.write("No nearby amenities data available.")