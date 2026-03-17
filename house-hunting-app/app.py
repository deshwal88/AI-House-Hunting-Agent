import streamlit as st
from llm_handler import get_rental_response

# Page config
st.set_page_config(page_title="Rental House Finder", page_icon="🏠")
st.title("🏠 Rental House Finder")
st.caption("Describe what you're looking for and I'll help you find it!")

# Load API key from Streamlit secrets
api_key = st.secrets["GEMINI_API_KEY"]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if prompt := st.chat_input("E.g. 2BHK in Mumbai under ₹20,000/month..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your Python function → Gemini
    with st.chat_message("assistant"):
        with st.spinner("Finding rentals..."):
            response = get_rental_response(prompt, api_key)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})