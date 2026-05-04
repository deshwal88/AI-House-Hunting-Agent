import pathlib
import types
import streamlit as st

APP_PATH = pathlib.Path(__file__).parent / "app.py"


def create_sample_properties(count=9):
    properties = []
    for idx in range(1, count + 1):
        properties.append({
            "property_id": f"prop-{idx}",
            "address": f"{100 + idx} Example Lane",
            "city": "Austin",
            "state": "TX",
            "zip_code": "78701",
            "rent": 1200 + idx * 150,
            "bedrooms": 1 + (idx % 4),
            "bathrooms": 1 + ((idx + 1) % 3) * 0.5,
            "property_type": "Apartment" if idx % 2 == 0 else "Condo",
            "sqft": 650 + idx * 50,
            "final_score": 0.9 - idx * 0.05,
            "hard_score": 0.7 + idx * 0.02,
            "soft_score": 0.6 + idx * 0.01,
            "rationale": f"This property is ranked {idx} because it fits the sample criteria.",
            "images": [],
        })
    return properties


def load_app_module():
    source = APP_PATH.read_text()
    marker = "_PAGES[st.session_state.page]()"
    marker_index = source.rfind(marker)
    if marker_index == -1:
        raise RuntimeError("Could not find app router call in app.py")

    executable_source = source[:marker_index]
    module = types.ModuleType("app")
    module.__file__ = str(APP_PATH)
    module.__package__ = ""
    module.__dict__["__name__"] = "app"
    exec(compile(executable_source, str(APP_PATH), "exec"), module.__dict__)
    return module


def main():
    sample_props = create_sample_properties()

    st.session_state.page = 7
    st.session_state.session_id = "sample-session"
    st.session_state.search_started = True
    st.session_state.awaiting_feedback = True
    st.session_state.ranked_list = sample_props
    st.session_state.iteration = 1
    st.session_state.converged = False
    st.session_state.preference_profile = ""
    st.session_state.iteration_at_feedback = -1
    st.session_state.search_error = None

    app = load_app_module()
    app.page_results()


if __name__ == "__main__":
    main()
