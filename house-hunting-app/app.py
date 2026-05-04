"""
AI House Finder — Streamlit UI

Multi-page wizard that collects user requirements (pages 0–6),
then delegates to the FastAPI/LangGraph backend (page 7 — results).

The results page:
  1. POSTs requirements to /start
  2. Polls /state/{session_id} until ranked_list is ready
  3. Shows property cards with score breakdown + images
  4. Offers drag-and-drop reordering (streamlit-sortables) for feedback
  5. POSTs reordering to /feedback and refreshes results (up to 5 loops)
  6. Shows convergence message when the agent has learned the user's taste

Run:
  Terminal 1: uvicorn backend.main:api --port 8000 --reload
  Terminal 2: streamlit run app.py
"""

import base64
import json
import os
import random
import time
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as stc

try:
    from streamlit_sortables import sort_items
    SORTABLES_AVAILABLE = True
except ImportError:
    SORTABLES_AVAILABLE = False

# ── Sortable CSS injector ──────────────────────────────────────────────────────
# streamlit_sortables renders in a same-origin iframe, so a sibling stc.html()
# frame can walk window.parent's iframes and patch their document's <head>.
_SORTABLE_CSS = """\
.sortable-container-body {
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
}


.sortable-item {
    color: #5c5959;
    background-color: white;
    cursor: grab;
    height: 41vh;
    margin: 5px;
    padding-bottom: 10px;
    padding-top: 160px;
    width: calc(20vw - 20px);
    background-size: contain;
    background-image: url(https://static.vecteezy.com/system/resources/thumbnails/047/022/946/small_2x/modern-two-story-house-with-stone-accents-and-garage-at-dusk-free-photo.jpeg);
    background-repeat: no-repeat;
    border-radius: 6px;
    overflow: clip;
    white-space: pre-line !important;
    font-size: 12px;
}
.sortable-item:active { cursor: grabbing; opacity: 0.85; }
.sortable-item:hover { 
background-image: none;
color: black;
opacity: 0.9;
margin: 0px;
padding: 0px;
background-color: #c1d4d9 !important; 
width: calc(20vw - 20px);
height: 40vh;
padding: 10px 0 0 0;
transition: none;
display: flex !important;
align-items: center !important;
}
"""


# ── Local image helpers ────────────────────────────────────────────────────────
_IMAGE_FOLDER: dict[str, str] = {
    "apartment": "Apartment Images",
    "condo":     "Apartment Images",
    "townhouse": "Townhouse images",
}

_APP_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def _load_local_images(folder_key: str) -> list[str]:
    """Read all images from a local folder and return as base64 data-URLs (cached)."""
    folder = os.path.join(_APP_DIR, "House Images", folder_key)
    urls: list[str] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".webp", ".jpg", ".jpeg", ".png")):
            continue
        with open(os.path.join(folder, fname), "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        ext  = fname.rsplit(".", 1)[-1].lower()
        mime = "image/webp" if ext == "webp" else f"image/{ext}"
        urls.append(f"data:{mime};base64,{data}")
    return urls


def _folder_for_type(property_type: str) -> str:
    return _IMAGE_FOLDER.get(property_type.lower(), "Apartment Images")


def _inject_sortable_css() -> None:
    """Inject custom CSS into the streamlit_sortables iframe via sibling-frame JS."""
    css_js = json.dumps(_SORTABLE_CSS)   # safely escaped for embedding in JS
    stc.html(
        f"""<script>
(function() {{
    var css = {css_js};
    function inject() {{
        var frames = window.parent.document.querySelectorAll("iframe");
        var done = false;
        frames.forEach(function(f) {{
            try {{
                var d = f.contentDocument || f.contentWindow.document;
                if (!d || d.getElementById("_sc_css")) return;
                if (!d.querySelector(".sortable-item")) return;
                var el = d.createElement("style");
                el.id = "_sc_css";
                el.textContent = css;
                d.head.appendChild(el);
                done = true;
            }} catch(e) {{}}
        }});
        if (!done) setTimeout(inject, 200);
    }}
    inject();
    [300, 700, 1500].forEach(function(t) {{ setTimeout(inject, t); }});
}})();
</script>""",
        height=0,
    )

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI House Finder",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { max-width: 95vw; padding-top: 2rem; }

/* Welcome hero */
iframe.stCustomComponentV1{
    height: 80vh !important;
}
.hero { text-align: center; padding: 48px 0 28px; }
.hero-icon  { font-size: 72px; line-height: 1; }
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f97316 0%, #8b5cf6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 16px 0 12px;
}
.hero-sub {
    color: #64748b; font-size: 1.05rem;
    max-width: 460px; margin: 0 auto 36px; line-height: 1.65;
}

/* Feature strip */
.features { display: flex; justify-content: center; gap: 48px; margin-top: 36px; }
.fi { text-align: center; }
.fi-icon  { font-size: 26px; }
.fi-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; letter-spacing: 0.04em; }

/* Section labels */
.section-label {
    font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
    letter-spacing: 0.07em; color: #475569; margin-bottom: 4px;
}

/* Loading house animation */
.house-wrap {
    display: flex; flex-direction: column;
    align-items: center; padding: 48px 0 32px;
}
.house-scene {
    position: relative; width: 200px; height: 230px;
}
/* Ground */
.h-ground {
    position: absolute; bottom: 0; left: -10px; right: -10px; height: 18px;
    background: linear-gradient(#4ade80, #16a34a); border-radius: 6px;
    animation: scaleIn 0.5s ease-out forwards; transform: scaleX(0);
}
/* Foundation */
.h-base {
    position: absolute; bottom: 14px; left: 10px; right: 10px; height: 16px;
    background: #92400e; border-radius: 2px;
    animation: scaleIn 0.5s ease-out 0.3s forwards; transform: scaleX(0);
}
/* Walls */
.h-walls {
    position: absolute; bottom: 28px; left: 20px; right: 20px; height: 100px;
    background: #fef3c7; border: 3px solid #d97706; border-radius: 2px;
    animation: riseUp 0.7s ease-out 0.8s forwards;
    transform: scaleY(0); transform-origin: bottom;
}
/* Window left */
.h-win-l, .h-win-r {
    position: absolute; top: 20px; width: 26px; height: 26px;
    background: #bae6fd; border: 2px solid #38bdf8;
    animation: popIn 0.3s ease-out 2.7s forwards;
    transform: scale(0); opacity: 0;
}
.h-win-l { left: 12px; }
.h-win-r { right: 12px; }
/* Window panes */
.h-win-l::before, .h-win-l::after,
.h-win-r::before, .h-win-r::after {
    content: ""; position: absolute; background: #7dd3fc;
}
.h-win-l::before, .h-win-r::before {
    top: 50%; left: 0; width: 100%; height: 2px; transform: translateY(-50%);
}
.h-win-l::after, .h-win-r::after {
    left: 50%; top: 0; width: 2px; height: 100%; transform: translateX(-50%);
}
/* Door */
.h-door {
    position: absolute; bottom: 0; left: 50%; transform: translateX(-50%) scaleY(0);
    width: 30px; height: 48px;
    background: #92400e; border-radius: 4px 4px 0 0;
    transform-origin: bottom;
    animation: riseUp 0.4s ease-out 2.4s forwards;
}
.h-door::after {
    content: ""; position: absolute;
    width: 5px; height: 5px; border-radius: 50%;
    background: #fbbf24; right: 5px; top: 50%;
}
/* Roof */
.h-roof {
    position: absolute; bottom: 126px; left: 10px; right: 10px; height: 70px;
    background: #ea580c;
    clip-path: polygon(50% 0%, 0% 100%, 100% 100%);
    animation: dropIn 0.7s cubic-bezier(0.34,1.56,0.64,1) 1.5s forwards;
    transform: translateY(-70px); opacity: 0;
}
/* Chimney */
.h-chimney {
    position: absolute; bottom: 182px; right: 52px;
    width: 20px; height: 32px;
    background: #7f1d1d; border-radius: 2px 2px 0 0;
    animation: riseChimney 0.4s ease-out 2.2s forwards;
    transform: scaleY(0); transform-origin: bottom;
}
/* Smoke puffs */
.h-smoke { position: absolute; border-radius: 50%; opacity: 0; }
.h-smoke-1 {
    bottom: 214px; right: 56px; width: 10px; height: 10px;
    background: #d1d5db;
    animation: smokeUp 2s ease-out 3.0s infinite;
}
.h-smoke-2 {
    bottom: 214px; right: 50px; width: 8px; height: 8px;
    background: #e2e8f0;
    animation: smokeUp 2s ease-out 3.5s infinite;
}
/* Tree */
.h-tree {
    position: absolute; bottom: 14px; left: -28px;
    animation: popIn 0.5s ease-out 3.2s forwards;
    transform: scale(0); opacity: 0;
    font-size: 36px; line-height: 1;
}

/* Keyframes */
@keyframes scaleIn    { to { transform: scaleX(1); } }
@keyframes riseUp     { to { transform: scaleY(1); } }
@keyframes dropIn     { to { transform: translateY(0); opacity: 1; } }
@keyframes riseChimney{ to { transform: scaleY(1); } }
@keyframes popIn      { to { transform: scale(1); opacity: 1; } }
@keyframes smokeUp {
    0%   { transform: translateY(0) scale(0.6); opacity: 0.8; }
    100% { transform: translateY(-48px) scale(2); opacity: 0; }
}

/* Loading dots */
.dots { display: flex; justify-content: center; gap: 8px; margin: 16px 0 8px; }
.dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg, #f97316, #8b5cf6);
    animation: bounce 1.4s ease-in-out infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40%         { transform: translateY(-14px); }
}

/* Loading text */
.load-title {
    text-align: center; font-size: 1.2rem; font-weight: 700;
    background: linear-gradient(135deg, #f97316, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.load-sub {
    text-align: center; color: #94a3b8; font-size: 0.85rem;
}

/* Score pill */
.pill {
    display: inline-block; color: #fff; border-radius: 20px;
    padding: 3px 14px; font-weight: 700; font-size: 1rem;
}

/* Preference profile card */
.pref-card {
    background: linear-gradient(135deg, #fff7ed, #faf5ff);
    border: 1px solid #fed7aa; border-radius: 12px;
    padding: 14px 18px; margin-bottom: 20px;
    font-size: 0.92rem; line-height: 1.6; color: #44403c;
}
.pref-label {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.07em; color: #f97316; margin-bottom: 6px;
}

/* Drag-drop hint */
.drag-hint {
    text-align: center; color: #94a3b8; font-size: 0.85rem;
    margin: 0 0 12px; font-style: italic;
}

/* ── Property card grid ───────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] { overflow: visible !important; }
.prop-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 14px; padding: 6px; margin-bottom: 16px;
}
.prop-card {
    border: 1px solid #e2e8f0; border-radius: 10px;
    background: #fff; position: relative;
    transition: transform 0.18s ease, background-color 0.18s ease;
    cursor: default;
}
.prop-card:hover { transform: scale(1.1); background-color: #f1f5f9; z-index: 10; }
.prop-img  { width: 100%; height: 50%; display: block; border-radius: 9px 9px 0 0; 
            background-size: cover; background-position: center;
            }
.prop-placeholder {
    width: 100%; height: 160px;
    background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
    display: flex; align-items: center; justify-content: center;
    font-size: 42px; border-radius: 9px 9px 0 0;
}
.prop-rank {
    position: absolute; top: 8px; left: 8px;
    background: rgba(0,0,0,0.6); color: #fff;
    width: 22px; height: 22px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; z-index: 2;
}
.prop-body  { padding: 7px 10px 9px; font-size: 14px; line-height: 1.45; color: #1e293b; }
.prop-addr  { font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.prop-sub   { color: #64748b; margin-bottom: 3px; }
.prop-meta  { display: flex; justify-content: space-between; align-items: center; }
.prop-badge { font-size: 9px; font-weight: 700; color: #fff; border-radius: 10px; padding: 2px 7px; }
.prop-rat   {
    font-size: 12px; color: #64748b; font-style: italic; margin-top: 3px;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ─────────────────────────────────────────────────────
DEFAULTS: dict = {
    "page": 0,
    # Location
    "city": "", "state_abbr": "", "zip_code": "", "radius": 5,
    # Commute
    "has_commute": False, "commute_destination": "", "max_commute_minutes": 30,
    # Property
    "property_type": "Any", "bedrooms": "Any", "bathrooms": "Any",
    # Budget
    "min_budget": 800, "max_budget": 3000,
    # Preferences
    "pet_friendly": False, "parking_required": False, "amenities": [],
    # Backend session
    "session_id": None, "search_started": False,
    "ranked_list": [], "preference_profile": "", "iteration": 0,
    "converged": False, "convergence_reason": "", "awaiting_feedback": False,
    "search_error": None, "iteration_at_feedback": -1,
    "prop_images": {},
}

for _k, _v in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Constants ──────────────────────────────────────────────────────────────────
US_STATES = [
    "", "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
    "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT",
    "VT","VA","WA","WV","WI","WY",
]
PROPERTY_TYPES = ["Any","Apartment","Condo","Townhouse","Single Family","Multi-Family"]
BED_OPTIONS    = ["Any","Studio","1","2","3","4","5+"]
BATH_OPTIONS   = ["Any","1","1.5","2","2.5","3+"]
AMENITY_OPTIONS = [
    "Gym / Fitness Center","Swimming Pool","In-unit Laundry","Laundry in Building",
    "Dishwasher","Air Conditioning","Balcony / Patio","Grocery Store Nearby",
    "Public Transit Nearby","Park Nearby","Good Schools Nearby","Hospital Nearby",
]
TOTAL_STEPS = 5


# ── Helpers ────────────────────────────────────────────────────────────────────
def render_progress(step: int) -> None:
    st.progress(step / TOTAL_STEPS)
    st.caption(f"Step {step} of {TOTAL_STEPS}")


def nav(*, back: bool = True, next_label: str = "Continue →",
        next_disabled: bool = False) -> None:
    st.markdown("")
    cols = st.columns([1, 3, 1])
    with cols[0]:
        if back and st.button("← Back", use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with cols[2]:
        if st.button(next_label, type="primary",
                     use_container_width=True, disabled=next_disabled):
            st.session_state.page += 1
            st.rerun()


def _parse_beds(v: str | None) -> int | None:
    if v in ("Any", None): return None
    if v == "Studio":      return 0
    if v == "5+":          return 5
    return int(v)


def _parse_baths(v: str | None) -> float | None:
    if v in ("Any", None): return None
    if v == "3+":          return 3.0
    return float(v)


def _build_requirements() -> dict:
    return {
        "location": {
            "city":                st.session_state.city,
            "state":               st.session_state.state_abbr,
            "zip":                 st.session_state.zip_code or None,
            "search_radius_miles": st.session_state.radius,
        },
        "property": {
            "bedrooms":      _parse_beds(st.session_state.bedrooms),
            "bathrooms":     _parse_baths(st.session_state.bathrooms),
            "price_min":     st.session_state.min_budget,
            "price_max":     st.session_state.max_budget,
            "property_type": (
                None if st.session_state.property_type == "Any"
                else st.session_state.property_type.lower().replace(" ", "_")
            ),
        },
        "amenities": {
            "grocery_max_miles":       1.0,
            "school_max_miles":        2.0,
            "gym_max_miles":           1.5,
            "public_transit_max_miles": 0.5,
        },
        "preferences": {
            "pet_friendly":        st.session_state.pet_friendly,
            "parking":             st.session_state.parking_required,
            "amenities_requested": st.session_state.amenities,
            "commute_destination": (
                st.session_state.commute_destination
                if st.session_state.has_commute else None
            ),
            "max_commute_minutes": (
                st.session_state.max_commute_minutes
                if st.session_state.has_commute else None
            ),
        },
        "additional_comments": "",
    }


def _poll_backend() -> None:
    sid = st.session_state.session_id
    if not sid:
        return
    try:
        resp = requests.get(f"{BACKEND_URL}/state/{sid}", timeout=5)
        if resp.ok:
            data = resp.json()
            new_iteration = data.get("iteration", 0)
            # Only surface ranked_list once the backend has processed the latest
            # feedback (iteration advances in update_weights).  Without this gate
            # the very first poll after submission returns the pre-feedback list
            # because agent_states[sid] still holds it while phase-2 is running.
            if new_iteration > st.session_state.get("iteration_at_feedback", -1):
                st.session_state.ranked_list = data.get("ranked_list", [])
            st.session_state.preference_profile = data.get("preference_profile", "")
            st.session_state.iteration          = new_iteration
            st.session_state.converged          = data.get("converged", False)
            st.session_state.convergence_reason = data.get("convergence_reason", "")
            st.session_state.awaiting_feedback  = data.get("awaiting_feedback", False)
    except Exception:
        pass   # backend not yet ready — keep polling


# ── Page 0 — Welcome ───────────────────────────────────────────────────────────
def page_welcome() -> None:
    st.markdown("""
    <div class="hero">
        <div class="hero-icon">🏠</div>
        <div class="hero-title">AI House Finder</div>
        <div class="hero-sub">
            Answer a few quick questions and our AI agent will find,
            score, and <em>learn</em> your perfect rental — adapting
            to your feedback in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("Get Started →", type="primary", use_container_width=True):
            st.session_state.page = 1
            st.rerun()

    st.markdown("""
    <div class="features">
        <div class="fi"><div class="fi-icon">🔍</div><div class="fi-label">Smart Search</div></div>
        <div class="fi"><div class="fi-icon">🧠</div><div class="fi-label">LLM Scoring</div></div>
        <div class="fi"><div class="fi-icon">🔄</div><div class="fi-label">Learns From You</div></div>
        <div class="fi"><div class="fi-icon">🗺️</div><div class="fi-label">Local Insights</div></div>
    </div>
    """, unsafe_allow_html=True)


# ── Page 1 — Location ──────────────────────────────────────────────────────────
def page_location() -> None:
    render_progress(1)
    st.markdown("## 📍 Where do you want to live?")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.session_state.city = st.text_input(
            "City", value=st.session_state.city,
            placeholder="e.g. Austin, Chicago, New Brunswick")
    with c2:
        idx = US_STATES.index(st.session_state.state_abbr) \
              if st.session_state.state_abbr in US_STATES else 0
        st.session_state.state_abbr = st.selectbox(
            "State", US_STATES, index=idx,
            format_func=lambda x: x if x else "Select state…")

    st.session_state.zip_code = st.text_input(
        "ZIP Code (optional — overrides city/state for precision)",
        value=st.session_state.zip_code, placeholder="e.g. 78701")

    st.session_state.radius = st.slider(
        "Search radius", 1, 25, st.session_state.radius, 1, format="%d miles")

    ok = bool(st.session_state.city and st.session_state.state_abbr) \
         or bool(st.session_state.zip_code)
    nav(back=False, next_label="Next: Commute →", next_disabled=not ok)


# ── Page 2 — Commute ───────────────────────────────────────────────────────────
def page_commute() -> None:
    render_progress(2)
    st.markdown("## 🚗 Do you have a regular commute?")
    st.caption("We'll use this to prioritise homes that keep your journey manageable.")

    st.session_state.has_commute = st.toggle(
        "Yes, I commute to a regular destination",
        value=st.session_state.has_commute)

    if st.session_state.has_commute:
        st.markdown("---")
        st.session_state.commute_destination = st.text_input(
            "Where do you commute to?",
            value=st.session_state.commute_destination,
            placeholder="e.g. Rutgers University · Google NYC · 100 Wall St, New York")
        st.session_state.max_commute_minutes = st.slider(
            "Maximum commute time", 5, 120,
            st.session_state.max_commute_minutes, 5, format="%d min")

    nav(next_label="Next: Property →")


# ── Page 3 — Property ──────────────────────────────────────────────────────────
def page_property() -> None:
    render_progress(3)
    st.markdown("## 🏡 What kind of place?")

    st.markdown('<div class="section-label">Property type</div>', unsafe_allow_html=True)
    pi = PROPERTY_TYPES.index(st.session_state.property_type) \
         if st.session_state.property_type in PROPERTY_TYPES else 0
    st.session_state.property_type = st.radio(
        "Property type", PROPERTY_TYPES, index=pi,
        horizontal=True, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Bedrooms</div>', unsafe_allow_html=True)
    bi = BED_OPTIONS.index(st.session_state.bedrooms) \
         if st.session_state.bedrooms in BED_OPTIONS else 0
    st.session_state.bedrooms = st.radio(
        "Bedrooms", BED_OPTIONS, index=bi,
        horizontal=True, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<div class="section-label">Bathrooms</div>', unsafe_allow_html=True)
    bai = BATH_OPTIONS.index(st.session_state.bathrooms) \
          if st.session_state.bathrooms in BATH_OPTIONS else 0
    st.session_state.bathrooms = st.radio(
        "Bathrooms", BATH_OPTIONS, index=bai,
        horizontal=True, label_visibility="collapsed")

    nav(next_label="Next: Budget →")


# ── Page 4 — Budget ────────────────────────────────────────────────────────────
def page_budget() -> None:
    render_progress(4)
    st.markdown("## 💰 What's your monthly budget?")
    st.caption("Drag both handles to set your comfortable rent range.")

    rng = st.slider("Monthly rent", 300, 10_000,
                    (st.session_state.min_budget, st.session_state.max_budget),
                    50, format="$%d")
    st.session_state.min_budget = rng[0]
    st.session_state.max_budget = rng[1]

    c1, c2 = st.columns(2)
    c1.metric("Minimum", f"${rng[0]:,}/mo")
    c2.metric("Maximum", f"${rng[1]:,}/mo")

    nav(next_label="Next: Preferences →")


# ── Page 5 — Preferences ───────────────────────────────────────────────────────
def page_preferences() -> None:
    render_progress(5)
    st.markdown("## ✨ Any must-haves?")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.pet_friendly = st.toggle(
            "🐾 Pet-friendly", value=st.session_state.pet_friendly)
    with c2:
        st.session_state.parking_required = st.toggle(
            "🅿️ Parking required", value=st.session_state.parking_required)

    st.markdown("---")
    st.markdown('<div class="section-label">Desired amenities</div>',
                unsafe_allow_html=True)
    st.session_state.amenities = st.multiselect(
        "Amenities", AMENITY_OPTIONS,
        default=[a for a in st.session_state.amenities if a in AMENITY_OPTIONS],
        label_visibility="collapsed", placeholder="Choose amenities…")

    nav(next_label="Review & Search 🔍")


# ── Page 6 — Review ────────────────────────────────────────────────────────────
def page_review() -> None:
    st.markdown("## 📋 Review your search")
    st.caption("Everything look right? Hit **Search** to launch the AI agent.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**📍 Location**")
        loc = f"{st.session_state.city}, {st.session_state.state_abbr}"
        if st.session_state.zip_code:
            loc += f"  ·  ZIP {st.session_state.zip_code}"
        st.info(loc)
        st.caption(f"Radius: {st.session_state.radius} miles")

        if st.session_state.has_commute and st.session_state.commute_destination:
            st.markdown("**🚗 Commute**")
            st.info(st.session_state.commute_destination)
            st.caption(f"Max {st.session_state.max_commute_minutes} min")

        st.markdown("**💰 Budget**")
        st.info(f"${st.session_state.min_budget:,} – ${st.session_state.max_budget:,}/mo")

    with c2:
        st.markdown("**🏠 Property**")
        parts = []
        if st.session_state.property_type != "Any": parts.append(st.session_state.property_type)
        if st.session_state.bedrooms != "Any":       parts.append(f"{st.session_state.bedrooms} bed")
        if st.session_state.bathrooms != "Any":      parts.append(f"{st.session_state.bathrooms} bath")
        st.info(", ".join(parts) if parts else "Any type / size")

        st.markdown("**✨ Preferences**")
        prefs: list[str] = []
        if st.session_state.pet_friendly:     prefs.append("🐾 Pet-friendly")
        if st.session_state.parking_required: prefs.append("🅿️ Parking")
        prefs.extend(st.session_state.amenities[:4])
        extra = len(st.session_state.amenities) - 4
        if extra > 0: prefs.append(f"…+{extra} more")
        st.info("\n".join(prefs) if prefs else "None")

    st.markdown("---")
    cb, _, cs = st.columns([1, 1, 2])
    with cb:
        if st.button("← Edit", use_container_width=True):
            st.session_state.page -= 1
            st.rerun()
    with cs:
        if st.button("🔍 Search for Properties", type="primary", use_container_width=True):
            # Reset backend state
            for k in ("ranked_list","preference_profile","iteration","converged",
                      "convergence_reason","awaiting_feedback","search_error",
                      "search_started","session_id","iteration_at_feedback","prop_images"):
                st.session_state[k] = DEFAULTS.get(k)

            sid = str(uuid.uuid4())
            st.session_state.session_id = sid
            req = _build_requirements()

            try:
                resp = requests.post(
                    f"{BACKEND_URL}/start",
                    json={"requirements": req, "session_id": sid},
                    timeout=5,
                )
                if resp.ok:
                    st.session_state.search_started = True
                    st.session_state.page = 7
                    st.rerun()
                else:
                    st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
            except requests.exceptions.ConnectionError:
                st.error(
                    "**Cannot connect to the backend.**  \n"
                    "Make sure you've started it in a separate terminal:  \n"
                    "```\nuvicorn backend.main:api --port 8000 --reload\n```"
                )


# ── Page 7 — Results ───────────────────────────────────────────────────────────
def page_results() -> None:
    sid = st.session_state.session_id

    # ── Loading state ─────────────────────────────────────────────────────
    if not st.session_state.ranked_list:
        _poll_backend()

        if st.session_state.search_error:
            st.error(f"**Agent error:** {st.session_state.search_error}")
            if st.button("← Try Again"):
                st.session_state.page = 6
                st.rerun()
            return

        # Show animated house while pipeline runs
        st.markdown("""
        <div class="house-wrap">
          <div class="house-scene">
            <div class="h-tree">🌲</div>
            <div class="h-ground"></div>
            <div class="h-base"></div>
            <div class="h-walls">
              <div class="h-win-l"></div>
              <div class="h-win-r"></div>
              <div class="h-door"></div>
            </div>
            <div class="h-roof"></div>
            <div class="h-chimney"></div>
            <div class="h-smoke h-smoke-1"></div>
            <div class="h-smoke h-smoke-2"></div>
          </div>
          <div class="load-title">Building your perfect home search…</div>
          <div class="dots">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
          </div>
          <div class="load-sub">Scanning listings · Scoring properties · Checking nearby amenities</div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(2)
        st.rerun()
        return

    # ── Results header ────────────────────────────────────────────────────
    loc = (f"ZIP {st.session_state.zip_code}" if st.session_state.zip_code
           else f"{st.session_state.city}, {st.session_state.state_abbr}")

    st.markdown(f"## 🏆 {len(st.session_state.ranked_list)} Properties Found")
    st.caption(
        f"{loc}  ·  ${st.session_state.min_budget:,}–${st.session_state.max_budget:,}/mo  ·  "
        f"Iteration {st.session_state.iteration}"
    )

    # ── Convergence banner ────────────────────────────────────────────────
    if st.session_state.converged:
        st.success(f"✅ {st.session_state.convergence_reason}")

    # ── Preference profile ────────────────────────────────────────────────
    if st.session_state.preference_profile:
        st.markdown(
            f'<div class="pref-card">'
            f'<div class="pref-label">🧠 AI Preference Profile</div>'
            f'{st.session_state.preference_profile}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Property cards ────────────────────────────────────────────────────
    props = st.session_state.ranked_list

    # Assign local images once per search (no repeats across the 10 cards)
    if not st.session_state.prop_images:
        folder  = _folder_for_type(st.session_state.property_type)
        pool    = _load_local_images(folder)
        picked  = random.sample(pool, min(len(pool), len(props)))
        st.session_state.prop_images = {
            p["property_id"]: picked[i] for i, p in enumerate(props)
        }

    # Drag-and-drop reordering (shown only while awaiting feedback)
    if st.session_state.awaiting_feedback and not st.session_state.converged:
        _render_feedback_ui(props, sid)
    else:
        _render_property_grid(props)

    # ── Bottom controls ───────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 New Search", use_container_width=True):
            for k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            st.rerun()
    with c2:
        if not st.session_state.converged and not st.session_state.awaiting_feedback:
            if st.button("🔃 Refresh", use_container_width=True):
                _poll_backend()
                st.rerun()


def _render_property_grid(props: list[dict]) -> None:
    """Render all properties as a 5×2 HTML card grid."""
    cards = []
    for i, prop in enumerate(props, start=1):
        score   = prop.get("final_score", 0)
        pct     = round(score * 100)
        color   = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 50 else "#ef4444"
        addr    = prop.get("address", "N/A")
        city    = prop.get("city", "")
        st_val  = prop.get("state", "")
        rent    = prop.get("rent") or 0
        beds    = prop.get("bedrooms")
        baths   = prop.get("bathrooms")

        images = prop.get("images") or []
        if not images:
            local = st.session_state.get("prop_images", {}).get(prop.get("property_id"))
            if local:
                images = [local]
        img_html = (
            f'<img class="prop-img" src="{images[0]}" alt="">'
            if images
            else '<div class="prop-placeholder">🏠</div>'
        )

        parts = []
        if rent:              parts.append(f"💰 ${rent:,}")
        if beds  is not None: parts.append(f"🛏 {beds}")
        if baths is not None: parts.append(f"🚿 {baths}")
        meta = " · ".join(parts)

        rationale = prop.get("rationale", "")
        rat_html  = f'<div class="prop-rat">{rationale}</div>' if rationale else ""

        cards.append(f"""<div class="prop-card">
  <div class="prop-rank">{i}</div>
  {img_html}
  <div class="prop-body">
    <div class="prop-addr">{addr}</div>
    <div class="prop-sub">{city}, {st_val}</div>
    <div class="prop-meta">
      <span>{meta}</span>
      <span class="prop-badge" style="background:{color}">{pct}%</span>
    </div>
    {rat_html}
  </div>
</div>""")

    st.markdown(
        '<div class="prop-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _render_feedback_ui(props: list[dict], sid: str) -> None:
    """Sortable feedback UI — cards merged into draggable one-line items."""
    # Banner header: default house background, 160 px, cover
    st.markdown("""
    <div style="
        width:100%; height:160px; border-radius:12px; margin-bottom:14px;
        background:linear-gradient(135deg,#e2e8f0,#cbd5e1);
        background-size:cover; background-position:center;
        display:flex; flex-direction:column;
        align-items:center; justify-content:center; gap:8px;
    ">
        <span style="font-size:48px;line-height:1;">🏠</span>
        <span style="font-weight:700;font-size:1.05rem;color:#475569;">
            Drag to Refine Your Ranking
        </span>
        <span style="font-size:0.8rem;color:#64748b;">
            Re-order the properties below — the AI will learn your taste and re-rank.
        </span>
    </div>
    """, unsafe_allow_html=True)

    if SORTABLES_AVAILABLE:
        # Format each property as a compact single line before passing to sortables
        id_map: dict[str, str] = {}
        labels: list[str] = []
        for i, p in enumerate(props):
            pct  = round(p.get("final_score", 0) * 100)
            beds = p.get("bedrooms")
            bath = p.get("bathrooms")
            bed_bath = f"{beds} Bed · {bath} Bath" if beds is not None and bath is not None else ""
            label = (
                f"{pct}% Match\n" +
                f"{p.get('address','?')}, {p.get('city','')}\n"
                + (f"{bed_bath} | " if bed_bath else "")
                + f"${p.get('rent', 0):,}/mo\n"
                + f"{p.get('rationale','')}" if p.get("rationale") else ""
            )
            labels.append(label)
            id_map[label] = p["property_id"]

        sorted_labels = sort_items(labels, direction="vertical", key="feedback_sort")
        _inject_sortable_css()
        sorted_ids = [id_map[lbl] for lbl in sorted_labels if lbl in id_map]

        if st.button("✅ Submit My Ranking", type="primary"):
            try:
                requests.post(
                    f"{BACKEND_URL}/feedback",
                    json={"session_id": sid, "ordered_property_ids": sorted_ids},
                    timeout=5,
                )
                st.session_state.iteration_at_feedback = st.session_state.iteration
                st.session_state.awaiting_feedback = False
                st.session_state.ranked_list = []
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Could not submit feedback: {e}")

    else:
        st.warning(
            "Install `streamlit-sortables` for drag-and-drop:  \n"
            "```\npip install streamlit-sortables\n```  \n"
            "Then restart Streamlit."
        )
        manual_order = st.text_input("Manual order (comma-separated ranks, e.g. 3,1,5,2,4):",
                                     key="manual_feedback")
        if st.button("Submit Manual Ranking"):
            try:
                indices    = [int(x.strip()) - 1 for x in manual_order.split(",") if x.strip()]
                sorted_ids = [props[i]["property_id"] for i in indices if 0 <= i < len(props)]
                requests.post(
                    f"{BACKEND_URL}/feedback",
                    json={"session_id": sid, "ordered_property_ids": sorted_ids},
                    timeout=5,
                )
                st.session_state.iteration_at_feedback = st.session_state.iteration
                st.session_state.awaiting_feedback = False
                st.session_state.ranked_list = []
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Could not submit: {e}")


# ── Router ─────────────────────────────────────────────────────────────────────
_PAGES = [
    page_welcome,     # 0
    page_location,    # 1
    page_commute,     # 2
    page_property,    # 3
    page_budget,      # 4
    page_preferences, # 5
    page_review,      # 6
    page_results,     # 7
]

_PAGES[st.session_state.page]()
