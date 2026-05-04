# Rental AI Agent — Full Implementation Context
> This document is written for a Claude agent that will implement this system from scratch.
> Read every section before writing any code. All decisions, tradeoffs, and architecture choices are explained here.

---

## 1. Project Summary

**Goal:** Build a personalized rental property recommendation system that tailors results to each individual user — not a generic search engine.

**Type:** LLM-powered agentic pipeline with a feedback loop.

**Context:** This is an AI/LLM class project. The LLM must be a *core, demonstrable component* — not a helper utility. The system must showcase: tool-use/function-calling, prompt engineering, preference learning, and iterative personalization.

**What is already built:** A Streamlit UI that collects user requirements through a sequential, click-based wizard. It successfully produces a structured requirements object. You do NOT need to rebuild this.

**What needs to be built:** The entire backend — the LangGraph agent, all tool integrations, scoring logic, feedback loop, and the bridge between Streamlit and the agent.

---

## 2. Framework Decision: LangGraph (not LangChain, not plain Python)

**Why not plain Python + Gemini/Claude API:**
Plain Python is fine for linear pipelines. This system has a *stateful feedback loop* — the agent must revisit earlier stages (re-scoring) based on user input, while preserving state across iterations. Plain Python requires you to manually manage state, loop control, and conditional routing. This is exactly what LangGraph solves.

**Why not LangChain (chains only):**
LangChain is linear. Once a chain runs, it's done. Our system needs to loop: show results → get feedback → update weights → re-score → show results again. This cycle repeats until convergence. LangGraph models this as a directed graph with conditional edges.

**Why LangGraph:**
- Models the pipeline as a stateful directed graph
- Each node is a Python function that reads and writes to a shared `AgentState`
- Conditional edges allow looping back to earlier nodes
- Natively supports tool-calling nodes
- Works well with any LLM (OpenAI, Anthropic, Gemini) via LangChain's model interfaces

**LLM to use:** Any of `gpt-4o`, `claude-sonnet-4-6`, or `gemini-1.5-pro`. All support tool/function calling. Use whichever your class has API access to. The code patterns are identical — only the model string changes.

---

## 3. System Architecture Overview

```
Streamlit UI (already built)
        │
        │ requirements.temp (JSON)
        ▼
┌─────────────────────────────────────────┐
│           LangGraph Agent               │
│                                         │
│  Node 1: ingest_requirements (LLM)      │
│  Node 2: fetch_properties (RentCast)    │
│  Node 3: enrich_properties (RentCafe    │
│           + ArcGIS Places)              │
│  Node 4: hard_score (pure Python)       │
│  Node 5: soft_score (LLM)              │
│  Node 6: rank_and_display              │
│  Node 7: collect_feedback              │
│  Node 8: update_weights (LLM)          │
│       │                                 │
│  [Conditional Edge]                     │
│  converged? → END                       │
│  not converged? → back to Node 5        │
└─────────────────────────────────────────┘
        │
        │ ranked properties + rationale
        ▼
Streamlit UI (displays results, drag-drop feedback)
```

**Streamlit ↔ LangGraph bridge:**
Run LangGraph as a **FastAPI backend**. Streamlit calls it via HTTP. `st.session_state` holds AgentState between Streamlit reruns. This is the cleanest approach for a class project and keeps the two systems decoupled.

---

## 4. Data Templates

### 4.1 requirements.temp
Produced by the Streamlit UI. Passed into Node 1.

```json
{
  "location": {
    "city": "Austin",
    "state": "TX",
    "zip": "78701",
    "search_radius_miles": 10
  },
  "property": {
    "bedrooms": 2,
    "bathrooms": 1,
    "price_min": 1200,
    "price_max": 2000,
    "property_type": "apartment"
  },
  "amenities": {
    "grocery_max_miles": 1.0,
    "school_max_miles": 2.0,
    "gym_max_miles": 1.5,
    "public_transit_max_miles": 0.5
  },
  "preferences": {
    "pet_friendly": true,
    "parking": true,
    "laundry": "in_unit"
  },
  "additional_comments": "I work from home and need a quiet space. I have a dog.",
  "session_id": "uuid-string"
}
```

### 4.2 information.temp
Built after Node 3 (enrichment). One entry per property.

```json
{
  "session_id": "uuid-string",
  "properties": [
    {
      "property_id": "rentcast-id-123",
      "address": "123 Main St, Austin TX 78701",
      "bedrooms": 2,
      "bathrooms": 1,
      "rent": 1650,
      "sqft": 920,
      "images": ["url1", "url2"],
      "floor_plan_url": "url",
      "amenity_distances": {
        "walmart": 0.8,
        "target": 1.2,
        "whole_foods": 2.1,
        "elementary_school": 1.5,
        "gym": 0.6,
        "bus_stop": 0.3
      },
      "raw_rentcast_data": {},
      "raw_rentcafe_data": {},
      "raw_arcgis_data": {}
    }
  ]
}
```

---

## 5. AgentState — The Shared State Object

This is the single object that flows through every node. Every node reads from it and writes to it.

```python
from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    # Input
    requirements: dict                  # from requirements.temp
    session_id: str

    # After Node 2
    properties: List[dict]              # raw from RentCast

    # After Node 3
    enriched_properties: List[dict]     # filled information.temp entries

    # After Node 4
    hard_scores: Dict[str, float]       # property_id -> score (0-1)

    # After Node 5 (updated each loop)
    soft_scores: Dict[str, float]       # property_id -> score (0-1)

    # After Node 6
    ranked_list: List[dict]             # sorted properties with scores + rationale

    # After Node 7 (user input)
    user_feedback: List[str]            # ordered list of property_ids from drag-drop

    # Learned over feedback loops
    soft_weights: Dict[str, float]      # feature_name -> weight (sum to 1.0)
    preference_profile: str             # LLM-maintained natural language summary

    # Control
    iteration: int                      # feedback loop counter (starts at 0)
    converged: bool                     # termination flag
    convergence_reason: str             # why it converged (for display)
```

---

## 6. Tool Definitions

All tools use the `@tool` decorator from LangChain. Register them with the LangGraph nodes that need them.

### Tool 1: RentCast API
```python
from langchain.tools import tool
import requests

@tool
def rentcast_search(
    city: str,
    state: str,
    bedrooms: int,
    bathrooms: float,
    min_price: int,
    max_price: int,
    radius_miles: float = 10.0
) -> list:
    """
    Search rental properties using RentCast API.
    Returns list of property dicts with address, rent, beds, baths, sqft.
    RentCast free tier: https://app.rentcast.io/app — get API key there.
    Endpoint: GET https://api.rentcast.io/v1/listings/rental/long-term
    """
    headers = {"X-Api-Key": RENTCAST_API_KEY}
    params = {
        "city": city, "state": state,
        "bedrooms": bedrooms, "bathrooms": bathrooms,
        "minPrice": min_price, "maxPrice": max_price,
        "radius": radius_miles, "limit": 20
    }
    resp = requests.get(
        "https://api.rentcast.io/v1/listings/rental/long-term",
        headers=headers, params=params
    )
    return resp.json().get("listings", [])
```

### Tool 2: RentCafe API
⚠️ **Important note:** RentCafe does not have a public API. Two options:
- **Option A (recommended for class project):** Use their website search scraper with `requests` + `BeautifulSoup`. Target: `https://www.rentcafe.com/apartments-for-rent/us/{state}/{city}/`
- **Option B:** Skip RentCafe, source images from RentCast listings directly (they include photo URLs in some responses) or use a placeholder image service.

```python
@tool
def rentcafe_get_images(address: str, city: str, state: str) -> dict:
    """
    Fetch property images and floor plan for a given address.
    Uses RentCafe website search. Returns dict with image_urls and floor_plan_url.
    """
    # Implementation: scrape rentcafe.com search results for this address
    # Return {"images": [...urls], "floor_plan_url": "..."}
    pass
```

### Tool 3: ArcGIS Places API
```python
@tool
def arcgis_amenity_distances(
    latitude: float,
    longitude: float,
    amenity_types: list  # e.g. ["grocery", "school", "gym"]
) -> dict:
    """
    Find nearest amenities to a property using ArcGIS Places API.
    Free tier available at: https://developers.arcgis.com/
    Returns dict of amenity_type -> distance_miles.
    Endpoint: https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/places/near-point
    """
    import arcgis  # pip install arcgis
    # or use requests directly with ArcGIS REST API
    results = {}
    for amenity in amenity_types:
        # query ArcGIS for nearest place of this category
        # calculate distance from property coords
        results[amenity] = distance_in_miles
    return results
```

### Tool 4: Scorer (not an LLM tool — pure Python functions)
```python
def hard_score(property: dict, requirements: dict, weights: dict) -> float:
    """
    Fixed-weight scoring. Weights never change.
    Score each feature 0-1, multiply by weight, sum.
    """
    HARD_WEIGHTS = {
        "price_match": 0.35,      # how close rent is to midpoint of user range
        "bedroom_match": 0.25,    # exact match = 1.0, off by 1 = 0.5, off by 2 = 0
        "bathroom_match": 0.15,   # same logic
        "sqft_per_dollar": 0.15,  # value metric
        "within_radius": 0.10     # binary: is it within search radius
    }
    score = 0.0
    # price score
    mid = (requirements["property"]["price_min"] + requirements["property"]["price_max"]) / 2
    price_dev = abs(property["rent"] - mid) / mid
    score += HARD_WEIGHTS["price_match"] * max(0, 1 - price_dev)
    # bedroom score
    bed_diff = abs(property["bedrooms"] - requirements["property"]["bedrooms"])
    score += HARD_WEIGHTS["bedroom_match"] * max(0, 1 - (bed_diff * 0.5))
    # ... add other features
    return round(score, 4)


def soft_score(property: dict, soft_weights: dict, amenity_distances: dict) -> float:
    """
    Learnable-weight scoring. Weights start equal, update via feedback.
    Features are amenity distances and preference matches.
    """
    # Initial equal weights (set in Node 1 or first run of Node 5)
    DEFAULT_SOFT_WEIGHTS = {
        "grocery_distance": 1/6,
        "school_distance": 1/6,
        "gym_distance": 1/6,
        "transit_distance": 1/6,
        "pet_friendly": 1/6,
        "quiet_neighborhood": 1/6
    }
    weights = soft_weights if soft_weights else DEFAULT_SOFT_WEIGHTS
    score = 0.0
    # score each feature and multiply by its weight
    # grocery: 0 miles = 1.0, 2 miles = 0.0 (linear decay)
    if "grocery_distance" in amenity_distances:
        d = amenity_distances["grocery_distance"]
        score += weights["grocery_distance"] * max(0, 1 - d/2.0)
    # ... add other features
    return round(score, 4)
```

---

## 7. Node Implementations

### Node 1: ingest_requirements (LLM node)

**Purpose:** Take requirements.temp, parse the free-text `additional_comments`, infer latent preferences the user didn't explicitly state, and build the initial `preference_profile` string.

**Why LLM here:** The `additional_comments` field is unstructured. "I work from home and have a dog" implies: needs quiet, needs pet-friendly, may prefer ground floor, home office space matters. The LLM extracts these signals.

```python
from langchain_openai import ChatOpenAI  # or ChatAnthropic / ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def ingest_requirements(state: AgentState) -> AgentState:
    req = state["requirements"]
    comments = req.get("additional_comments", "")

    prompt = f"""
You are a rental requirements analyst.

The user filled out a rental search form with these structured requirements:
{req}

They also left this free-text comment: "{comments}"

Your tasks:
1. Identify any latent preferences implied by their comments (e.g. "I have a dog" → pet-friendly, ground floor preferred)
2. Write a concise natural language preference profile (3-5 sentences) summarizing what matters most to this user.
3. Return ONLY the preference profile string. No preamble.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    state["preference_profile"] = response.content
    state["iteration"] = 0
    state["converged"] = False
    # Initialize equal soft weights
    state["soft_weights"] = {
        "grocery_distance": 1/6,
        "school_distance": 1/6,
        "gym_distance": 1/6,
        "transit_distance": 1/6,
        "pet_friendly": 1/6,
        "quiet_neighborhood": 1/6
    }
    return state
```

### Node 2: fetch_properties (Tool node)

**Purpose:** Call RentCast API with structured parameters from requirements.

```python
def fetch_properties(state: AgentState) -> AgentState:
    req = state["requirements"]
    properties = rentcast_search.invoke({
        "city": req["location"]["city"],
        "state": req["location"]["state"],
        "bedrooms": req["property"]["bedrooms"],
        "bathrooms": req["property"]["bathrooms"],
        "min_price": req["property"]["price_min"],
        "max_price": req["property"]["price_max"],
        "radius_miles": req["location"]["search_radius_miles"]
    })
    state["properties"] = properties
    return state
```

### Node 3: enrich_properties (Tool node)

**Purpose:** For each property from RentCast, call RentCafe for images and ArcGIS for amenity distances. Build `information.temp`.

```python
def enrich_properties(state: AgentState) -> AgentState:
    enriched = []
    amenity_types = list(state["requirements"]["amenities"].keys())

    for prop in state["properties"]:
        # Get images
        images_data = rentcafe_get_images.invoke({
            "address": prop["address"],
            "city": prop["city"],
            "state": prop["state"]
        })
        # Get amenity distances
        distances = arcgis_amenity_distances.invoke({
            "latitude": prop["latitude"],
            "longitude": prop["longitude"],
            "amenity_types": amenity_types
        })
        enriched_prop = {**prop, **images_data, "amenity_distances": distances}
        enriched.append(enriched_prop)

    state["enriched_properties"] = enriched
    return state
```

### Node 4: hard_score (Pure Python node)

**Purpose:** Score every property with fixed weights. This never changes across iterations.

```python
def hard_score_node(state: AgentState) -> AgentState:
    scores = {}
    for prop in state["enriched_properties"]:
        scores[prop["property_id"]] = hard_score(
            prop, state["requirements"], {}
        )
    state["hard_scores"] = scores
    return state
```

### Node 5: soft_score (LLM node)

**Purpose:** Score properties against the user's `preference_profile` using the current `soft_weights`. This runs on every feedback loop iteration with updated weights.

```python
def soft_score_node(state: AgentState) -> AgentState:
    scores = {}
    rationales = {}

    for prop in state["enriched_properties"]:
        # Compute numeric soft score
        numeric_score = soft_score(
            prop, state["soft_weights"], prop.get("amenity_distances", {})
        )
        scores[prop["property_id"]] = numeric_score

        # LLM generates rationale for top properties (optional, for display)
        if numeric_score > 0.6:
            prompt = f"""
User preference profile: {state['preference_profile']}

Property details:
- Address: {prop['address']}
- Rent: ${prop['rent']}/month
- Amenity distances: {prop.get('amenity_distances', {})}

In 1-2 sentences, explain why this property matches the user's preferences.
"""
            resp = llm.invoke([HumanMessage(content=prompt)])
            rationales[prop["property_id"]] = resp.content

    state["soft_scores"] = scores
    state["soft_rationales"] = rationales
    return state
```

### Node 6: rank_and_display

**Purpose:** Combine hard and soft scores into a final ranking. Push to Streamlit.

```python
HARD_WEIGHT = 0.6  # hard scoring contributes 60% of final score
SOFT_WEIGHT = 0.4  # soft scoring contributes 40%

def rank_and_display(state: AgentState) -> AgentState:
    ranked = []
    for prop in state["enriched_properties"]:
        pid = prop["property_id"]
        final_score = (
            HARD_WEIGHT * state["hard_scores"].get(pid, 0) +
            SOFT_WEIGHT * state["soft_scores"].get(pid, 0)
        )
        ranked.append({
            **prop,
            "hard_score": state["hard_scores"].get(pid, 0),
            "soft_score": state["soft_scores"].get(pid, 0),
            "final_score": round(final_score, 4),
            "rationale": state.get("soft_rationales", {}).get(pid, "")
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    state["ranked_list"] = ranked
    # Push to Streamlit via session state or FastAPI endpoint
    return state
```

### Node 7: collect_feedback

**Purpose:** Wait for user drag-drop reordering from Streamlit. Receives an ordered list of property_ids.

```python
def collect_feedback(state: AgentState) -> AgentState:
    # In FastAPI mode: this node blocks until the /feedback endpoint receives a POST
    # from Streamlit with the user's reordering.
    # state["user_feedback"] is set by the FastAPI endpoint handler before
    # this node resumes.
    # In practice: use asyncio.Event or a queue to synchronize.
    return state
```

### Node 8: update_weights (LLM node)

**Purpose:** The most important LLM node. Interprets the user's reordering to infer what they actually value. Updates `soft_weights` and `preference_profile`. Checks convergence.

```python
def update_weights(state: AgentState) -> AgentState:
    system_ranking = [p["property_id"] for p in state["ranked_list"]]
    user_ranking = state["user_feedback"]

    if system_ranking == user_ranking:
        state["converged"] = True
        state["convergence_reason"] = "User ordering matches system ranking."
        return state

    # Build comparison context for LLM
    comparison = []
    for i, pid in enumerate(user_ranking):
        prop = next(p for p in state["ranked_list"] if p["property_id"] == pid)
        system_rank = system_ranking.index(pid) + 1
        comparison.append(
            f"User rank {i+1} (System rank {system_rank}): "
            f"{prop['address']} | Rent: ${prop['rent']} | "
            f"Amenities: {prop.get('amenity_distances', {})}"
        )

    prompt = f"""
You are a preference learning system for rental recommendations.

Current preference profile: {state['preference_profile']}

Current soft weights: {state['soft_weights']}

The system ranked properties in one order, but the user reordered them.
Here is the user's preferred order with each property's details:

{chr(10).join(comparison)}

Based on the differences between system ranking and user ranking:
1. What preferences does the user seem to value MORE than currently weighted?
2. What preferences does the user seems to value LESS?
3. Write an updated preference profile (3-5 sentences).
4. Return updated soft weights as JSON. Weights must sum to 1.0. Keys: grocery_distance, school_distance, gym_distance, transit_distance, pet_friendly, quiet_neighborhood.

Respond in this exact JSON format:
{{
  "updated_preference_profile": "...",
  "updated_soft_weights": {{
    "grocery_distance": 0.0,
    "school_distance": 0.0,
    "gym_distance": 0.0,
    "transit_distance": 0.0,
    "pet_friendly": 0.0,
    "quiet_neighborhood": 0.0
  }},
  "reasoning": "one sentence explanation of key weight change"
}}
"""
    resp = llm.invoke([HumanMessage(content=prompt)])
    import json
    data = json.loads(resp.content)

    state["preference_profile"] = data["updated_preference_profile"]
    state["soft_weights"] = data["updated_soft_weights"]
    state["iteration"] += 1

    # Convergence: stop after 5 iterations regardless
    if state["iteration"] >= 5:
        state["converged"] = True
        state["convergence_reason"] = "Maximum iterations reached."

    return state
```

---

## 8. Graph Assembly

```python
from langgraph.graph import StateGraph, END

def check_convergence(state: AgentState) -> str:
    return "converged" if state["converged"] else "continue"

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ingest_requirements", ingest_requirements)
    graph.add_node("fetch_properties", fetch_properties)
    graph.add_node("enrich_properties", enrich_properties)
    graph.add_node("hard_score", hard_score_node)
    graph.add_node("soft_score", soft_score_node)
    graph.add_node("rank_and_display", rank_and_display)
    graph.add_node("collect_feedback", collect_feedback)
    graph.add_node("update_weights", update_weights)

    graph.set_entry_point("ingest_requirements")

    graph.add_edge("ingest_requirements", "fetch_properties")
    graph.add_edge("fetch_properties", "enrich_properties")
    graph.add_edge("enrich_properties", "hard_score")
    graph.add_edge("hard_score", "soft_score")
    graph.add_edge("soft_score", "rank_and_display")
    graph.add_edge("rank_and_display", "collect_feedback")
    graph.add_edge("collect_feedback", "update_weights")

    graph.add_conditional_edges(
        "update_weights",
        check_convergence,
        {
            "converged": END,
            "continue": "soft_score"   # loop back — skips re-fetching properties
        }
    )

    return graph.compile()

app = build_graph()
```

**Note on the loop:** When the feedback loop cycles back, it goes to `soft_score` — NOT back to `fetch_properties`. Properties are only fetched once. Only scoring and ranking update across iterations.

---

## 9. FastAPI Backend

```python
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

api = FastAPI()
agent_states = {}  # session_id -> AgentState
feedback_events = {}  # session_id -> asyncio.Event

class RequirementsInput(BaseModel):
    requirements: dict
    session_id: str

class FeedbackInput(BaseModel):
    session_id: str
    ordered_property_ids: list

@api.post("/start")
async def start_agent(input: RequirementsInput):
    """Called by Streamlit when user submits requirements."""
    state = AgentState(
        requirements=input.requirements,
        session_id=input.session_id,
        properties=[], enriched_properties=[],
        hard_scores={}, soft_scores={},
        ranked_list=[], user_feedback=[],
        soft_weights={}, preference_profile="",
        iteration=0, converged=False, convergence_reason=""
    )
    # Run graph asynchronously
    asyncio.create_task(run_graph(input.session_id, state))
    return {"status": "started"}

@api.post("/feedback")
async def receive_feedback(input: FeedbackInput):
    """Called by Streamlit when user submits drag-drop reordering."""
    if input.session_id in agent_states:
        agent_states[input.session_id]["user_feedback"] = input.ordered_property_ids
        if input.session_id in feedback_events:
            feedback_events[input.session_id].set()
    return {"status": "received"}

@api.get("/state/{session_id}")
async def get_state(session_id: str):
    """Streamlit polls this to get current ranked_list."""
    state = agent_states.get(session_id, {})
    return {
        "ranked_list": state.get("ranked_list", []),
        "converged": state.get("converged", False),
        "iteration": state.get("iteration", 0),
        "preference_profile": state.get("preference_profile", "")
    }

async def run_graph(session_id: str, initial_state: AgentState):
    agent_states[session_id] = initial_state
    result = await app.ainvoke(initial_state)
    agent_states[session_id] = result
```

---

## 10. Streamlit Integration Points

The Streamlit UI (already built) needs these additions:

```python
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

# After requirements are collected:
def submit_requirements(requirements: dict):
    session_id = st.session_state.get("session_id", str(uuid.uuid4()))
    st.session_state["session_id"] = session_id
    requests.post(f"{BACKEND_URL}/start", json={
        "requirements": requirements,
        "session_id": session_id
    })

# Polling for results (use st.rerun() on a timer):
def poll_results():
    sid = st.session_state.get("session_id")
    if sid:
        resp = requests.get(f"{BACKEND_URL}/state/{sid}")
        data = resp.json()
        st.session_state["ranked_list"] = data["ranked_list"]
        st.session_state["converged"] = data["converged"]

# After drag-drop reordering:
def submit_feedback(ordered_ids: list):
    sid = st.session_state.get("session_id")
    requests.post(f"{BACKEND_URL}/feedback", json={
        "session_id": sid,
        "ordered_property_ids": ordered_ids
    })
```

For drag-and-drop in Streamlit, use the `streamlit-sortables` package:
```bash
pip install streamlit-sortables
```
```python
from streamlit_sortables import sort_items
sorted_ids = sort_items([p["property_id"] for p in st.session_state["ranked_list"]])
```

---

## 11. Scoring Details

### Hard Scoring (fixed, never changes)

| Feature | Weight | How scored |
|---|---|---|
| Price match | 0.35 | `1 - abs(rent - midpoint) / midpoint`, clamped 0-1 |
| Bedroom match | 0.25 | `1 - (diff * 0.5)`, clamped 0-1 |
| Bathroom match | 0.15 | same as bedroom |
| Sqft per dollar | 0.15 | `sqft/rent` normalized across all properties |
| Within radius | 0.10 | binary 1.0 or 0.0 |

**Total hard weights = 1.0**

### Soft Scoring (learnable)

| Feature | Initial weight | How scored |
|---|---|---|
| Grocery distance | 1/6 | `max(0, 1 - distance/2.0)` |
| School distance | 1/6 | `max(0, 1 - distance/3.0)` |
| Gym distance | 1/6 | `max(0, 1 - distance/2.0)` |
| Transit distance | 1/6 | `max(0, 1 - distance/0.5)` |
| Pet friendly | 1/6 | binary 1.0 or 0.0 |
| Quiet neighborhood | 1/6 | derived from ArcGIS noise/density data |

**Weights always sum to 1.0. After each feedback loop, LLM recalculates weights.**

### Final Combined Score
```
final_score = 0.6 * hard_score + 0.4 * soft_score
```

The 0.6/0.4 split keeps objective criteria dominant but gives personalization meaningful influence. These constants can be made configurable.

---

## 12. Convergence Logic

The feedback loop terminates under any of these conditions:

1. **User ordering matches system ranking** — `system_ranking == user_ranking`
2. **Weight stability** — cosine similarity between previous and current `soft_weights` > 0.97
3. **Max iterations** — `iteration >= 5` (hard cap to prevent infinite loops in demo)

```python
import numpy as np

def weights_converged(old_weights: dict, new_weights: dict, threshold=0.97) -> bool:
    old_vec = np.array(list(old_weights.values()))
    new_vec = np.array(list(new_weights.values()))
    cosine_sim = np.dot(old_vec, new_vec) / (np.linalg.norm(old_vec) * np.linalg.norm(new_vec))
    return cosine_sim > threshold
```

---

## 13. API Keys Needed

| Service | Free tier? | Where to get |
|---|---|---|
| RentCast | Yes (limited) | https://app.rentcast.io/app |
| ArcGIS Places | Yes (developer) | https://developers.arcgis.com |
| RentCafe | No public API | Scrape or skip — use RentCast images |
| OpenAI / Anthropic / Google | Yes (free credits) | respective developer portals |

Store all keys in a `.env` file:
```
RENTCAST_API_KEY=...
ARCGIS_API_KEY=...
OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY or GOOGLE_API_KEY
```

---

## 14. File Structure

```
rental-agent/
├── streamlit_app.py          # already built — add integration points from Section 10
├── backend/
│   ├── main.py               # FastAPI app (Section 9)
│   ├── graph.py              # LangGraph assembly (Section 8)
│   ├── state.py              # AgentState TypedDict (Section 5)
│   ├── nodes/
│   │   ├── ingest.py         # Node 1
│   │   ├── fetch.py          # Node 2
│   │   ├── enrich.py         # Node 3
│   │   ├── hard_score.py     # Node 4
│   │   ├── soft_score.py     # Node 5
│   │   ├── rank.py           # Node 6
│   │   ├── feedback.py       # Node 7
│   │   └── update_weights.py # Node 8
│   ├── tools/
│   │   ├── rentcast.py       # @tool definitions
│   │   ├── rentcafe.py
│   │   ├── arcgis.py
│   │   └── scorer.py         # hard_score + soft_score functions
│   └── templates/
│       ├── requirements_template.py
│       └── information_template.py
├── .env
└── requirements.txt
```

---

## 15. Implementation Order (recommended)

Build in this order — each step unblocks the next:

1. `state.py` — define AgentState
2. `scorer.py` — implement hard_score and soft_score (no API needed, fully testable)
3. `tools/rentcast.py` — get real property data flowing
4. `nodes/ingest.py` + `nodes/fetch.py` + `nodes/hard_score.py` — first 4 nodes
5. `graph.py` — wire the graph, test with mock data for nodes 3-8
6. `tools/arcgis.py` — add enrichment
7. `nodes/soft_score.py` + `nodes/update_weights.py` — LLM nodes
8. `main.py` — FastAPI wrapper
9. Update `streamlit_app.py` with integration points
10. End-to-end test with real APIs

---

## 16. Key Design Decisions Summary

| Decision | Choice | Reason |
|---|---|---|
| Framework | LangGraph | Stateful feedback loop requires conditional graph edges |
| LLM role | Agent brain for 3 nodes + tool orchestration | Makes LLM central to project, not a helper |
| Hard/soft split | 60/40 | Objective criteria should dominate; soft adds personalization |
| Soft weight update | LLM interprets reordering | Richer than pure math; demonstrates NLP preference learning |
| Convergence | 3 conditions (ordering match, weight stability, max iterations) | Robust termination |
| Streamlit bridge | FastAPI + HTTP polling | Decoupled; Streamlit's synchronous model needs async backend |
| RentCafe | Scrape or use RentCast images | No public API exists |
| Loop re-entry point | Node 5 (soft_score), not Node 2 | Properties don't need re-fetching on each loop |

---

*End of context document. All information needed to implement this system is contained above.*
