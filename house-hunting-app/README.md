# 🏠 AI House Finder

An intelligent rental property search agent that learns your preferences in real time. You describe what you're looking for through a guided wizard, and the agent fetches live listings, scores them against your criteria, and refines its rankings based on your drag-and-drop feedback — getting smarter with every round.

---

## Major Components

### `app.py` — Streamlit Frontend
A multi-page wizard UI (7 steps) that collects search criteria (location, commute, property type, budget, preferences), submits them to the backend, polls for results, and renders an interactive property card grid. Supports drag-and-drop reordering via `streamlit-sortables` for feedback submission.

### `backend/main.py` — FastAPI Backend
Exposes three REST endpoints (`POST /start`, `POST /feedback`, `GET /state/{session_id}`) that bridge the Streamlit UI and the LangGraph agent. Manages per-session state and uses `asyncio.Event` to synchronize the feedback loop.

### `backend/graph.py` — LangGraph Pipeline
Wires 8 nodes into a directed state graph with a feedback loop. Phase 1 (ingest → fetch → enrich → hard_score → soft_score → rank) runs once. Phase 2 (update_weights → soft_score → rank) re-runs after each feedback submission until convergence.

### `backend/nodes/` — Agent Nodes

| Node | Role |
|---|---|
| `ingest.py` | Parses requirements and builds an initial preference profile via Gemini LLM |
| `fetch.py` | Fetches up to 20 active rental listings from the RentCast API |
| `enrich.py` | Calls ArcGIS Places (in parallel) to get distances to grocery, school, gym, and transit for each property |
| `hard_score.py` | Scores properties on fixed objective criteria: price match, beds, baths, sqft/dollar |
| `soft_score.py` | Scores properties on learnable preference criteria; generates per-property rationales via a batched Gemini call |
| `rank.py` | Combines hard + soft scores (60/40 initially, 30/70 after feedback) and surfaces the top 10 |
| `feedback.py` | No-op node; feedback is injected into state by the FastAPI `/feedback` endpoint |
| `update_weights.py` | Uses Gemini to infer updated soft weights from the gap between system and user rankings; checks convergence |

### `backend/tools/` — External API Integrations

- **`rentcast.py`** — Wraps the RentCast long-term rental listings API. Normalizes raw responses into a consistent property dict format.
- **`arcgis.py`** — Calls the ArcGIS Places near-point API to measure walking/driving distances to nearby amenities.
- **`scorer.py`** — Pure-Python scoring functions (`hard_score`, `soft_score`, `weights_converged`) with no API dependencies. Convergence is detected via cosine similarity of weight vectors.

---

## How to Run

### Prerequisites

- Python 3.10+
- API keys for **Gemini**, **RentCast**, and **ArcGIS** (set in a `.env` file)

### 1. Install dependencies

```bash
cd house-hunting-app
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the `house-hunting-app/` directory:

```env
GEMINI_API_KEY=your_gemini_api_key
RENTCAST_API_KEY=your_rentcast_api_key
ARCGIS_API_KEY=your_arcgis_api_key
```

### 3. Start the backend (Terminal 1)

```bash
cd house-hunting-app
python start_backend.py
```

Or directly with uvicorn:

```bash
uvicorn backend.main:api --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Streamlit frontend (Terminal 2)

```bash
cd house-hunting-app
streamlit run app.py
```

### 5. Open the app

Navigate to [http://localhost:8501](http://localhost:8501) in your browser.

---

## Notes

- The backend must be running before submitting a search from the UI.
- `google-genai` is listed in `requirements.txt` but unused — it can be safely removed.
- In-memory session state is used by default; swap for Redis in multi-worker deployments.
