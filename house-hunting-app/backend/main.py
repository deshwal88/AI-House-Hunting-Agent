import sys, os
# Ensure the house-hunting-app directory is on sys.path so that
# `import backend.xyz` works regardless of where uvicorn is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
FastAPI backend — bridges Streamlit and the LangGraph agent.

Endpoints:
  POST /start              — Streamlit submits requirements → starts the agent graph
  POST /feedback           — Streamlit submits user drag-drop reordering
  GET  /state/{session_id} — Streamlit polls for current ranked_list + status

Feedback synchronisation:
  collect_feedback node awaits an asyncio.Event stored in `feedback_events`.
  The /feedback endpoint populates state["user_feedback"] then sets that event,
  allowing the graph to resume from where it was waiting.

Run with:
  uvicorn backend.main:api --host 0.0.0.0 --port 8000 --reload
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()  # load .env so all os.getenv() calls in tools/nodes work

# Logging is configured in start_backend.py via uvicorn's log_config.
# All agent.* loggers inherit from the "agent" logger defined there.
log = logging.getLogger("agent.main")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.state import AgentState
from backend.graph import agent_graph

api = FastAPI(title="AI House Finder — Agent Backend", version="1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Streamlit runs on localhost — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state stores (in-memory; replace with Redis for multi-worker) ───────
agent_states:    dict[str, AgentState] = {}
feedback_events: dict[str, asyncio.Event] = {}


# ── Request / response models ─────────────────────────────────────────────────
class StartInput(BaseModel):
    requirements: dict
    session_id:   str

class FeedbackInput(BaseModel):
    session_id:           str
    ordered_property_ids: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@api.post("/start")
async def start_agent(body: StartInput):
    """
    Called by Streamlit when the user submits their requirements.
    Initialises state and kicks off the LangGraph pipeline asynchronously.
    """
    sid = body.session_id
    log.info(f"[API] POST /start  session={sid}")
    req = body.requirements
    loc = req.get("location", {})
    log.info(f"[API]   Location: {loc.get('city')}, {loc.get('state')}  ZIP={loc.get('zip', 'n/a')}")

    # Cancel any existing session for this ID
    if sid in feedback_events:
        log.info(f"[API]   Cancelling previous session {sid}")
        feedback_events[sid].set()
    feedback_events[sid] = asyncio.Event()

    initial_state: AgentState = {
        "requirements":       body.requirements,
        "session_id":         sid,
        "properties":         [],
        "enriched_properties": [],
        "hard_scores":        {},
        "soft_scores":        {},
        "soft_rationales":    {},
        "ranked_list":        [],
        "user_feedback":      [],
        "soft_weights":       {},
        "preference_profile": "",
        "iteration":          0,
        "converged":          False,
        "convergence_reason": "",
    }
    agent_states[sid] = initial_state

    asyncio.create_task(_run_graph(sid, initial_state))
    log.info(f"[API]   Graph task created for session {sid}")
    return {"status": "started", "session_id": sid}


@api.post("/feedback")
async def receive_feedback(body: FeedbackInput):
    """
    Called by Streamlit after the user reorders the property cards.
    Populates user_feedback in state and unblocks the collect_feedback node.
    """
    sid = body.session_id
    log.info(f"[API] POST /feedback  session={sid}  items={len(body.ordered_property_ids)}")
    if sid not in agent_states:
        log.error(f"[API] ✗ Session {sid} not found")
        raise HTTPException(status_code=404, detail="Session not found")

    agent_states[sid]["user_feedback"] = body.ordered_property_ids
    log.info(f"[API]   User ranking: {body.ordered_property_ids[:5]}")

    if sid in feedback_events:
        feedback_events[sid].set()
        feedback_events[sid] = asyncio.Event()
        log.info(f"[API]   Feedback event set — graph resuming")

    return {"status": "received"}


@api.get("/state/{session_id}")
async def get_state(session_id: str):
    """
    Streamlit polls this every ~2 seconds to get the current pipeline state.
    Returns ranked_list, preference_profile, iteration count, and converged flag.
    """
    state = agent_states.get(session_id)
    if state is None:
        return {"ranked_list": [], "converged": False, "iteration": 0,
                "preference_profile": "", "convergence_reason": ""}

    return {
        "ranked_list":        state.get("ranked_list", []),
        "converged":          state.get("converged", False),
        "iteration":          state.get("iteration", 0),
        "preference_profile": state.get("preference_profile", ""),
        "convergence_reason": state.get("convergence_reason", ""),
        "awaiting_feedback":  _is_awaiting_feedback(session_id),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_awaiting_feedback(sid: str) -> bool:
    """True when the graph has produced a ranked_list and is waiting for user feedback."""
    state = agent_states.get(sid, {})
    event = feedback_events.get(sid)
    return bool(
        state.get("ranked_list")
        and not state.get("converged")
        and event is not None
        and not event.is_set()
    )


async def _run_graph(sid: str, initial_state: AgentState) -> None:
    """
    Runs the LangGraph agent asynchronously.
    The collect_feedback node calls `await feedback_events[sid].wait()` internally;
    we inject the event-wait by monkey-patching collect_feedback before graph runs.
    """
    # Patch collect_feedback to actually await the asyncio.Event
    from backend.nodes import feedback as fb_module

    original_collect = fb_module.collect_feedback.__wrapped__ \
        if hasattr(fb_module.collect_feedback, "__wrapped__") \
        else None

    async def _waiting_collect_feedback(state: AgentState) -> AgentState:
        event = feedback_events.get(sid)
        if event:
            await event.wait()
        # state["user_feedback"] has been set by /feedback endpoint
        return state

    # Temporarily override the node function reference used by the graph
    # LangGraph looks up the function at invoke time from the node dict
    # so we need to rebuild or patch at the state level instead.
    # Simpler: run the graph until collect_feedback, then resume manually.

    try:
        log.info(f"[GRAPH] ═══ Phase 1 start  session={sid} ═══")
        log.info("[GRAPH]   Steps: ingest → fetch → enrich → hard_score → soft_score → rank")
        phase1_result = await _run_phase1(sid, initial_state)
        agent_states[sid] = phase1_result
        n = len(phase1_result.get("ranked_list", []))
        log.info(f"[GRAPH] ═══ Phase 1 complete — {n} properties ranked, awaiting feedback ═══")

        # Feedback loop
        iteration = 0
        while not agent_states[sid].get("converged", False):
            iteration += 1
            log.info(f"[GRAPH]   Waiting for user feedback (iteration {iteration})...")
            event = feedback_events.get(sid)
            if event:
                await event.wait()

            log.info(f"[GRAPH] ═══ Phase 2 start  iteration={iteration} ═══")
            current = agent_states[sid]
            phase2_result = await _run_phase2(sid, current)
            agent_states[sid] = phase2_result

            if phase2_result.get("converged"):
                reason = phase2_result.get("convergence_reason", "")
                log.info(f"[GRAPH] ═══ CONVERGED — {reason} ═══")
            else:
                log.info(f"[GRAPH] ═══ Phase 2 complete — re-ranked, awaiting next feedback ═══")

    except Exception as exc:
        log.error(f"[GRAPH] ✗ Unhandled exception in session {sid}: {exc}", exc_info=True)
        if sid in agent_states:
            agent_states[sid]["convergence_reason"] = f"Error: {exc}"
            agent_states[sid]["converged"] = True


async def _run_phase1(sid: str, state: AgentState) -> AgentState:
    """
    Run ingest → fetch → enrich → hard_score → soft_score → rank_and_display.
    Returns state after rank node (ready to show results).
    """
    from backend.nodes.ingest     import ingest_requirements
    from backend.nodes.fetch      import fetch_properties
    from backend.nodes.enrich     import enrich_properties
    from backend.nodes.hard_score import hard_score_node
    from backend.nodes.soft_score import soft_score_node
    from backend.nodes.rank       import rank_and_display

    loop = asyncio.get_event_loop()

    log.info("[GRAPH]   [1/6] ingest_requirements...")
    state = await loop.run_in_executor(None, ingest_requirements, state)
    agent_states[sid] = state

    log.info("[GRAPH]   [2/6] fetch_properties...")
    state = await loop.run_in_executor(None, fetch_properties, state)
    agent_states[sid] = state
    log.info(f"[GRAPH]         → {len(state.get('properties', []))} properties fetched")

    log.info("[GRAPH]   [3/6] enrich_properties...")
    state = await loop.run_in_executor(None, enrich_properties, state)
    agent_states[sid] = state

    log.info("[GRAPH]   [4/6] hard_score_node...")
    state = await loop.run_in_executor(None, hard_score_node, state)
    agent_states[sid] = state

    log.info("[GRAPH]   [5/6] soft_score_node...")
    state = await loop.run_in_executor(None, soft_score_node, state)
    agent_states[sid] = state

    log.info("[GRAPH]   [6/6] rank_and_display...")
    state = await loop.run_in_executor(None, rank_and_display, state)
    agent_states[sid] = state

    return state


async def _run_phase2(sid: str, state: AgentState) -> AgentState:
    """
    Run update_weights → [if not converged: soft_score → rank].
    Returns updated state after ranking.
    """
    from backend.nodes.update_weights import update_weights
    from backend.nodes.soft_score     import soft_score_node
    from backend.nodes.rank           import rank_and_display

    loop = asyncio.get_event_loop()

    log.info("[GRAPH]   [1/3] update_weights...")
    state = await loop.run_in_executor(None, update_weights, state)
    agent_states[sid] = state

    if not state.get("converged", False):
        log.info("[GRAPH]   [2/3] soft_score_node...")
        state = await loop.run_in_executor(None, soft_score_node, state)
        agent_states[sid] = state

        log.info("[GRAPH]   [3/3] rank_and_display...")
        state = await loop.run_in_executor(None, rank_and_display, state)
        agent_states[sid] = state
    else:
        log.info("[GRAPH]   Converged — skipping re-score and re-rank")

    return state
