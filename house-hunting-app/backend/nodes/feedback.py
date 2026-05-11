"""
Node 7 — collect_feedback

Waits for the user to submit a drag-and-drop reordering of the ranked
properties in Streamlit. The FastAPI /feedback endpoint receives the
user's ordered list of property_ids and signals this node to resume
via an asyncio.Event.

The event and feedback data are stored in shared dicts in main.py and
injected into the state before this node is called.
"""

from backend.state import AgentState


async def collect_feedback(state: AgentState) -> AgentState:
    """
    This node is a no-op at the graph level — the actual waiting is done
    by the FastAPI endpoint layer.  By the time this node executes,
    state["user_feedback"] has already been populated by the /feedback
    endpoint (via the feedback_events asyncio.Event mechanism in main.py).
    """
    # user_feedback is already set; nothing to do here.
    return state
