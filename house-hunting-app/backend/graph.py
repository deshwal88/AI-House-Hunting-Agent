"""
LangGraph assembly — wires all 8 nodes into a stateful directed graph
with a feedback loop.

Loop re-entry point: soft_score (Node 5) — NOT fetch_properties.
Properties are fetched only once; only scoring and ranking update per iteration.

Graph flow:
  ingest → fetch → enrich → hard_score → soft_score → rank → feedback → update_weights
                                                 ↑                              |
                                                 └──── (not converged) ─────────┘
                                                              ↓ (converged)
                                                             END
"""

from langgraph.graph import StateGraph, END

from backend.state import AgentState
from backend.nodes.ingest        import ingest_requirements
from backend.nodes.fetch         import fetch_properties
from backend.nodes.enrich        import enrich_properties
from backend.nodes.hard_score    import hard_score_node
from backend.nodes.soft_score    import soft_score_node
from backend.nodes.rank          import rank_and_display
from backend.nodes.feedback      import collect_feedback
from backend.nodes.update_weights import update_weights


# ── Conditional edge: check convergence ───────────────────────────────────────
def check_convergence(state: AgentState) -> str:
    return "converged" if state.get("converged", False) else "continue"


# ── Build and compile the graph ───────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("ingest_requirements", ingest_requirements)
    graph.add_node("fetch_properties",    fetch_properties)
    graph.add_node("enrich_properties",   enrich_properties)
    graph.add_node("hard_score",          hard_score_node)
    graph.add_node("soft_score",          soft_score_node)
    graph.add_node("rank_and_display",    rank_and_display)
    graph.add_node("collect_feedback",    collect_feedback)
    graph.add_node("update_weights",      update_weights)

    graph.set_entry_point("ingest_requirements")

    graph.add_edge("ingest_requirements", "fetch_properties")
    graph.add_edge("fetch_properties",    "enrich_properties")
    graph.add_edge("enrich_properties",   "hard_score")
    graph.add_edge("hard_score",          "soft_score")
    graph.add_edge("soft_score",          "rank_and_display")
    graph.add_edge("rank_and_display",    "collect_feedback")
    graph.add_edge("collect_feedback",    "update_weights")

    graph.add_conditional_edges(
        "update_weights",
        check_convergence,
        {
            "converged": END,
            "continue":  "soft_score",   # loop back — skips re-fetching properties
        },
    )

    return graph.compile()


# Module-level compiled graph instance (imported by main.py)
agent_graph = build_graph()
