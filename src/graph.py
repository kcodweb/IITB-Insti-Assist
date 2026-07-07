"""
Builds the LangGraph StateGraph implementing the Supervisor pattern:

                     ┌────────────┐
        ┌───────────▶│ supervisor │◀─────────────┐
        │            └─────┬──────┘              │
        │        RETRIEVE  │ANSWER│CRITIQUE│FINISH│
        │                  ▼      ▼        ▼      │
        │           ┌──────────┐┌────────┐┌───────┐
        │           │retrieval ││ answer ││critic │
        │           │  agent   ││ agent  ││ agent │
        │           └────┬─────┘└───┬────┘└───┬───┘
        └────────────────┴──────────┴─────────┘
                                                  │
                                                  ▼
                                              ┌───────┐
                                              │finish │
                                              └───────┘

Every worker node routes back to the supervisor when it's done; the
supervisor is the only node that decides where to go next. This keeps
each worker agent single-purpose and makes the control flow easy to
trace via state["history"].
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents.supervisor import supervisor_node
from src.agents.retrieval_agent import retrieval_agent_node
from src.agents.answer_agent import answer_agent_node
from src.agents.critic_agent import critic_agent_node
from src.agents.finish_agent import finish_node


def _route(state: AgentState) -> str:
    return state["next"]


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("answer_agent", answer_agent_node)
    graph.add_node("critic_agent", critic_agent_node)
    graph.add_node("finish", finish_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route,
        {
            "RETRIEVE": "retrieval_agent",
            "ANSWER": "answer_agent",
            "CRITIQUE": "critic_agent",
            "FINISH": "finish",
        },
    )

    # Every worker reports back to the supervisor for the next decision.
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("answer_agent", "supervisor")
    graph.add_edge("critic_agent", "supervisor")
    graph.add_edge("finish", END)

    return graph.compile()


def run(question: str, max_revisions: int = 2) -> AgentState:
    app = build_graph()
    initial_state: AgentState = {
        "question": question,
        "revision_count": 0,
        "max_revisions": max_revisions,
        "history": [],
    }
    final_state = app.invoke(initial_state)
    return final_state
