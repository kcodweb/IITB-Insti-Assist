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

from functools import lru_cache
from typing import Iterator

from langgraph.graph import END, StateGraph

from src import config
from src.agents.answer_agent import answer_agent_node
from src.agents.critic_agent import critic_agent_node
from src.agents.finish_agent import finish_node
from src.agents.retrieval_agent import retrieval_agent_node
from src.agents.supervisor import supervisor_node
from src.state import AgentState, ChatMessage


def _route(state: AgentState) -> str:
    return state["next"]


@lru_cache(maxsize=1)
def build_graph():
    """Compile the graph once per process; it's stateless between runs."""
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


def _initial_state(
    question: str, chat_history: list[ChatMessage] | None, max_revisions: int
) -> AgentState:
    return {
        "question": question,
        "chat_history": list(chat_history or []),
        "revision_count": 0,
        "max_revisions": max_revisions,
        "history": [],
    }


def run(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    max_revisions: int = config.MAX_REVISIONS,
) -> AgentState:
    """Answer one question (optionally as a follow-up in a conversation)."""
    return build_graph().invoke(_initial_state(question, chat_history, max_revisions))


def stream(
    question: str,
    chat_history: list[ChatMessage] | None = None,
    max_revisions: int = config.MAX_REVISIONS,
) -> Iterator[tuple[list[str], AgentState]]:
    """
    Like run(), but yields (new_trace_lines, state) after every node so a
    UI can show the agents working live. The last state yielded is final.
    """
    seen = 0
    for state in build_graph().stream(
        _initial_state(question, chat_history, max_revisions), stream_mode="values"
    ):
        trace = state.get("history", [])
        yield trace[seen:], state
        seen = len(trace)
