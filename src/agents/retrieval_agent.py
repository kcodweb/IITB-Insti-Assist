"""
Retrieval agent.

Responsibility: given the user's question, query the vector store and
decide whether we actually found grounded, relevant context. This is
the ONLY agent that talks to the vector store — the answer agent and
critic agent only ever see the chunks this agent hands them.
"""

from __future__ import annotations

from src.state import AgentState
from src.rag.retriever import retrieve


def retrieval_agent_node(state: AgentState) -> AgentState:
    history = state.get("history", [])
    question = state["question"]

    chunks = retrieve(question)
    grounded = len(chunks) > 0

    history.append(
        f"[retrieval_agent] found {len(chunks)} relevant chunk(s) "
        f"(grounded={grounded})"
    )

    return {
        **state,
        "retrieved_chunks": chunks,
        "grounded": grounded,
        "history": history,
    }
