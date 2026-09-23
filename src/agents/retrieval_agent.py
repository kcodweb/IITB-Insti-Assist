"""
Retrieval agent.

Responsibility: given the user's question, query the vector store and
decide whether we actually found grounded, relevant context. This is
the ONLY agent that talks to the vector store — the answer agent and
critic agent only ever see the chunks this agent hands them.

For follow-up questions ("what about for Spring?") it first rewrites the
question into a standalone search query using the conversation so far;
searching for "what about for Spring?" on its own would find nothing.
"""

from __future__ import annotations

from src import config
from src.rag.retriever import retrieve
from src.state import AgentState
from src.utils import llm as llm_utils
from src.utils.grounding import format_history

REWRITE_PROMPT = """You rewrite follow-up questions for a search engine over \
IIT Bombay's UG academic rulebook and academic calendar.

Using the conversation, rewrite the user's LATEST question as one standalone \
search query: resolve references like "it", "that", "those dates" or "what \
about Spring?", and keep every specific (semester, programme, grade, course \
code, number). If the question is already standalone, return it unchanged.

Reply with the query only — no quotes, no explanation."""


def _standalone_query(question: str, chat_history: list[dict]) -> str:
    llm = llm_utils.get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {
                "role": "user",
                "content": f"CONVERSATION:\n{format_history(chat_history, config.HISTORY_WINDOW)}"
                f"\n\nLATEST QUESTION: {question}",
            },
        ]
    )
    lines = llm_utils.extract_text(response).strip().splitlines()
    query = lines[0].strip().strip('"').strip() if lines else ""
    return query or question


def retrieval_agent_node(state: AgentState) -> AgentState:
    question = state["question"]
    trace: list[str] = []

    query = question
    if state.get("chat_history"):
        try:
            query = _standalone_query(question, state["chat_history"])
        except Exception as e:  # a failed rewrite shouldn't sink the whole answer
            trace.append(f"[retrieval_agent] couldn't rewrite follow-up ({type(e).__name__}); searching as asked")
        if query != question:
            trace.append(f'[retrieval_agent] rewrote follow-up as: "{query}"')

    chunks = retrieve(query)
    grounded = len(chunks) > 0
    trace.append(f"[retrieval_agent] found {len(chunks)} relevant chunk(s) (grounded={grounded})")

    return {
        "search_query": query,
        "retrieved_chunks": chunks,
        "grounded": grounded,
        "history": trace,
    }
