"""
Answer agent.

Responsibility: draft an answer USING ONLY the retrieved chunks. It
never uses outside knowledge, and it must cite which source(s) it drew
from. If the critic sends it back with feedback, it revises the draft
taking that feedback into account.
"""

from __future__ import annotations

from src.state import AgentState
from src.utils.llm import get_chat_model, extract_text

SYSTEM_PROMPT = """You are the Answer Agent inside IITB Insti-Assist, a RAG \
assistant for IIT Bombay's academic policies (course registration, grading, \
academic calendar, exam rules).

Hard rules:
1. Answer ONLY using the CONTEXT provided below. Never use outside knowledge \
about IIT Bombay or any other institute, even if you think you know the answer.
2. If the context does not fully answer the question, say so explicitly \
rather than filling gaps with assumptions.
3. Every factual claim you make must be traceable to one of the context \
chunks. Mention the source file name(s) you relied on at the end, under \
"Sources:".
4. Be concise and direct — this is a policy lookup assistant, not an essay.
"""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        page_note = f", page {c['page']}" if "page" in c else ""
        parts.append(f"[{c['chunk_id']}] (source: {c['source']}{page_note})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def answer_agent_node(state: AgentState) -> AgentState:
    history = state.get("history", [])
    question = state["question"]
    chunks = state.get("retrieved_chunks", [])
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    context = _format_context(chunks)
    sources_used = sorted({c["source"] for c in chunks})

    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    if critique:
        user_prompt += (
            f"\n\nA reviewer flagged this issue with your previous draft: "
            f"\"{critique}\"\nRevise your answer to fix that issue, still using "
            f"only the CONTEXT above."
        )

    llm = get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    draft = extract_text(response)

    draft_version = state.get("draft_version", 0) + 1

    if critique:
        revision_count += 1
        history.append(f"[answer_agent] revised draft (revision #{revision_count})")
    else:
        history.append("[answer_agent] wrote initial draft")

    return {
        **state,
        "draft_answer": draft,
        "sources_used": sources_used,
        "draft_version": draft_version,
        "revision_count": revision_count,
        "history": history,
    }
