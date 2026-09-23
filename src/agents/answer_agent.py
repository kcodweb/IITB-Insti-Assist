"""
Answer agent.

Responsibility: draft an answer USING ONLY the retrieved chunks, citing
the numbered passage behind every claim inline ([1], [2]...). It never
uses outside knowledge. If the critic sends it back with feedback, it
revises its previous draft taking that feedback into account.
"""

from __future__ import annotations

from src import config
from src.state import AgentState
from src.utils import llm as llm_utils
from src.utils.grounding import format_context, format_history, parse_citations

SYSTEM_PROMPT = """You are the Answer Agent inside IITB Insti-Assist, a RAG \
assistant for IIT Bombay's academic rules (UG Rules & Regulations) and \
academic calendar.

Hard rules:
1. Answer ONLY using the numbered CONTEXT passages. Never use outside knowledge \
about IIT Bombay or anything else, even if you think you know the answer.
2. Cite the passage(s) behind every factual sentence inline, like [1] or [2][3]. \
Don't add a separate "Sources" list — the app shows sources itself.
3. If the context only partly answers the question, answer that part and say \
plainly what the documents don't specify. If it doesn't answer it at all, say \
"The documents I have don't cover this." and nothing else.
4. Copy numbers, dates and names exactly as the context gives them, and keep \
track of which semester (Autumn / Spring / Summer), programme or category a \
rule or date belongs to — the passage headers in [brackets] tell you.
5. Be concise: lead with the direct answer in 1-3 sentences; use a short \
bullet list only when there are several conditions, steps or dates.
"""


def answer_agent_node(state: AgentState) -> AgentState:
    question = state["question"]
    search_query = state.get("search_query") or question
    chunks = state.get("retrieved_chunks", [])
    revision_count = state.get("revision_count", 0)
    is_revision = "draft_answer" in state and state.get("critique")

    prompt = ""
    if state.get("chat_history"):
        prompt += (
            "CONVERSATION SO FAR (only for understanding the question — facts must "
            f"still come from the CONTEXT):\n{format_history(state['chat_history'], config.HISTORY_WINDOW)}\n\n"
        )
    prompt += f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}"
    if search_query != question:
        prompt += f"\n(Interpreted as: {search_query})"
    if is_revision:
        prompt += (
            f"\n\nYOUR PREVIOUS DRAFT:\n{state['draft_answer']}\n\n"
            f"A reviewer rejected it: \"{state['critique']}\"\n"
            "Rewrite the answer to fix that problem, still using only the CONTEXT."
        )

    llm = llm_utils.get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    draft = llm_utils.extract_text(response).strip()

    if is_revision:
        revision_count += 1
        line = f"[answer_agent] revised draft (revision #{revision_count})"
    else:
        line = "[answer_agent] wrote initial draft"

    return {
        "draft_answer": draft,
        "citations": [n for n in parse_citations(draft) if 1 <= n <= len(chunks)],
        "draft_version": state.get("draft_version", 0) + 1,
        "revision_count": revision_count,
        "history": [line],
    }
