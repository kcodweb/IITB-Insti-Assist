"""
Finish node.

Not really an "agent" — just the terminal step that packages whatever
the graph produced into a clean final_answer, handling the three
possible endings:
  1. Nothing grounded was found -> polite "I don't know".
  2. Draft got approved by the critic -> return it with sources.
  3. Draft exhausted its revision budget without approval -> return it
     but flagged as unverified, so the user isn't misled into thinking
     it was fully checked.
"""

from __future__ import annotations

from src.state import AgentState

REFUSAL_MESSAGE = (
    "I don't know. I couldn't find anything in my academic policy documents "
    "that answers this question. I only answer from the documents I've been "
    "given, so I won't guess. Try rephrasing, or ask an academic office / "
    "the institute website for anything outside course registration, "
    "grading, the academic calendar, or exam rules."
)


def finish_node(state: AgentState) -> AgentState:
    history = state.get("history", [])

    if not state.get("grounded", False):
        final = REFUSAL_MESSAGE
        history.append("[finish] no grounded context found — returning refusal")

    elif state.get("approved", False):
        sources = ", ".join(state.get("sources_used", [])) or "unknown"
        final = f"{state['draft_answer']}\n\n(Verified against: {sources})"
        history.append("[finish] draft approved by critic — returning final answer")

    else:
        sources = ", ".join(state.get("sources_used", [])) or "unknown"
        final = (
            f"{state['draft_answer']}\n\n"
            f"⚠️ Note: this answer went through {state.get('revision_count', 0)} "
            f"revision(s) but the reviewer still had unresolved concerns: "
            f"\"{state.get('critique', '')}\". Verify important details against "
            f"the source document(s): {sources}."
        )
        history.append("[finish] revision budget exhausted without approval")

    return {**state, "final_answer": final, "history": history}
