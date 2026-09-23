"""
Finish node.

Not really an "agent" — just the terminal step that packages whatever
the graph produced into a clean final_answer + status, handling the
three possible endings:
  1. Nothing grounded was found -> polite "I don't know"  (status "refused")
  2. Draft got approved by the critic -> return it         (status "verified")
  3. Draft exhausted its revision budget without approval -> return it
     but flagged, so the user isn't misled into thinking it was fully
     checked                                               (status "unverified")

Sources aren't pasted into the answer text: the draft cites passages
inline ([1], [2]) and the UI/CLI render `citations` against
`retrieved_chunks` with document names and page numbers.
"""

from __future__ import annotations

from src.rag.ingest import load_source_info
from src.state import AgentState


def refusal_message() -> str:
    titles = [v.get("title", k) for k, v in load_source_info().items()]
    covered = " and the ".join(titles) if titles else "my academic policy documents"
    return (
        "I don't know. I couldn't find anything that answers this in the documents "
        f"I've been given (the {covered}), and I only answer from those documents, "
        "so I won't guess. Try rephrasing, or check with the Academic Office for "
        "anything outside course registration, grading, exams and the academic calendar."
    )


def finish_node(state: AgentState) -> AgentState:
    if not state.get("grounded", False):
        return {
            "final_answer": refusal_message(),
            "status": "refused",
            "history": ["[finish] no grounded context found — returning refusal"],
        }

    if state.get("approved", False):
        return {
            "final_answer": state["draft_answer"],
            "status": "verified",
            "history": ["[finish] draft approved by critic — returning final answer"],
        }

    return {
        "final_answer": (
            f"{state['draft_answer']}\n\n"
            f"⚠️ This answer went through {state.get('revision_count', 0)} revision(s), "
            f"but the reviewer still had a concern: \"{state.get('critique', '')}\" "
            "Please verify important details against the cited pages."
        ),
        "status": "unverified",
        "history": ["[finish] revision budget exhausted without approval"],
    }
