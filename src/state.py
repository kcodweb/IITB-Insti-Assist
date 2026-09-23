"""
Shared "blackboard" state passed between every node in the graph.

This is the standard way to implement a Supervisor-orchestrated
multi-agent system in LangGraph: one TypedDict is threaded through the
whole run, each agent reads the fields it needs and returns only the
fields it owns, and the supervisor node is the only one allowed to set
`next` (control flow).
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RetrievedChunk(TypedDict, total=False):
    text: str         # "[document | section]" header + chunk text
    source: str       # file name of the source document
    title: str        # human-friendly document title
    section: str      # e.g. "6 EXAMINATION / ASSESSMENT > 6.5 Grading"
    chunk_id: str     # e.g. "ugrulebook.pdf#p28-chunk-0"
    score: Optional[float]  # dense cosine similarity (None for keyword-only hits)
    page: int         # present only for PDF-derived chunks


class AgentState(TypedDict, total=False):
    # --- input ------------------------------------------------------------
    question: str
    chat_history: List[ChatMessage]  # earlier turns, oldest first

    # --- retrieval_agent output --------------------------------------------
    search_query: str  # standalone version of the question (follow-ups resolved)
    retrieved_chunks: List[RetrievedChunk]
    grounded: bool  # False if nothing relevant enough was retrieved

    # --- answer_agent output -------------------------------------------------
    draft_answer: str
    citations: List[int]  # 1-based indices into retrieved_chunks cited by the draft
    draft_version: int  # incremented every time answer_agent writes a draft
    revision_count: int

    # --- critic_agent output ---------------------------------------------------
    approved: bool
    critique: str
    critiqued_version: int  # which draft_version the critic last reviewed
    # NOTE: LangGraph merges a node's returned dict into state key-by-key —
    # omitting a key from the return value does NOT delete it, it just
    # leaves the previous value in place. So rather than trying to "clear"
    # approved/critique after a revision, we compare draft_version to
    # critiqued_version to know whether the CURRENT draft has been reviewed
    # yet, instead of relying on key presence/absence.

    # --- control flow (owned only by supervisor_agent) ---------------------------
    next: str  # "RETRIEVE" | "ANSWER" | "CRITIQUE" | "FINISH"
    max_revisions: int
    # Human-readable trace. The reducer means each node returns just its own
    # new lines and LangGraph appends them, instead of every node copying
    # and re-returning the whole list.
    history: Annotated[List[str], operator.add]

    # --- final output (finish node) ------------------------------------------------
    final_answer: Optional[str]
    status: Literal["verified", "unverified", "refused"]
