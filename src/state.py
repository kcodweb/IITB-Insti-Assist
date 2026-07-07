"""
Shared "blackboard" state passed between every node in the graph.

This is the standard way to implement a Supervisor-orchestrated
multi-agent system in LangGraph: one TypedDict is threaded through the
whole run, each agent reads the fields it needs and writes back only
the fields it owns, and the supervisor node is the only one allowed to
set `next` (control flow).
"""

from __future__ import annotations

from typing import List, Optional, TypedDict


class RetrievedChunk(TypedDict, total=False):
    text: str
    source: str      # file name of the source document
    chunk_id: str     # e.g. "grading_policy.txt#chunk-2" or "policy.pdf#p3-chunk-0"
    score: float      # similarity score from the vector store
    page: int         # present only for PDF-derived chunks


class AgentState(TypedDict, total=False):
    # --- input ------------------------------------------------------------
    question: str

    # --- retrieval_agent output --------------------------------------------
    retrieved_chunks: List[RetrievedChunk]
    grounded: bool  # False if nothing relevant enough was retrieved

    # --- answer_agent output -------------------------------------------------
    draft_answer: str
    sources_used: List[str]
    draft_version: int  # incremented every time answer_agent writes a draft

    # --- critic_agent output ---------------------------------------------------
    critique: str
    approved: bool
    critiqued_version: int  # which draft_version the critic last reviewed
    # NOTE: LangGraph merges a node's returned dict into state key-by-key —
    # omitting a key from the return value does NOT delete it, it just
    # leaves the previous value in place. So rather than trying to "clear"
    # approved/critique after a revision, we compare draft_version to
    # critiqued_version to know whether the CURRENT draft has been reviewed
    # yet, instead of relying on key presence/absence.

    # --- control flow (owned only by supervisor_agent) ---------------------------
    next: str              # "RETRIEVE" | "ANSWER" | "CRITIQUE" | "FINISH"
    revision_count: int
    max_revisions: int
    history: List[str]     # human-readable trace, one entry per node run

    # --- final output -----------------------------------------------------------
    final_answer: Optional[str]
