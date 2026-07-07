"""
Supervisor agent.

This is the orchestrator in our Supervisor pattern. It never touches
the vector store or drafts an answer itself — its only job is to look
at the current state and decide which worker node runs next, by
setting `state["next"]`.

The routing logic here is deterministic (plain Python), not an LLM
call, because the control-flow rules are simple and we want them to be
100% predictable and cheap. (You could swap this for an LLM-based
router — see README "Extending this project" — but a rules-based
supervisor is easier to reason about and debug for a graded project.)
"""

from __future__ import annotations

from src.state import AgentState

MAX_REVISIONS_DEFAULT = 2


def supervisor_node(state: AgentState) -> AgentState:
    history = state.get("history", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", MAX_REVISIONS_DEFAULT)

    # Step 1: nothing retrieved yet -> go retrieve.
    if "retrieved_chunks" not in state:
        next_node = "RETRIEVE"

    # Step 2: retrieval happened but found nothing grounded -> refuse and stop.
    elif not state.get("grounded", False):
        next_node = "FINISH"

    # Step 3: we have grounded chunks but no draft yet -> write one.
    elif "draft_answer" not in state:
        next_node = "ANSWER"

    # Step 4: we have a draft, but the critic hasn't reviewed *this* version yet.
    elif state.get("draft_version", 0) != state.get("critiqued_version", -1):
        next_node = "CRITIQUE"

    # Step 5: critic rejected the current draft and we still have revision budget -> rewrite.
    elif not state.get("approved", False) and revision_count < max_revisions:
        next_node = "ANSWER"

    # Step 6: approved, or out of revision budget -> done.
    else:
        next_node = "FINISH"

    history.append(f"[supervisor] -> routing to {next_node}")
    return {**state, "next": next_node, "history": history, "max_revisions": max_revisions}
