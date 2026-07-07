"""
Critic agent (groundedness checker).

Responsibility: read the draft answer next to the retrieved context and
decide if every claim in the draft is actually supported by that
context. This is what prevents hallucination from slipping through —
it's a second, independent LLM call whose only job is to be skeptical
of the first one.

Outputs a structured verdict: approved (bool) + critique (str).
"""

from __future__ import annotations

import json

from src.state import AgentState
from src.utils.llm import get_chat_model, extract_text

SYSTEM_PROMPT = """You are the Critic Agent inside IITB Insti-Assist. You do \
NOT know anything about IIT Bombay yourself. Your only job is to check \
whether the DRAFT ANSWER is fully and only supported by the CONTEXT it was \
given — nothing more, nothing less.

Reject the draft if:
- It states any fact not present in the context (hallucination).
- It answers a part of the question the context doesn't actually cover, \
without saying so.
- It contradicts the context.

Approve the draft if it sticks strictly to what the context supports, even \
if that means the draft says "the context doesn't specify this."

Respond with ONLY a JSON object, no other text, in this exact shape:
{"approved": true or false, "critique": "one sentence explaining your verdict, \
or empty string if approved"}
"""


def critic_agent_node(state: AgentState) -> AgentState:
    history = state.get("history", [])
    question = state["question"]
    draft = state["draft_answer"]
    chunks = state.get("retrieved_chunks", [])

    context = "\n\n---\n\n".join(c["text"] for c in chunks)
    user_prompt = (
        f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nDRAFT ANSWER:\n{draft}"
    )

    llm = get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    raw = extract_text(response)

    approved, critique = _parse_verdict(raw)

    history.append(
        f"[critic_agent] verdict: {'APPROVED' if approved else 'REJECTED'}"
        + (f" — {critique}" if critique else "")
    )

    return {
        **state,
        "approved": approved,
        "critique": critique,
        "critiqued_version": state.get("draft_version", 0),
        "history": history,
    }


def _parse_verdict(raw: str) -> tuple[bool, str]:
    """Parse the critic's JSON verdict, failing safe (reject) if malformed."""
    text = raw.strip()
    # Strip markdown code fences if the model added them despite instructions.
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json\n", "", 1).replace("json", "", 1)
    try:
        data = json.loads(text)
        return bool(data.get("approved", False)), str(data.get("critique", ""))
    except (json.JSONDecodeError, AttributeError):
        # Fail safe: if we can't parse the critic's verdict, treat it as a
        # rejection with a generic note rather than silently approving.
        return False, "Critic response could not be parsed; treated as rejection."
