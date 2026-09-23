"""
Critic agent (groundedness checker).

Responsibility: read the draft answer next to the retrieved context and
decide if every claim in the draft is actually supported by that
context. This is what prevents hallucination from slipping through —
it's a second, independent check whose only job is to be skeptical of
the first one.

It works in two layers:
  1. Deterministic checks (no LLM): citations must point at passages
     that exist — an invalid one is an automatic rejection — and any
     number in the draft that appears nowhere in the context is flagged.
  2. An LLM review of the draft against the numbered passages, told
     about anything layer 1 flagged.

Outputs a structured verdict: approved (bool) + critique (str).
"""

from __future__ import annotations

import json
import re

from src.state import AgentState
from src.utils import llm as llm_utils
from src.utils.grounding import format_context, parse_citations, unsupported_numbers

SYSTEM_PROMPT = """You are the Critic Agent inside IITB Insti-Assist. You do \
NOT know anything about IIT Bombay yourself. Your only job is to check \
whether the DRAFT ANSWER is fully and only supported by the numbered CONTEXT \
passages it was given — nothing more, nothing less.

Reject the draft if:
- It states any fact, number or date not present in the context (hallucination).
- A citation like [2] is attached to a claim that passage [2] doesn't support.
- It attributes a rule or date to the wrong semester, programme or category.
- It answers a part of the question the context doesn't actually cover, \
without saying so.
- It contradicts the context.

Approve the draft if it sticks strictly to what the context supports, even \
if that means the draft says the documents don't cover the question.

Respond with ONLY a JSON object, no other text, in this exact shape:
{"approved": true or false, "critique": "one sentence: what is wrong and how \
to fix it, or empty string if approved"}
"""


def critic_agent_node(state: AgentState) -> AgentState:
    question = state.get("search_query") or state["question"]
    draft = state["draft_answer"]
    chunks = state.get("retrieved_chunks", [])
    context = format_context(chunks)

    invalid = [n for n in parse_citations(draft) if not 1 <= n <= len(chunks)]
    if invalid:
        approved = False
        critique = (
            f"The draft cites passage(s) {', '.join(f'[{n}]' for n in invalid)}, "
            f"but only [1]-[{len(chunks)}] exist; cite only the passages provided."
        )
        method = "rule check"
    else:
        approved, critique = _llm_review(question, context, draft)
        method = "LLM review"

    line = f"[critic_agent] verdict: {'APPROVED' if approved else 'REJECTED'} ({method})"
    if critique and not approved:
        line += f" — {critique}"

    return {
        "approved": approved,
        "critique": critique,
        "critiqued_version": state.get("draft_version", 0),
        "history": [line],
    }


def _llm_review(question: str, context: str, draft: str) -> tuple[bool, str]:
    prompt = f"QUESTION: {question}\n\nCONTEXT:\n{context}\n\nDRAFT ANSWER:\n{draft}"

    flags = []
    numbers = unsupported_numbers(draft, context, question)
    if numbers:
        flags.append(
            "These numbers in the draft don't appear anywhere in the context: "
            f"{', '.join(numbers)}. They may be legitimate rewordings (e.g. 'eighty' "
            "-> '80'); verify each one."
        )
    if not parse_citations(draft):
        flags.append("The draft has no [n] citations; check every claim especially carefully.")
    if flags:
        prompt += "\n\nAUTOMATED CHECKS:\n- " + "\n- ".join(flags)

    llm = llm_utils.get_chat_model(temperature=0.0)
    response = llm.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    return parse_verdict(llm_utils.extract_text(response))


def parse_verdict(raw: str) -> tuple[bool, str]:
    """
    Parse the critic's JSON verdict, failing safe (reject) if malformed.
    Tolerates code fences / prose around the JSON, and only a real boolean
    true (or the string "true") counts as approval — a model answering
    {"approved": "false"} must not slip through as truthy.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict):
        return False, "Critic response could not be parsed; treated as rejection."

    approved = data.get("approved")
    if isinstance(approved, str):
        approved = approved.strip().lower() == "true"
    approved = approved is True
    critique = str(data.get("critique") or "").strip()
    if not approved and not critique:
        critique = "The draft is not fully supported by the context."
    return approved, critique
