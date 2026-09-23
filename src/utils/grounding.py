"""
Deterministic helpers shared by the answer and critic agents: how the
retrieved context is shown to the LLM, how inline citations are parsed,
and a cheap rule-based check for numbers that appear in a draft but
nowhere in its sources.
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def format_context(chunks: list[dict]) -> str:
    """Numbered passages, so answers can cite them as [1], [2], ..."""
    parts = []
    for i, c in enumerate(chunks, start=1):
        page = f", page {c['page']}" if "page" in c else ""
        parts.append(f"[{i}] ({c['source']}{page})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def format_history(history: list[dict], window: int) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-window:])


def parse_citations(text: str) -> list[int]:
    """[1], [2, 3] and [2][3] -> [1, 2, 3] (first-seen order, no duplicates)."""
    found: list[int] = []
    for group in _CITATION.findall(text):
        for n in (int(x) for x in group.split(",")):
            if n not in found:
                found.append(n)
    return found


def _normalize(number: str) -> str:
    if "." in number:
        number = number.rstrip("0").rstrip(".")
    return number.lstrip("0") or "0"


def unsupported_numbers(draft: str, context: str, question: str = "") -> list[str]:
    """
    Numbers in the draft that appear neither in the context nor in the
    question ("7.0" and "7" count as the same). Citation markers are
    ignored. A hit isn't proof of a hallucination — the draft may have
    rephrased "eighty percent" as "80%" — so the critic treats these as
    things to double-check rather than automatic failures.
    """
    allowed = {_normalize(n) for n in _NUMBER.findall(context + " " + question)}
    found: list[str] = []
    for n in _NUMBER.findall(_CITATION.sub(" ", draft)):
        if _normalize(n) not in allowed and n not in found:
            found.append(n)
    return found
