"""
Shared fixtures. Nothing here touches the network or a real LLM: the
chat model is replaced by a scripted fake, and retrieval by canned chunks
(or, for the index tests, by a throwaway Chroma store with fake embeddings).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import llm as llm_utils  # noqa: E402

CHUNKS = [
    {
        "text": "[UG Rules & Regulations | 6 EXAMINATION / ASSESSMENT > 6.5 Grading]\n"
        "If attendance falls below eighty percent the instructor may award a DX grade. "
        "A student with 80% attendance or more cannot be awarded DX.",
        "source": "ugrulebook.pdf",
        "title": "UG Rules & Regulations",
        "section": "6 EXAMINATION / ASSESSMENT > 6.5 Grading",
        "chunk_id": "ugrulebook.pdf#p29-chunk-1",
        "score": 0.81,
        "page": 29,
    },
    {
        "text": "[Academic Calendar 2026-27 | (A) SCHEDULE FOR AUTUMN SEMESTER]\n"
        "Award of DX grades for first half-semester courses 9 Sep 2026 (Wednesday) - 11 Sep 2026 (Friday)",
        "source": "Academic_Calendar_2026-27_FINAL.pdf",
        "title": "Academic Calendar 2026-27",
        "section": "(A) SCHEDULE FOR AUTUMN SEMESTER",
        "chunk_id": "Academic_Calendar_2026-27_FINAL.pdf#p4-chunk-1",
        "score": 0.74,
        "page": 4,
    },
]


class ScriptedLLM:
    """
    Stands in for a chat model. Each agent is recognised by its system
    prompt and gets the next scripted reply for its role; every call is
    recorded so tests can assert on what the agents actually sent.
    """

    ROLES = {"Answer Agent": "answer", "Critic Agent": "critic", "rewrite follow-up": "rewrite"}

    def __init__(self, **replies: list[str]):
        self.replies = {role: list(r) for role, r in replies.items()}
        self.calls: list[tuple[str, list[dict]]] = []

    def invoke(self, messages):
        system = messages[0]["content"]
        role = next((r for marker, r in self.ROLES.items() if marker in system), None)
        assert role is not None, f"unrecognised prompt: {system[:80]}"
        self.calls.append((role, messages))
        queue = self.replies.get(role)
        assert queue, f"no scripted reply left for the {role} agent"
        reply = queue.pop(0) if len(queue) > 1 else queue[0]  # last reply repeats
        return type("AIMessage", (), {"content": reply})()

    def calls_for(self, role: str) -> list[list[dict]]:
        return [m for r, m in self.calls if r == role]


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a ScriptedLLM: `llm = fake_llm(answer=[...], critic=[...])`."""

    def install(**replies):
        llm = ScriptedLLM(**replies)
        monkeypatch.setattr(llm_utils, "get_chat_model", lambda temperature=0.0: llm)
        return llm

    return install


@pytest.fixture
def fake_retrieval(monkeypatch):
    """Replace vector search: `queries = fake_retrieval(chunks)` records each query."""

    def install(chunks=CHUNKS):
        queries: list[str] = []

        def retrieve(query, *args, **kwargs):
            queries.append(query)
            return list(chunks)

        monkeypatch.setattr("src.agents.retrieval_agent.retrieve", retrieve)
        return queries

    return install
