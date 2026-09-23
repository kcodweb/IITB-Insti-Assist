"""
End-to-end routing tests for the Supervisor graph, with a scripted LLM.

These are the scenarios that caught the stale-verdict infinite loop
described in WRITEUP.md: happy path, refusal when ungrounded, one
revision then approval, and the revision budget running out.
"""

from __future__ import annotations

from src.graph import run, stream

APPROVE = '{"approved": true, "critique": ""}'
REJECT = '{"approved": false, "critique": "The 75% figure is not in the context."}'


def routes(state) -> list[str]:
    return [line.split("-> routing to ")[1] for line in state["history"] if "routing to" in line]


def test_happy_path_is_verified_with_citations(fake_llm, fake_retrieval):
    fake_retrieval()
    fake_llm(answer=["You need at least 80% attendance to avoid a DX grade [1]."], critic=[APPROVE])

    state = run("What attendance do I need to avoid DX?")

    assert routes(state) == ["RETRIEVE", "ANSWER", "CRITIQUE", "FINISH"]
    assert state["status"] == "verified"
    assert state["citations"] == [1]
    assert state["final_answer"].startswith("You need at least 80%")


def test_refuses_without_calling_the_llm_when_nothing_is_grounded(fake_llm, fake_retrieval):
    fake_retrieval(chunks=[])
    llm = fake_llm()

    state = run("What is the capital of France?")

    assert routes(state) == ["RETRIEVE", "FINISH"]
    assert state["status"] == "refused"
    assert "I don't know" in state["final_answer"]
    assert llm.calls == []


def test_rejected_draft_is_revised_then_approved(fake_llm, fake_retrieval):
    fake_retrieval()
    llm = fake_llm(
        answer=["You need 75% attendance [1].", "You need at least 80% attendance [1]."],
        critic=[REJECT, APPROVE],
    )

    state = run("What attendance do I need?")

    assert routes(state) == ["RETRIEVE", "ANSWER", "CRITIQUE", "ANSWER", "CRITIQUE", "FINISH"]
    assert state["status"] == "verified"
    assert state["revision_count"] == 1
    # The revision prompt shows the answer agent its own draft and the critique.
    revision_prompt = llm.calls_for("answer")[1][1]["content"]
    assert "You need 75% attendance [1]." in revision_prompt
    assert "The 75% figure is not in the context." in revision_prompt


def test_revision_budget_is_respected(fake_llm, fake_retrieval):
    fake_retrieval()
    llm = fake_llm(answer=["You need 75% attendance [1]."], critic=[REJECT])

    state = run("What attendance do I need?", max_revisions=2)

    assert len(llm.calls_for("answer")) == 3  # initial draft + 2 revisions
    assert routes(state)[-1] == "FINISH"
    assert state["status"] == "unverified"
    assert "reviewer still had a concern" in state["final_answer"]


def test_invalid_citation_is_rejected_by_rule_without_an_llm_call(fake_llm, fake_retrieval):
    fake_retrieval()
    llm = fake_llm(answer=["80% attendance is required [7].", "80% attendance is required [1]."], critic=[APPROVE])

    state = run("What attendance do I need?")

    assert state["status"] == "verified"
    assert len(llm.calls_for("critic")) == 1  # only the second draft reached the LLM critic
    assert any("(rule check)" in line and "[7]" in line for line in state["history"])


def test_unsupported_numbers_are_flagged_to_the_critic(fake_llm, fake_retrieval):
    fake_retrieval()
    llm = fake_llm(answer=["You need 75% attendance [1]."], critic=[APPROVE])

    run("What attendance do I need?")

    critic_prompt = llm.calls_for("critic")[0][1]["content"]
    assert "AUTOMATED CHECKS" in critic_prompt and "75" in critic_prompt


def test_follow_up_questions_are_rewritten_before_searching(fake_llm, fake_retrieval):
    queries = fake_retrieval()
    fake_llm(
        rewrite=["When are DX grades awarded for full-semester courses in the Spring semester?"],
        answer=["DX grades are awarded ... [2]."],
        critic=[APPROVE],
    )
    history = [
        {"role": "user", "content": "When are DX grades awarded in Autumn?"},
        {"role": "assistant", "content": "9-11 Sep 2026 for first half-semester courses [2]."},
    ]

    state = run("what about spring?", chat_history=history)

    assert queries == ["When are DX grades awarded for full-semester courses in the Spring semester?"]
    assert state["search_query"] == queries[0]
    assert any("rewrote follow-up" in line for line in state["history"])


def test_first_question_is_not_rewritten(fake_llm, fake_retrieval):
    queries = fake_retrieval()
    llm = fake_llm(answer=["80% [1]."], critic=[APPROVE])

    run("What attendance do I need?")

    assert queries == ["What attendance do I need?"]
    assert llm.calls_for("rewrite") == []


def test_stream_yields_each_step_and_ends_with_the_final_state(fake_llm, fake_retrieval):
    fake_retrieval()
    fake_llm(answer=["80% [1]."], critic=[APPROVE])

    steps = list(stream("What attendance do I need?"))
    streamed_lines = [line for lines, _ in steps for line in lines]
    final = steps[-1][1]

    assert final["status"] == "verified"
    assert streamed_lines == final["history"]
    assert sum(1 for lines, _ in steps if lines) >= 8  # one update per node run
