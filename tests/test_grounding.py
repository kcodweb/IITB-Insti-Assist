from __future__ import annotations

import pytest

from src.agents.critic_agent import parse_verdict
from src.utils.grounding import parse_citations, unsupported_numbers


@pytest.mark.parametrize(
    "text, expected",
    [
        ("No citations here.", []),
        ("80% is required [1].", [1]),
        ("Dates differ [2][3], see also [2].", [2, 3]),
        ("Both apply [1, 3].", [1, 3]),
    ],
)
def test_parse_citations(text, expected):
    assert parse_citations(text) == expected


def test_numbers_present_in_context_or_question_are_fine():
    context = "Attendance below 80% ... CPI of at least 7.0 ... 12 Sep 2026"
    assert unsupported_numbers("You need 80% [1] and a CPI of 7 [2] by 12 Sep 2026.", context) == []
    assert unsupported_numbers("For semester 3 you need 80%.", context, question="semester 3?") == []


def test_invented_numbers_are_flagged_but_citation_markers_are_not():
    context = "Attendance below 80% leads to DX."
    assert unsupported_numbers("You need 75% attendance [4].", context) == ["75"]


@pytest.mark.parametrize(
    "raw, approved",
    [
        ('{"approved": true, "critique": ""}', True),
        ('```json\n{"approved": true, "critique": ""}\n```', True),
        ('Here is my verdict:\n{"approved": false, "critique": "Wrong date."}', False),
        ('{"approved": "true", "critique": ""}', True),
        ('{"approved": "false", "critique": "x"}', False),  # a truthy string must not approve
        ('{"approved": 1}', False),
        ("I think it's fine", False),
        ("", False),
    ],
)
def test_parse_verdict(raw, approved):
    ok, critique = parse_verdict(raw)
    assert ok is approved
    if not ok:
        assert critique  # a rejection always explains itself


def test_parse_verdict_keeps_the_critique():
    assert parse_verdict('{"approved": false, "critique": "Wrong semester."}') == (False, "Wrong semester.")
