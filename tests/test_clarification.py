"""
The vague-question gate.

This runs before every dashboard question, so a false positive costs the user
an answer they asked for. The heuristic stage must never call the LLM: these
tests would fail with a network error if it did.
"""

import pytest

from src.rag.clarification import is_vague


@pytest.mark.parametrize("question", [
    "What is our average order value in 2022?",
    "How does our return rate compare to the industry?",
    "Top 5 product categories by revenue in 2021",
])
def test_specific_questions_are_not_blocked(question):
    # Each of these is under the 10-word LLM-bypass threshold, so they reach
    # stage 2 and the LLM classified them as vague. With use_llm=False they run.
    vague, _ = is_vague(question, use_llm=False)
    assert vague is False


@pytest.mark.parametrize("question", [
    "tell me more",
    "what about risks?",
    "elaborate",
])
def test_known_vague_openers_are_caught_without_an_llm(question):
    vague, prompt = is_vague(question, use_llm=False)
    assert vague is True
    assert prompt


def test_a_short_follow_up_with_history_is_allowed_through():
    # "How does that compare?" after a revenue answer resolves fine.
    vague, _ = is_vague("what about 2023?",
                        chat_history="Human: revenue in 2022?\nAI: 1.2M",
                        use_llm=False)
    assert vague is False
