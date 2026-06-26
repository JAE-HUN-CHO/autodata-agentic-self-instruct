"""Subagent wrappers: Challenger, Solver, QualityVerifier.

Each pairs a provider with the appropriate prompt and parses the structured output into the
project's dataclasses.
"""
from __future__ import annotations

import json

from .schemas import QAItem, RubricCriterion, QualityVerdict
from .llm import LLMProvider
from . import prompts
from .rubric_eval import _extract_json


class Challenger:
    def __init__(self, provider: LLMProvider, temperature: float = 0.9):
        self.provider = provider
        self.temperature = temperature

    def generate(self, paper_text: str, failures_block: str, round_no: int) -> QAItem:
        raw = self.provider.complete(
            system=prompts.challenger_system_for_round(round_no),
            user=prompts.challenger_user_prompt(paper_text, failures_block, round_no),
            temperature=self.temperature,
            json_mode=True,
        )
        data = _extract_json(raw)
        rubric = [
            RubricCriterion(
                criterion=str(c["criterion"]),
                weight=int(c["weight"]),
                category=str(c.get("category", "positive")),
            )
            for c in data.get("rubric", [])
        ]
        qa = QAItem(
            context=str(data.get("context", "")),
            question=str(data.get("question", "")),
            reference_answer=str(data.get("reference_answer", "")),
            rubric=rubric,
            question_type=str(data.get("question_type", "")),
            reasoning_tags=list(data.get("reasoning_tags", [])),
        )
        # carry the mock difficulty hint if present, else derive from round number
        qa._difficulty = int(data.get("_mock_difficulty", min(100, 30 + (round_no - 1) * 12)))  # type: ignore[attr-defined]
        return qa


class Solver:
    """A solver (weak or strong) parameterized by its provider + sampling."""
    def __init__(self, provider: LLMProvider, temperature: float = 0.7):
        self.provider = provider
        self.temperature = temperature

    def answer(self, qa: QAItem) -> str:
        difficulty = getattr(qa, "_difficulty", 50)
        return self.provider.complete(
            system=prompts.SOLVER_SYSTEM,
            user=prompts.solver_user_prompt(qa, difficulty),
            temperature=self.temperature,
        )


class QualityVerifier:
    def __init__(self, provider: LLMProvider, temperature: float = 0.0):
        self.provider = provider
        self.temperature = temperature

    def check(self, qa: QAItem) -> QualityVerdict:
        raw = self.provider.complete(
            system=prompts.QUALITY_VERIFIER_SYSTEM,
            user=prompts.quality_verifier_user_prompt(qa),
            temperature=self.temperature,
            json_mode=True,
        )
        data = _extract_json(raw)
        overall = str(data.get("overall", "FAIL")).upper()
        checks = {k: str(v) for k, v in data.items() if k.startswith("check_")}
        return QualityVerdict(
            passed=(overall == "PASS"),
            feedback=str(data.get("feedback", "")),
            checks=checks,
        )
