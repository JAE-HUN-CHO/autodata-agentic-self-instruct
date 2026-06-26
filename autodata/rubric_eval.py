"""Rubric-based judge: the core measurement primitive (Sec 3.1).

Given a solver answer and a rubric, returns a normalized score in [0, 1]:
    (sum satisfied positive weights - sum triggered negative weights) / (sum positive weights),
clamped to [0, 1]. The judge LLM decides per-criterion satisfaction; for robustness we accept
either a full per-criterion map or a direct normalized_score from the model and recompute when
per-criterion data is present.
"""
from __future__ import annotations

import json

from .schemas import QAItem, SolverEval
from .prompts import JUDGE_SYSTEM, judge_user_prompt
from .llm import LLMProvider


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (handles code fences / preamble)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in judge response: {text[:120]!r}")
    return json.loads(text[start:end + 1])


def _score_from_per_criterion(qa: QAItem, marks: dict[str, int]) -> float:
    earned = 0
    penalty = 0
    for c in qa.rubric:
        satisfied = int(marks.get(c.criterion, 0))
        if not satisfied:
            continue
        if c.is_positive:
            earned += c.weight
        else:
            penalty += abs(c.weight)
    raw = (earned - penalty) / qa.max_positive_weight()
    return max(0.0, min(1.0, raw))


class RubricJudge:
    def __init__(self, provider: LLMProvider, temperature: float = 0.0):
        self.provider = provider
        self.temperature = temperature

    def evaluate(self, qa: QAItem, solver_answer: str) -> SolverEval:
        raw = self.provider.complete(
            system=JUDGE_SYSTEM,
            user=judge_user_prompt(qa, solver_answer),
            temperature=self.temperature,
            json_mode=True,
        )
        data = _extract_json(raw)

        per_criterion: dict[str, int] = {}
        if isinstance(data.get("per_criterion"), dict):
            per_criterion = {k: int(v) for k, v in data["per_criterion"].items()}
            score = _score_from_per_criterion(qa, per_criterion)
        elif "normalized_score" in data:
            score = max(0.0, min(1.0, float(data["normalized_score"])))
        else:
            raise ValueError(f"judge returned neither per_criterion nor normalized_score: {data}")

        return SolverEval(score=score, per_criterion=per_criterion, raw_answer=solver_answer)
