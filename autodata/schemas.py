"""Core data structures for Agentic Self-Instruct.

Faithful to the CS-paper instantiation (Autodata, arXiv 2606.25996v1, Sec 3.1 / App C.1):
an example is a (context, question, reference_answer, weighted rubric) tuple, and the unit
of measurement is a normalized rubric score in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import json


# ---------------------------------------------------------------------------
# Failure modes used to give the challenger targeted feedback (Sec 3.1).
# ---------------------------------------------------------------------------
TOO_EASY = "TOO_EASY"            # weak solver scored too high -> not discriminative
FAILED_STRONG = "FAILED_STRONG"  # strong solver too low / gap too small
FAILED_QV = "FAILED_QV"          # quality verifier rejected the package
ACCEPTED = "ACCEPTED"
REJECTED = "REJECTED"


@dataclass
class RubricCriterion:
    criterion: str
    weight: int                          # positive for positive criteria, negative for negative
    category: str = "positive"           # "positive" | "negative"

    @property
    def is_positive(self) -> bool:
        return self.category == "positive"


@dataclass
class QAItem:
    """A single generated training/eval example."""
    context: str
    question: str
    reference_answer: str
    rubric: list[RubricCriterion]
    question_type: str = ""
    reasoning_tags: list[str] = field(default_factory=list)

    def max_positive_weight(self) -> int:
        return sum(c.weight for c in self.rubric if c.is_positive) or 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "QAItem":
        rubric = [RubricCriterion(**c) for c in d.get("rubric", [])]
        return QAItem(
            context=d.get("context", ""),
            question=d.get("question", ""),
            reference_answer=d.get("reference_answer", ""),
            rubric=rubric,
            question_type=d.get("question_type", ""),
            reasoning_tags=d.get("reasoning_tags", []),
        )


@dataclass
class SolverEval:
    """Result of judging one solver attempt against the rubric."""
    score: float                          # normalized [0, 1]
    per_criterion: dict[str, int]         # criterion text -> satisfied (1) / not (0)
    raw_answer: str = ""


@dataclass
class QualityVerdict:
    passed: bool
    feedback: str = ""
    checks: dict[str, str] = field(default_factory=dict)


@dataclass
class RoundResult:
    round: int
    status: str                           # one of the failure modes or ACCEPTED
    qa: Optional[QAItem] = None
    weak_scores: list[float] = field(default_factory=list)
    strong_scores: list[float] = field(default_factory=list)
    gap: Optional[float] = None
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "status": self.status,
            "qa": self.qa.to_dict() if self.qa else None,
            "weak_scores": self.weak_scores,
            "strong_scores": self.strong_scores,
            "gap": self.gap,
            "feedback": self.feedback,
        }


@dataclass
class PaperResult:
    paper_id: str
    accepted: bool
    accepted_qa: Optional[QAItem]
    rounds: list[RoundResult] = field(default_factory=list)

    @property
    def n_rounds(self) -> int:
        return len(self.rounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "accepted": self.accepted,
            "accepted_qa": self.accepted_qa.to_dict() if self.accepted_qa else None,
            "n_rounds": self.n_rounds,
            "rounds": [r.to_dict() for r in self.rounds],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
