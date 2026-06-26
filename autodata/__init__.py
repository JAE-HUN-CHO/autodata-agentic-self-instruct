"""Agentic Self-Instruct: an implementation of the CS-paper variant of Autodata
(arXiv 2606.25996v1, FAIR at Meta)."""

from .schemas import (
    QAItem, RubricCriterion, SolverEval, RoundResult, PaperResult,
    TOO_EASY, FAILED_STRONG, FAILED_QV, ACCEPTED, REJECTED,
)
from .orchestrator import AgenticSelfInstruct, AcceptanceCriteria
from .subagents import Challenger, Solver, QualityVerifier
from .rubric_eval import RubricJudge
from .llm import OpenAICompatibleProvider, MockProvider, build_provider, resolve_api_key

__all__ = [
    "QAItem", "RubricCriterion", "SolverEval", "RoundResult", "PaperResult",
    "TOO_EASY", "FAILED_STRONG", "FAILED_QV", "ACCEPTED", "REJECTED",
    "AgenticSelfInstruct", "AcceptanceCriteria",
    "Challenger", "Solver", "QualityVerifier", "RubricJudge",
    "OpenAICompatibleProvider", "MockProvider", "build_provider", "resolve_api_key",
]
