"""Agentic Self-Instruct orchestrator (CS-paper variant, Sec 3.1 / App C.1).

Per source paper, loop: challenger -> quality verifier -> weak eval -> strong eval -> gap
check, with compute-saving early exits (weak evaluated before strong) and failure-mode
feedback routed back to the challenger. Accept the first question that separates the weak and
strong solver by the target margin.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from statistics import mean

from .schemas import (
    QAItem, RoundResult, PaperResult,
    TOO_EASY, FAILED_STRONG, FAILED_QV, ACCEPTED,
)
from .subagents import Challenger, Solver, QualityVerifier
from .rubric_eval import RubricJudge

# Cap on concurrent solver→judge calls per round, independent of n_attempts. Keeps a
# misconfigured high n_attempts from opening a burst of network calls all at once
# (NIM rate-limits, and the per-key window is easy to trip — see provider 429 handling).
MAX_PARALLEL_ATTEMPTS = 8


@dataclass
class AcceptanceCriteria:
    """Exact thresholds from Sec 3.1 'Criteria'. (strong_avg_min defaults to App C.1's 0.60;
    set to 0.65 to match the Sec 3.1 prose.)"""
    weak_avg_max: float = 0.65
    weak_attempt_max: float = 0.75
    strong_avg_min: float = 0.60
    strong_avg_max: float = 0.95
    gap_min: float = 0.20

    def weak_passes(self, weak_scores: list[float]) -> tuple[bool, str]:
        if not weak_scores:
            return False, "no weak scores"
        if mean(weak_scores) > self.weak_avg_max:
            return False, f"weak_avg {mean(weak_scores):.3f} > {self.weak_avg_max}"
        if max(weak_scores) > self.weak_attempt_max:
            return False, f"max_weak {max(weak_scores):.3f} > {self.weak_attempt_max}"
        if all(s == 0 for s in weak_scores):
            return False, "degenerate all-zero weak rollouts"
        return True, "WEAK_PASSED"

    def strong_passes(self, strong_scores: list[float]) -> tuple[bool, str]:
        if not strong_scores:
            return False, "no strong scores"
        avg = mean(strong_scores)
        if avg < self.strong_avg_min:
            return False, f"strong_avg {avg:.3f} < {self.strong_avg_min}"
        if avg >= self.strong_avg_max:
            return False, f"strong_avg {avg:.3f} >= {self.strong_avg_max} (saturated)"
        return True, "STRONG_PASSED"

    def gap_passes(self, weak_scores, strong_scores) -> tuple[bool, float, str]:
        gap = mean(strong_scores) - mean(weak_scores)
        if gap < self.gap_min:
            return False, gap, f"gap {gap:.3f} < {self.gap_min}"
        return True, gap, "GAP_PASSED"


class AgenticSelfInstruct:
    def __init__(
        self,
        challenger: Challenger,
        weak_solver: Solver,
        strong_solver: Solver,
        judge: RubricJudge,
        quality_verifier: QualityVerifier,
        criteria: AcceptanceCriteria | None = None,
        n_attempts: int = 3,
        max_rounds: int = 12,
        verbose: bool = True,
        parallel_attempts: bool = True,
    ):
        self.challenger = challenger
        self.weak_solver = weak_solver
        self.strong_solver = strong_solver
        self.judge = judge
        self.qv = quality_verifier
        self.criteria = criteria or AcceptanceCriteria()
        self.n_attempts = n_attempts
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.parallel_attempts = parallel_attempts

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _eval_solver(self, solver: Solver, qa: QAItem) -> list[float]:
        # Each attempt is solver(qa) -> judge(qa, answer). Attempts are mutually independent
        # so we fan them out across a small thread pool when running against a real network
        # provider (NIM/vLLM/OpenAI). For the mock provider this is fast either way.
        def one_attempt(_i: int) -> float:
            ans = solver.answer(qa)
            return self.judge.evaluate(qa, ans).score

        if self.parallel_attempts and self.n_attempts > 1:
            max_workers = min(self.n_attempts, MAX_PARALLEL_ATTEMPTS)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                return list(ex.map(one_attempt, range(self.n_attempts)))
        return [one_attempt(i) for i in range(self.n_attempts)]

    @staticmethod
    def _failures_block(rounds: list[RoundResult]) -> str:
        groups: dict[str, list[str]] = {TOO_EASY: [], FAILED_STRONG: [], FAILED_QV: []}
        for r in rounds:
            if r.status in groups and r.qa is not None:
                groups[r.status].append(f"  Q: {r.qa.question}  ({r.feedback})")
        block = ""
        labels = {TOO_EASY: "TOO EASY (weak scored too high)",
                  FAILED_STRONG: "FAILED ON STRONG (gap too small / strong too low)",
                  FAILED_QV: "FAILED QUALITY CHECK"}
        for mode, items in groups.items():
            if items:
                block += f"[{labels[mode]}]\n" + "\n".join(items) + "\n"
        return block

    def run_paper(self, paper_id: str, paper_text: str) -> PaperResult:
        rounds: list[RoundResult] = []
        for round_no in range(1, self.max_rounds + 1):
            t0 = time.time()
            failures = self._failures_block(rounds)
            self._log(f"  [{paper_id}] r{round_no}: challenger…")
            qa = self.challenger.generate(paper_text, failures, round_no)

            # 0) deterministic rubric sanity check (runs before the LLM QV).
            # _parse_rubric silently drops malformed entries, which on a noisy model output
            # can leave an empty or no-positive-weight rubric. The LLM QV doesn't inspect
            # rubric shape, and the judge will still emit a `normalized_score` for an empty
            # rubric -- so without this gate, a degenerate rubric can ride all the way to
            # dataset.jsonl. Reject deterministically and let the failure feed back to the
            # challenger as a regular FAILED_QV.
            n_pos = sum(1 for c in qa.rubric if c.is_positive)
            n_neg = sum(1 for c in qa.rubric if not c.is_positive)
            # Sum raw positive weights directly. QAItem.max_positive_weight() applies an
            # `or 1` fallback for the divide-by-zero guard in scoring, which would let a
            # rubric of only `weight=0` positive criteria slip past this gate.
            positive_weight_sum = sum(c.weight for c in qa.rubric if c.is_positive)
            if not qa.rubric or n_pos < 1 or positive_weight_sum <= 0:
                msg = (
                    f"rubric is unusable after parse: total={len(qa.rubric)}, "
                    f"positive={n_pos}, negative={n_neg}, "
                    f"positive_weight_sum={positive_weight_sum}"
                )
                rounds.append(RoundResult(round_no, FAILED_QV, qa=qa, feedback=msg))
                self._log(f"  [{paper_id}] r{round_no}: FAILED_QV ({msg}) [{time.time()-t0:.1f}s]")
                continue

            # 1) quality verifier
            self._log(f"  [{paper_id}] r{round_no}: quality_verifier… (+{time.time()-t0:.1f}s)")
            verdict = self.qv.check(qa)
            if not verdict.passed:
                rounds.append(RoundResult(round_no, FAILED_QV, qa=qa, feedback=verdict.feedback))
                self._log(f"  [{paper_id}] r{round_no}: FAILED_QV ({verdict.feedback}) [{time.time()-t0:.1f}s]")
                continue

            # 2) weak solver first (compute saving, per paper)
            self._log(f"  [{paper_id}] r{round_no}: weak×{self.n_attempts}… (+{time.time()-t0:.1f}s)")
            weak = self._eval_solver(self.weak_solver, qa)
            wpass, wmsg = self.criteria.weak_passes(weak)
            if not wpass:
                rounds.append(RoundResult(round_no, TOO_EASY, qa=qa, weak_scores=weak, feedback=wmsg))
                self._log(f"  [{paper_id}] r{round_no}: TOO_EASY ({wmsg}) [{time.time()-t0:.1f}s]")
                continue

            # 3) strong solver
            self._log(f"  [{paper_id}] r{round_no}: strong×{self.n_attempts}… (+{time.time()-t0:.1f}s)")
            strong = self._eval_solver(self.strong_solver, qa)
            spass, smsg = self.criteria.strong_passes(strong)
            if not spass:
                rounds.append(RoundResult(round_no, FAILED_STRONG, qa=qa,
                                          weak_scores=weak, strong_scores=strong, feedback=smsg))
                self._log(f"  [{paper_id}] r{round_no}: FAILED_STRONG ({smsg}) [{time.time()-t0:.1f}s]")
                continue

            # 4) gap
            gpass, gap, gmsg = self.criteria.gap_passes(weak, strong)
            if not gpass:
                rounds.append(RoundResult(round_no, FAILED_STRONG, qa=qa, weak_scores=weak,
                                          strong_scores=strong, gap=gap, feedback=gmsg))
                self._log(f"  [{paper_id}] r{round_no}: FAILED_STRONG ({gmsg}) [{time.time()-t0:.1f}s]")
                continue

            # accepted
            rounds.append(RoundResult(round_no, ACCEPTED, qa=qa, weak_scores=weak,
                                      strong_scores=strong, gap=gap, feedback="accepted"))
            self._log(f"  [{paper_id}] r{round_no}: ACCEPTED "
                      f"(weak={mean(weak):.3f} strong={mean(strong):.3f} gap={gap:.3f}) [{time.time()-t0:.1f}s]")
            return PaperResult(paper_id, accepted=True, accepted_qa=qa, rounds=rounds)

        self._log(f"  [{paper_id}] REJECTED after {self.max_rounds} rounds")
        return PaperResult(paper_id, accepted=False, accepted_qa=None, rounds=rounds)
