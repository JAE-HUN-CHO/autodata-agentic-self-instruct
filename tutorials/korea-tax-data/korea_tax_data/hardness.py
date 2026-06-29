"""Hardness gate — the reranker-data analog of autodata's ``AcceptanceCriteria``.

autodata accepts a QA only when ``strong_avg - weak_avg >= gap_min`` (the question genuinely
separates a strong from a weak solver). Here we accept a training triple only when the
**baseline reranker FAILS to separate** the negative from the positive — i.e. the negative is
confusable and therefore worth training on.

Concretely, with ``min_pos = min(score over positives)``:

* a negative is **hard** iff ``neg_score >= min_pos - margin`` (the reranker ranks it at, above,
  or just below the weakest positive);
* the triple is **ACCEPTED** iff it has at least ``min_hard_neg`` hard negatives;
* otherwise it is **TOO_EASY** and the challenger escalates (wider sibling window) next round —
  the analog of autodata feeding a ``TOO_EASY`` failure back to its challenger.

A few easy negatives are still kept as filler (cross-law contrast) up to ``max_neg``, but the
*accept decision* is driven by hard negatives only.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import Candidate, ACCEPTED, TOO_EASY, NO_NEGATIVES


@dataclass
class HardnessCriteria:
    margin: float = 0.10        # neg is "hard" if score >= min_pos_score - margin
    min_hard_neg: int = 4       # accept only when at least this many hard negatives exist
    max_neg: int = 12           # cap on negatives emitted per row
    keep_easy_filler: bool = True   # pad with easy negatives (contrast) up to max_neg


@dataclass
class HardnessVerdict:
    """Outcome of the hardness gate for one group: status, kept negatives, and diagnostics."""
    status: str                 # ACCEPTED / TOO_EASY / NO_NEGATIVES
    kept: list[Candidate]       # negatives to emit (hard first, then optional easy filler)
    n_hard: int
    min_pos_score: float
    feedback: str = ""


def evaluate(
    pos_scores: list[float],
    neg_candidates: list[Candidate],
    criteria: HardnessCriteria,
) -> HardnessVerdict:
    """Decide whether this (query, pos, neg) group is hard enough to keep.

    ``neg_candidates`` must already carry baseline reranker ``.score`` values.
    """
    if not neg_candidates:
        return HardnessVerdict(NO_NEGATIVES, [], 0, 0.0, "no candidate negatives")
    min_pos = min(pos_scores) if pos_scores else 0.0
    threshold = min_pos - criteria.margin

    ranked = sorted(neg_candidates, key=lambda c: c.score, reverse=True)
    hard = [c for c in ranked if c.score >= threshold]
    easy = [c for c in ranked if c.score < threshold]
    n_hard = len(hard)

    kept = list(hard[: criteria.max_neg])
    if criteria.keep_easy_filler and len(kept) < criteria.max_neg:
        kept.extend(easy[: criteria.max_neg - len(kept)])

    if n_hard >= criteria.min_hard_neg:
        return HardnessVerdict(ACCEPTED, kept, n_hard, min_pos,
                               f"{n_hard} hard negatives (>= {criteria.min_hard_neg})")
    return HardnessVerdict(
        TOO_EASY, kept, n_hard, min_pos,
        f"only {n_hard} hard negatives (< {criteria.min_hard_neg}); "
        f"min_pos={min_pos:.3f} margin={criteria.margin}",
    )
