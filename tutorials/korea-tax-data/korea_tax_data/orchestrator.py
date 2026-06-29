"""Agentic reranker-data loop — the synthesis of autodata + the CE builder.

Per (query, positives), loop:

    negative challenger -> [LLM positive-discovery] -> [LLM false-negative teacher]
        -> baseline reranker scores pos & neg -> hardness gate

Accept the first round whose negatives are hard enough (the baseline reranker confuses them
with the positive). On a ``TOO_EASY`` round, escalate the challenger (wider sibling window) and
try again — directly mirroring autodata's challenger/accept loop, with the *reranker* standing
in for the weak/strong solver pair and the *hardness gate* standing in for the gap criterion.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import CorpusProvider
from .doc_text import article_text
from .hardness import HardnessCriteria, evaluate
from .llm_roles import LLMRoles
from .negatives import NegativeChallenger, article_identity
from .reranker import RerankerScorer
from .schemas import (
    Article, Candidate, RerankerExample, RoundResult,
    ACCEPTED, TOO_EASY, NO_NEGATIVES, NEG_SIBLING, NEG_POOL, NEG_AUTHORITY,
)


@dataclass
class LoopConfig:
    max_rounds: int = 4
    emit_on_exhaust: bool = True   # keep the last attempt even if never fully "hard" (yield)
    llm_refine_topk: int = 20      # how many top candidates the LLM teacher inspects


class AgenticRerankerData:
    def __init__(
        self,
        corpus: CorpusProvider,
        reranker: RerankerScorer,
        challenger: NegativeChallenger,
        criteria: HardnessCriteria | None = None,
        llm: LLMRoles | None = None,
        loop: LoopConfig | None = None,
        verbose: bool = False,
    ):
        self.corpus = corpus
        self.reranker = reranker
        self.challenger = challenger
        self.criteria = criteria or HardnessCriteria()
        self.llm = llm or LLMRoles(provider=None)
        self.loop = loop or LoopConfig()
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def run_query(
        self,
        query: str,
        positives: list[Article],
        issue_id: str,
        gold_ids: frozenset[tuple[str, tuple[str, ...]]] = frozenset(),
    ) -> tuple[RerankerExample | None, list[RoundResult]]:
        """Build one training row for ``query``; return (example or None, round log)."""
        rounds: list[RoundResult] = []
        pos_texts = [article_text(p) for p in positives]
        if not pos_texts:
            return None, [RoundResult(0, NO_NEGATIVES, feedback="no positive text")]
        pos_ids = {article_identity(p) for p in positives}
        pos_titles = [p.clause_title for p in positives]
        last_example: RerankerExample | None = None

        for round_no in range(1, self.loop.max_rounds + 1):
            exclude = set(pos_ids) | set(gold_ids)
            cands = self.challenger.generate(positives, query, round_no, exclude)
            if not cands:
                rounds.append(RoundResult(round_no, NO_NEGATIVES, feedback="challenger empty"))
                continue

            # LLM role 3 — promote unlabelled positives out of the negative pool.
            promoted_texts: list[str] = []
            top = cands[: self.loop.llm_refine_topk]
            sel = self.llm.find_positives(
                query, pos_titles, [{"i": j, "text": c.text} for j, c in enumerate(top)])
            promote_idx = set(sel)
            for j in promote_idx:
                exclude.add(top[j].identity)
                if top[j].text not in pos_texts and top[j].text not in promoted_texts:
                    promoted_texts.append(top[j].text)
            cands = [c for j, c in enumerate(cands)
                     if not (j < len(top) and j in promote_idx) and c.identity not in exclude]

            # LLM role 2 — drop false negatives (candidates that are actually answers).
            fn = set(self.llm.false_negatives(query, [c.text for c in cands]))
            cands = [c for j, c in enumerate(cands) if j not in fn]
            if not cands:
                rounds.append(RoundResult(round_no, NO_NEGATIVES, feedback="all denoised away"))
                continue

            # Baseline reranker scores positives + negatives; hardness gate decides.
            full_pos = pos_texts + promoted_texts
            pos_scores = self.reranker.score(query, full_pos)
            neg_scores = self.reranker.score(query, [c.text for c in cands])
            for c, s in zip(cands, neg_scores):
                c.score = s
            verdict = evaluate(pos_scores, cands, self.criteria)

            example = RerankerExample(
                query=query, pos=full_pos, neg=[c.text for c in verdict.kept],
                issue_id=issue_id, neg_sources=_count_sources(verdict.kept),
            )
            last_example = example
            rounds.append(RoundResult(round_no, verdict.status, n_hard_neg=verdict.n_hard,
                                      margin=round(verdict.min_pos_score, 4),
                                      feedback=verdict.feedback))
            self._log(f"    q={query[:24]!r} r{round_no}: {verdict.status} "
                      f"(hard={verdict.n_hard}, neg={len(example.neg)})")
            if verdict.status == ACCEPTED and example.neg:
                return example, rounds

        # escalation exhausted — keep the best attempt if allowed and non-empty
        if self.loop.emit_on_exhaust and last_example and last_example.neg:
            return last_example, rounds
        return None, rounds


def _count_sources(cands: list[Candidate]) -> dict[str, int]:
    counts = {NEG_SIBLING: 0, NEG_POOL: 0, NEG_AUTHORITY: 0}
    for c in cands:
        counts[c.source] = counts.get(c.source, 0) + 1
    return {k: v for k, v in counts.items() if v}
