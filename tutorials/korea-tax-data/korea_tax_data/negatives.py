"""Negative challenger — the home of **Fix #1**.

The v2 builder's ``negatives_from_pool`` emitted the *entire* clause bucket of the live
retrieve pool (median ~79 articles scattered across many laws). The analysis showed that
teaches the model "query vs unrelated article", not the actual failure "소득세법시행령 133 vs
131·132·134". 92% of the reranker's regressions were that same-law confusion (problem B).

:class:`NegativeChallenger` inverts the priority:

1. **sibling** (primary) — same law, adjacent article numbers around each positive. These are
   the confusable negatives that target B directly.
2. **pool** (auxiliary, capped) — a few scattered retrieve-pool articles, so the model still
   sees cross-law contrast.
3. **authority** (small) — a couple of 판례/해석례, since type-bucketing already separates them.

It also *escalates*: when the accept loop reports the current negatives were "too easy", the
challenger widens the sibling window and asks for more siblings — the reranker-data analog of
autodata's challenger generating a harder question after a ``TOO_EASY`` round.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import CorpusProvider
from .doc_text import article_text, authority_text
from .schemas import (
    Article, Candidate, NEG_SIBLING, NEG_POOL, NEG_AUTHORITY,
)


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def _digits(s) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", str(s or "")))


def article_identity(a: Article) -> tuple[str, tuple[str, ...]]:
    return (_norm(a.law_name), _digits(a.clause_num))


@dataclass
class NegConfig:
    """Negative-mix knobs. Defaults encode the analysis's prescription (sibling-heavy)."""
    sibling_window: int = 6      # initial same-law article-number radius
    sibling_k: int = 6           # siblings per positive
    pool_k: int = 2              # auxiliary scattered pool negatives (small)
    authority_k: int = 1         # authority negatives (smallest)
    window_step: int = 6         # how much the window grows per escalation round
    max_window: int = 40


class NegativeChallenger:
    def __init__(self, corpus: CorpusProvider, cfg: NegConfig | None = None):
        self.corpus = corpus
        self.cfg = cfg or NegConfig()

    def generate(
        self,
        positives: list[Article],
        query: str,
        round_no: int,
        exclude_ids: set[tuple[str, tuple[str, ...]]],
    ) -> list[Candidate]:
        """Return candidate negatives for one (query, positives) group at a difficulty level.

        ``round_no`` (1-based) escalates the sibling window so re-rolls dig wider for harder,
        rarer confusables. ``exclude_ids`` holds positives (+ teacher-promoted positives +
        held-out gold) that must never appear as negatives.
        """
        window = min(self.cfg.max_window,
                     self.cfg.sibling_window + (round_no - 1) * self.cfg.window_step)
        # escalating rounds also pull a few more siblings each time
        sibling_k = self.cfg.sibling_k + (round_no - 1) * 2

        out: list[Candidate] = []
        # Always exclude the positives we were handed (plus any caller-supplied ids), so no
        # source — sibling, pool, or authority — can ever emit a labelled positive as a negative.
        seen: set[tuple[str, tuple[str, ...]]] = set(exclude_ids)
        seen |= {article_identity(p) for p in positives}

        def _add(art: Article, source: str) -> None:
            ident = article_identity(art)
            if ident in seen:
                return
            text = article_text(art)
            if len(text) < 30:
                return
            seen.add(ident)
            out.append(Candidate(text=text, identity=ident, source=source))

        # 1) sibling — primary. The corpus sibling API excludes by (law_norm, str(num)).
        pos_keys = {(_norm(a.law_name), str(a.clause_num)) for a in positives}
        for pos in positives:
            for sib in self.corpus.siblings(pos, window=window, k=sibling_k, exclude=pos_keys):
                _add(sib, NEG_SIBLING)

        # 2) pool — auxiliary, capped. Skipped when disabled (pool_k <= 0) or when the provider
        #    has no retriever wired (Neo4jCorpusProvider.retrieve_pool is an opt-in stub), so the
        #    sibling/authority-only real path runs without a live retriever instead of crashing.
        if self.cfg.pool_k > 0:
            try:
                pool = self.corpus.retrieve_pool(query, k=self.cfg.pool_k * 6)
            except NotImplementedError:
                pool = []
            added = 0
            for art in pool:
                if added >= self.cfg.pool_k:
                    break
                before = len(out)
                _add(art, NEG_POOL)
                if len(out) > before:
                    added += 1

        # 3) authority — smallest.
        added = 0
        for auth in self.corpus.authorities(query, k=self.cfg.authority_k * 3):
            if added >= self.cfg.authority_k:
                break
            ident = (_norm(auth.law), _digits(auth.case_number))
            if ident in seen:
                continue
            text = authority_text(auth)
            if len(text) < 30:
                continue
            seen.add(ident)
            out.append(Candidate(text=text, identity=ident, source=NEG_AUTHORITY))
            added += 1

        return out
