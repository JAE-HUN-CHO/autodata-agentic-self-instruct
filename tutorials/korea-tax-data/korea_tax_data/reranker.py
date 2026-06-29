"""Reranker scorer — the "judge" of the loop.

In autodata the weak/strong *solvers* answer a question and a rubric judge scores them; an
example is accepted when the scores separate by a margin. Here the **baseline reranker** plays
that role: it scores ``(query, document)`` pairs, and a training triple is accepted when the
reranker *fails* to separate the hard negative from the positive (the negative is confusable —
exactly the mistake fine-tuning should fix).

Two implementations:

* :class:`MockReranker` — deterministic, offline, lexical. By default it is **law-name blind**
  (scores on title/body overlap only), which reproduces the analysis's problem B: two articles
  with the same number/topic from different laws look almost identical. Set ``law_aware=True``
  to see how the law-prefixed :func:`doc_text.candidate_text` *adds* a separating signal.
* :class:`CrossEncoderReranker` — wraps ``sentence_transformers.CrossEncoder`` (the real
  bge-reranker-v2-m3, or a fine-tuned checkpoint). Imported lazily.
"""
from __future__ import annotations

import re
from typing import Protocol


class RerankerScorer(Protocol):
    def score(self, query: str, docs: list[str]) -> list[float]: ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", str(text or ""))


class MockReranker:
    """Lexical, deterministic stand-in for bge-reranker-v2-m3 (offline / tests).

    ``law_aware=False`` (default) intentionally ignores the law-name prefix so sibling articles
    score nearly as high as the positive — the hard-negative regime the analysis is about.
    """

    def __init__(self, law_aware: bool = False):
        """``law_aware=False`` (default) ignores the law-name prefix, reproducing problem B."""
        self.law_aware = law_aware

    @staticmethod
    def _strip_law_prefix(doc: str) -> str:
        # Drop a leading "법령명 제N조" header so the law-blind scorer can't use it. The law name
        # may contain spaces/parens ("상속세 및 증여세법", "국세기본법 시행령"), so allow them (non-greedy
        # up to the 제N조 token).
        return re.sub(r"^[가-힣A-Za-z0-9·ㆍ()\- ]+?\s*제\s*\d+\s*조(?:의\s*\d+)?", "", doc, count=1)

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Soft lexical overlap of query tokens against each doc, in roughly [0, 1+]."""
        q = set(_tokens(query))
        if not q:
            return [0.0] * len(docs)
        out: list[float] = []
        for doc in docs:
            text = doc if self.law_aware else self._strip_law_prefix(doc)
            d = set(_tokens(text))
            if not d:
                out.append(0.0)
                continue
            inter = len(q & d)
            # soft overlap: also credit substring matches (조사/어미 변형 흡수)
            soft = sum(1 for t in q if t not in d and any(t in w or w in t for w in d))
            out.append((inter + 0.5 * soft) / len(q))
        return out


class CrossEncoderReranker:
    """Real reranker: ``sentence_transformers.CrossEncoder`` over (query, doc) pairs.

    Used both as the build-time hardness judge and by ``eval.py``. Pass a HF model id
    (``BAAI/bge-reranker-v2-m3``) or a local fine-tuned checkpoint path.
    """

    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3", max_length: int = 2048):
        """Load a CrossEncoder (HF id or local fine-tuned checkpoint). Imported lazily."""
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
        self._ce = CrossEncoder(model_path, max_length=max_length)

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Score each ``(query, doc)`` pair; higher = more relevant. Empty docs -> ``[]``."""
        if not docs:
            return []
        import math  # noqa: PLC0415
        raw = self._ce.predict([(query, d) for d in docs])
        # NaN guard (mirrors eval.py): map NaN to -inf so it never beats a real score.
        # predict() returns a numpy array, so normalize via float() BEFORE isnan — a bare
        # isinstance(s, float) check would let a np.float32('nan') slip through.
        out: list[float] = []
        for s in raw:
            v = float(s)
            out.append(-math.inf if math.isnan(v) else v)
        return out
