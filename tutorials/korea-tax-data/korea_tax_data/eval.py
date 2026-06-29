"""Offline reranker A/B — score stored ``{query, pos, neg}`` rows, no live graph needed.

For each row the reranker scores ``pos + neg``, ranks them, and we compute recall@k, MRR, and
nDCG@10 with ``gold = pos`` (the same shape as the CE builder's ``eval.py``). Because the rows
are self-contained, this is fast and deterministic and lets you compare a baseline vs a
fine-tuned checkpoint directly::

    python -m korea_tax_data.cli eval --model mock --split test
    python -m korea_tax_data.cli eval --model BAAI/bge-reranker-v2-m3 --split test
    python -m korea_tax_data.cli eval --model output/ft-bge-reranker-v2-m3 --split test

``--law-aware`` (mock only) toggles whether the mock reranker reads the law-name prefix, which
demonstrates the signal Fix #2 restores.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from .reranker import MockReranker, RerankerScorer


def _ndcg(ranked_is_pos: list[bool], n_pos: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, ok in enumerate(ranked_is_pos[:10]) if ok)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_pos, 10)))
    return dcg / idcg if idcg else 0.0


def build_scorer(model: str, law_aware: bool = False) -> RerankerScorer:
    if model.lower() in ("mock", "mock-reranker"):
        return MockReranker(law_aware=law_aware)
    from .reranker import CrossEncoderReranker  # noqa: PLC0415
    return CrossEncoderReranker(model)


def eval_rows(rows: list[dict[str, Any]], scorer: RerankerScorer,
              k_list=(1, 5, 10)) -> dict[str, Any]:
    rec = {k: [] for k in k_list}
    mrr, ndcg = [], []
    used = 0
    for r in rows:
        pos, neg = list(r.get("pos") or []), list(r.get("neg") or [])
        if not pos or not neg:
            continue
        cands = pos + neg
        is_pos = [True] * len(pos) + [False] * len(neg)
        scores = scorer.score(r["query"], cands)

        def _key(i: int, _s=scores) -> float:
            s = float(_s[i])
            return math.inf if math.isnan(s) else -s

        order = sorted(range(len(cands)), key=_key)
        ranked = [is_pos[i] for i in order]
        used += 1
        for k in k_list:
            rec[k].append(sum(ranked[:k]) / len(pos))
        first = next((i for i, ok in enumerate(ranked) if ok), None)
        mrr.append(1 / (first + 1) if first is not None else 0.0)
        ndcg.append(_ndcg(ranked, len(pos)))

    return {
        "rows": used,
        **{f"recall@{k}": round(st.mean(rec[k]), 4) for k in k_list if rec[k]},
        "MRR": round(st.mean(mrr), 4) if mrr else None,
        "nDCG@10": round(st.mean(ndcg), 4) if ndcg else None,
    }


def eval_split(split_path: str | Path, model: str, law_aware: bool = False) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(split_path).open(encoding="utf-8")]
    scorer = build_scorer(model, law_aware=law_aware)
    out = {"model": model, "law_aware": law_aware, "split": str(split_path), **eval_rows(rows, scorer)}
    return out
