"""Issue-level train/eval/test split (0.64 / 0.16 / 0.20).

Splitting by ``issue_id`` (not by row) keeps every query shape of one issue on the same side of
the split, so paraphrases / rewrites of a held-out question can't leak across train and test.
Ported from the CE builder's ``split.py``.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def split_rows(rows: list[dict[str, Any]], seed: int = 42,
               test: float = 0.2, eval_frac: float = 0.2) -> dict[str, list[dict[str, Any]]]:
    """Group rows by issue_id, shuffle issues by ``seed``, and slice into train/eval/test."""
    by_issue: dict[str, list] = defaultdict(list)
    for r in rows:
        by_issue[str(r.get("issue_id"))].append(r)
    issues = list(by_issue)
    random.Random(seed).shuffle(issues)

    n = len(issues)
    n_test = max(1, round(n * test)) if n >= 3 else (1 if n >= 2 else 0)
    train_pool = issues[n_test:]
    n_eval = max(1, round(len(train_pool) * eval_frac)) if len(train_pool) >= 2 else 0
    if len(train_pool) - n_eval < 1:
        n_eval = max(0, len(train_pool) - 1)

    # Keep the shuffled slice ORDER (not a set) so JSONL row order is reproducible per seed.
    issue_lists = {
        "test": issues[:n_test],
        "eval": train_pool[:n_eval],
        "train": train_pool[n_eval:],
    }
    return {name: [r for i in iss for r in by_issue[i]] for name, iss in issue_lists.items()}


def write_splits(in_path: str | Path, out_dir: str | Path, **kw) -> dict[str, dict[str, int]]:
    """Split a trainset JSONL into train/eval/test files; raise on any cross-split issue leak."""
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in in_path.open(encoding="utf-8")]
    splits = split_rows(rows, **kw)

    # leak check: no issue_id shared across splits. Use explicit raises (not assert) so the
    # guard survives `python -O`, which strips assert statements.
    id_sets = {name: {str(r.get("issue_id")) for r in rs} for name, rs in splits.items()}
    if id_sets["train"] & id_sets["test"]:
        raise ValueError("train/test issue leak")
    if id_sets["train"] & id_sets["eval"]:
        raise ValueError("train/eval issue leak")
    if id_sets["eval"] & id_sets["test"]:
        raise ValueError("eval/test issue leak")

    counts: dict[str, dict[str, int]] = {}
    for name, rs in splits.items():
        (out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        counts[name] = {"issues": len(id_sets[name]), "rows": len(rs)}
    return counts
