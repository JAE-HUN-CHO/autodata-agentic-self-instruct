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

    sets = {
        "test": set(issues[:n_test]),
        "eval": set(train_pool[:n_eval]),
        "train": set(train_pool[n_eval:]),
    }
    return {name: [r for i in iss for r in by_issue[i]] for name, iss in sets.items()}


def write_splits(in_path: str | Path, out_dir: str | Path, **kw) -> dict[str, dict[str, int]]:
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in in_path.open(encoding="utf-8")]
    splits = split_rows(rows, **kw)

    # leak check: no issue_id shared across splits
    id_sets = {name: {str(r.get("issue_id")) for r in rs} for name, rs in splits.items()}
    assert not (id_sets["train"] & id_sets["test"]), "train/test issue leak"
    assert not (id_sets["train"] & id_sets["eval"]), "train/eval issue leak"
    assert not (id_sets["eval"] & id_sets["test"]), "eval/test issue leak"

    counts: dict[str, dict[str, int]] = {}
    for name, rs in splits.items():
        (out_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs), encoding="utf-8")
        counts[name] = {"issues": len(id_sets[name]), "rows": len(rs)}
    return counts
