"""Held-out leak guard (string-bridge), ported from the CE builder's ``load_heldout``.

The held-out gold (``data/sample_heldout.json``, analog of csv64) has questions but **no
issue_id**. So we bridge by string: any corpus issue whose ``user_expressions`` exactly match a
gold question is held out, and we then drop

* that issue entirely (``issue_ids``),
* every *sibling expression* of that issue — its other user_expressions, aliases, keywords —
  so a paraphrase can't sneak the same issue back in (``query_strings``), and
* any positive whose ``(law, digits)`` overlaps a gold answer article (``arts_keys``), which
  catches a *different* query that happens to share a held-out positive.

Fail-closed: if the gold file is missing the builder refuses to run (never risk train-on-test).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .schemas import Issue


def _norm(s) -> str:
    return re.sub(r"\s+", "", str(s or ""))


def _digits(s) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+", str(s or "")))


@dataclass
class HeldOut:
    issue_ids: set[str] = field(default_factory=set)
    query_strings: set[str] = field(default_factory=set)
    arts_keys: set[tuple[str, tuple[str, ...]]] = field(default_factory=set)


def load_heldout(gold_path: str | Path, issues: list[Issue]) -> HeldOut:
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else (data.get("items") or [])
    gold_qs = {str(r.get("q") or r.get("question") or "").strip()
               for r in rows if (r.get("q") or r.get("question"))}
    arts_keys: set[tuple[str, tuple[str, ...]]] = set()
    for r in rows:
        for a in r.get("arts") or []:
            if isinstance(a, (list, tuple)) and len(a) >= 2:
                law, dig = _norm(a[0]), _digits(a[1])
                if law and dig:
                    arts_keys.add((law, dig))

    held = HeldOut(arts_keys=arts_keys)
    held.query_strings |= gold_qs
    for it in issues:
        exprs = {str(v).strip() for v in it.user_expressions}
        if exprs & gold_qs:  # this issue owns a gold question -> held out
            held.issue_ids.add(it.issue_id)
            held.query_strings |= exprs
            held.query_strings |= {str(v).strip() for v in it.aliases}
            held.query_strings |= {str(v).strip() for v in it.search_keywords}
    return held


def pos_overlaps_heldout(positives, arts_keys: set[tuple[str, tuple[str, ...]]]) -> bool:
    keys = {(_norm(p.law_name), _digits(p.clause_num)) for p in positives}
    return bool(keys & arts_keys)
