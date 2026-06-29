"""Builder — drives the agentic loop over a corpus and writes the FlagEmbedding JSONL.

Flow per issue (leak-guarded):

1. resolve positives,
2. expand query shapes (user_expressions + aliases + keywords + optional LLM paraphrase),
3. for each shape, run :class:`AgenticRerankerData.run_query` to get one hard-negative row,
4. stream rows to ``output/ce_trainset.jsonl`` plus a per-issue trajectory for inspection.

Issue-level dedup of query shapes and the held-out drop both happen here, mirroring the
original builder, so the agentic loop stays focused on hardness alone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .corpus import CorpusProvider
from .leakguard import HeldOut, pos_overlaps_heldout
from .llm_roles import LLMRoles
from .negatives import article_identity
from .orchestrator import AgenticRerankerData
from .schemas import Issue, IssueResult, LEAKED, RoundResult


@dataclass
class BuildConfig:
    max_query_shapes: int = 4
    paraphrase_n: int = 0      # LLM paraphrases per issue (0 = off; offline default)


def query_shapes(issue: Issue, cfg: BuildConfig, llm: LLMRoles,
                 held_strings: set[str]) -> list[str]:
    """Conversational expressions first, then alias, then a keyword bag — deduped, leak-filtered."""
    shapes: list[str] = []
    for q in issue.user_expressions[:2]:
        if q and q.strip():
            shapes.append(q.strip())
    for q in issue.aliases[:1]:
        if q and q.strip():
            shapes.append(q.strip())
    if issue.search_keywords:
        shapes.append(" ".join(str(k) for k in issue.search_keywords[:6]).strip())
    if cfg.paraphrase_n and shapes:
        shapes += llm.paraphrase(shapes[0], cfg.paraphrase_n)

    seen, out = set(), []
    for q in shapes:
        if q and q not in seen and q not in held_strings:
            seen.add(q)
            out.append(q)
    if not out and issue.name and issue.name not in held_strings:
        out = [issue.name]
    return out[: cfg.max_query_shapes]


class Builder:
    def __init__(self, corpus: CorpusProvider, engine: AgenticRerankerData,
                 heldout: HeldOut, llm: LLMRoles, cfg: BuildConfig | None = None):
        self.corpus = corpus
        self.engine = engine
        self.heldout = heldout
        self.llm = llm
        self.cfg = cfg or BuildConfig()

    def build_issue(self, issue: Issue) -> IssueResult:
        res = IssueResult(issue_id=issue.issue_id)
        if issue.issue_id in self.heldout.issue_ids:
            res.leaked = True
            res.rounds.append(RoundResult(0, LEAKED, feedback="issue_id is held-out"))
            return res
        positives = self.corpus.positives(issue)
        if positives and pos_overlaps_heldout(positives, self.heldout.arts_keys):
            res.leaked = True
            res.rounds.append(RoundResult(0, LEAKED, feedback="positive overlaps held-out gold"))
            return res
        if not positives:
            return res

        gold_ids = frozenset(self.heldout.arts_keys)
        shapes = query_shapes(issue, self.cfg, self.llm, self.heldout.query_strings)
        for q in shapes:
            example, rounds = self.engine.run_query(q, positives, issue.issue_id, gold_ids)
            res.rounds.extend(rounds)
            if example:
                res.examples.append(example)
        return res

    def run(self, out_path: str | Path, traj_dir: str | Path | None = None,
            limit: int | None = None) -> dict[str, Any]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        traj = Path(traj_dir) if traj_dir else None
        if traj:
            traj.mkdir(parents=True, exist_ok=True)

        issues = self.corpus.issues(limit)
        n_rows = n_leaked = n_issues_with_rows = 0
        src_totals: dict[str, int] = {}
        with out_path.open("w", encoding="utf-8") as f:
            for issue in issues:
                r = self.build_issue(issue)
                if r.leaked:
                    n_leaked += 1
                if traj:
                    (traj / f"{issue.issue_id}.json").write_text(r.to_json(), encoding="utf-8")
                if r.examples:
                    n_issues_with_rows += 1
                for ex in r.examples:
                    f.write(json.dumps(ex.to_row(), ensure_ascii=False) + "\n")
                    n_rows += 1
                    for k, v in ex.neg_sources.items():
                        src_totals[k] = src_totals.get(k, 0) + v
        stats = {
            "issues_processed": len(issues),
            "issues_leaked_dropped": n_leaked,
            "issues_with_rows": n_issues_with_rows,
            "rows_written": n_rows,
            "negative_source_totals": dict(sorted(src_totals.items())),
            "out": str(out_path),
        }
        return stats
