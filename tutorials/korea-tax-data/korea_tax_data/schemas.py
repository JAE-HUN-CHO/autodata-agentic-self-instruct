"""Core data structures for the Korea-tax reranker-data generator.

The generator combines two lineages:

* **autodata** (this repo) contributes the *agentic accept / refine loop* — a candidate
  is kept only when it is hard enough for the current model, otherwise the failure mode is
  fed back and a new candidate is generated.
* **the CE-trainset builder** contributes the *reranker training shape* —
  ``{"query", "pos": [...], "neg": [...]}`` in the FlagEmbedding reranker format.

The unit of work here is a :class:`RerankerExample` (one query + its positives + hard
negatives), the analog of autodata's ``QAItem``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json
import re


# ---------------------------------------------------------------------------
# Negative-difficulty *acceptance* outcomes — mirror autodata's failure modes
# (TOO_EASY / FAILED_STRONG / ACCEPTED) but phrased for reranker triples.
# ---------------------------------------------------------------------------
TOO_EASY = "TOO_EASY"          # baseline reranker already separates pos from negs cleanly
NO_NEGATIVES = "NO_NEGATIVES"  # no usable hard negative survived denoise
LEAKED = "LEAKED"              # query / positive overlaps the held-out gold set
ACCEPTED = "ACCEPTED"

# Where a negative came from — lets us cap pool/authority negatives and report the mix.
NEG_SIBLING = "sibling"        # same law, adjacent article (the analysis's primary fix)
NEG_POOL = "pool"              # scattered retrieve-pool article (the v2 default; auxiliary)
NEG_AUTHORITY = "authority"    # 판례 / 해석례 etc. (small)


@dataclass(frozen=True)
class Article:
    """A statute article (조문) — the rerankable document unit.

    ``law_name`` is kept as a first-class field precisely because the analysis found the v2
    builder dropped it from the document text. :func:`korea_tax_data.doc_text.candidate_text`
    puts it back at the head of the rendered text.
    """
    law_name: str
    clause_num: str
    clause_title: str = ""
    clause_content: str = ""

    @property
    def article_int(self) -> int | None:
        """First integer in the article number (133의2 -> 133, 제70조 -> 70). Sibling windows.

        Uses ``search`` (not ``match``) so a Korean-prefixed form like "제70조" still yields 70;
        otherwise ``corpus.siblings`` would see ``None`` and silently skip sibling negatives.
        """
        m = re.search(r"\d+", str(self.clause_num))
        return int(m.group(0)) if m else None


@dataclass(frozen=True)
class Authority:
    """A non-statute authority document (판례/해석례/심판례/기본통칙/집행기준)."""
    law: str
    case_number: str
    title: str = ""
    body: str = ""


@dataclass
class Issue:
    """A tax issue (쟁점): user-side query expressions plus labelled primary/secondary statutes."""
    issue_id: str
    name: str = ""
    user_expressions: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    search_keywords: list[str] = field(default_factory=list)
    primary_statutes: list[str] = field(default_factory=list)
    secondary_statutes: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    """A scored candidate document within one query's group."""
    text: str
    identity: tuple[str, tuple[str, ...]]   # (law_name_norm, article_digits) — dedup / leak key
    source: str                             # NEG_SIBLING / NEG_POOL / NEG_AUTHORITY
    score: float = 0.0                      # baseline reranker score (filled by hardness gate)


@dataclass
class RerankerExample:
    """One training row in FlagEmbedding reranker format, plus provenance metadata."""
    query: str
    pos: list[str]
    neg: list[str]
    issue_id: str = ""
    neg_sources: dict[str, int] = field(default_factory=dict)   # {"sibling": 6, "pool": 2, ...}

    def to_row(self) -> dict[str, Any]:
        """The exact JSONL row FlagEmbedding's reranker finetuner consumes (+ provenance)."""
        return {
            "query": self.query,
            "pos": self.pos,
            "neg": self.neg,
            "issue_id": self.issue_id,
            "neg_sources": self.neg_sources,
        }


@dataclass
class RoundResult:
    """One round of the per-query accept/refine loop (analog of autodata.RoundResult)."""
    round: int
    status: str
    n_hard_neg: int = 0
    margin: float | None = None
    feedback: str = ""


@dataclass
class IssueResult:
    """Aggregate outcome for one issue across all its query shapes."""
    issue_id: str
    examples: list[RerankerExample] = field(default_factory=list)
    rounds: list[RoundResult] = field(default_factory=list)
    leaked: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializable view: examples + per-round log for trajectory inspection."""
        return {
            "issue_id": self.issue_id,
            "leaked": self.leaked,
            "n_examples": len(self.examples),
            "examples": [e.to_row() for e in self.examples],
            "rounds": [asdict(r) for r in self.rounds],
        }

    def to_json(self, indent: int = 2) -> str:
        """Pretty JSON of :meth:`to_dict` (UTF-8, non-ASCII preserved)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
