"""Korea-tax reranker training-data generator.

Combines the agentic accept/refine loop from *Agentic Self-Instruct* (this repo's ``autodata``
package) with the CE-reranker trainset shape, and applies the two fixes from the data analysis:

* **Fix #1** — same-law *sibling-article* hard negatives (``negatives.NegativeChallenger``),
* **Fix #2** — law-name-prefixed document text (``doc_text.candidate_text``).

See ``docs/DESIGN.md`` for the mapping to autodata and ``docs/ANALYSIS.md`` for the diagnosis.
"""
from .schemas import (
    Article, Authority, Issue, Candidate, RerankerExample, IssueResult,
    ACCEPTED, TOO_EASY, NO_NEGATIVES, LEAKED,
    NEG_SIBLING, NEG_POOL, NEG_AUTHORITY,
)
from .doc_text import candidate_text, article_text, law_prefix
from .corpus import JsonlCorpusProvider, parse_statute_ref
from .reranker import MockReranker, CrossEncoderReranker
from .negatives import NegativeChallenger, NegConfig, article_identity
from .hardness import HardnessCriteria, evaluate
from .llm_roles import LLMRoles
from .orchestrator import AgenticRerankerData, LoopConfig
from .leakguard import load_heldout, HeldOut, pos_overlaps_heldout
from .builder import Builder, BuildConfig

__all__ = [
    "Article", "Authority", "Issue", "Candidate", "RerankerExample", "IssueResult",
    "ACCEPTED", "TOO_EASY", "NO_NEGATIVES", "LEAKED",
    "NEG_SIBLING", "NEG_POOL", "NEG_AUTHORITY",
    "candidate_text", "article_text", "law_prefix",
    "JsonlCorpusProvider", "parse_statute_ref",
    "MockReranker", "CrossEncoderReranker",
    "NegativeChallenger", "NegConfig", "article_identity",
    "HardnessCriteria", "evaluate",
    "LLMRoles",
    "AgenticRerankerData", "LoopConfig",
    "load_heldout", "HeldOut", "pos_overlaps_heldout",
    "Builder", "BuildConfig",
]
