"""Corpus access — the data layer behind the generator.

A :class:`CorpusProvider` answers the questions the builder needs:

* the issues (쟁점) to turn into training rows,
* the labelled positive / secondary articles for an issue,
* **same-law sibling articles** for an article (the raw material for Fix #1), and
* a retrieve pool + authority docs (auxiliary negatives).

Two implementations, mirroring autodata's offline/real split:

* :class:`JsonlCorpusProvider` — reads ``data/sample_corpus.json``; pure, deterministic,
  no network. The default for the tutorial and tests.
* :class:`Neo4jCorpusProvider` — runs the same Cypher shape as the original
  ``build_ce_trainset.py`` against the 7693 graph. Imported lazily so offline runs need no
  ``neo4j`` dependency.

The sibling lookup is the new capability the v2 builder never had: given 소득세법시행령 제133조
it returns 제131·132·134·135조 (same law, adjacent article numbers) as confusable negatives.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

from .schemas import Article, Authority, Issue

_STATUTE = re.compile(r"^(?P<law>.+?)\s*제\s*(?P<num>\d+)\s*조(?:의\s*(?P<sub>\d+))?")


def _norm(s: Any) -> str:
    """Whitespace-stripped string key for law-name comparison."""
    return re.sub(r"\s+", "", str(s or ""))


def parse_statute_ref(ref: str) -> tuple[str, str] | None:
    """``"소득세법시행령 제133조"`` -> ``("소득세법시행령", "133")``. ``제133조의2`` -> ``("...","133의2")``."""
    m = _STATUTE.match(str(ref or "").strip())
    if not m:
        return None
    num = m.group("num") + (f"의{m.group('sub')}" if m.group("sub") else "")
    return m.group("law").strip(), num


class CorpusProvider(Protocol):
    """Data layer the builder reads from (offline JSONL or real Neo4j)."""

    def issues(self, limit: int | None = None) -> list[Issue]:
        """The tax issues (쟁점) to turn into training rows."""
        ...

    def positives(self, issue: Issue) -> list[Article]:
        """Labelled primary-statute articles for an issue (the gold positives)."""
        ...

    def secondaries(self, issue: Issue) -> list[Article]:
        """Secondary-statute articles for an issue (bonus negatives)."""
        ...

    def siblings(self, article: Article, window: int, k: int,
                 exclude: set[tuple[str, str]]) -> list[Article]:
        """Same-law articles within ``window`` of ``article`` (the Fix #1 raw material)."""
        ...

    def retrieve_pool(self, query: str, k: int) -> list[Article]:
        """Scattered retrieve-pool articles for ``query`` (auxiliary negatives)."""
        ...

    def authorities(self, query: str, k: int) -> list[Authority]:
        """Authority docs (판례/해석례 …) for ``query`` (small negatives)."""
        ...


# ---------------------------------------------------------------------------
# Offline provider — bundled synthetic corpus.
# ---------------------------------------------------------------------------
class JsonlCorpusProvider:
    """Deterministic offline corpus backed by a bundled JSON file (no network)."""

    def __init__(self, corpus_path: str | Path):
        """Load articles/authorities/issues and build the (law, num) and per-law indexes."""
        data = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self._articles = [Article(a["law_name"], str(a["clause_num"]),
                                  a.get("clause_title", ""), a.get("clause_content", ""))
                          for a in data.get("articles", [])]
        self._authorities = [Authority(a["law"], a["case_number"], a.get("title", ""),
                                       a.get("body", ""))
                            for a in data.get("authorities", [])]
        self._issues = [Issue(
            issue_id=str(i["issue_id"]), name=i.get("name", ""),
            user_expressions=list(i.get("user_expressions", [])),
            aliases=list(i.get("aliases", [])),
            search_keywords=list(i.get("search_keywords", [])),
            primary_statutes=list(i.get("primary_statutes", [])),
            secondary_statutes=list(i.get("secondary_statutes", [])),
        ) for i in data.get("issues", [])]
        # index: (law_norm, num) -> Article, and law_norm -> [Article] sorted by article_int
        self._by_id: dict[tuple[str, str], Article] = {}
        self._by_law: dict[str, list[Article]] = {}
        for a in self._articles:
            self._by_id[(_norm(a.law_name), str(a.clause_num))] = a
            self._by_law.setdefault(_norm(a.law_name), []).append(a)
        for law in self._by_law:
            self._by_law[law].sort(key=lambda x: (x.article_int if x.article_int is not None else 1e9))

    def issues(self, limit: int | None = None) -> list[Issue]:
        """All issues (optionally capped at ``limit`` for smoke runs)."""
        return self._issues[:limit] if limit else list(self._issues)

    def _resolve_refs(self, refs: list[str]) -> list[Article]:
        """Resolve statute refs like ``"소득세법 제70조"`` to indexed :class:`Article` rows."""
        out: list[Article] = []
        for ref in refs:
            parsed = parse_statute_ref(ref)
            if not parsed:
                continue
            art = self._by_id.get((_norm(parsed[0]), parsed[1]))
            if art:
                out.append(art)
        return out

    def positives(self, issue: Issue) -> list[Article]:
        """Resolve the issue's primary statutes to gold positive articles."""
        return self._resolve_refs(issue.primary_statutes)

    def secondaries(self, issue: Issue) -> list[Article]:
        """Resolve the issue's secondary statutes (used as bonus negatives)."""
        return self._resolve_refs(issue.secondary_statutes)

    def siblings(self, article: Article, window: int, k: int,
                 exclude: set[tuple[str, str]]) -> list[Article]:
        """Same-law articles whose number is within ``window`` of ``article``, nearest first."""
        base = article.article_int
        law = _norm(article.law_name)
        pool = self._by_law.get(law, [])
        if base is None:
            return []
        scored: list[tuple[int, Article]] = []
        for a in pool:
            ai = a.article_int
            if ai is None:
                continue
            dist = abs(ai - base)
            if dist == 0 or dist > window:
                continue
            if (law, str(a.clause_num)) in exclude:
                continue
            scored.append((dist, a))
        scored.sort(key=lambda t: (t[0], t[1].article_int or 0))
        return [a for _, a in scored[:k]]

    def retrieve_pool(self, query: str, k: int) -> list[Article]:
        """Offline stand-in for the live multi-lane retrieve: lexical top-k over all laws.

        This deliberately mixes laws (like the v2 pool) so the contrast with sibling negatives
        is visible. The real path replaces this with ``graph.astream(interrupt_after=retrieve)``.
        """
        q = set(_tokens(query))
        scored = [(_overlap(q, _tokens(a.clause_title + " " + a.clause_content)), a)
                  for a in self._articles]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [a for s, a in scored[:k] if s > 0]

    def authorities(self, query: str, k: int) -> list[Authority]:
        q = set(_tokens(query))
        scored = [(_overlap(q, _tokens(a.title + " " + a.body)), a) for a in self._authorities]
        scored.sort(key=lambda t: t[0], reverse=True)
        return [a for s, a in scored[:k] if s > 0] or self._authorities[:k]


def _tokens(text: str) -> list[str]:
    """Hangul/alnum tokens of ``text`` (for the offline lexical retrieve stand-in)."""
    return re.findall(r"[가-힣A-Za-z0-9]+", str(text or ""))


def _overlap(q: set[str], doc_tokens: list[str]) -> int:
    """Count query tokens that appear in (or substring-match) the doc tokens."""
    d = set(doc_tokens)
    return sum(1 for t in q if any(t in w or w in t for w in d))


# ---------------------------------------------------------------------------
# Real provider — Neo4j 7693 (same Cypher shape as build_ce_trainset.py).
# ---------------------------------------------------------------------------
class Neo4jCorpusProvider:
    """Reads issues/positives/siblings/authorities from the 7693 graph.

    Mirrors the original builder's Cypher (TaxIssue.primary_statutes ∩ GOVERNED_BY clause),
    and adds a sibling query (same law_name, numeric clause_num within a window) that the v2
    builder lacked. ``neo4j`` is imported lazily so the offline path needs no driver.
    """

    CLAUSE_LABELS = ("LawClause", "SubLawClause", "RulesLawClause")
    AUTHORITY_LABELS = ("Interpretation", "CourtPrecedent", "TaxTribunal",
                        "BasicRule", "ExecutionStandardSection")

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None):
        """Open a Neo4j driver (args fall back to NEO4J_* env vars). ``neo4j`` imported lazily."""
        from neo4j import GraphDatabase  # noqa: PLC0415
        self._drv = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7693"),
            auth=(user or os.environ.get("NEO4J_USERNAME", "neo4j"),
                  password or os.environ.get("NEO4J_PASSWORD", "")),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._drv.close()

    def issues(self, limit: int | None = None) -> list[Issue]:
        """Fetch TaxIssue nodes that carry primary statutes (optionally capped at ``limit``)."""
        cy = """
        MATCH (t:TaxIssue)
        WHERE t.primary_statutes IS NOT NULL AND size(t.primary_statutes) > 0
        RETURN t.issue_id AS issue_id, coalesce(t.name,'') AS name,
               coalesce(t.user_expressions,[]) AS ue, coalesce(t.aliases,[]) AS al,
               coalesce(t.search_keywords,[]) AS kw,
               coalesce(t.primary_statutes,[]) AS prim, coalesce(t.secondary_statutes,[]) AS sec
        """ + (f"LIMIT {int(limit)}" if limit else "")
        with self._drv.session() as s:
            return [Issue(issue_id=str(r["issue_id"]), name=r["name"],
                          user_expressions=list(r["ue"]), aliases=list(r["al"]),
                          search_keywords=list(r["kw"]), primary_statutes=list(r["prim"]),
                          secondary_statutes=list(r["sec"])) for r in s.run(cy)]

    def _clauses_for_refs(self, refs: list[str]) -> list[Article]:
        pairs = [p for p in (parse_statute_ref(r) for r in refs) if p]
        if not pairs:
            return []
        # Match each (law, num) PAIR exactly. Unwinding only the numbers and filtering by a law
        # *set* would admit the cross-product (LawA 제2조 for a "LawA 제1조 | LawB 제2조" issue),
        # injecting wrong gold positives. We narrow the query by num, then keep only real pairs.
        want = {(_norm(law), num) for law, num in pairs}
        nums = list({num for _, num in pairs})
        cy = """
        UNWIND $nums AS n
        MATCH (c) WHERE any(l IN labels(c) WHERE l IN $labels) AND c.clause_num = n
        RETURN c.law_name AS law, c.clause_num AS num, coalesce(c.clause_title,'') AS title,
               coalesce(c.clause_content,'') AS body
        """
        out: list[Article] = []
        with self._drv.session() as s:
            for r in s.run(cy, nums=nums, labels=list(self.CLAUSE_LABELS)):
                if (_norm(r["law"]), str(r["num"])) in want and r["body"]:
                    out.append(Article(r["law"], str(r["num"]), r["title"], r["body"]))
        return out

    def positives(self, issue: Issue) -> list[Article]:
        """Resolve the issue's primary statutes to gold positive articles (exact pair match)."""
        return self._clauses_for_refs(issue.primary_statutes)

    def secondaries(self, issue: Issue) -> list[Article]:
        """Resolve the issue's secondary statutes (bonus negatives)."""
        return self._clauses_for_refs(issue.secondary_statutes)

    def siblings(self, article: Article, window: int, k: int,
                 exclude: set[tuple[str, str]]) -> list[Article]:
        base = article.article_int
        if base is None:
            return []
        lo, hi = base - window, base + window
        cy = """
        MATCH (c) WHERE any(l IN labels(c) WHERE l IN $labels)
          AND replace(c.law_name,' ','') = $law
          AND c.clause_content IS NOT NULL
        WITH c, toInteger(apoc.text.regexGroups(c.clause_num, '\\\\d+')[0][0]) AS ai
        WHERE ai >= $lo AND ai <= $hi AND ai <> $base
        RETURN c.law_name AS law, c.clause_num AS num, coalesce(c.clause_title,'') AS title,
               coalesce(c.clause_content,'') AS body, abs(ai - $base) AS dist
        ORDER BY dist ASC LIMIT $k2
        """
        out: list[Article] = []
        try:
            with self._drv.session() as s:
                for r in s.run(cy, labels=list(self.CLAUSE_LABELS), law=_norm(article.law_name),
                               lo=lo, hi=hi, base=base, k2=k * 3):
                    if (_norm(r["law"]), str(r["num"])) in exclude:
                        continue
                    out.append(Article(r["law"], str(r["num"]), r["title"], r["body"]))
                    if len(out) >= k:
                        break
        except Exception:  # noqa: BLE001 -- no APOC: fall back to a plain numeric scan
            return self._siblings_no_apoc(article, window, k, exclude)
        return out

    def _siblings_no_apoc(self, article: Article, window: int, k: int,
                          exclude: set[tuple[str, str]]) -> list[Article]:
        """Sibling lookup fallback when APOC is unavailable: scan same-law clauses in Python."""
        cy = """
        MATCH (c) WHERE any(l IN labels(c) WHERE l IN $labels)
          AND replace(c.law_name,' ','') = $law AND c.clause_content IS NOT NULL
        RETURN c.law_name AS law, c.clause_num AS num, coalesce(c.clause_title,'') AS title,
               coalesce(c.clause_content,'') AS body
        """
        base = article.article_int
        cands: list[tuple[int, Article]] = []
        with self._drv.session() as s:
            for r in s.run(cy, labels=list(self.CLAUSE_LABELS), law=_norm(article.law_name)):
                m = re.match(r"\d+", str(r["num"]))
                if not m:
                    continue
                dist = abs(int(m.group(0)) - base)
                if 0 < dist <= window and (_norm(r["law"]), str(r["num"])) not in exclude:
                    cands.append((dist, Article(r["law"], str(r["num"]), r["title"], r["body"])))
        cands.sort(key=lambda t: t[0])
        return [a for _, a in cands[:k]]

    def retrieve_pool(self, query: str, k: int) -> list[Article]:
        """Real retrieve pool — wire your live graph here.

        The original builder ran ``graph.astream(interrupt_after=["retrieve"])`` and read
        ``all_retrieval_candidates``. That depends on the live orchestrator app, so it is left
        as an integration point: import your graph and map its rows to :class:`Article`.
        """
        raise NotImplementedError(
            "Wire the live retrieve graph here (see build_ce_trainset.pool_for_query). "
            "Offline runs use JsonlCorpusProvider.retrieve_pool instead."
        )

    def authorities(self, query: str, k: int) -> list[Authority]:
        """Authority (판례/해석례 …) negatives — opt-in stub, like :meth:`retrieve_pool`.

        Raising (rather than returning ``[]``) makes the offline/real divergence explicit:
        ``NegativeChallenger`` guards this lane with ``authority_k > 0`` + ``try/except
        NotImplementedError``, so the real path runs sibling-only until you wire a full-text /
        vector query over authority nodes here.
        """
        raise NotImplementedError(
            "Wire an authority lookup (full-text/vector over 판례/해석례 nodes) to enable "
            "authority negatives on the real path; sibling negatives work without it."
        )
