"""Document-text builder for the reranker — train == inference.

This is the home of **Fix #2** from the analysis.

The v2 CE builder rendered a statute as::

    best_sub_text   or   f"{clause_title}: {content}"

`clause_title` is e.g. ``"제133조(장기보유특별공제)"`` — it carries the *article number* but
**not the law name**. The analysis measured that only 0.73% of training docs had their own law
name anywhere in the text, so the cross-encoder literally could not tell whether "제133조" was
소득세법시행령 133 or 부가가치세법 133. That removed the single most decisive token for the
"same article number, different law" failure (problem B).

:func:`candidate_text` keeps the v2 coalesce chain byte-for-byte (so the body content matches
inference) and **prepends ``법령명 + 조문번호``**. The same function is used for positives,
negatives, and — on the real path — for the live serving text, so train and inference never
diverge.
"""
from __future__ import annotations

import re
from typing import Any

from .schemas import Article, Authority

# Inference serving truncates to 3000 chars (RERANKER_MAX_DOC_CHARS in the live vLLM pooling).
# Keep None to train on full text, or set to 3000 to match serving exactly. The analysis flags
# the train(full) != inference(3000) mismatch; expose it as one honest knob rather than hide it.
RERANKER_MAX_DOC_CHARS: int | None = None

# The body coalesce chain, identical to the live `_candidate_text` (clause + authority share it).
_BODY_FIELDS = (
    "clause_content", "chunk_text", "summary", "query_summary",
    "answer", "holding", "full_text",
)


def _norm(s: Any) -> str:
    """Whitespace-stripped string."""
    return re.sub(r"\s+", "", str(s or ""))


def _truncate(text: str) -> str:
    """Apply the optional serving-parity char cap (no-op when RERANKER_MAX_DOC_CHARS is None)."""
    return text[:RERANKER_MAX_DOC_CHARS] if RERANKER_MAX_DOC_CHARS else text


def _coalesce_body(row: dict[str, Any]) -> str:
    """First non-empty body field in the live ``_candidate_text`` order."""
    for f in _BODY_FIELDS:
        v = str(row.get(f) or "").strip()
        if v:
            return v
    return ""


def law_prefix(law_name: Any, clause_num: Any) -> str:
    """``"소득세법시행령 제133조"`` — the discriminating header that v2 omitted.

    Falls back gracefully: law-name only, or article-number only, if one is missing.
    """
    law = str(law_name or "").strip()
    num = str(clause_num or "").strip()
    # Only statute article numbers (123, 133의2) get the 제…조 wrapper. Authority case numbers
    # (서면법규과-733, 대법원2018두12345) are kept verbatim.
    m = re.fullmatch(r"(\d+)(?:의(\d+))?", num) if num else None
    if m and not num.startswith("제"):
        num = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
    return " ".join(p for p in (law, num) if p)


def candidate_text(row: dict[str, Any]) -> str:
    """Render a raw row (statute or authority) exactly as the reranker will see it.

    Order: ``best_sub_text`` (if the retriever set it) wins, otherwise
    ``"{law_prefix} {clause_title}: {body}"``. Law prefix is prepended in *both* branches so a
    retriever-provided ``best_sub_text`` still gets its law name.
    """
    prefix = law_prefix(row.get("law_name") or row.get("law"),
                        row.get("clause_num") or row.get("num"))

    best = str(row.get("best_sub_text") or "").strip()
    if best:
        text = f"{prefix} {best}".strip() if prefix else best
        return _truncate(text)

    title = str(row.get("clause_title") or row.get("title") or "").strip()
    body = _coalesce_body(row)
    head = " ".join(p for p in (prefix, title) if p)
    if head and body:
        text = f"{head}: {body}"
    else:
        text = head or body
    return _truncate(text)


def article_text(a: Article) -> str:
    """Render an :class:`Article` exactly as the reranker sees it (law-prefixed)."""
    return candidate_text({
        "law_name": a.law_name, "clause_num": a.clause_num,
        "clause_title": a.clause_title, "clause_content": a.clause_content,
    })


def authority_text(a: Authority) -> str:
    """Render an :class:`Authority` (판례/해석례 …) with its law-type prefix."""
    return candidate_text({
        "law": a.law, "num": a.case_number, "title": a.title, "answer": a.body,
    })
