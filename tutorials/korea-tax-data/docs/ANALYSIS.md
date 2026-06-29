# Post-mortem confirmation — why v2 FT regressed, and what this codebase changes

This file records the diagnosis the codebase is built around, **verified against the v2 builder
source** (`2026-06-25-ce-trainset/`). Both claims check out; the VERDICT doc confirms the
resulting fine-tune regressed.

## Symptom (measured, from the v2 bundle)

`eval_data/output/VERDICT_ce_ft_20260629.md`:

| basis (deduped clause bucket, distinct gold 295) | recall@10 |
|---|---|
| retrieve (search score) | 49.5% |
| **base CE (bge-reranker-v2-m3)** | **58.3%** |
| **FT-v2** | **53.6%**  (**−4.7pp**) |

Global lens (all candidate types mixed): base 29.9% → FT-v2 **5.7%** (**−24.2pp**). The
fine-tune lost to the base reranker on *every* basis. So the trainset, not the trainer, is the
suspect.

## Claim 1 — the hard negatives are pool negatives, not same-law siblings

**Verified.** `build_ce_trainset.py :: negatives_from_pool()` emits the *entire* clause bucket of
the live retrieve pool, with no notion of "same law, adjacent article":

```python
for r in pool:
    if str(r.get("node_label") or "") not in CLAUSE_LABELS:
        continue                      # keep any clause-type row …
    ident = row_identity(r)
    if ident in pos_ids or ident in gold_keys:
        continue                      # … that isn't a positive / held-out
    body = doc_text(r)
    ...
    out.append(body)                  # → median ~79 articles, scattered across many laws
```

There is no query that fetches articles by `(same law_name, clause_num ± window)`. The negative
set is therefore "whatever the multi-lane retriever surfaced", which teaches *query vs unrelated
article*, not the actual confusion *소득세법시행령 133 vs 131·132·134*. The analysis measured the
neg article-number spread (median 432, far beyond any single law's article count) and that only
17.7% of negs were within ±5 of the positive — i.e. same-law clustering was incidental, not by
design.

**Fix in this codebase** — [`negatives.py`](../korea_tax_data/negatives.py)
`NegativeChallenger` makes *sibling* the primary source via `CorpusProvider.siblings(article,
window, k)` (same `law_name`, `|article_int − base| ≤ window`), with pool/authority capped to a
small auxiliary. `test_negatives.py::test_sibling_is_primary_source` asserts the mix is
sibling-heavy.

## Claim 2 — the document text has no law name

**Verified.** `build_ce_trainset.py :: doc_text()`:

```python
def doc_text(row):
    text = str(row.get("best_sub_text") or "").strip()
    if not text:
        title = str(row.get("clause_title") or "").strip()      # e.g. "제133조(장기보유특별공제)"
        content = str(row.get("clause_content") or ... or "").strip()
        text = f"{title}: {content}" if (title and content) else (content or title)
    return text[:MAX] if MAX else text
```

`law_name` is present on the row but is used **only** for identity matching (`row_identity`),
never in the rendered text. So a document reads `"제133조(...): 본문"` — the article number is
there (inside `clause_title`), but the *law name* is not. The analysis measured this directly:
only 1541/211405 ≈ **0.73%** of training docs contained their own law name. The cross-encoder
literally cannot tell whether "제133조" is 소득세법시행령 or 부가가치세법.

This is also a **train/inference mismatch risk**: if eval-time docs are rendered with a
`법령명+조문번호` head while training docs are not, the model sees two different formats.

**Fix in this codebase** — [`doc_text.py`](../korea_tax_data/doc_text.py) `candidate_text()`
keeps the v2 body-coalesce chain byte-for-byte and **prepends `법령명 제N조`**. The same function
renders positives, negatives, *and* (on the real path) the live serving text, so train ==
inference by construction. `test_doc_text.py::test_v2_failure_is_fixed` asserts two same-number
articles from different laws now render differently.

## The third lever — keep clause-vs-clause FT from breaking the global lens

The VERDICT notes the FT collapsed *worse* in the global lens (−24pp) than the bucket lens
(−4.7pp): "학습셋이 조문-vs-조문만 봐서 조문-vs-authority 점수 보정 파괴". A clause-only trainset
teaches nothing about how a clause should score *against* a 판례/해석례, so the FT buried clauses
deeper when types mix.

**Mitigation here** — authority negatives are a first-class (small) source (`NEG_AUTHORITY`,
`authority_k`), so the model still sees a little clause-vs-authority contrast instead of zero.
Kept deliberately small, matching the analysis's "판례/해석은 소량만" (type bucketing already
separates them).

## What this codebase does *not* claim

- It generates and leak-guards **data**; the real-CE A/B is still the gate. Given two prior
  failed fine-tunes, treat a green A/B as required before shipping a checkpoint.
- The offline `MockReranker` demonstrates the *loop and negative mix*; the numeric payoff of
  Fix #2 needs the real cross-encoder (the mock is lexical and queries rarely contain law names).
- listwise/NLI post-processing (lever C in the source plan, ~7%) is out of scope, consistent
  with the analysis ranking it last.
