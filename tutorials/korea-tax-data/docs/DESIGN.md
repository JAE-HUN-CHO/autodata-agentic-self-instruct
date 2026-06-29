# Design — how autodata and the CE builder combine

The goal is one codebase that produces reranker fine-tuning data, built by porting the
*Agentic Self-Instruct* loop onto reranker triples and folding in the CE builder's
domain machinery (statute parsing, leak guard, FlagEmbedding output, splits).

## The central analogy

Agentic Self-Instruct's core idea: **only keep an example that is hard for the current model.**
It measures "hard" as a *capability gap* — a strong solver clears the rubric, a weak solver
doesn't. A question that both answer well is `TOO_EASY` and gets regenerated.

For reranker data the "current model" is the **baseline reranker**, and "hard" means the
reranker **cannot separate the positive from the negative**. A negative the reranker already
scores far below the positive teaches nothing; a negative it ranks at/above the positive is
precisely the mistake fine-tuning must correct. So:

```text
autodata:   accept if  strong_avg − weak_avg ≥ gap_min
here:       accept if  count{ neg : score(neg) ≥ min(score(pos)) − margin } ≥ min_hard_neg
```

Both are "keep only the discriminative item"; the difference is what does the discriminating.

## Component mapping

| autodata module / concept            | this codebase                              | role |
|--------------------------------------|--------------------------------------------|------|
| `schemas.QAItem`                     | `schemas.RerankerExample`                  | the generated unit |
| `TOO_EASY / FAILED_STRONG / ACCEPTED`| `TOO_EASY / NO_NEGATIVES / LEAKED / ACCEPTED` | round outcomes |
| `subagents.Challenger`               | `negatives.NegativeChallenger`             | proposes the hard item |
| weak + strong `Solver`               | `reranker.RerankerScorer`                  | scores candidates |
| `rubric_eval.RubricJudge`            | `hardness.evaluate` + `HardnessCriteria`   | turns scores into accept/reject |
| `orchestrator.AcceptanceCriteria`    | `hardness.HardnessCriteria`                | the thresholds |
| `orchestrator.AgenticSelfInstruct`   | `orchestrator.AgenticRerankerData`         | the per-unit loop |
| `subagents.QualityVerifier`          | `llm_roles.LLMRoles` (false-neg + discovery) | label hygiene |
| `llm.build_provider` / providers     | **reused as-is** by `llm_roles`            | OpenAI/vLLM/NIM access |
| `llm.MockProvider` (offline)         | `reranker.MockReranker` + `corpus.JsonlCorpusProvider` | deterministic offline |
| `cli.py` (config → pipeline → JSONL) | `cli.py` (same shape, reranker pipeline)   | runner |

The LLM provider layer is **literally reused** (`from autodata.llm import build_provider`),
which is the concrete "combine the two codebases" seam: the host repo owns provider quirks
(reasoning-model param swaps, NIM 202/429 handling), and this tutorial inherits all of it for
its paraphrase / false-negative / positive-discovery roles.

## The per-query loop (`AgenticRerankerData.run_query`)

```text
for round = 1..max_rounds:
    cands = NegativeChallenger.generate(positives, query, round, exclude)   # sibling-first
    cands = LLM.find_positives(...)   → promote unlabelled positives out of negatives
    cands = LLM.false_negatives(...)  → drop negatives that are actually answers
    pos_scores, neg_scores = reranker.score(...)
    verdict = hardness.evaluate(pos_scores, cands, criteria)
    if verdict == ACCEPTED: return example
    else (TOO_EASY): escalate — widen sibling window, pull more siblings, retry
return best attempt (emit_on_exhaust) or None
```

Escalation is the reranker analog of autodata regenerating a harder question: a `TOO_EASY`
round means the current siblings were too distinguishable, so round *n+1* widens the article
window (`sibling_window + (n−1)·window_step`) to dig for rarer, closer confusables.

## Why a data layer (`CorpusProvider`) instead of inlined Cypher

The v2 builder hard-wired Neo4j calls throughout. Splitting a `CorpusProvider` Protocol with a
`JsonlCorpusProvider` lets the whole loop run offline and under tests (autodata's MockProvider
philosophy), while `Neo4jCorpusProvider` carries the real Cypher — including the **new sibling
query** the v2 builder never had. Same interface, two backends, identical loop on top.

## Knobs that encode the analysis

| knob (config)                | default | what it controls |
|------------------------------|---------|------------------|
| `negatives.sibling_k`        | 6       | siblings per positive (primary negatives) — Fix #1 |
| `negatives.pool_k`           | 2       | scattered pool negatives (small auxiliary) |
| `negatives.authority_k`      | 1       | 판례/해석례 negatives (smallest) — keeps global-lens contrast |
| `negatives.sibling_window`   | 6       | initial same-law article radius |
| `hardness.margin`            | 0.10    | how close a neg must score to the positive to count as "hard" |
| `hardness.min_hard_neg`      | 3       | accept threshold |
| `loop.max_rounds`            | 4       | escalation budget |
| `doc_text.RERANKER_MAX_DOC_CHARS` | None | None = full text; 3000 = match live serving truncation exactly |
