# OpenAI paper-faithful run on tutorial_paper — 2026-06-26

Companion to `../openai_paper_faithful_2026-06-26/`. Same config and same
model lineup; only the input paper differs. Result: **0/1 accepted, REJECTED
after 12 rounds.**

## Setup

- Config: [`config/openai_paper_faithful_config.yaml`](../../config/openai_paper_faithful_config.yaml)
  (n_attempts=3, max_rounds=12, strong_avg_min=0.65)
- Models: gpt-4.1 (strong) vs gpt-4.1-nano (weak); gpt-4.1-mini orchestration.
- Input: `tutorial/papers/tutorial_paper.txt` (variance-coupled gradient
  clipping for fp8 training of 96-layer transformers).

## Reproduce

```bash
export OPENAI_API_KEY=sk-...
python -m autodata.cli \
    --config config/openai_paper_faithful_config.yaml \
    --papers tutorial/papers \
    --out   output \
    --limit 1
```

## Result

```json
{
  "papers_processed": 1,
  "papers_accepted": 0,
  "acceptance_rate": 0.0,
  "failure_mode_counts": { "FAILED_STRONG": 9, "TOO_EASY": 3 }
}
```

Per-round strong solver average (* = within 0.10 of the 0.65 acceptance line):

| Round | Status | strong_avg |
|------:|:-------|-----------:|
| 1  | TOO_EASY (weak 0.679) | — |
| 2  | FAILED_STRONG          | 0.517 |
| 3  | TOO_EASY (degenerate weak 0.000) | — |
| 4  | FAILED_STRONG          | 0.028 |
| 5  | FAILED_STRONG          | 0.241 |
| 6  | TOO_EASY (weak 0.793) | — |
| 7  | FAILED_STRONG          | 0.190 |
| 8  | FAILED_STRONG          | 0.274 |
| 9  | FAILED_STRONG          | 0.259 |
| 10 | FAILED_STRONG          | 0.370 |
| 11 | FAILED_STRONG          | **0.614** * (0.036 short of acceptance) |
| 12 | FAILED_STRONG          | 0.218 |

Wall-clock: 395 s (~6.6 min).

## The interesting finding: domain matters more than expected

Comparing **all three OpenAI runs on the same lineup** (paper-faithful config,
gpt-4.1 strong vs gpt-4.1-nano weak):

| Paper                | Topic                                            | Outcome             |
|----------------------|--------------------------------------------------|---------------------|
| `sample_paper_beta`  | legal-document reranker (information retrieval)  | **ACCEPTED, round 2** |
| `sample_paper_alpha` | adaptive gradient clipping (fp8 training)        | REJECTED after 12   |
| `tutorial_paper`     | variance-coupled clipping (fp8, 96-layer)        | REJECTED after 12   |

Two fp8-training-internals papers rejected, one information-retrieval paper
accepted. Same orchestrator, same models, same loop knobs — the only thing
that varied was the source paper's domain.

### Why we think this is happening

The fp8-training papers carry their content as *very specific numerical
mechanisms* (`k = 6`, `~400 steps`, `7/8 seeds`, `beta = 0.98`, `EMA on a
CPU fp32 shadow copy`). The judge scores by whether the answer hits the
rubric criteria — which essentially asks "did you mention these specific
mechanisms and numbers?" The strong solver (gpt-4.1) tends to give
*plausible-sounding* but generic answers about gradient clipping and fp8,
which hit some rubric items but not enough to reach 0.65 on average.

So the failure mode isn't "strong is too weak overall" — it's
"strong can't produce the exact specific-numbers answer this rubric
wants, while weak (nano) also can't, so neither separates cleanly." This
is also why we see two near-misses at 0.517 and 0.614: the strong solver
*can* get most of the way there, but reliably nailing every rubric item
across three attempts is hard for this kind of question.

### Implications

Across our small 3-paper corpus the acceptance rate is **33% (1/3)**, and
that is dominated by topic, not by model gap. For richer paper-faithful
results you'd want a corpus that is either (a) larger and topic-diverse
enough that the acceptance rate averages out toward the paper's reported
~50–60%, or (b) intentionally weighted toward conceptual / discussion-style
papers (the kind where rubrics reward *understanding* rather than
*exact-detail recall*).

Either of these is a natural next experiment.
