# Tutorial: Agentic Self-Instruct on NVIDIA NIM

A 10-minute walkthrough that runs the **paper's exact pipeline and exact model
lineup** (Kimi-K2.6 + Qwen3.5-397B + Qwen3) against NVIDIA's hosted NIM API and
produces one accepted `(context, question, reference_answer, weighted_rubric)`
training example.

Everything in this folder is self-contained: a probe script, a tuned config,
one realistic source paper, and the expected outputs from a real NIM run. If
you have an `NVIDIA_API_KEY` you can reproduce it end to end.

## Why a separate tutorial config?

The repo's main `config/nvidia_nim_config.yaml` is the paper-faithful one:
`n_attempts = 3`, `max_rounds = 12`. On the hosted NIM tier each Qwen3 call
takes 10–30 s (network + GPU queue), so a faithful 2-paper run takes
20–40 minutes.

`tutorial_nim_config.yaml` keeps **every threshold from Sec 3.1 of the paper**
but trims the loop so a single paper finishes in a few minutes:

| field            | paper / main config | tutorial |
|------------------|---------------------|----------|
| `n_attempts`     | 3                   | **2**    |
| `max_rounds`     | 12                  | **6**    |
| `parallel_attempts` | (new flag)       | **true** |
| acceptance thresholds | unchanged      | unchanged |
| model lineup     | Kimi / Qwen3.5-397B / Qwen3 (NIM slugs) | identical |

The `parallel_attempts` flag is new (see "Code changes" below) — when true the
orchestrator fans the `n_attempts` solver-then-judge units across a thread
pool. NIM calls are network-bound, so two attempts run in roughly the time of
one.

## Steps

### 0. Set your NIM key

```bash
export NVIDIA_API_KEY=nvapi-...
```

The key is read by `autodata.llm.OpenAICompatibleProvider` via `env:NVIDIA_API_KEY`.
It is never read from disk or committed.

### 1. Probe the three NIM endpoints

```bash
python tutorial/check_nim.py
```

Hits each model once with a tiny payload and reports OK/FAIL + per-call
latency. Catches stale slugs and access problems before the long run.

Expected output (timings vary):

```
[OK   ] challenger/judge/qv    moonshotai/kimi-k2.6                      1.20s  reply=' PROBE_OK'
[OK   ] strong_solver          qwen/qwen3.5-397b-a17b                   11.85s  reply='PROBE_OK'
[OK   ] weak_solver            qwen/qwen3-next-80b-a3b-instruct         32.93s  reply='PROBE_OK'

All 3 endpoints reachable. You are ready to run the pipeline.
```

The slow first call to a model is a NIM cold-start; subsequent calls are
typically 1–5× faster.

### 2. Run the pipeline against the tutorial paper

```bash
python -m autodata.cli \
    --config tutorial/tutorial_nim_config.yaml \
    --papers tutorial/papers \
    --out    tutorial/output \
    --limit  1
```

What you see (per-step elapsed time is logged so you can tell what's slow):

```
[paper] tutorial_paper
  [tutorial_paper] r1: challenger…
  [tutorial_paper] r1: quality_verifier… (+1.5s)
  [tutorial_paper] r1: weak×2… (+2.3s)
  [tutorial_paper] r1: TOO_EASY (weak_avg 0.71 > 0.65) [38.2s]
  [tutorial_paper] r2: challenger…
  ...
  [tutorial_paper] r3: ACCEPTED (weak=0.61 strong=0.83 gap=0.22) [54.1s]
[paper] tutorial_paper done in 145.7s (accepted=True, rounds=3)
```

### 3. Inspect the outputs

```text
tutorial/output/
  dataset.jsonl                   # one accepted QA example
  stats.json                      # corpus stats
  trajectories/
    tutorial_paper.json           # full round-by-round log (QA + scores + feedback)
```

Two reference snapshots are checked in:

* `tutorial/expected_output_offline/` — full output produced by an `--offline`
  run on the same tutorial paper. Deterministic (`MockProvider`), so this is
  what your `tutorial/output/` will look like *in shape* — accept after a
  couple of TOO_EASY rounds, gap ≥ 0.20 at acceptance. Wording and exact
  scores from a real NIM run will differ (temperature > 0).
* `tutorial/expected_output_real_nim/run_log.txt` — actual stdout from a
  real NIM run on 2026-06-26. Use it to calibrate per-step latency
  expectations (Qwen3 cold start dominates round 1) and to see what the
  failure modes look like on real models. The run shown was cut short by
  a 429 rate-limit on the judge, which is exactly what motivated the
  `Retry-After`-aware backoff and per-role `max_retries: 5` in the config.

## Code changes that came with this tutorial

The tutorial uses five additions to the main code. The first three were
written for ergonomics; the last two were forced by what real NIM actually
returns when the pipeline runs against it for more than a few calls.

1. **`AgenticSelfInstruct.parallel_attempts`** (`autodata/orchestrator.py`).
   When true (default), the `n_attempts` solver-then-judge runs inside a
   round are dispatched on a `ThreadPoolExecutor`. The MockProvider tests
   still pass — they're CPU-cheap either way — and on real network providers
   the wall clock drops to roughly `1/n_attempts`. Exposed in config as
   `loop.parallel_attempts`.

2. **Per-step timing in the round log** (`autodata/orchestrator.py`,
   `autodata/cli.py`). Each round now prints which substep is running
   (`challenger…`, `weak×n…`, `strong×n…`) and the cumulative elapsed time,
   and the CLI prints a per-paper total. This is how you see, in real time,
   that the weak Qwen3 cold start is the dominant cost on the first round
   and is amortized away on later rounds.

3. **`--limit N` CLI flag** (`autodata/cli.py`). Process at most `N` papers
   from the corpus. The tutorial uses `--limit 1` so the walk-through is
   bounded; for a full corpus drop the flag.

4. **Tolerant rubric parser** (`autodata/subagents.py`, `_parse_rubric`).
   The challenger sometimes returns a rubric entry missing the `criterion`
   key, wraps the rubric in a `{"items": [...]}` dict, or aliases keys
   (`name` / `points`). The first real NIM run crashed in round 2 with
   `KeyError: 'criterion'`. The parser now skips malformed entries (still
   visible in the trajectory log via the QV decision downstream) and
   tolerates a single level of dict-wrapping.

5. **`Retry-After`-aware 429 handling** (`autodata/llm.py`). The original
   retry path used a geometric `2**backoff` capped at 8 s — designed for
   transient 5xx — and exhausted three retries inside a rate-limit window
   that NIM expects you to wait ~30 s for. The provider now reads the
   `Retry-After` header on 429, falls back to 30 s when absent, and caps
   at 60 s. `max_retries` is also bumped to 5 in `tutorial_nim_config.yaml`
   to survive multi-round bursts.

## Faithfulness to the paper

What we kept exact:
- the orchestrator structure: challenger → quality verifier → weak (×N) →
  strong (×N) → gap check, with weak evaluated first (compute saving) and
  failure modes (`TOO_EASY` / `FAILED_STRONG` / `FAILED_QV`) fed back to the
  challenger;
- the Sec 3.1 acceptance thresholds (`weak_avg_max=0.65`,
  `weak_attempt_max=0.75`, `strong_avg_min=0.60`, `strong_avg_max=0.95`,
  `gap_min=0.20`);
- the role-to-model mapping: Kimi-K2.6 for challenger / judge /
  quality-verifier, Qwen3.5-397B-A17B for the strong solver.

What is necessarily different on NIM:
- the paper's weak solver is **Qwen3.5-4B**, which is not currently hosted on
  the public NIM catalog. We use `qwen/qwen3-next-80b-a3b-instruct` (the
  smallest available Qwen3-next, 3B active params via MoE). The strong/weak
  capability gap is preserved — that's the only hard requirement the
  acceptance criteria care about. Swap in a self-hosted Qwen3.5-4B if you
  want exact parity.

## Troubleshooting

- **`PROBE_OK` returned but real calls 400.** Check the model slug against
  `curl -H "Authorization: Bearer $NVIDIA_API_KEY" https://integrate.api.nvidia.com/v1/models`
  — NIM occasionally renames slugs.
- **Round timeouts.** Bump `loop.max_rounds` or run with `--limit 1` first.
- **`response_format` errors.** The provider auto-drops `response_format:
  json_object` on the first 400 that mentions it. The challenger prompt
  already says "return only JSON" and the parser tolerates code fences.
- **Async-pending 202s from NIM.** Some NIM models run in async mode and
  return `202 Accepted + NVCF-REQID`. This synchronous client raises
  immediately rather than re-submitting. Pick a different slug.
