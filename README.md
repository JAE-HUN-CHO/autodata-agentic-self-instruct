# Agentic Self-Instruct

A runnable implementation of the **CS-research-paper instantiation** of *Autodata: An agentic
data scientist to create high quality synthetic data* (FAIR at Meta, arXiv `2606.25996v1`,
Sec 3.1 / App C.1). The source paper is included verbatim at
[`paper/Autodata_Agentic_Self_Instruct_arxiv_2606.25996v1.pdf`](paper/Autodata_Agentic_Self_Instruct_arxiv_2606.25996v1.pdf).

An orchestrator agent directs a **challenger**, a **weak solver**, a **strong solver**, a
**quality verifier**, and a **rubric judge** in a loop that produces synthetic
`(context, question, reference_answer, weighted_rubric)` training examples. An example is kept
only if it **separates the weak and strong solver** by a target margin; on failure the
failure mode is fed back to the challenger and a new question is generated from a different
angle, until acceptance or the round budget runs out.

## Why this design

The whole pipeline is **provider-agnostic**: every subagent talks to an OpenAI-compatible chat
endpoint, which is exactly what vLLM serves. So the same code runs:

- **offline**, against a deterministic `MockProvider` (no network, no models) — for testing the
  orchestration logic and demonstrating the accept/reject loop, and
- **for real**, against your own vLLM endpoints — just change the config.

## Install

```bash
pip install -r requirements.txt   # pyyaml, requests
```

## Run offline (no models needed)

```bash
python -m autodata.cli --config config/cs_config.yaml --offline --papers examples --out output
python tests/test_orchestrator.py        # 6 tests (loop / acceptance criteria)
python tests/test_providers.py           # 6 tests (OpenAI / NIM provider, no network)
```

The mock starts each paper with an "easy, high-level" question and escalates difficulty each
round, reproducing the paper's qualitative behaviour: round 1 is usually rejected as
`TOO_EASY`, and `TOO_EASY` dominates the failure-mode counts (paper: ~80% of failed rounds).
Outputs:

- `output/dataset.jsonl` — accepted examples (one JSON object per line)
- `output/trajectories/<paper>.json` — full round-by-round log per paper
- `output/stats.json` — corpus stats (acceptance rate, mean rounds, accepted weak/strong/gap, failure modes)

## Run against your vLLM (H200) endpoints

Edit `config/cs_config.yaml` — each role can point at its own `base_url`/`model`:

```yaml
models:
  challenger:    { model: "kimi-k2.6",          base_url: "http://localhost:8001/v1" }
  judge:         { model: "kimi-k2.6",          base_url: "http://localhost:8001/v1" }
  quality_verifier: { model: "kimi-k2.6",       base_url: "http://localhost:8001/v1" }
  strong_solver: { model: "Qwen3.5-397B-A17B",  base_url: "http://localhost:8002/v1" }
  weak_solver:   { model: "Qwen3.5-4B",         base_url: "http://localhost:8003/v1" }
```

then drop `--offline`:

```bash
python -m autodata.cli --config config/cs_config.yaml --papers /path/to/papers --out output
```

Put one `.txt`/`.md` source document per paper in the `--papers` directory.

## Run against hosted APIs (OpenAI / NVIDIA NIM)

Every subagent speaks the OpenAI-compatible `/v1/chat/completions` interface, so the same code
runs unchanged against OpenAI and NVIDIA NIM — only the config differs. Two ready-made configs
ship in `config/`, and API keys are read from the environment (never committed):

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
python -m autodata.cli --config config/openai_config.yaml --papers examples --out output

# NVIDIA NIM (build.nvidia.com hosted endpoint)
export NVIDIA_API_KEY=nvapi-...
python -m autodata.cli --config config/nvidia_nim_config.yaml --papers examples --out output
```

`api_key` accepts `env:VAR_NAME`, `${VAR_NAME}`, or a literal value. Each role can point at its
own `base_url`/`model`, so you can even mix providers (e.g. strong solver on OpenAI, weak solver
on NIM). The only hard requirement is a genuine capability gap between `strong_solver` and
`weak_solver` — that gap is what the acceptance criteria select for.

Backend quirks are handled automatically, so you don't pick a code path per model family:

- **OpenAI reasoning models** (o1/o3/gpt-5 family) reject a custom `temperature` and require
  `max_completion_tokens` instead of `max_tokens`. On the first `400`, the provider drops
  `temperature` and switches the token field, then retries — without spending a retry. (Standard
  chat models like `gpt-4.1`/`gpt-4o` need none of this and are the cheaper default.)
- **NIM models that can't constrain output to JSON** (`response_format: json_object`) get that
  field dropped automatically; the prompts still ask for JSON and the parser tolerates code
  fences / preamble.

## Acceptance criteria (exact, Sec 3.1)

A question is accepted only when **all** hold across `n_attempts` (=3) solver runs:

| criterion        | threshold                        |
|------------------|----------------------------------|
| weak average     | ≤ 0.65                           |
| best weak attempt| ≤ 0.75 (no high outlier)         |
| weak not degenerate | not all-zero                  |
| strong average   | ≥ 0.60 (set 0.65 for Sec 3.1 prose) and < 0.95 |
| gap              | (strong_avg − weak_avg) ≥ 0.20   |

Compute saving: the weak solver is evaluated **before** the strong solver, and the strong
solver is skipped if the weak criterion already fails (as in the paper).

## File map

```
autodata/
  schemas.py        dataclasses (QAItem, RubricCriterion, RoundResult, PaperResult)
  llm.py            LLMProvider: OpenAICompatibleProvider (vLLM) + MockProvider; build_provider()
  prompts.py        system prompts for challenger / solver / quality_verifier / judge
  subagents.py      Challenger, Solver, QualityVerifier
  rubric_eval.py    RubricJudge: per-criterion -> normalized weighted score
  orchestrator.py   AgenticSelfInstruct loop + AcceptanceCriteria (exact thresholds)
  cli.py            corpus runner -> dataset.jsonl + trajectories + stats
config/cs_config.yaml          vLLM / self-hosted (paper's model lineup)
config/openai_config.yaml      OpenAI API
config/nvidia_nim_config.yaml  NVIDIA NIM (build.nvidia.com)
examples/            two tiny demo "papers"
tests/test_orchestrator.py     loop + acceptance-criteria tests
tests/test_providers.py        OpenAI/NIM provider tests (offline, fake requests)
ai-docs/             paper2code intermediate artifacts (phases 1-3)
paper/               source paper PDF (arXiv 2606.25996v1, verbatim)
```

## Scope & faithfulness notes

- Implements the **CS variant** (hard-coded acceptance criteria), the cleanest of the paper's
  three settings. The legal variant (flexible loop-judge with `grpo_suitability`) and the
  scientific-reasoning variant (binary `≤1/4 weak`, `≥3/4 strong`) are natural extensions of
  the same `AcceptanceCriteria` + orchestrator scaffolding.
- `strong_avg_min` is configurable because the paper states 0.65 in the Sec 3.1 prose and 0.60
  in the App C.1 prompt; default is 0.60.
- System prompts are a functional reimplementation of the described subagent behaviour, not
  verbatim copies of the paper's appendix figures.
- Meta-optimization of the data-scientist agent (Sec 4) is out of scope for this first cut.
```
