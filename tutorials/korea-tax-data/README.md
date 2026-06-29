# Korea-Tax Reranker Trainset — Agentic Self-Instruct ×  CE Builder

A data-generation codebase for fine-tuning a Korean-tax GraphRAG reranker
(`BAAI/bge-reranker-v2-m3`). It fuses two things:

1. the **agentic accept/refine loop** from this repo's `autodata` package
   (*Agentic Self-Instruct*, arXiv 2606.25996v1), and
2. the **CE-reranker trainset shape** (`{query, pos[], neg[]}`, FlagEmbedding format) from the
   `2026-06-25-ce-trainset` builder,

and it bakes in the two fixes the trainset post-mortem identified (see
[`docs/ANALYSIS.md`](docs/ANALYSIS.md)):

- **Fix #1 — same-law *sibling-article* hard negatives.** The previous builder used the whole
  retrieve pool (~79 scattered articles) as negatives, so the model learned "query vs unrelated
  article", not the real failure "소득세법시행령 133 vs 131·132·134" (92% of regressions).
- **Fix #2 — law-name-prefixed document text.** The previous builder rendered docs as
  `"{제목}: 본문"`, dropping the law name (only 0.73% of docs had it), so the cross-encoder could
  not tell which *law* an article number belonged to.

## How the two lineages combine

`autodata` keeps a generated example only when it **separates a weak from a strong solver** by a
margin. Here, the **baseline reranker** plays that role and we keep a training triple only when
the reranker **fails to separate** the negative from the positive — i.e. the negative is
genuinely *confusable*, which is exactly what fine-tuning should fix.

| autodata (QA self-instruct)        | here (reranker data)                                |
|------------------------------------|-----------------------------------------------------|
| Challenger generates a question    | `NegativeChallenger` generates hard negatives       |
| Weak + strong solver answer        | Baseline reranker scores `pos` vs `neg`             |
| Accept if `strong − weak ≥ gap`    | Accept if negatives are *hard* (`score ≥ min_pos − margin`) |
| `TOO_EASY` → harder question       | `TOO_EASY` → wider sibling window, re-roll          |
| Quality verifier / rubric judge    | LLM teacher roles (false-neg + positive discovery)  |
| `MockProvider` (offline)           | `MockReranker` + `JsonlCorpusProvider` (offline)    |
| OpenAI/vLLM/NIM provider           | Neo4j corpus + CrossEncoder + OpenAI LLM (real)     |

Full mapping in [`docs/DESIGN.md`](docs/DESIGN.md).

## Quickstart (offline — no Neo4j, no GPU, no network)

```bash
cd tutorials/korea-tax-data
pip install pyyaml

python -m korea_tax_data.cli build --config config/offline.yaml
python -m korea_tax_data.cli split --config config/offline.yaml
python -m korea_tax_data.cli eval  --config config/offline.yaml --split test
```

`build` runs the agentic loop over the bundled synthetic corpus
(`data/sample_corpus.json`) and writes:

- `output/ce_trainset.jsonl` — FlagEmbedding rows `{query, pos, neg, issue_id, neg_sources}`
- `output/trajectories/<issue>.json` — per-issue round log (accept / too-easy / leaked)

A successful run prints a negative-source breakdown where **`sibling` dominates** `pool` and
`authority` — Fix #1 in action — and shows the held-out issue (`A-003`) dropped by the leak
guard.

### Tests

```bash
pip install pytest && python -m pytest -q     # 19 tests
```

## Real path (Neo4j 7693 + bge-reranker-v2-m3 + OpenAI)

Swap the config; the commands are identical. The LLM-role provider is reused from this repo's
`autodata.llm`, which speaks the OpenAI-compatible `/v1/chat/completions` interface — so the same
config runs against **OpenAI, a self-hosted vLLM endpoint, or NVIDIA NIM** by changing only
`llm.provider.base_url`/`model` (see the host repo's
[`paper/`](../../paper/) and NIM tutorial for the provider lineup). The hardness-judge reranker
is independent and stays `bge-reranker-v2-m3` (or a fine-tuned checkpoint).

```bash
export NEO4J_URI=bolt://127.0.0.1:7693 NEO4J_USERNAME=neo4j NEO4J_PASSWORD=...
export OPENAI_API_KEY=sk-...
pip install neo4j sentence-transformers openai

python -m korea_tax_data.cli build --config config/neo4j_crossencoder.yaml
python -m korea_tax_data.cli split --config config/neo4j_crossencoder.yaml
```

Then fine-tune in an **isolated** env (never touches a live reranker server):

```bash
CUDA_VISIBLE_DEVICES=<free gpu> PY=/path/to/ce-train/python bash scripts/train.sh
python -m korea_tax_data.cli eval --config config/neo4j_crossencoder.yaml --split test \
    --model output/ft-bge-reranker-v2-m3        # vs --model BAAI/bge-reranker-v2-m3
```

Two integration seams are left as explicit stubs (documented in code):
`Neo4jCorpusProvider.retrieve_pool` (wire your live `graph.astream(interrupt_after=["retrieve"])`)
and the held-out gold path in the config. Sibling + authority negatives — the Fix #1 emphasis —
work **without** the live graph.

## Layout

```text
korea_tax_data/
  schemas.py        Article / Issue / Candidate / RerankerExample (+ accept-status constants)
  doc_text.py       Fix #2 — law-name-prefixed candidate_text (train == inference)
  corpus.py         CorpusProvider: JsonlCorpusProvider (offline) + Neo4jCorpusProvider (real)
  reranker.py       RerankerScorer: MockReranker (offline) + CrossEncoderReranker (real)
  negatives.py      Fix #1 — NegativeChallenger (sibling-first, escalating)
  hardness.py       HardnessCriteria — autodata's accept/reject ported to rerankers
  llm_roles.py      paraphrase / false-negative / positive-discovery on autodata.llm providers
  orchestrator.py   AgenticRerankerData — the per-query accept/refine loop
  leakguard.py      held-out string-bridge (issue_id + sibling expr + gold-article guard)
  builder.py        corpus -> JSONL driver
  split.py / eval.py / cli.py
config/             offline.yaml + neo4j_crossencoder.yaml
data/               sample_corpus.json + sample_heldout.json (synthetic)
scripts/train.sh    FlagEmbedding fine-tune (isolated env)
docs/               DESIGN.md (mapping) + ANALYSIS.md (post-mortem confirmation)
tests/              19 tests (offline, deterministic)
```

## Honest limits

- The bundled corpus is **synthetic** (no real legal text); it exists to make the pipeline run
  offline and to keep tests deterministic. The real path reads the same fields from Neo4j.
- The offline `MockReranker` is lexical. It demonstrates the *loop and the negative mix*, not
  the numeric payoff of Fix #2 — that requires the real cross-encoder. In particular, on
  sibling-only rows the law-name prefix doesn't change the mock's ranking (siblings share the
  law); the prefix earns its keep across laws and via the real CE's learned associations.
- This generates and validates **data**; it does not claim a green A/B. The post-mortem
  ([`docs/ANALYSIS.md`](docs/ANALYSIS.md)) is explicit that the prior FT regressed, so treat the
  real-CE A/B as the gate before shipping any checkpoint.
