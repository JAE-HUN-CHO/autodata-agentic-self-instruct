"""CLI: build / split / eval the reranker training set.

    python -m korea_tax_data.cli build  --config config/offline.yaml
    python -m korea_tax_data.cli split  --config config/offline.yaml
    python -m korea_tax_data.cli eval   --config config/offline.yaml --split test
    python -m korea_tax_data.cli eval   --config config/offline.yaml --split test --law-aware

Everything is driven by a YAML config (corpus / reranker / llm / negatives / hardness / loop),
so the same commands run the offline mock path and the real Neo4j + CrossEncoder + OpenAI path
by swapping the config file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # tutorials/korea-tax-data
# make `autodata` (host repo) importable for the real LLM provider path
sys.path.insert(0, str(ROOT.parents[1]))

from .builder import Builder, BuildConfig
from .corpus import JsonlCorpusProvider
from .hardness import HardnessCriteria
from .leakguard import load_heldout
from .llm_roles import LLMRoles
from .negatives import NegativeChallenger, NegConfig
from .orchestrator import AgenticRerankerData, LoopConfig
from .reranker import MockReranker


def _resolve(path: str) -> Path:
    """Resolve a relative path against the tutorial root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _expand_env(obj):
    """Recursively expand ``$VAR`` / ``${VAR}`` in config string values from the environment.

    Without this, ``uri: ${NEO4J_URI}`` reaches the Neo4j driver as the literal string. Unknown
    variables are left untouched (os.path.expandvars semantics). The ``env:VAR`` api_key form has
    no ``$`` and is handled later by autodata's ``resolve_api_key``.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def _load_config(path: str) -> dict:
    # Resolve the config path CWD-first, then fall back to the tutorial root, so
    # `--config config/offline.yaml` works whether or not the CWD is the tutorial dir
    # (consistent with how _resolve() treats the relative paths inside the config).
    p = Path(path)
    if not p.exists():
        p = _resolve(path)
    return _expand_env(yaml.safe_load(p.read_text(encoding="utf-8")) or {})


def _build_corpus(cfg: dict):
    """Construct the corpus provider (jsonl offline or neo4j real) from config."""
    c = cfg.get("corpus", {})
    ctype = c.get("type", "jsonl")
    if ctype == "jsonl":
        return JsonlCorpusProvider(_resolve(c.get("path", "data/sample_corpus.json")))
    if ctype == "neo4j":
        from .corpus import Neo4jCorpusProvider  # noqa: PLC0415
        return Neo4jCorpusProvider(c.get("uri"), c.get("user"), c.get("password"))
    raise SystemExit(f"unknown corpus.type {ctype!r}")


def _build_reranker(cfg: dict, model_override: str | None = None, law_aware: bool | None = None):
    """Construct the reranker scorer (mock or CrossEncoder) from config / overrides."""
    r = cfg.get("reranker", {})
    model = model_override or r.get("model", "mock")
    la = r.get("law_aware", False) if law_aware is None else law_aware
    if model.lower() in ("mock", "mock-reranker") or r.get("type") == "mock":
        return MockReranker(law_aware=la)
    from .reranker import CrossEncoderReranker  # noqa: PLC0415
    return CrossEncoderReranker(model)


def _build_llm(cfg: dict) -> LLMRoles:
    """Construct LLM roles; disabled config yields deterministic no-op roles."""
    lc = cfg.get("llm", {})
    if not lc.get("enabled"):
        return LLMRoles(provider=None)
    from autodata.llm import build_provider  # noqa: PLC0415
    provider = build_provider(lc.get("provider", {}), role="judge", offline=False)
    return LLMRoles(provider=provider, enabled=True)


def _engine(cfg: dict, corpus, reranker, llm) -> AgenticRerankerData:
    """Assemble the agentic engine (challenger + hardness criteria + loop) from config."""
    challenger = NegativeChallenger(corpus, NegConfig(**cfg.get("negatives", {})))
    criteria = HardnessCriteria(**cfg.get("hardness", {}))
    loop = LoopConfig(**cfg.get("loop", {}))
    return AgenticRerankerData(corpus, reranker, challenger, criteria, llm, loop,
                               verbose=cfg.get("verbose", True))


def cmd_build(args) -> int:
    """`build` subcommand: run the agentic loop over the corpus and write the trainset."""
    cfg = _load_config(args.config)
    corpus = _build_corpus(cfg)
    reranker = _build_reranker(cfg)
    llm = _build_llm(cfg)
    engine = _engine(cfg, corpus, reranker, llm)
    heldout = load_heldout(_resolve(cfg["heldout"]), corpus.issues())
    print(f"[heldout] issues={len(heldout.issue_ids)} "
          f"sibling_strings={len(heldout.query_strings)} arts={len(heldout.arts_keys)}",
          file=sys.stderr)
    builder = Builder(corpus, engine, heldout, llm, BuildConfig(**cfg.get("build", {})))
    out_dir = _resolve(cfg.get("output", "output"))
    stats = builder.run(out_dir / "ce_trainset.jsonl", out_dir / "trajectories", limit=args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0 if stats["rows_written"] else 1


def cmd_split(args) -> int:
    """`split` subcommand: issue-level train/eval/test split of the trainset."""
    from .split import write_splits  # noqa: PLC0415
    cfg = _load_config(args.config)
    out_dir = _resolve(cfg.get("output", "output"))
    inp = _resolve(args.inp) if args.inp else (out_dir / "ce_trainset.jsonl")
    counts = write_splits(inp, out_dir, seed=args.seed)
    print(json.dumps({"splits": counts}, ensure_ascii=False, indent=2))
    print("[ok] issue-level split, no cross-split issue overlap")
    return 0


def cmd_eval(args) -> int:
    """`eval` subcommand: offline A/B (recall/MRR/nDCG) over a split."""
    from .eval import eval_split  # noqa: PLC0415
    cfg = _load_config(args.config)
    out_dir = _resolve(cfg.get("output", "output"))
    split_path = out_dir / f"{args.split}.jsonl"
    model = args.model or cfg.get("reranker", {}).get("model", "mock")
    out = eval_split(split_path, model, law_aware=args.law_aware)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    """Parse args and dispatch to the build / split / eval subcommand."""
    ap = argparse.ArgumentParser(description="Korea-tax reranker training-data generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate the training set")
    b.add_argument("--config", required=True)
    b.add_argument("--limit", type=int, default=None)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("split", help="issue-level train/eval/test split")
    s.add_argument("--config", required=True)
    s.add_argument("--in", dest="inp", default=None)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    e = sub.add_parser("eval", help="offline A/B over a split")
    e.add_argument("--config", required=True)
    e.add_argument("--split", default="test", choices=["train", "eval", "test"])
    e.add_argument("--model", default=None, help="'mock' or a CrossEncoder model id/path")
    e.add_argument("--law-aware", action="store_true", help="mock only: read law-name prefix")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
