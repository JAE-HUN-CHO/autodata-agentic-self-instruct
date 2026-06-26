"""CLI tests -- run fully offline with the MockProvider.

Covers the --limit boundary: a positive limit slices the corpus, --limit 0 is an honest
"process zero papers" dry run (empty outputs, no expensive provider calls), and a negative
limit is rejected.

Run:  python tests/test_cli.py   (or)   python -m pytest -q
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodata.cli import main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, "config", "cs_config.yaml")
PAPERS = os.path.join(REPO_ROOT, "examples")  # ships two demo papers


def _run(out_dir, *extra) -> str:
    """Run the CLI offline and return its captured stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["--config", CONFIG, "--offline", "--papers", PAPERS, "--out", out_dir, *extra])
    return buf.getvalue()


def test_limit_one_processes_single_paper():
    with tempfile.TemporaryDirectory() as out:
        stdout = _run(out, "--limit", "1")
        with open(os.path.join(out, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["papers_processed"] == 1
        # The per-paper completion line is a user-visible contract added with --limit;
        # pin it so a format change or a dropped log line is caught here.
        assert "done in" in stdout
        assert "accepted=" in stdout and "rounds=" in stdout


def test_limit_zero_processes_nothing():
    # --limit 0 must honor "at most N" exactly: zero papers, empty dataset, no crash.
    with tempfile.TemporaryDirectory() as out:
        _run(out, "--limit", "0")
        with open(os.path.join(out, "dataset.jsonl"), encoding="utf-8") as f:
            assert f.read() == ""
        with open(os.path.join(out, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["papers_processed"] == 0
        assert stats["acceptance_rate"] == 0
        assert stats["mean_rounds_to_accept"] is None


def test_negative_limit_is_rejected():
    with tempfile.TemporaryDirectory() as out:
        try:
            _run(out, "--limit", "-1")
        except SystemExit as e:
            assert "--limit" in str(e)
        else:
            raise AssertionError("expected SystemExit for negative --limit")


def test_build_pipeline_parallel_attempts_from_config():
    """build_pipeline must pass parallel_attempts from cfg['loop'] to
    AgenticSelfInstruct. When the key is present and True, the pipeline's flag
    is True; when it's False, the flag is False; when absent, it defaults to True
    (the AgenticSelfInstruct default matches the new CLI default)."""
    from autodata.cli import build_pipeline

    base_cfg = {
        "models": {
            "challenger":       {"model": "m"},
            "weak_solver":      {"model": "m"},
            "strong_solver":    {"model": "m"},
            "judge":            {"model": "m"},
            "quality_verifier": {"model": "m"},
        },
        "acceptance_criteria": {},
        "loop": {"n_attempts": 2, "max_rounds": 3, "parallel_attempts": True},
        "sampling": {},
    }

    pipe_true = build_pipeline(dict(base_cfg, loop={"parallel_attempts": True}), offline=True)
    assert pipe_true.parallel_attempts is True

    pipe_false = build_pipeline(dict(base_cfg, loop={"parallel_attempts": False}), offline=True)
    assert pipe_false.parallel_attempts is False

    # When the key is absent, build_pipeline defaults to True
    pipe_default = build_pipeline(dict(base_cfg, loop={}), offline=True)
    assert pipe_default.parallel_attempts is True


def test_limit_processes_exactly_n_papers():
    """--limit N must process exactly N papers when the corpus is larger than N.
    The examples directory has 2 papers; limit=2 must process both."""
    with tempfile.TemporaryDirectory() as out:
        _run(out, "--limit", "2")
        with open(os.path.join(out, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["papers_processed"] == 2


def test_stdout_contains_timing_lines_for_each_paper():
    """The per-paper 'done in X.Xs' lines must appear once for each processed paper."""
    with tempfile.TemporaryDirectory() as out:
        stdout = _run(out, "--limit", "2")
    done_lines = [l for l in stdout.splitlines() if "done in" in l]
    assert len(done_lines) == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
