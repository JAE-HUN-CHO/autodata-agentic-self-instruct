"""CLI tests -- run fully offline with the MockProvider.

Covers the --limit boundary: a positive limit slices the corpus, --limit 0 is an honest
"process zero papers" dry run (empty outputs, no expensive provider calls), and a negative
limit is rejected.

Run:  python tests/test_cli.py   (or)   python -m pytest -q
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodata.cli import main

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(REPO_ROOT, "config", "cs_config.yaml")
PAPERS = os.path.join(REPO_ROOT, "examples")  # ships two demo papers


def _run(out_dir, *extra):
    main(["--config", CONFIG, "--offline", "--papers", PAPERS, "--out", out_dir, *extra])


def test_limit_one_processes_single_paper():
    with tempfile.TemporaryDirectory() as out:
        _run(out, "--limit", "1")
        with open(os.path.join(out, "stats.json"), encoding="utf-8") as f:
            stats = json.load(f)
        assert stats["papers_processed"] == 1


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
