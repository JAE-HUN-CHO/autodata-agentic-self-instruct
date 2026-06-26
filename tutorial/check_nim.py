"""Probe each NIM model used by the tutorial. Run this first.

What it does: sends a one-token request to the three NIM models in the tutorial
config and reports OK/FAIL + latency. If a model is unreachable or its slug
changed in the NIM catalog, you see it here in ~10 seconds instead of after a
long pipeline run.

Usage:
    export NVIDIA_API_KEY=nvapi-...
    python tutorial/check_nim.py
"""
from __future__ import annotations

import os
import sys
import time

# Make `autodata` importable when run as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autodata.llm import OpenAICompatibleProvider  # noqa: E402

BASE = "https://integrate.api.nvidia.com/v1"
MODELS = [
    ("challenger/judge/qv", "moonshotai/kimi-k2.6"),
    ("strong_solver",       "qwen/qwen3.5-397b-a17b"),
    ("weak_solver",         "qwen/qwen3-next-80b-a3b-instruct"),
]


def main() -> int:
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("ERROR: NVIDIA_API_KEY is not set. Export it before running.", file=sys.stderr)
        return 2

    bad = 0
    for role, model in MODELS:
        p = OpenAICompatibleProvider(
            model=model, base_url=BASE, api_key=key,
            name=role, timeout=60, max_retries=1,
        )
        t0 = time.time()
        try:
            out = p.complete(
                system="You are a terse assistant. Reply with the exact phrase: PROBE_OK",
                user="say PROBE_OK",
                temperature=0.0,
                max_tokens=8,
            )
            dt = time.time() - t0
            status = "OK" if "PROBE_OK" in out else "WEIRD"
            if status == "WEIRD":
                bad += 1
            print(f"[{status:<5}] {role:<22} {model:<40} {dt:5.2f}s  reply={out!r}")
        except Exception as e:
            bad += 1
            dt = time.time() - t0
            print(f"[FAIL ] {role:<22} {model:<40} {dt:5.2f}s  err={e}")

    if bad:
        print(f"\n{bad}/{len(MODELS)} endpoints did not return a clean PROBE_OK. "
              f"Fix slugs or access before running the pipeline.")
        return 1
    print(f"\nAll {len(MODELS)} endpoints reachable. You are ready to run the pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
