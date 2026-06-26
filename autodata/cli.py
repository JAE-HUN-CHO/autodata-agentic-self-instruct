"""CLI runner.

Loads config + a corpus of papers, runs Agentic Self-Instruct over each, and writes:
  * dataset.jsonl       -- accepted (context, question, reference_answer, rubric) examples
  * trajectories/*.json -- full per-paper round-by-round log
  * stats.json          -- corpus-level statistics (Table 1 style)

Offline (mock) usage:
  python -m autodata.cli --config config/cs_config.yaml --offline --papers examples
Real vLLM usage: set provider base_url/model in the config and drop --offline.
"""
from __future__ import annotations

import argparse
import json
import os
from statistics import mean

import yaml

from .llm import build_provider
from .subagents import Challenger, Solver, QualityVerifier
from .rubric_eval import RubricJudge
from .orchestrator import AgenticSelfInstruct, AcceptanceCriteria


def load_papers(path: str) -> list[tuple[str, str]]:
    papers = []
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith((".txt", ".md")):
                with open(os.path.join(path, fn), encoding="utf-8") as f:
                    papers.append((os.path.splitext(fn)[0], f.read()))
    else:
        with open(path, encoding="utf-8") as f:
            papers.append((os.path.splitext(os.path.basename(path))[0], f.read()))
    if not papers:
        raise SystemExit(f"no .txt/.md papers found at {path}")
    return papers


def build_pipeline(cfg: dict, offline: bool) -> AgenticSelfInstruct:
    roles = cfg["models"]
    sampling = cfg.get("sampling", {})
    challenger = Challenger(build_provider(roles["challenger"], "challenger", offline),
                            temperature=sampling.get("challenger", 0.9))
    weak = Solver(build_provider(roles["weak_solver"], "weak_solver", offline),
                  temperature=sampling.get("weak_solver", 0.7))
    strong = Solver(build_provider(roles["strong_solver"], "strong_solver", offline),
                    temperature=sampling.get("strong_solver", 0.7))
    judge = RubricJudge(build_provider(roles["judge"], "judge", offline),
                        temperature=sampling.get("judge", 0.0))
    qv = QualityVerifier(build_provider(roles["quality_verifier"], "quality_verifier", offline),
                         temperature=sampling.get("quality_verifier", 0.0))
    crit = AcceptanceCriteria(**cfg.get("acceptance_criteria", {}))
    loop = cfg.get("loop", {})
    return AgenticSelfInstruct(
        challenger=challenger, weak_solver=weak, strong_solver=strong,
        judge=judge, quality_verifier=qv, criteria=crit,
        n_attempts=loop.get("n_attempts", 3),
        max_rounds=loop.get("max_rounds", 12),
        verbose=True,
    )


def corpus_stats(results) -> dict:
    accepted = [r for r in results if r.accepted]
    rounds_to_accept = [r.n_rounds for r in accepted]
    acc_weak, acc_strong, acc_gap = [], [], []
    for r in accepted:
        last = r.rounds[-1]
        acc_weak.append(mean(last.weak_scores))
        acc_strong.append(mean(last.strong_scores))
        acc_gap.append(last.gap)
    # failure-mode breakdown across all pre-acceptance rounds
    modes: dict[str, int] = {}
    for r in results:
        for rd in r.rounds:
            if rd.status != "ACCEPTED":
                modes[rd.status] = modes.get(rd.status, 0) + 1

    def avg(xs):
        return round(mean(xs), 4) if xs else None

    return {
        "papers_processed": len(results),
        "papers_accepted": len(accepted),
        "acceptance_rate": round(len(accepted) / len(results), 4) if results else 0,
        "mean_rounds_to_accept": avg(rounds_to_accept),
        "accepted_weak_avg": avg(acc_weak),
        "accepted_strong_avg": avg(acc_strong),
        "accepted_gap_avg": avg(acc_gap),
        "failure_mode_counts": modes,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Agentic Self-Instruct data generator")
    ap.add_argument("--config", required=True)
    ap.add_argument("--papers", default=None, help="dir or file of source papers (.txt/.md)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--offline", action="store_true", help="use deterministic MockProvider")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    papers_path = args.papers or cfg.get("papers_path", "examples")
    papers = load_papers(papers_path)
    pipeline = build_pipeline(cfg, args.offline)

    os.makedirs(args.out, exist_ok=True)
    traj_dir = os.path.join(args.out, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)

    results = []
    dataset_path = os.path.join(args.out, "dataset.jsonl")
    with open(dataset_path, "w", encoding="utf-8") as ds:
        for pid, text in papers:
            print(f"[paper] {pid}")
            res = pipeline.run_paper(pid, text)
            results.append(res)
            with open(os.path.join(traj_dir, f"{pid}.json"), "w", encoding="utf-8") as tf:
                tf.write(res.to_json())
            if res.accepted and res.accepted_qa is not None:
                row = res.accepted_qa.to_dict()
                row["paper_id"] = pid
                ds.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = corpus_stats(results)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as sf:
        json.dump(stats, sf, ensure_ascii=False, indent=2)

    print("\n=== corpus stats ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\ndataset -> {dataset_path}")


if __name__ == "__main__":
    main()
