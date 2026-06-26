# OpenAI paper-faithful run — 2026-06-26

First **real accepted training example** generated end-to-end on the OpenAI API.

## Setup

- Config: [`config/openai_paper_faithful_config.yaml`](../../config/openai_paper_faithful_config.yaml)
- Paper-faithful loop knobs (Sec 3.1 / App C.1):
  - `n_attempts = 3`
  - `max_rounds = 12`
  - `strong_avg_min = 0.65` (Sec 3.1 prose, tighter than App C.1's 0.60)
- Models: `gpt-4.1` (strong solver) vs `gpt-4.1-nano` (weak); `gpt-4.1-mini`
  for challenger / judge / quality_verifier.
- Input: `examples/` (`sample_paper_alpha`, `sample_paper_beta`)

## Reproduce

```bash
export OPENAI_API_KEY=sk-...
python -m autodata.cli \
    --config config/openai_paper_faithful_config.yaml \
    --papers examples \
    --out output
```

## Result summary

```json
{
  "papers_processed": 2,
  "papers_accepted": 1,
  "acceptance_rate": 0.5,
  "mean_rounds_to_accept": 2,
  "accepted_weak_avg": 0.3962,
  "accepted_strong_avg": 0.6918,
  "accepted_gap_avg": 0.2956,
  "failure_mode_counts": { "FAILED_STRONG": 10, "TOO_EASY": 3 }
}
```

- `sample_paper_beta` → **ACCEPTED at round 2** (weak 0.396 / strong 0.692 / gap 0.296)
- `sample_paper_alpha` → REJECTED after 12 rounds (strong never crossed 0.65)
- Wall-clock: ~7.5 min total
- The one accepted item is in `dataset.jsonl`; full round-by-round logs in
  `trajectories/`; the original stdout is in `run_log.txt`.

## What's interesting

- Acceptance rate (50%) lines up with the paper's general range (~50–60%).
- Accepted weak / strong / gap (0.40 / 0.69 / 0.30) all sit cleanly inside the
  Sec 3.1 thresholds.
- Failure-mode mix is **inverted** from the paper. Paper: TOO_EASY dominates
  (~80%). Here: FAILED_STRONG dominates (10/13). The challenger
  (`gpt-4.1-mini`) tends to overshoot the difficulty just past what the
  strong solver (`gpt-4.1`) can solve — a real, lineup-specific observation.
- Paper-alpha vs paper-beta behaved very differently with the same orchestrator
  — paper-alpha never opened the gap (strong stuck at 0.31–0.61), paper-beta
  accepted on the second round. Domain familiarity of the strong solver matters
  more than rate-of-progress through rounds.
