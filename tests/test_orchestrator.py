"""Tests for the Agentic Self-Instruct loop (run offline with the MockProvider).

Run:  python -m pytest -q   (or)   python tests/test_orchestrator.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodata import (
    AgenticSelfInstruct, AcceptanceCriteria,
    Challenger, Solver, QualityVerifier, RubricJudge, MockProvider,
    ACCEPTED, TOO_EASY,
)
from autodata.subagents import _parse_rubric


def make_pipeline(max_rounds=12, criteria=None, strong_strength="strong", weak_strength="weak"):
    challenger = Challenger(MockProvider("c", "tool"))
    weak = Solver(MockProvider("w", weak_strength))
    strong = Solver(MockProvider("s", strong_strength))
    judge = RubricJudge(MockProvider("j", "tool"))
    qv = QualityVerifier(MockProvider("q", "tool"))
    return AgenticSelfInstruct(
        challenger=challenger, weak_solver=weak, strong_solver=strong,
        judge=judge, quality_verifier=qv,
        criteria=criteria or AcceptanceCriteria(),
        max_rounds=max_rounds, verbose=False,
    )


# --- acceptance criteria boundary tests ------------------------------------
def test_weak_passes_boundary():
    c = AcceptanceCriteria(weak_avg_max=0.65, weak_attempt_max=0.75)
    assert c.weak_passes([0.6, 0.65, 0.6])[0] is True
    assert c.weak_passes([0.7, 0.7, 0.7])[0] is False          # avg too high
    assert c.weak_passes([0.5, 0.5, 0.8])[0] is False          # single attempt > 0.75
    assert c.weak_passes([0.0, 0.0, 0.0])[0] is False          # degenerate all-zero


def test_strong_passes_boundary():
    c = AcceptanceCriteria(strong_avg_min=0.60, strong_avg_max=0.95)
    assert c.strong_passes([0.7, 0.7, 0.7])[0] is True
    assert c.strong_passes([0.5, 0.5, 0.5])[0] is False        # below min
    assert c.strong_passes([0.97, 0.98, 0.99])[0] is False     # saturated


def test_gap_threshold():
    c = AcceptanceCriteria(gap_min=0.20)
    ok, gap, _ = c.gap_passes([0.40], [0.70])
    assert ok and abs(gap - 0.30) < 1e-9
    bad, gap2, _ = c.gap_passes([0.55], [0.65])
    assert not bad and abs(gap2 - 0.10) < 1e-9


# --- end-to-end loop behaviour ---------------------------------------------
def test_discriminative_paper_is_accepted():
    # default mock has a real weak/strong separation -> should accept within budget
    pipe = make_pipeline()
    res = pipe.run_paper("p1", "some paper text")
    assert res.accepted is True
    assert res.accepted_qa is not None
    assert res.rounds[-1].status == ACCEPTED
    last = res.rounds[-1]
    assert (sum(last.strong_scores) / 3) - (sum(last.weak_scores) / 3) >= 0.20


def test_no_separation_is_rejected():
    # make weak just as strong -> no gap -> never accepted, exhausts rounds
    pipe = make_pipeline(max_rounds=4, weak_strength="strong", strong_strength="strong")
    res = pipe.run_paper("p2", "some paper text")
    assert res.accepted is False
    assert res.n_rounds == 4
    assert all(r.status != ACCEPTED for r in res.rounds)


def test_dataset_serialization_roundtrip():
    pipe = make_pipeline()
    res = pipe.run_paper("p3", "text")
    d = res.to_dict()
    assert d["paper_id"] == "p3"
    assert "rounds" in d and len(d["rounds"]) >= 1


def test_parse_rubric_skips_malformed_entries():
    # Real NIM challenger outputs occasionally drop a key, wrap the rubric in a dict, or
    # use an alias like "name"/"points". The parser should keep valid entries and drop the
    # rest rather than raising mid-pipeline.
    rubric = _parse_rubric([
        {"criterion": "good entry", "weight": 5, "category": "positive"},
        {"weight": 3, "category": "positive"},                   # missing criterion -> drop
        {"criterion": "missing weight"},                          # missing weight -> drop
        {"criterion": "non-numeric weight", "weight": "five"},   # bad weight -> drop
        {"name": "alias key", "points": -2},                      # name+points aliases -> kept
        "not a dict",                                            # wrong type -> drop
        {"criterion": "second good", "weight": -4},               # neg weight -> auto-category
    ])
    crits = [(c.criterion, c.weight, c.category) for c in rubric]
    assert crits == [
        ("good entry", 5, "positive"),
        ("alias key", -2, "negative"),
        ("second good", -4, "negative"),
    ]


def test_parse_rubric_forces_category_to_match_weight_sign():
    # The judge keys scoring off `category` alone, so a contradictory pair must be normalized:
    # a negative weight is always "negative" (a penalty), a non-negative weight "positive",
    # regardless of what the model claimed. Otherwise a penalty would be miscounted as credit.
    rubric = _parse_rubric([
        {"criterion": "penalty mislabeled positive", "weight": -4, "category": "positive"},
        {"criterion": "credit mislabeled negative", "weight": 6, "category": "negative"},
        {"criterion": "garbage category", "weight": 3, "category": "banana"},
    ])
    cats = {c.criterion: c.category for c in rubric}
    assert cats["penalty mislabeled positive"] == "negative"
    assert cats["credit mislabeled negative"] == "positive"
    assert cats["garbage category"] == "positive"


def test_parse_rubric_handles_dict_wrapper_and_garbage():
    assert _parse_rubric(None) == []
    assert _parse_rubric("foo") == []
    # one level of dict unwrap
    out = _parse_rubric({"items": [{"criterion": "x", "weight": 1}]})
    assert len(out) == 1 and out[0].criterion == "x"


def test_parallel_attempts_matches_serial():
    # Parallel and serial attempt dispatch must produce identical accept/reject decisions
    # and matching per-round score sets when the upstream providers are deterministic.
    pipe_serial = make_pipeline()
    pipe_serial.parallel_attempts = False
    pipe_parallel = make_pipeline()
    pipe_parallel.parallel_attempts = True

    res_s = pipe_serial.run_paper("p4", "deterministic text")
    res_p = pipe_parallel.run_paper("p4", "deterministic text")

    assert res_s.accepted == res_p.accepted
    assert res_s.n_rounds == res_p.n_rounds
    # MockProvider is deterministic in input -> output, so the per-round score multisets
    # should match exactly regardless of dispatch order.
    for rs, rp in zip(res_s.rounds, res_p.rounds):
        assert rs.status == rp.status
        assert sorted(rs.weak_scores) == sorted(rp.weak_scores)
        assert sorted(rs.strong_scores) == sorted(rp.strong_scores)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
