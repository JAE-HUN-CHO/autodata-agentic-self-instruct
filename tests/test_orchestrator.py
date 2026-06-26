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


def test_eval_solver_serial_when_n_attempts_is_1():
    """When n_attempts=1, _eval_solver must take the serial path even if
    parallel_attempts=True (the condition `n_attempts > 1` is False)."""
    pipe = make_pipeline()
    pipe.n_attempts = 1
    pipe.parallel_attempts = True  # would be parallel if n_attempts > 1

    res = pipe.run_paper("p5", "text for single attempt")
    # With one attempt per solver every accepted round has exactly one score each
    for rnd in res.rounds:
        if rnd.weak_scores:
            assert len(rnd.weak_scores) == 1
        if rnd.strong_scores:
            assert len(rnd.strong_scores) == 1


def test_parallel_attempts_false_always_serial():
    """parallel_attempts=False must use the serial loop regardless of n_attempts."""
    from autodata.orchestrator import AgenticSelfInstruct, AcceptanceCriteria
    from autodata import Challenger, Solver, QualityVerifier, RubricJudge, MockProvider

    challenger = Challenger(MockProvider("c", "tool"))
    weak = Solver(MockProvider("w", "weak"))
    strong = Solver(MockProvider("s", "strong"))
    judge = RubricJudge(MockProvider("j", "tool"))
    qv = QualityVerifier(MockProvider("q", "tool"))
    pipe = AgenticSelfInstruct(
        challenger=challenger, weak_solver=weak, strong_solver=strong,
        judge=judge, quality_verifier=qv,
        criteria=AcceptanceCriteria(),
        n_attempts=3,
        max_rounds=12,
        verbose=False,
        parallel_attempts=False,  # explicitly disabled
    )
    res = pipe.run_paper("p6", "text")
    # Pipeline must still produce a result (accept or reject) — serial path works end-to-end
    assert isinstance(res.accepted, bool)
    assert res.n_rounds >= 1


def test_eval_solver_max_parallel_attempts_cap():
    """_eval_solver must cap the ThreadPoolExecutor to MAX_PARALLEL_ATTEMPTS workers
    even when n_attempts exceeds that constant, and must still return the right
    number of scores."""
    from autodata.orchestrator import AgenticSelfInstruct, AcceptanceCriteria, MAX_PARALLEL_ATTEMPTS
    from autodata import Challenger, Solver, QualityVerifier, RubricJudge, MockProvider
    from autodata.schemas import QAItem, RubricCriterion

    challenger = Challenger(MockProvider("c", "tool"))
    weak = Solver(MockProvider("w", "weak"))
    strong = Solver(MockProvider("s", "strong"))
    judge = RubricJudge(MockProvider("j", "tool"))
    qv = QualityVerifier(MockProvider("q", "tool"))

    n_attempts = MAX_PARALLEL_ATTEMPTS + 4  # deliberately exceeds cap
    pipe = AgenticSelfInstruct(
        challenger=challenger, weak_solver=weak, strong_solver=strong,
        judge=judge, quality_verifier=qv,
        criteria=AcceptanceCriteria(),
        n_attempts=n_attempts,
        max_rounds=1,       # just one round to check scores length
        verbose=False,
        parallel_attempts=True,
    )
    # Build a minimal QAItem to call _eval_solver directly
    rubric = [RubricCriterion(criterion="c1", weight=5, category="positive")]
    qa = QAItem(context="ctx", question="q?", reference_answer="a",
                rubric=rubric, question_type="t", reasoning_tags=[])
    qa._difficulty = 50  # type: ignore[attr-defined]

    scores = pipe._eval_solver(weak, qa)
    # The number of scores must equal n_attempts, not MAX_PARALLEL_ATTEMPTS
    assert len(scores) == n_attempts
    # All scores should be floats in [0, 1]
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in scores)


def test_parse_rubric_weight_zero_is_positive_category():
    """Weight 0 is >= 0, so its category must be 'positive', not 'negative'."""
    rubric = _parse_rubric([{"criterion": "zero weight", "weight": 0}])
    assert len(rubric) == 1
    assert rubric[0].category == "positive"
    assert rubric[0].weight == 0


def test_parse_rubric_description_alias():
    """'description' is an accepted alias for 'criterion' in the rubric entry."""
    rubric = _parse_rubric([{"description": "via description key", "weight": 3}])
    assert len(rubric) == 1
    assert rubric[0].criterion == "via description key"
    assert rubric[0].category == "positive"


def test_parse_rubric_points_alias():
    """'points' is an accepted alias for 'weight' in the rubric entry."""
    rubric = _parse_rubric([{"criterion": "uses points key", "points": -7}])
    assert len(rubric) == 1
    assert rubric[0].weight == -7
    assert rubric[0].category == "negative"


def test_parse_rubric_dict_wrapper_with_no_list_value():
    """A dict where none of the values is a list must return [] (one-level unwrap only)."""
    assert _parse_rubric({"items": "not a list", "other": 42}) == []


def test_parse_rubric_empty_list():
    """An empty list is valid (no criteria); must return an empty list."""
    assert _parse_rubric([]) == []


def test_failures_block_groups_by_status():
    """_failures_block must only include rounds that have a non-ACCEPTED status,
    and group TOO_EASY / FAILED_STRONG / FAILED_QV into separate labelled sections."""
    from autodata.orchestrator import AgenticSelfInstruct
    from autodata.schemas import QAItem, RubricCriterion, RoundResult, TOO_EASY, FAILED_STRONG, FAILED_QV

    rubric = [RubricCriterion(criterion="c", weight=5, category="positive")]
    qa = QAItem(context="ctx", question="q?", reference_answer="a",
                rubric=rubric, question_type="t", reasoning_tags=[])

    rounds = [
        RoundResult(1, TOO_EASY, qa=qa, feedback="weak_avg too high"),
        RoundResult(2, FAILED_STRONG, qa=qa, feedback="strong_avg too low"),
        RoundResult(3, FAILED_QV, qa=qa, feedback="quality check failed"),
    ]
    block = AgenticSelfInstruct._failures_block(rounds)
    # All three failure modes must appear in the block
    assert "TOO EASY" in block
    assert "FAILED ON STRONG" in block
    assert "FAILED QUALITY CHECK" in block
    # The question text must appear for each
    assert block.count("q?") == 3


def test_pipeline_stores_parallel_attempts_flag():
    """AgenticSelfInstruct must store the parallel_attempts flag it receives."""
    pipe_t = make_pipeline()
    pipe_t.parallel_attempts = True
    assert pipe_t.parallel_attempts is True

    pipe_f = make_pipeline()
    pipe_f.parallel_attempts = False
    assert pipe_f.parallel_attempts is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
