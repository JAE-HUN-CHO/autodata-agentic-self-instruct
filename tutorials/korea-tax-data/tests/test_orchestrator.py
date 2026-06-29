"""Accept/refine loop + leak guard + end-to-end build."""
import tempfile
from pathlib import Path

from conftest import CORPUS, HELDOUT

from korea_tax_data.builder import Builder, BuildConfig
from korea_tax_data.corpus import JsonlCorpusProvider
from korea_tax_data.hardness import HardnessCriteria, evaluate
from korea_tax_data.leakguard import load_heldout
from korea_tax_data.llm_roles import LLMRoles
from korea_tax_data.negatives import NegativeChallenger, NegConfig
from korea_tax_data.orchestrator import AgenticRerankerData, LoopConfig
from korea_tax_data.reranker import MockReranker
from korea_tax_data.schemas import Candidate, ACCEPTED, TOO_EASY, NEG_SIBLING


def _engine(law_aware=False, criteria=None, loop=None):
    c = JsonlCorpusProvider(CORPUS)
    return c, AgenticRerankerData(
        c, MockReranker(law_aware=law_aware),
        NegativeChallenger(c, NegConfig()),
        criteria or HardnessCriteria(min_hard_neg=1, margin=0.5),
        LLMRoles(provider=None),
        loop or LoopConfig(),
    )


# --- hardness gate boundaries (analog of autodata acceptance-criteria tests) ---
def test_hardness_accepts_confusable_negatives():
    negs = [Candidate("n1", ("a", ("1",)), NEG_SIBLING, score=0.9),
            Candidate("n2", ("b", ("2",)), NEG_SIBLING, score=0.85)]
    v = evaluate([0.95], negs, HardnessCriteria(margin=0.2, min_hard_neg=2))
    assert v.status == ACCEPTED and v.n_hard == 2


def test_hardness_rejects_easy_negatives():
    negs = [Candidate("n1", ("a", ("1",)), NEG_SIBLING, score=0.1)]
    v = evaluate([0.95], negs, HardnessCriteria(margin=0.2, min_hard_neg=1))
    assert v.status == TOO_EASY and v.n_hard == 0


def test_run_query_emits_example():
    c, eng = _engine()
    issue = next(i for i in c.issues() if i.issue_id == "A-001")
    ex, rounds = eng.run_query(issue.user_expressions[0], c.positives(issue), "A-001")
    assert ex is not None
    assert ex.pos and ex.neg
    assert all(p != n for p in ex.pos for n in ex.neg)   # pos and neg disjoint


def test_too_easy_triggers_escalation():
    # impossible bar -> never accepts -> uses every round, then emits on exhaust
    c, eng = _engine(criteria=HardnessCriteria(min_hard_neg=999, margin=0.0),
                     loop=LoopConfig(max_rounds=3, emit_on_exhaust=True))
    issue = next(i for i in c.issues() if i.issue_id == "A-001")
    ex, rounds = eng.run_query(issue.user_expressions[0], c.positives(issue), "A-001")
    assert len(rounds) == 3
    assert all(r.status == TOO_EASY for r in rounds)


# --- leak guard ---
def test_leak_guard_drops_held_issue():
    c = JsonlCorpusProvider(CORPUS)
    held = load_heldout(HELDOUT, c.issues())
    assert "A-003" in held.issue_ids                       # bridged by user_expression
    assert ("소득세법", ("70",)) in held.arts_keys


# --- end to end ---
def test_build_writes_rows_and_drops_leak():
    c = JsonlCorpusProvider(CORPUS)
    held = load_heldout(HELDOUT, c.issues())
    reranker = MockReranker()
    eng = AgenticRerankerData(c, reranker, NegativeChallenger(c, NegConfig()),
                              HardnessCriteria(min_hard_neg=1, margin=0.5),
                              LLMRoles(provider=None), LoopConfig())
    builder = Builder(c, eng, held, LLMRoles(provider=None), BuildConfig())
    with tempfile.TemporaryDirectory() as d:
        stats = builder.run(Path(d) / "out.jsonl", Path(d) / "traj")
        assert stats["rows_written"] > 0
        assert stats["issues_leaked_dropped"] >= 1            # A-003 held out
        assert stats["negative_source_totals"].get("sibling", 0) > 0
        rows = (Path(d) / "out.jsonl").read_text(encoding="utf-8").splitlines()
        assert all("A-003" not in r for r in rows)            # no leaked issue in output
