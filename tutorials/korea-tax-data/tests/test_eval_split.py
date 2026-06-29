"""Eval metrics + issue-level split integrity."""
from korea_tax_data.eval import eval_rows
from korea_tax_data.reranker import MockReranker
from korea_tax_data.split import split_rows


def test_eval_perfect_ranking():
    # positive lexically matches the query; negatives don't -> recall@1 == 1.0
    rows = [{"query": "장기보유특별공제 공제율",
             "pos": ["소득세법시행령 제133조 장기보유특별공제 공제율을 곱한 금액"],
             "neg": ["부가가치세법 제40조 공통매입세액 안분"]}]
    out = eval_rows(rows, MockReranker(), k_list=(1,))
    assert out["recall@1"] == 1.0
    assert out["MRR"] == 1.0


def test_eval_handles_empty_neg():
    rows = [{"query": "x", "pos": ["a" * 40], "neg": []}]
    out = eval_rows(rows, MockReranker(), k_list=(1,))
    assert out["rows"] == 0      # rows with no negatives are skipped


def test_split_no_issue_leak():
    rows = [{"query": f"q{i}", "pos": ["p"], "neg": ["n"], "issue_id": f"I-{i % 5}"}
            for i in range(20)]
    sp = split_rows(rows, seed=1)
    ids = {name: {r["issue_id"] for r in rs} for name, rs in sp.items()}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["eval"])
    assert not (ids["eval"] & ids["test"])
    # every row preserved
    assert sum(len(rs) for rs in sp.values()) == 20
