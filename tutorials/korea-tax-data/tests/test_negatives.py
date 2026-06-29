"""Fix #1 — negatives are same-law siblings first, with capped pool/authority."""
from conftest import CORPUS

from korea_tax_data.corpus import JsonlCorpusProvider
from korea_tax_data.negatives import NegativeChallenger, NegConfig
from korea_tax_data.schemas import NEG_SIBLING, NEG_POOL, NEG_AUTHORITY, Article


def _corpus():
    return JsonlCorpusProvider(CORPUS)


def test_siblings_are_same_law_adjacent():
    c = _corpus()
    pos = Article("소득세법시행령", "133", "제133조(장기보유특별공제)", "...")
    sibs = c.siblings(pos, window=4, k=10, exclude={("소득세법시행령", "133")})
    laws = {s.law_name for s in sibs}
    nums = {s.clause_num for s in sibs}
    assert laws == {"소득세법시행령"}                 # never crosses into another law
    assert "133" not in nums                          # the positive itself is excluded
    assert {"131", "132", "134", "135"} & nums        # adjacent articles surface


def test_sibling_is_primary_source():
    c = _corpus()
    issue = next(i for i in c.issues() if i.issue_id == "A-001")
    ch = NegativeChallenger(c, NegConfig(sibling_k=6, pool_k=2, authority_k=1))
    cands = ch.generate(c.positives(issue), issue.user_expressions[0], round_no=1, exclude_ids=set())
    by_src = {}
    for cand in cands:
        by_src[cand.source] = by_src.get(cand.source, 0) + 1
    assert by_src.get(NEG_SIBLING, 0) >= by_src.get(NEG_POOL, 0)     # sibling-heavy
    assert by_src.get(NEG_POOL, 0) <= 2                              # pool capped
    assert by_src.get(NEG_AUTHORITY, 0) <= 1


def test_escalation_widens_window():
    c = _corpus()
    issue = next(i for i in c.issues() if i.issue_id == "A-001")
    ch = NegativeChallenger(c, NegConfig(sibling_window=2, sibling_k=4, window_step=10))
    r1 = ch.generate(c.positives(issue), "장기보유특별공제", 1, set())
    r3 = ch.generate(c.positives(issue), "장기보유특별공제", 3, set())
    sib1 = sum(1 for x in r1 if x.source == NEG_SIBLING)
    sib3 = sum(1 for x in r3 if x.source == NEG_SIBLING)
    assert sib3 >= sib1                                # wider window -> at least as many siblings


def test_positives_never_appear_as_negatives():
    c = _corpus()
    issue = next(i for i in c.issues() if i.issue_id == "A-002")  # two positives
    pos = c.positives(issue)
    pos_ids = {(p.law_name.replace(" ", ""), tuple([p.clause_num])) for p in pos}
    ch = NegativeChallenger(c, NegConfig())
    cands = ch.generate(pos, issue.user_expressions[0], 2, set())
    for cand in cands:
        assert cand.identity not in pos_ids
