"""Fix #2 — the law name must be in the rendered document text (train == inference)."""
from korea_tax_data.doc_text import candidate_text, law_prefix, article_text
from korea_tax_data.schemas import Article


def test_law_prefix_present():
    row = {"law_name": "소득세법시행령", "clause_num": "133",
           "clause_title": "제133조(장기보유특별공제)", "clause_content": "공제율을 곱한 금액..."}
    text = candidate_text(row)
    assert text.startswith("소득세법시행령 제133조"), text
    assert "장기보유특별공제" in text
    assert "공제율" in text


def test_v2_failure_is_fixed():
    # Two different laws, same article number -> v2 ("{title}: {body}") would be ambiguous.
    a = article_text(Article("소득세법시행령", "40", "제40조(가)", "가나다"))
    b = article_text(Article("부가가치세법", "40", "제40조(가)", "가나다"))
    assert a != b                       # law name disambiguates identical title/body
    assert a.startswith("소득세법시행령 제40조")
    assert b.startswith("부가가치세법 제40조")


def test_clause_num_normalized_to_je_jo():
    assert law_prefix("소득세법", "70") == "소득세법 제70조"
    assert law_prefix("소득세법", "제70조") == "소득세법 제70조"


def test_best_sub_text_still_gets_prefix():
    row = {"law_name": "법인세법", "clause_num": "19", "best_sub_text": "손금의 범위는..."}
    assert candidate_text(row).startswith("법인세법 제19조 손금의 범위")


def test_coalesce_body_chain():
    # falls through clause_content -> ... -> answer (authority field)
    row = {"law": "해석례", "num": "서면-1", "title": "제목", "answer": "본문내용입니다"}
    assert "본문내용입니다" in candidate_text(row)
