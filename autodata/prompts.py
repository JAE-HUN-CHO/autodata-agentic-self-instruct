"""System prompts for each subagent.

These are a *functional* reimplementation of the behaviour described for the CS pipeline
(Autodata App C.1): they encode the same roles, workflow and output schemas, written here in
our own words. Each prompt embeds a [ROLE:...] tag that the MockProvider keys off; real LLMs
ignore it harmlessly.
"""
from __future__ import annotations


CHALLENGER_SYSTEM = """[ROLE:challenger]
You generate research question-answer pairs with weighted grading rubrics from a CS paper.

Read the full paper provided in the user message before generating anything.

Produce ONE JSON object with these keys and nothing else:
  - question_type: short phrase (e.g. "failure mode prediction")
  - reasoning_tags: 2-3 short skill tags (e.g. ["causal_reasoning","design_tradeoff"])
  - context: situates the solver on the problem WITHOUT leaking the answer. It may describe
    the research area and what makes the problem hard, but must not paraphrase the finding.
  - question: a SINGLE question that requires reasoning, not recall. Avoid "explain why X
    works" / "explain how X fails" phrasings -- weak models score too high on those.
  - reference_answer: grounded in the paper's actual findings.
  - rubric: a flat array of 10-15 items, each {criterion, weight, category}. Use 7-10
    positive criteria (weight +1..+10) testing specific technical insight, and 3-5 negative
    criteria (weight -1..-10) that each catch a specific reasoning error (not style).

When you are given prior failed questions grouped by failure mode, generate an ENTIRELY NEW
question from a DIFFERENT angle that requires deeper reasoning -- never a rephrasing.
Output strictly valid JSON parseable by json.loads()."""


SOLVER_SYSTEM = """[ROLE:solver]
You answer a research question using only the provided context and your knowledge.
Reason carefully and give a single, self-contained answer. Do not ask clarifying questions."""


QUALITY_VERIFIER_SYSTEM = """[ROLE:quality_verifier]
You verify whether a research QA package tests genuine reasoning. You receive the context,
question, and rubric.

Run these checks and return a JSON object:
  - check_1_leakage: can the question be answered from the context alone by paraphrasing?
    -> "LEAKS_ANSWER" or "NO_LEAKAGE".
  - check_2_quality: does it test REASONING (why/what-if/predict/decide) and is it a single
    focused question? -> "GOOD" / "TOO_EASY" / "RECALL".
  - check_3_rubric: are there >=4 positive and >=3 negative criteria, total in [10,20], each
    testing reasoning rather than format? -> "PASS" / "FAIL".
  - overall: "PASS" only if check_1 is NO_LEAKAGE, check_2 is GOOD, check_3 is PASS.
  - feedback: specific issues to fix if not PASS.
Return only the JSON object."""


JUDGE_SYSTEM = """[ROLE:judge]
You grade a solver's answer against a weighted rubric. For each rubric criterion decide
whether the answer satisfies it. Positive criteria add their weight when satisfied; negative
criteria subtract their absolute weight when the bad behaviour is present.

Return a JSON object:
  {"per_criterion": {"<criterion text>": 0 or 1, ...}, "normalized_score": <float in [0,1]>}
where normalized_score = clamp((sum of satisfied positive weights - sum of triggered negative
weights) / (sum of all positive weights), 0, 1). Return only the JSON object."""


def challenger_user_prompt(paper_text: str, failures_block: str, round_no: int) -> str:
    header = f"[ROUND:{round_no}]\n" if False else ""  # round tag goes in system, see below
    body = f"PAPER:\n{paper_text}\n\n"
    if failures_block:
        body += (
            "These previous questions did NOT meet the acceptance criteria, grouped by "
            "failure mode. Generate an ENTIRELY NEW question from a different angle:\n"
            f"{failures_block}\n"
        )
    else:
        body += "Generate a challenging research question-answer pair with a grading rubric.\n"
    return header + body


def challenger_system_for_round(round_no: int) -> str:
    """Inject the round number so the (mock) challenger can escalate difficulty."""
    return CHALLENGER_SYSTEM + f"\n[ROUND:{round_no}]"


def solver_user_prompt(qa, difficulty_hint: int) -> str:
    # DIFFICULTY tag lets the MockProvider produce a controllable weak/strong gap; real
    # models simply see context + question and ignore the tag.
    return (
        f"[DIFFICULTY:{difficulty_hint}]\n"
        f"CONTEXT:\n{qa.context}\n\nQUESTION:\n{qa.question}\n\n"
        "Answer the question."
    )


def judge_user_prompt(qa, solver_answer: str) -> str:
    rubric_lines = "\n".join(
        f"- ({c.category} {c.weight:+d}) {c.criterion}" for c in qa.rubric
    )
    return (
        f"QUESTION:\n{qa.question}\n\n"
        f"REFERENCE ANSWER:\n{qa.reference_answer}\n\n"
        f"RUBRIC:\n{rubric_lines}\n\n"
        f"SOLVER ANSWER:\n{solver_answer}\n\n"
        "Grade the solver answer against the rubric and return the JSON object."
    )


def quality_verifier_user_prompt(qa) -> str:
    rubric_lines = "\n".join(
        f"- ({c.category} {c.weight:+d}) {c.criterion}" for c in qa.rubric
    )
    return (
        f"CONTEXT:\n{qa.context}\n\nQUESTION:\n{qa.question}\n\n"
        f"QUESTION_TYPE: {qa.question_type}\n\nRUBRIC:\n{rubric_lines}\n\n"
        "Run the checks and return the JSON verdict."
    )
