"""LLM provider abstraction.

Each logical role (challenger, weak solver, strong solver, judge, quality verifier) gets
its own provider instance, so they can point at different endpoints/models -- mirroring the
paper's setup (Kimi-K2.6 as challenger/judge, Qwen3.5-397B strong, Qwen3.5-4B weak).

Two implementations:
  * OpenAICompatibleProvider -- talks to any OpenAI /v1/chat/completions endpoint, which is
    exactly what vLLM serves. Plug in your H200 vLLM base_url and you are running real models.
  * MockProvider -- deterministic, no network. Lets the whole orchestration loop run and be
    tested offline, and lets us *demonstrate* acceptance/rejection by controlling how well the
    weak vs strong solver answer.
"""
from __future__ import annotations

from typing import Protocol, Optional
import hashlib
import json
import time


class LLMProvider(Protocol):
    """Minimal chat interface used by every subagent."""
    name: str

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Real provider: OpenAI-compatible (works with vLLM, OpenAI, most gateways).
# ---------------------------------------------------------------------------
class OpenAICompatibleProvider:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        name: str = "",
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.name = name or model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.7,
        json_mode: bool = False,
        max_tokens: int = 2048,
    ) -> str:
        import requests  # imported lazily so offline/mock runs need no deps

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # vLLM and OpenAI both honor this; harmless if the backend ignores it.
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:  # network / 5xx / malformed -> retry with backoff
                last_err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"[{self.name}] completion failed after {self.max_retries} tries: {last_err}")


# ---------------------------------------------------------------------------
# Mock provider: deterministic, offline, role-aware.
# ---------------------------------------------------------------------------
# The mock keys its behaviour off a ROLE tag that subagents embed in the system prompt
# (e.g. "[ROLE:challenger]"). For solvers it also reads a "[STRENGTH:weak|strong]" tag and
# a "[DIFFICULTY:0..100]" tag injected by the orchestrator, so we can drive a controllable
# weak/strong gap and exercise both the accept and reject branches without any model.
class MockProvider:
    def __init__(self, name: str = "mock", strength: str = "weak", seed: int = 0):
        self.name = name
        self.strength = strength          # "weak" | "strong"
        self.seed = seed

    @staticmethod
    def _tag(text: str, key: str, default: str = "") -> str:
        marker = f"[{key}:"
        i = text.find(marker)
        if i == -1:
            return default
        j = text.find("]", i)
        return text[i + len(marker):j].strip() if j != -1 else default

    def _hash01(self, *parts: str) -> float:
        h = hashlib.sha256(("|".join(parts) + f"|{self.seed}").encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def complete(self, system, user, temperature=0.7, json_mode=False, max_tokens=2048) -> str:
        role = self._tag(system, "ROLE", "solver")

        if role == "challenger":
            return self._mock_challenger(system, user)
        if role == "quality_verifier":
            return self._mock_quality_verifier(system, user)
        if role == "judge":
            return self._mock_judge(system, user)
        # default: a solver answer (judge will later score it)
        return self._mock_solver(system, user)

    # --- challenger: emit a QAItem as JSON. The first attempt is an "easy, high-level"
    #     question (as the paper notes) and difficulty escalates each round, mirroring the
    #     agent moving toward harder, more specific questions until the gap opens up. ---
    def _mock_challenger(self, system, user) -> str:
        rnd = int(self._tag(system, "ROUND", "1"))
        # per-paper offset + slope so different papers take different numbers of rounds
        paper_off = self._hash01("paper", user[:200]) * 8.0          # 0..8
        slope = 6.0 + self._hash01("slope", user[:200]) * 4.0        # 6..10 per round
        difficulty = min(100, int(2 + paper_off + (rnd - 1) * slope))
        rubric = [
            {"criterion": f"Correctly identifies key mechanism #{k}", "weight": 5 + (k % 3),
             "category": "positive"} for k in range(1, 9)
        ] + [
            {"criterion": f"Avoids reasoning error #{k}", "weight": -(2 + k), "category": "negative"}
            for k in range(1, 4)
        ]
        qa = {
            "context": f"[mock r{rnd}] Problem setup grounded in the source paper; no answer leaked.",
            "question": f"[mock round {rnd}] Predict the outcome under the paper's constraints.",
            "reference_answer": "[mock] The paper's specific finding applied to the scenario.",
            "rubric": rubric,
            "question_type": "outcome prediction",
            "reasoning_tags": ["causal_reasoning", "design_tradeoff"],
            "_mock_difficulty": difficulty,   # carried so solver/judge can read it back
        }
        return json.dumps(qa, ensure_ascii=False)

    def _mock_quality_verifier(self, system, user) -> str:
        # Accept by default; flag leakage only on a deterministic minority.
        leak = self._hash01("qv", user[:64]) < 0.1
        return json.dumps({
            "check_1_leakage": "LEAKS_ANSWER" if leak else "NO_LEAKAGE",
            "check_2_quality": "GOOD",
            "check_3_rubric": "PASS",
            "overall": "FAIL" if leak else "PASS",
            "feedback": "context leaks the answer" if leak else "",
        })

    def _mock_solver(self, system, user) -> str:
        # Strong solver answers better than weak; embed a self-score the mock judge reads.
        difficulty = float(self._tag(user, "DIFFICULTY", "50") or "50")
        base = 0.90 if self.strength == "strong" else 0.75
        # higher difficulty lowers both, but lowers the weak solver much more.
        slope = 0.0030 if self.strength == "strong" else 0.0085
        noise = (self._hash01(self.strength, user[:48]) - 0.5) * 0.06
        target = max(0.0, min(1.0, base - slope * difficulty + noise))
        return f"[mock {self.strength} answer] [SELFSCORE:{target:.3f}] ..."

    def _mock_judge(self, system, user) -> str:
        # Read the solver's embedded self-score and turn it into per-criterion 0/1 marks
        # whose normalized weighted value matches that score.
        target = float(self._tag(user, "SELFSCORE", "0.5") or "0.5")
        return json.dumps({"normalized_score": round(target, 3)})


def build_provider(cfg: dict, role: str, offline: bool) -> LLMProvider:
    """Factory used by the CLI. `cfg` is the resolved per-role config block."""
    if offline:
        strength = "strong" if role == "strong_solver" else (
            "weak" if role == "weak_solver" else "tool")
        return MockProvider(name=f"mock-{role}", strength=strength, seed=cfg.get("seed", 0))
    return OpenAICompatibleProvider(
        model=cfg["model"],
        base_url=cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=cfg.get("api_key", "EMPTY"),
        name=role,
    )
