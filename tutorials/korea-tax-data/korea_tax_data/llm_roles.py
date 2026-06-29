"""LLM helper roles — reuse autodata's provider layer.

This is the concrete reuse seam with the host repo: the three LLM roles from the original CE
builder (paraphrase / false-negative teacher / unlabelled-positive discovery) run on
``autodata.llm`` providers. The real path passes an ``OpenAICompatibleProvider`` (OpenAI / vLLM
/ NIM); the offline path passes ``None`` and every role degrades to a conservative no-op so the
pipeline stays deterministic and dependency-free.

Iron rule (from SPEC §6): the LLM only ever judges/relabels documents that already exist in the
corpus. It never invents authorities.
"""
from __future__ import annotations

import json
import sys
from typing import Any

try:  # reuse the host repo's provider abstraction
    from autodata.llm import LLMProvider  # noqa: F401  (type hint only)
except Exception:  # noqa: BLE001 -- tutorial may run standalone
    LLMProvider = Any  # type: ignore[misc,assignment]

_ALLOWLIST = "조문·별표·부칙·판례·심판례·해석례·기본통칙·집행기준"


def _extract_json(text: str) -> dict:
    """Best-effort parse of the first JSON object in a model response ({} on failure)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return {}


class LLMRoles:
    """Paraphrase / false-negative / positive-discovery helpers over an optional provider."""

    def __init__(self, provider=None, enabled: bool = True):
        """Wrap an optional provider; with no provider every role becomes a deterministic no-op."""
        self.provider = provider
        self.enabled = enabled and provider is not None

    def _json(self, prompt: str, default: Any) -> Any:
        """Call the provider in JSON mode, returning ``default`` if disabled or on any error."""
        if not self.enabled:
            return default
        try:
            raw = self.provider.complete(system="", user=prompt, temperature=0.0, json_mode=True)
        except Exception as e:  # noqa: BLE001
            print(f"[llm][warn] {e}", file=sys.stderr)
            return default
        data = _extract_json(raw)
        return data if data else default

    def paraphrase(self, query: str, n: int = 2) -> list[str]:
        """Role 1 — return up to ``n`` meaning-preserving single-question rewrites of ``query``."""
        if not self.enabled or n <= 0:
            return []
        p = (f"다음 세무 질문을 의미가 같은 다른 표현 {n}개로 바꿔라(각각 단일 질문). "
             f'원질문: "{query}"\n반드시 JSON: {{"queries":["...","..."]}}')
        out = self._json(p, {"queries": []})
        return [str(q).strip() for q in (out.get("queries") or []) if str(q).strip()][:n]

    def find_positives(self, query: str, pos_titles: list[str],
                       candidates: list[dict[str, Any]]) -> list[int]:
        """Role 3 — indices of candidate docs that are actually answers (unlabelled positives)."""
        if not self.enabled or not candidates:
            return []
        lst = "\n".join(f'{c["i"]}: {c["text"][:90]}' for c in candidates)
        p = (f'질문: "{query}"\n이미 정답표시된 자료: {[t[:40] for t in pos_titles[:5]]}\n'
             f"아래 후보({_ALLOWLIST})에서 이 질문에 **사실상 정답**인 번호만 골라라"
             f"(정답표시 안 됐어도). 무관하면 제외. 후보 밖/가상 금지.\n"
             f'후보:\n{lst}\n반드시 JSON: {{"positives":[번호...]}}')
        out = self._json(p, {"positives": []})
        idxs = {c["i"] for c in candidates}
        return [i for i in (out.get("positives") or []) if isinstance(i, int) and i in idxs]

    def false_negatives(self, query: str, negatives: list[str], k: int = 8) -> list[int]:
        """Role 2 — indices (within the first ``k``) of negatives that are actually answers."""
        if not self.enabled or not negatives:
            return []
        sub = negatives[:k]
        lst = "\n".join(f"{i}: {n[:200]}" for i, n in enumerate(sub))
        p = (f'질문: "{query}"\n아래는 오답(negative) 후보다. 이 중 **사실은 질문의 직접 정답**'
             f"인 것의 번호만 골라라(세법 근거로만, 추측 금지). 없으면 빈 배열.\n"
             f'후보:\n{lst}\n반드시 JSON: {{"false_negatives":[번호...]}}')
        out = self._json(p, {"false_negatives": []})
        return [i for i in (out.get("false_negatives") or [])
                if isinstance(i, int) and 0 <= i < len(sub)]
