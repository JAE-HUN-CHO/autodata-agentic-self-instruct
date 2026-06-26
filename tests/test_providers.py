"""Provider tests for the OpenAI / NVIDIA NIM path -- run fully offline.

We inject a fake `requests` module so no network is used: it lets us assert that
OpenAICompatibleProvider resolves keys from the environment and adapts its request shape to
backend 400s (reasoning models that reject `temperature` / want `max_completion_tokens`, and
NIM models that reject `response_format`).

Run:  python tests/test_providers.py   (or)   python -m pytest -q
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodata.llm import OpenAICompatibleProvider, resolve_api_key


# --- fake requests -----------------------------------------------------------
class _FakeResp:
    def __init__(self, status_code: int, body: str = "", content: str = "ok"):
        self.status_code = status_code
        self.text = body
        self._content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.text}")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class _FakeRequests:
    """Replays a scripted list of responses and records every payload it was sent."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.payloads = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.payloads.append(json)
        return self._responses.pop(0)


def _with_fake_requests(fake):
    sys.modules["requests"] = fake  # provider does `import requests` lazily inside complete()


def _provider(**kw):
    kw.setdefault("model", "test-model")
    kw.setdefault("base_url", "https://example/v1")
    kw.setdefault("api_key", "EMPTY")
    return OpenAICompatibleProvider(**kw)


# --- resolve_api_key ---------------------------------------------------------
def test_resolve_api_key_env_and_literal():
    os.environ["AUTODATA_TEST_KEY"] = "sk-secret"
    assert resolve_api_key("env:AUTODATA_TEST_KEY") == "sk-secret"
    assert resolve_api_key("${AUTODATA_TEST_KEY}") == "sk-secret"
    assert resolve_api_key("sk-literal") == "sk-literal"
    assert resolve_api_key("") == "EMPTY"
    del os.environ["AUTODATA_TEST_KEY"]


def test_resolve_api_key_missing_env_raises():
    os.environ.pop("AUTODATA_MISSING_KEY", None)
    try:
        resolve_api_key("env:AUTODATA_MISSING_KEY")
    except RuntimeError as e:
        assert "AUTODATA_MISSING_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError for unset env var")


# --- happy path: standard chat model ----------------------------------------
def test_complete_happy_path_sends_expected_payload():
    fake = _FakeRequests([_FakeResp(200, content="hello")])
    _with_fake_requests(fake)
    out = _provider().complete("sys", "usr", temperature=0.9, json_mode=True, max_tokens=100)
    assert out == "hello"
    p = fake.payloads[0]
    assert p["temperature"] == 0.9
    assert p["max_tokens"] == 100
    assert p["response_format"] == {"type": "json_object"}


# --- reasoning model: rejects temperature, then wants max_completion_tokens ---
def test_adapts_to_reasoning_model_400s():
    fake = _FakeRequests([
        _FakeResp(400, body="Unsupported value: 'temperature' does not support 0.9"),
        _FakeResp(400, body="Use 'max_completion_tokens' instead of 'max_tokens'."),
        _FakeResp(200, content="reasoned"),
    ])
    _with_fake_requests(fake)
    out = _provider().complete("sys", "usr", temperature=0.9, max_tokens=64)
    assert out == "reasoned"
    final = fake.payloads[-1]
    assert "temperature" not in final              # dropped
    assert "max_tokens" not in final               # swapped
    assert final["max_completion_tokens"] == 64
    # adaptations cost no retry budget: all 3 scripted responses were consumed
    assert fake._responses == []


# --- NIM model that can't constrain to JSON ---------------------------------
def test_adapts_when_response_format_unsupported():
    fake = _FakeRequests([
        _FakeResp(400, body="response_format is not supported by this model"),
        _FakeResp(200, content="{\"ok\": 1}"),
    ])
    _with_fake_requests(fake)
    out = _provider().complete("sys", "usr", json_mode=True)
    assert out == "{\"ok\": 1}"
    assert "response_format" not in fake.payloads[-1]


# --- adaptation is remembered for later calls -------------------------------
def test_adaptation_is_sticky_across_calls():
    fake = _FakeRequests([
        _FakeResp(400, body="temperature is not supported"),
        _FakeResp(200, content="a"),
        _FakeResp(200, content="b"),
    ])
    _with_fake_requests(fake)
    prov = _provider()
    assert prov.complete("s", "u") == "a"
    assert prov.complete("s", "u") == "b"          # second call already omits temperature
    assert "temperature" not in fake.payloads[-1]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
