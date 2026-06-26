"""Provider tests for the OpenAI / NVIDIA NIM path -- run fully offline.

We inject a fake `requests` module so no network is used: it lets us assert that
OpenAICompatibleProvider resolves keys from the environment and adapts its request shape to
backend 400s (reasoning models that reject `temperature` / want `max_completion_tokens`, and
NIM models that reject `response_format`), and that an async 202 is surfaced rather than
silently retried.

Run:  python tests/test_providers.py   (or)   python -m pytest -q
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autodata.llm import OpenAICompatibleProvider, resolve_api_key, _parse_retry_after


# --- fake requests -----------------------------------------------------------
class _FakeResp:
    def __init__(self, status_code: int, body: str = "", content: str = "ok", headers=None):
        self.status_code = status_code
        self.text = body
        self._content = content
        self.headers = headers or {}

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


@contextmanager
def _with_fake_requests(fake):
    # provider does `import requests` lazily inside complete(); save & restore the real module
    # so a fake never leaks into later tests sharing this pytest process.
    old = sys.modules.get("requests")
    sys.modules["requests"] = fake
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop("requests", None)
        else:
            sys.modules["requests"] = old


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
    with _with_fake_requests(fake):
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
    with _with_fake_requests(fake):
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
    with _with_fake_requests(fake):
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
    with _with_fake_requests(fake):
        prov = _provider()
        assert prov.complete("s", "u") == "a"
        assert prov.complete("s", "u") == "b"      # second call already omits temperature
    assert "temperature" not in fake.payloads[-1]


# --- Retry-After parsing: both RFC 7231 forms -------------------------------
def test_parse_retry_after_seconds_form():
    assert _parse_retry_after("30") == 30.0
    assert _parse_retry_after(" 12 ") == 12.0
    assert _parse_retry_after("-5") == 0.0          # clamped to 0
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("not-a-number") is None


def test_parse_retry_after_http_date_form():
    # HTTP-date form: delay = (date - now). Inject `now` so the test is deterministic.
    # 2015-10-21 07:28:00 GMT == 1445412480 epoch seconds.
    target_epoch = 1445412480
    # now is 100s before the target -> ~100s delay
    delay = _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=target_epoch - 100)
    assert delay is not None and abs(delay - 100.0) < 1.0
    # a date already in the past -> clamped to 0
    past = _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=target_epoch + 500)
    assert past == 0.0
    # malformed date -> None (caller falls back to default)
    assert _parse_retry_after("Someday, never o'clock") is None


# --- NIM 429: rate-limited; honor Retry-After and recover ------------------
# These tests patch autodata.llm.time.sleep by hand (save/restore) rather than via the
# pytest `monkeypatch` fixture, because this module is also run as a plain script
# (`python tests/test_providers.py`) where the __main__ runner calls each test with no args.
def test_429_retries_and_honors_retry_after():
    import autodata.llm as llm_mod
    fake = _FakeRequests([
        _FakeResp(429, body="rate limited", headers={"Retry-After": "0"}),
        _FakeResp(200, content="recovered"),
    ])
    slept = []
    original_sleep = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: slept.append(s)  # capture without actually waiting
    try:
        with _with_fake_requests(fake):
            out = _provider().complete("s", "u")
    finally:
        llm_mod.time.sleep = original_sleep
    assert out == "recovered"
    assert slept == [0.0]                # honored the Retry-After: 0
    assert len(fake.payloads) == 2       # retried after the rate limit


def test_429_falls_back_to_default_when_no_retry_after():
    import autodata.llm as llm_mod
    fake = _FakeRequests([
        _FakeResp(429, body="rate limited"),
        _FakeResp(200, content="recovered"),
    ])
    slept = []
    original_sleep = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: slept.append(s)
    try:
        with _with_fake_requests(fake):
            out = _provider().complete("s", "u")
    finally:
        llm_mod.time.sleep = original_sleep
    assert out == "recovered"
    assert slept == [30]                 # default rate-limit wait when header absent


# --- NIM async 202: surfaced loudly, not silently retried -------------------
def test_async_202_raises_without_retrying():
    fake = _FakeRequests([_FakeResp(202, headers={"NVCF-REQID": "req-123"})])
    with _with_fake_requests(fake):
        try:
            _provider(name="nim").complete("s", "u")
        except RuntimeError as e:
            assert "202" in str(e) and "req-123" in str(e)
        else:
            raise AssertionError("expected RuntimeError on 202 async response")
    # the request was sent exactly once -- no duplicate re-submission on retry
    assert len(fake.payloads) == 1
    assert fake._responses == []


# --- _parse_retry_after edge cases: float, zero, and boundary -----------------
def test_parse_retry_after_zero_and_float():
    # "0" must return 0.0, not None (it's a valid non-negative integer per RFC 7231)
    assert _parse_retry_after("0") == 0.0
    # float strings are accepted (not part of RFC 7231 but sent by some gateways)
    result = _parse_retry_after("1.5")
    assert result is not None and abs(result - 1.5) < 1e-9
    # "0.0" should also parse as 0.0
    assert _parse_retry_after("0.0") == 0.0


# --- 429 with all retries exhausted raises RuntimeError ----------------------
def test_429_exhausted_retries_raises():
    import autodata.llm as llm_mod
    # Three 429s; max_retries=3 -> all consumed, must raise
    fake = _FakeRequests([
        _FakeResp(429, body="rate limited", headers={"Retry-After": "0"}),
        _FakeResp(429, body="rate limited", headers={"Retry-After": "0"}),
        _FakeResp(429, body="rate limited", headers={"Retry-After": "0"}),
    ])
    original_sleep = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: None
    try:
        with _with_fake_requests(fake):
            _provider(max_retries=3).complete("s", "u")
    except RuntimeError as e:
        assert "429" in str(e) or "completion failed" in str(e)
    else:
        raise AssertionError("expected RuntimeError when all retries are 429s")
    finally:
        llm_mod.time.sleep = original_sleep


# --- 429: retry-after delay is capped at 60 seconds -------------------------
def test_429_retry_after_capped_at_60():
    import autodata.llm as llm_mod
    # Server sends Retry-After: 999 (far too large); we must cap at 60
    fake = _FakeRequests([
        _FakeResp(429, body="rate limited", headers={"Retry-After": "999"}),
        _FakeResp(200, content="ok"),
    ])
    slept = []
    original_sleep = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: slept.append(s)
    try:
        with _with_fake_requests(fake):
            out = _provider().complete("s", "u")
    finally:
        llm_mod.time.sleep = original_sleep
    assert out == "ok"
    assert len(slept) == 1 and slept[0] == 60  # capped at 60, not 999


# --- thread-safe adaptation: concurrent 400s converge without double-flip ----
def test_thread_safe_adaptation_concurrent():
    """Two threads that both see a temperature-400 must each return True from
    _adapt_from_400 once (the first write), and neither must un-flip the flag
    after the first thread flipped it, because transitions are one-way."""
    import threading
    prov = _provider()
    results = []

    def do_adapt():
        # body mentions "temperature" so the temperature adaptation triggers
        changed = prov._adapt_from_400("temperature is not supported", json_mode=False)
        results.append(changed)

    threads = [threading.Thread(target=do_adapt) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread should have observed changed=True (the first to flip)
    # All subsequent threads see _send_temperature already False, so changed=False.
    assert results.count(True) == 1
    assert results.count(False) == 7
    # The flag is now False and stays False (one-way transition)
    assert prov._send_temperature is False


# --- _build_payload reflects adapted state consistently ----------------------
def test_build_payload_reflects_adapted_state():
    """After adaptation disables temperature and json_format, _build_payload must
    omit those fields and use the updated token param."""
    prov = _provider()
    # initial state: temperature, max_tokens, response_format all present
    p_before = prov._build_payload("sys", "usr", 0.9, True, 128)
    assert "temperature" in p_before
    assert "max_tokens" in p_before
    assert "response_format" in p_before

    # adapt: disable temperature, swap token param, disable json format
    prov._adapt_from_400("temperature is unsupported max_completion_tokens json mode", json_mode=True)

    p_after = prov._build_payload("sys", "usr", 0.9, True, 128)
    assert "temperature" not in p_after
    assert "max_tokens" not in p_after
    assert p_after.get("max_completion_tokens") == 128
    assert "response_format" not in p_after


# --- 429 with Retry-After as HTTP-date form ----------------------------------
def test_429_honors_http_date_retry_after():
    """A Retry-After header in HTTP-date format should also be honored; because
    we can't control 'now' inside the provider, we just verify the provider
    recovers correctly when the delay resolves to a small positive value."""
    import autodata.llm as llm_mod
    # Use a fixed past HTTP-date; _parse_retry_after will clamp it to 0.0
    fake = _FakeRequests([
        _FakeResp(429, body="rate limited",
                  headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}),
        _FakeResp(200, content="ok"),
    ])
    slept = []
    original_sleep = llm_mod.time.sleep
    llm_mod.time.sleep = lambda s: slept.append(s)
    try:
        with _with_fake_requests(fake):
            out = _provider().complete("s", "u")
    finally:
        llm_mod.time.sleep = original_sleep
    assert out == "ok"
    # The HTTP-date is in the past -> clamped to 0.0
    assert len(slept) == 1 and slept[0] == 0.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
