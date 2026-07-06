"""Every model call is bounded — an explicit timeout and retry count reach ChatOpenAI.

Without these the OpenAI client's defaults apply, and a proxied provider can stall one
request long enough to pin a research unit (and its concurrency slot) for the rest of
the run. Pins: the dataclass defaults, both override paths (``configurable`` beats env),
and that a deliberate ``max_retries=0`` survives instead of being coerced back to the
default by a falsy-or chain.

Runs with plain Python (``python tests/test_model_request_limits.py``) — no pytest
needed — and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import os

from deep_research_agent.config import ResearchConfig
from deep_research_agent.models import build_chat_model

_ENV_KEYS = ("DRA_REQUEST_TIMEOUT", "DRA_MAX_RETRIES")


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    """Build a config with the request-limit env vars masked (so ambient values can't
    leak into the assertions), optionally setting controlled ones via `env`."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in env or {}:
            os.environ.pop(k, None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_defaults_match_dataclass() -> None:
    cfg = _cfg()
    assert cfg.request_timeout == ResearchConfig.request_timeout
    assert cfg.max_retries == ResearchConfig.max_retries


def test_env_overrides_defaults() -> None:
    cfg = _cfg({"DRA_REQUEST_TIMEOUT": "45", "DRA_MAX_RETRIES": "1"})
    assert cfg.request_timeout == 45.0
    assert cfg.max_retries == 1


def test_configurable_beats_env() -> None:
    cfg = _cfg({"DRA_REQUEST_TIMEOUT": "45", "DRA_MAX_RETRIES": "1"},
               request_timeout=90, max_retries=5)
    assert cfg.request_timeout == 90.0
    assert cfg.max_retries == 5


def test_explicit_zero_retries_survives() -> None:
    """0 means "don't retry" — a real choice, not an absent value."""
    assert _cfg(max_retries=0).max_retries == 0
    assert _cfg({"DRA_MAX_RETRIES": "0"}).max_retries == 0
    # Negative input is nonsense to the SDK; clamp rather than pass it through.
    assert _cfg(max_retries=-3).max_retries == 0


def test_limits_reach_the_chat_model() -> None:
    cfg = _cfg(request_timeout=30, max_retries=2)
    model = build_chat_model("openai/gpt-4o-mini", cfg)
    assert model.request_timeout == 30.0   # ChatOpenAI's field name for `timeout`
    assert model.max_retries == 2


if __name__ == "__main__":
    test_defaults_match_dataclass()
    test_env_overrides_defaults()
    test_configurable_beats_env()
    test_explicit_zero_retries_survives()
    test_limits_reach_the_chat_model()
    print("OK — model request timeout/retries verified.")
