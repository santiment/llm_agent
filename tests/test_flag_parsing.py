"""Boolean-knob parsing — only an explicit spelling flips a flag.

``_flag`` resolves every boolean knob (streaming, offload_results, sandbox_network).
The invariant: a recognized on/off spelling ("1/true/yes/on", "0/false/no/off") flips
the value; ANYTHING else keeps the dataclass default, with a warning. This matters most
for ``LLM_SANDBOX_NETWORK`` — a security-relevant flag whose default is off — where a
typo like ``flase`` must not silently open the sandbox's network.

Runs with plain Python (``python tests/test_flag_parsing.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import os

from deep_research_agent.config import ResearchConfig, _flag

_ENV_KEYS = ("DRA_STREAMING", "DRA_OFFLOAD_RESULTS", "LLM_SANDBOX_NETWORK")


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    """Build a config with the flag env vars masked (so ambient values can't leak
    into the assertions), optionally setting controlled ones via `env`."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


def test_explicit_spellings_flip_both_ways():
    for s in ("1", "true", "YES", " on "):
        assert _flag({"x": s}, "x", default=False) is True
    for s in ("0", "false", "No", " OFF "):
        assert _flag({"x": s}, "x", default=True) is False


def test_unrecognized_string_keeps_the_default():
    # A typo is not consent — whichever way the default points.
    assert _flag({"x": "flase"}, "x", default=False) is False
    assert _flag({"x": "flase"}, "x", default=True) is True
    assert _flag({"x": "enable-it"}, "x", default=False) is False


def test_non_strings_pass_through_as_bool():
    assert _flag({"x": True}, "x", default=False) is True
    assert _flag({"x": False}, "x", default=True) is False
    assert _flag({}, "x", default=True) is True


def test_sandbox_network_typo_stays_closed():
    # The finding that motivated this file: garbage input must fail CLOSED here.
    assert _cfg(env={"LLM_SANDBOX_NETWORK": "flase"}).sandbox_network is False
    assert _cfg(env={"LLM_SANDBOX_NETWORK": "true"}).sandbox_network is True
    assert _cfg().sandbox_network is False


def test_streaming_typo_keeps_streaming_on():
    # Same rule, opposite default: garbage falls back to the default (on), as before.
    assert _cfg(env={"DRA_STREAMING": "maybe"}).streaming is True
    assert _cfg(env={"DRA_STREAMING": "off"}).streaming is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all flag-parsing tests passed")
