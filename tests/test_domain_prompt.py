"""Domain-prompt slot — the seam that keeps the base prompts domain-neutral.

Deployment-specific guidance (cfg.domain_prompt) lands in BOTH system prompts at the
``<<DOMAIN>>`` slot, wrapped in a labeled DOMAIN CONTEXT block; an empty value
collapses the slot so the prompts read exactly as before. Also pins the config
resolution order (configurable -> DRA_DOMAIN_PROMPT -> DRA_DOMAIN_PROMPT_FILE) and
the installed-package fallback of the default skills/custom_tools dirs.

Runs with plain Python (``python tests/test_domain_prompt.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import os
import tempfile

from deep_research_agent.config import ResearchConfig, _repo_dir
from deep_research_agent.prompts import (
    ORCHESTRATOR_PROMPT,
    SUBAGENT_PROMPT,
    orchestrator_prompt,
    subagent_prompt,
)

_ENV_KEYS = ("DRA_DOMAIN_PROMPT", "DRA_DOMAIN_PROMPT_FILE")

_DOMAIN_TEXT = "Focus on BDC loan portfolios: yield, non-accruals, PIK income."


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    """Build a config with the domain env vars masked (so ambient values can't leak
    into the assertions), optionally setting controlled ones via `env`."""
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


# --- slot injection -----------------------------------------------------------------


def test_domain_text_lands_in_both_prompts():
    for build in (orchestrator_prompt, subagent_prompt):
        p = build("", _DOMAIN_TEXT)
        assert "DOMAIN CONTEXT" in p
        assert _DOMAIN_TEXT in p
        assert "<<DOMAIN>>" not in p


def test_empty_domain_collapses_slot():
    for build in (orchestrator_prompt, subagent_prompt):
        for empty in ("", None, "   \n"):
            p = build("", empty)
            assert "DOMAIN CONTEXT" not in p
            assert "<<DOMAIN>>" not in p


def test_domain_prompt_defaults_to_empty():
    # One-arg call keeps working (existing call sites / external callers).
    assert orchestrator_prompt("") == orchestrator_prompt("", "")
    assert subagent_prompt("") == subagent_prompt("", "")


def test_domain_and_mcp_slots_are_independent():
    p = orchestrator_prompt("MCP-BLOCK-HERE", _DOMAIN_TEXT)
    assert "MCP-BLOCK-HERE" in p and _DOMAIN_TEXT in p


# --- base prompts stay domain-neutral -------------------------------------------------


def test_base_prompts_carry_no_domain_terms():
    # Domain color belongs in the deployment's domain_prompt, never in the engine.
    # If one of these creeps back into the base prompt, move it to a domain prompt.
    banned = ("bitcoin", "btc", "mvrv", "on-chain", "tokenomics", "crypto")
    for prompt in (ORCHESTRATOR_PROMPT, SUBAGENT_PROMPT):
        low = prompt.lower()
        for term in banned:
            assert term not in low, f"domain term {term!r} leaked into a base prompt"


# --- config resolution ----------------------------------------------------------------


def test_configurable_wins_over_env():
    cfg = _cfg(env={"DRA_DOMAIN_PROMPT": "from-env"}, domain_prompt="from-configurable")
    assert cfg.domain_prompt == "from-configurable"


def test_env_inline_wins_over_file():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write("from-file")
    try:
        cfg = _cfg(env={"DRA_DOMAIN_PROMPT": "from-env",
                        "DRA_DOMAIN_PROMPT_FILE": f.name})
        assert cfg.domain_prompt == "from-env"
    finally:
        os.unlink(f.name)


def test_env_file_is_read():
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(_DOMAIN_TEXT)
    try:
        cfg = _cfg(env={"DRA_DOMAIN_PROMPT_FILE": f.name})
        assert cfg.domain_prompt == _DOMAIN_TEXT
    finally:
        os.unlink(f.name)


def test_missing_file_degrades_to_no_domain_prompt():
    cfg = _cfg(env={"DRA_DOMAIN_PROMPT_FILE": "/nonexistent/domain.md"})
    assert cfg.domain_prompt == ""


def test_unset_everywhere_is_empty():
    assert _cfg().domain_prompt == ""


# --- default content dirs: checkout vs installed package -------------------------------


def test_repo_dir_resolves_in_checkout():
    # In a checkout ./skills exists; the default must point at it.
    assert _repo_dir("skills").endswith("skills")


def test_repo_dir_empty_when_absent():
    # Installed as a dependency, parents[2] is site-packages: the content dir does not
    # exist and the default must be "" (feature off), never a garbage path.
    assert _repo_dir("no-such-content-dir") == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all domain-prompt tests passed")


def test_every_prompt_carries_the_current_date():
    from datetime import date
    from deep_research_agent.prompts import (coding_prompt, current_date_line, extract_prompt,
                                             orchestrator_prompt, subagent_prompt)
    today = date.today().isoformat()
    for prompt in (orchestrator_prompt(""), subagent_prompt(""), extract_prompt()):
        assert f"CURRENT DATE: {today} (UTC)" in prompt
        assert "never against the year your training data suggests" in prompt
    assert "CURRENT DATE" not in coding_prompt()   # code has no calendar
    assert "2026-01-31" in current_date_line(date(2026, 1, 31))

