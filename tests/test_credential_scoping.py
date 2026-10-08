"""The deployment's credentials only go where the deployment pointed them.

``configurable`` comes from the calling app, so a per-run override must never carry the
server's own secrets to a host the caller chose — the same key-exfiltration guard
``base_url`` already has (``DRA_ALLOWED_BASE_URLS``):

  - ``LLM_SANDBOX_TOKEN`` is bound to ``LLM_SANDBOX_URL``; a caller-supplied
    ``sandbox_url`` gets only a ``sandbox_token`` the caller supplied too.
  - ``DRA_MCP_BEARER`` is attached only to MCP servers configured by env, never to
    ``configurable.mcp_servers`` / ``mcp_config``.

Also pins MCP connection-name de-duplication: two bare hosts both normalize to
``…/mcp`` and used to share the key ``mcp``, silently dropping the first server.

Runs with plain Python (``python tests/test_credential_scoping.py``) — no pytest needed —
and is also pytest-discoverable. No network, no API keys.
"""

from __future__ import annotations

import os

from deep_research_agent.config import ResearchConfig

_ENV_KEYS = ("LLM_SANDBOX_URL", "LLM_SANDBOX_TOKEN", "DRA_MCP_BEARER", "DRA_MCP_SERVERS",
             "DRA_MCP_URL", "DRA_MCP_LABEL")


def _cfg(env: dict[str, str] | None = None, **configurable) -> ResearchConfig:
    saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
    os.environ.update(env or {})
    try:
        return ResearchConfig.from_runnable_config({"configurable": configurable})
    finally:
        for k in _ENV_KEYS:
            os.environ.pop(k, None)
        os.environ.update({k: v for k, v in saved.items() if v is not None})


_SANDBOX_ENV = {"LLM_SANDBOX_URL": "http://sandbox:8080/", "LLM_SANDBOX_TOKEN": "SECRET"}


def test_env_sandbox_gets_env_token():
    cfg = _cfg(_SANDBOX_ENV)
    assert (cfg.sandbox_url, cfg.sandbox_token) == ("http://sandbox:8080", "SECRET")


def test_caller_sandbox_url_never_receives_env_token():
    cfg = _cfg(_SANDBOX_ENV, sandbox_url="https://evil.example")
    assert cfg.sandbox_url == "https://evil.example"
    assert cfg.sandbox_token == ""


def test_caller_sandbox_url_with_own_token_is_honored():
    cfg = _cfg(_SANDBOX_ENV, sandbox_url="https://other.example", sandbox_token="THEIRS")
    assert (cfg.sandbox_url, cfg.sandbox_token) == ("https://other.example", "THEIRS")


def test_caller_repeating_env_sandbox_url_keeps_env_token():
    cfg = _cfg(_SANDBOX_ENV, sandbox_url="http://sandbox:8080")
    assert cfg.sandbox_token == "SECRET"


def test_metadata_sandbox_url_disables_the_sandbox():
    cfg = _cfg(_SANDBOX_ENV, sandbox_url="http://169.254.169.254/latest")
    assert (cfg.sandbox_url, cfg.sandbox_token) == ("", "")


def test_mcp_bearer_goes_to_env_servers_only():
    env = {"DRA_MCP_BEARER": "MCPSECRET", "DRA_MCP_URL": "http://mcp.internal:8000/mcp/data"}
    from_env = _cfg(env).mcp_servers
    assert from_env[0]["headers"]["Authorization"] == "Bearer MCPSECRET"
    from_caller = _cfg(env, mcp_servers=[{"url": "https://evil.example/mcp"}]).mcp_servers
    assert "Authorization" not in (from_caller[0].get("headers") or {})
    compat = _cfg(env, mcp_config={"url": "https://evil.example"}).mcp_servers
    assert "Authorization" not in (compat[0].get("headers") or {})


def test_mcp_names_are_unique_connection_keys():
    servers = _cfg({"DRA_MCP_SERVERS": '[{"url": "http://a:8000"}, {"url": "http://b:9000"}, '
                                       '{"url": "http://c:7000"}]'}).mcp_servers
    assert [s["name"] for s in servers] == ["mcp", "mcp_2", "mcp_3"]


if __name__ == "__main__":
    test_env_sandbox_gets_env_token()
    test_caller_sandbox_url_never_receives_env_token()
    test_caller_sandbox_url_with_own_token_is_honored()
    test_caller_repeating_env_sandbox_url_keeps_env_token()
    test_metadata_sandbox_url_disables_the_sandbox()
    test_mcp_bearer_goes_to_env_servers_only()
    test_mcp_names_are_unique_connection_keys()
    print("OK — credentials stay with the endpoints the deployment configured.")
