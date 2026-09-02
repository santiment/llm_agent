"""Skill helper modules (``skills/<skill>/*.py``) are seeded into every sandbox session as
``/workspace/<file>`` right after the session is created, so the model imports tested code
(``import recipes as R``) instead of retyping recipes out of a skill's markdown. HTTP is
stubbed — nothing here talks to a sandbox."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import deep_research_agent.agent as agent_mod
from deep_research_agent.agent import SANDBOX_SEED_DIR, skill_seed_files
from deep_research_agent.sandbox import HttpSandboxBackend

REPO = Path(__file__).resolve().parents[1]


def _backend(seeds):
    calls: list[tuple] = []
    b = HttpSandboxBackend("http://sbx.invalid", seed_files=seeds)

    def fake_http(method, path, body=None, *, timeout=None):
        calls.append((method, path, body))
        if method == "POST" and path == "/sessions":
            return {"session_id": "s1"}
        if path.endswith("/exec"):
            return {"stdout": "ok", "exit_code": 0}
        return {}

    b._http = fake_http
    return b, calls


def test_skill_seed_files_collects_public_py_helpers_only(tmp_path, caplog):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "recipes.py").write_text("A = 1\n")
    (tmp_path / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: x\n---\n")
    (tmp_path / "alpha" / "_private.py").write_text("hidden\n")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "recipes.py").write_text("B = 2\n")      # clashes with alpha's
    (tmp_path / "beta" / "helpers.py").write_text("H = 3\n")
    (tmp_path / "stray.py").write_text("not inside a skill\n")

    with caplog.at_level("WARNING"):
        seeds = skill_seed_files(str(tmp_path))

    assert [p for p, _ in seeds] == [f"{SANDBOX_SEED_DIR}/recipes.py", f"{SANDBOX_SEED_DIR}/helpers.py"]
    assert dict(seeds)[f"{SANDBOX_SEED_DIR}/recipes.py"] == b"A = 1\n"   # first skill wins
    assert "clashes" in caplog.text and "beta" in caplog.text
    assert skill_seed_files("") == []
    assert skill_seed_files(str(tmp_path / "missing")) == []


def test_repo_skills_ship_the_crowd_recipes_module():
    seeds = dict(skill_seed_files(str(REPO / "skills")))
    src = seeds[f"{SANDBOX_SEED_DIR}/recipes.py"]
    compile(src, "recipes.py", "exec")                       # valid Python
    assert b"def card(" in src and b"def dedup_report(" in src
    assert b"import pandas" not in src                       # stdlib only — no image dependency


def test_seeds_upload_once_right_after_session_creation():
    b, calls = _backend([("/workspace/recipes.py", b"X = 1\n")])
    assert b.id.startswith("sbx-") and calls == []           # reading id must not open a session

    out = b.execute("python3 -c 'import recipes'")
    assert out.exit_code == 0
    assert [f"{m} {p}" for m, p, _ in calls] == [
        "POST /sessions", "PUT /sessions/s1/files", "POST /sessions/s1/exec"]
    put = calls[1][2]
    assert put["path"] == "/workspace/recipes.py" and put["encoding"] == "base64"
    assert base64.b64decode(put["content"]) == b"X = 1\n"

    b.execute("echo again")
    assert sum(1 for m, _, _ in calls if m == "PUT") == 1     # once per session, not per call


def test_seed_failure_is_logged_not_raised(caplog):
    b, calls = _backend([("/workspace/recipes.py", b"X")])
    real = b._http

    def flaky(method, path, body=None, *, timeout=None):
        if method == "PUT":
            raise RuntimeError("HTTP 500")
        return real(method, path, body, timeout=timeout)

    b._http = flaky
    with caplog.at_level("WARNING"):
        assert b.execute("echo hi").exit_code == 0
    assert "seed" in caplog.text and "failed" in caplog.text


def test_no_seeds_means_no_put():
    b, calls = _backend([])
    b.execute("echo hi")
    assert [m for m, _, _ in calls] == ["POST", "POST"]


def test_make_graph_hands_skill_seeds_to_the_sandbox(monkeypatch):
    captured: dict = {}

    class _Stub:
        def with_config(self, *a, **k):
            return self

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _Stub()

    monkeypatch.setattr(agent_mod, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.delenv("LLM_SANDBOX_URL", raising=False)
    config = {"configurable": {"openai_api_key": "test-key", "mcp_servers": [],
                               "sandbox_url": "http://sandbox.invalid:8080"}}
    asyncio.run(agent_mod.make_graph(config))

    sandbox = captured["backend"].default
    assert isinstance(sandbox, HttpSandboxBackend)
    assert f"{SANDBOX_SEED_DIR}/recipes.py" in [p for p, _ in sandbox._seed_files]
