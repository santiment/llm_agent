"""A sandbox session that died mid-run (service timeout, lost container) is replaced once,
the call retried, and the model told that earlier files are gone. HTTP is stubbed."""

from __future__ import annotations

from deep_research_agent.sandbox import _RESET_NOTE, HttpSandboxBackend


def _backend(exec_outputs_by_session, *, raise_404_for=None):
    calls: list[tuple] = []
    sessions = iter(["s1", "s2", "s3"])
    b = HttpSandboxBackend("http://sbx.invalid", seed_files=[("/workspace/recipes.py", b"R=1")])

    def fake_http(method, path, body=None, *, timeout=None):
        calls.append((method, path))
        if method == "POST" and path == "/sessions":
            return {"session_id": next(sessions)}
        sid = path.split("/")[2]
        if raise_404_for and sid == raise_404_for:
            raise RuntimeError(f"sandbox {method} {path} -> HTTP 404: session not found")
        if path.endswith("/exec"):
            return {"stdout": exec_outputs_by_session[sid], "exit_code": 0}
        return {}

    b._http = fake_http
    return b, calls


def test_dead_container_in_exec_output_reopens_and_retries(capture_events, caplog):
    b, calls = _backend({"s1": "Error response from daemon: No such container: llmsbx_abc\n",
                         "s2": "42\n"})
    with caplog.at_level("WARNING"):
        res = b.execute("python /workspace/x.py")
    assert res.output == _RESET_NOTE + "42\n" and res.exit_code == 0
    assert [f"{m} {p}" for m, p in calls] == [
        "POST /sessions", "PUT /sessions/s1/files", "POST /sessions/s1/exec",
        "POST /sessions", "PUT /sessions/s2/files", "POST /sessions/s2/exec"]  # re-seeded
    assert b._session_id == "s2"
    resets = [e for e in capture_events if e.get("state") == "sandbox_reset"]
    assert len(resets) == 1 and "No such container" in resets[0]["detail"]
    assert "is gone" in caplog.text


def test_http_404_on_session_reopens_too():
    b, calls = _backend({"s2": "ok\n"}, raise_404_for="s1")
    res = b.execute("echo hi")
    assert res.output.endswith("ok\n") and res.output.startswith(_RESET_NOTE)
    assert b._session_id == "s2"


def test_healthy_session_is_untouched():
    b, calls = _backend({"s1": "fine\n"})
    assert b.execute("echo").output == "fine\n"
    assert b.execute("echo").output == "fine\n"
    assert sum(1 for m, p in calls if p == "/sessions") == 1


def test_other_errors_still_raise():
    b, _ = _backend({})
    def boom(method, path, body=None, *, timeout=None):
        if path == "/sessions":
            return {"session_id": "s1"}
        raise RuntimeError("sandbox POST /sessions/s1/exec -> HTTP 500: boom")
    b._http = boom
    try:
        b.execute("echo")
    except RuntimeError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("a non-session error must propagate")
