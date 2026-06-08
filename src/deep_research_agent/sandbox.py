"""Connect the agent's `execute`/filesystem tools to the llm-sandbox service.

deepagents enables the built-in `execute` tool only when the agent's backend is a
`SandboxBackendProtocol`, and routes ALL file ops through that backend — so wiring the sandbox
means making it the agent's default backend. Three pieces:

  - `HttpSandboxBackend(BaseSandbox)` — implements only `execute`/`upload_files`/
    `download_files`/`id`; `BaseSandbox` derives ls/read/write/edit/grep/glob from those by
    running shell/`python3` in the container. Each method is a SYNC HTTP call to the
    llm-sandbox service (stdlib urllib, no new dep) — correct because deepagents reaches them
    from its async loop via `asyncio.to_thread`. A session is created LAZILY on first use and
    reused for the run.
  - `SandboxCompositeBackend(CompositeBackend, SandboxBackendProtocol)` — keeps `/skills/`
    routing while still being recognized as a sandbox so `execute` turns on (it delegates
    execute/id to the sandbox default).
  - `SandboxCleanupMiddleware` — destroys the run's session in `after_agent` (the service's
    own session timeout is the backstop).
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from langchain.agents.middleware import AgentMiddleware

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
)
from deepagents.backends.sandbox import BaseSandbox

log = logging.getLogger("deep_research_agent.sandbox")

_DEFAULT_EXEC_TIMEOUT = 60


class HttpSandboxBackend(BaseSandbox):
    """SandboxBackendProtocol backed by the llm-sandbox HTTP API (provider-agnostic)."""

    def __init__(self, base_url: str, token: str = "", *, network: bool = False,
                 session_timeout: int = 900, http_timeout: float = 120.0) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._network = network
        self._session_timeout = int(session_timeout)
        self._http_timeout = http_timeout
        # `id` must not create a session (deepagents may read it eagerly), so it's a stable
        # instance id; the actual sandbox session is created lazily on first execute/file op.
        import uuid
        self._id = "sbx-" + uuid.uuid4().hex[:12]
        self._session_id: str | None = None
        self._lock = threading.Lock()

    def _http(self, method: str, path: str, body: dict | None = None, *,
              timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self._base + path, data=data, method=method,
                                     headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._http_timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise RuntimeError(f"sandbox {method} {path} -> HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"sandbox unreachable at {self._base}: {exc.reason}") from exc

    def _ensure_session(self) -> str:
        if self._session_id is None:
            with self._lock:
                if self._session_id is None:
                    data = self._http("POST", "/sessions", {
                        "network": self._network, "timeout_seconds": self._session_timeout})
                    self._session_id = data["session_id"]
                    log.info("sandbox session %s opened (%s)", self._session_id, self._base)
        return self._session_id

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        sid = self._ensure_session()
        secs = int(timeout) if timeout is not None else _DEFAULT_EXEC_TIMEOUT
        data = self._http("POST", f"/sessions/{sid}/exec",
                          {"command": command, "timeout_seconds": secs}, timeout=secs + 30)
        return ExecuteResponse(
            output=(data.get("stdout") or "") + (data.get("stderr") or ""),
            exit_code=data.get("exit_code"),
            truncated=bool(data.get("truncated")),
        )

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        sid = self._ensure_session()
        out: list[FileUploadResponse] = []
        for path, content in files:
            try:
                b64 = base64.b64encode(content).decode("ascii")
                self._http("PUT", f"/sessions/{sid}/files",
                           {"path": path, "content": b64, "encoding": "base64"})
                out.append(FileUploadResponse(path=path))
            except Exception as exc:  # partial-success contract: never raise
                out.append(FileUploadResponse(path=path, error=str(exc)))
        return out

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        sid = self._ensure_session()
        out: list[FileDownloadResponse] = []
        for path in paths:
            try:
                data = self._http("GET", f"/sessions/{sid}/files?path={quote(path)}")
                content = data.get("content", "")
                raw = (base64.b64decode(content) if data.get("encoding") == "base64"
                       else content.encode("utf-8"))
                out.append(FileDownloadResponse(path=path, content=raw))
            except Exception as exc:
                out.append(FileDownloadResponse(path=path, error=str(exc)))
        return out

    def close(self) -> None:
        with self._lock:
            sid, self._session_id = self._session_id, None
        if sid is not None:
            try:
                self._http("DELETE", f"/sessions/{sid}", timeout=15)
                log.info("sandbox session %s closed", sid)
            except Exception as exc:  # the service's session timeout reaps it as a backstop
                log.warning("sandbox session %s close failed: %s", sid, exc)


class SandboxCompositeBackend(CompositeBackend, SandboxBackendProtocol):
    """A CompositeBackend that is ALSO a SandboxBackendProtocol, so deepagents enables the
    `execute` tool. File ops route by prefix (e.g. `/skills/` → its FilesystemBackend, the rest
    → the sandbox default); `execute`/`id` delegate to the sandbox default."""

    @property
    def id(self) -> str:
        return self.default.id  # type: ignore[attr-defined]


class SandboxCleanupMiddleware(AgentMiddleware):
    """Destroy the per-run sandbox session at the end of the run."""

    def __init__(self, backend: HttpSandboxBackend) -> None:
        super().__init__()
        self._backend = backend

    def after_agent(self, state: dict, runtime) -> dict[str, Any] | None:
        try:
            self._backend.close()
        except Exception as exc:  # never break a run on cleanup
            log.warning("sandbox cleanup failed: %s", exc)
        return None
