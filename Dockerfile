# The service image for the deep-research-agent (LangGraph HTTP/SSE server).
# Targets:
#   prod (default) — for k8s: non-root, venv + app + runtime assets, nothing else.
#   test           — CI only (Jenkinsfile): the offline pytest suite.
#
#   docker build -t llm-agent .                # prod
#   docker run --rm -p 2024:2024 --env-file .env llm-agent
#
# slim (glibc) over alpine: the dependency tree carries compiled wheels
# (tiktoken, orjson, ...) that do not all publish musl builds.
FROM python:3.11-slim AS build

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
# Pinned: needs >=0.12 for the relative exclude-newer window in pyproject.toml,
# and an exact version keeps builds reproducible (and honours the same ~10-day
# freshness policy the window itself enforces).
RUN pip install --no-cache-dir uv==0.12.5

WORKDIR /app
# Dependencies from the lockfile (uv), project last — keeps the dep layer
# cache-stable. --extra dev is deliberate even for prod: langgraph-cli[inmem]
# is what serves the graph (`langgraph dev`, same bring-up as run.sh); pytest
# rides along, tolerated for the simplicity of a single resolved venv.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra dev --no-install-project
COPY src ./src
RUN uv sync --frozen --extra dev


# --- test: CI only (Jenkinsfile) — never shipped; `prod` stays the default
# target because it is the last stage in the file.
FROM build AS test
COPY tests ./tests
# The suite asserts the checkout-default content dirs resolve (test_domain_prompt's
# test_repo_dir_resolves_in_checkout) — mirror the checkout layout.
COPY skills ./skills
# The build stage never puts the venv on PATH (uv addresses it directly);
# pytest needs it.
ENV PATH="/app/.venv/bin:$PATH"
CMD ["pytest"]


# --- prod (default): non-root, no build tooling -----------------------------
FROM python:3.11-slim AS prod
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/src /app/src
# Runtime assets the server loads from its working directory: the graph
# manifest, drop-in tools (DRA_CUSTOM_TOOLS_DIR=./custom_tools) and skills.
COPY langgraph.json ./
COPY custom_tools ./custom_tools
COPY skills ./skills

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Explicit uid AND gid so the Deployment can pin runAsUser/runAsGroup to
# matching numbers.
RUN groupadd -g 10001 app && useradd -r -u 10001 -g app -s /usr/sbin/nologin app

# The server writes its state dir (.langgraph_api) into the CWD. Pre-create it
# owned by the app user for plain `docker run`; on k8s the root filesystem is
# read-only and the Deployment mounts an emptyDir over this path (see devops
# stage/k8s-apps/llm_agent/deployment.yaml).
RUN mkdir .langgraph_api && chown app:app .langgraph_api

EXPOSE 2024
USER app
# Shell form so DRA_HOST/DRA_PORT from the environment are honoured; exec so
# the server is PID 1 and receives SIGTERM directly.
CMD ["sh", "-c", "exec langgraph dev --host ${DRA_HOST:-0.0.0.0} --port ${DRA_PORT:-2024} --no-browser --no-reload"]
