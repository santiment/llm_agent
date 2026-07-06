# Custom tools (drop-in)

The agent stays generic. Deployment-specific tools live as `*.py` files in the
`custom_tools/` directory and are auto-loaded at startup — **no edits to
`config.py` / `agent.py` / `prompts.py`**. Drop a file in, restart.

## TL;DR — add a tool in 4 lines

```bash
cp custom_tools/_template.py custom_tools/weather.py   # start from the template
```

```python
from deep_research_agent.tools.custom import CustomTool

class WeatherNow(CustomTool):
    name = "weather_now"
    description = "Current weather for a city. Cite as 'OpenWeather'."

    async def run(self, city: str) -> str:
        # self.cfg is the run config; hardcoded result shown for illustration
        return f"{city}: 21°C, clear skies, humidity 48%. Source: OpenWeather."
```

Restart the agent — `weather_now` is now callable by the orchestrator and every
research sub-agent. Nothing else changed.

## The contract: subclass `CustomTool`

This is the easy path for ~all tools. Set two class attributes, implement one
method:

| Member | Required | Purpose |
|--------|----------|---------|
| `name` | ✅ | snake_case, stable, unique — what the model calls. |
| `description` | ✅ | What it does, WHEN to use it, how to cite. The model reads **only** this to decide. Be specific. |
| `run(self, ...)` | ✅ | The tool body. Sync `def` or `async def`. |
| `enabled(cls, cfg)` | optional | Return `False` to skip loading (e.g. a required env var is unset). Default: always on. |

Rules that make it "just work":

- **Args = `run`'s typed params.** `def run(self, query: str, limit: int = 10)`
  gives the model a `{query: string, limit: integer=10}` schema. Avoid `**kwargs`
  — it produces no schema. Sync or async both work.
- **`self.cfg`** is the live `ResearchConfig` — read API keys / URLs / flags off
  it or the environment.
- Define **multiple** `CustomTool` subclasses in one file — all are picked up.

### Return format

The model **always ends up seeing a string** (the tool's `ToolMessage` content).
LangChain coerces the return value for you, so you may return any of:

| Return type | What happens | Use when |
|-------------|--------------|----------|
| `str` | passed through as-is | the default — `json.dumps(...)` structured data yourself, like `social_messages` |
| `list` (of dicts) | JSON-encoded; **also** the shape the agent offloads to a file (row count, columns, head) the `execute` tool reads back | large tabular results |
| `dict` / other JSON-serializable | JSON-encoded automatically | small structured blobs |

Put any numbers/facts the model should cite directly in the returned text. Keep
results lean — a result over `DRA_MAX_RESULT_CHARS` / `DRA_MAX_RESULT_ROWS` is
offloaded to a sandbox file (or truncated if no sandbox), not shown inline.

```python
async def run(self, query: str, limit: int = 10) -> str:
    return json.dumps({
        "query": query,
        "results": [{"title": "Example A", "score": 0.92}],
        "source": "My Source (cite this)",
    })
```

### Conditional loading

Only load a tool when its prerequisite exists:

```python
class SocialMessages(CustomTool):
    name = "social_messages"
    description = "..."

    @classmethod
    def enabled(cls, cfg) -> bool:
        return bool(os.environ.get("DRA_METRICS_HUB_URL"))

    async def run(self, asset: str) -> str:
        base = os.environ["DRA_METRICS_HUB_URL"]
        ...
```

## Escape hatch: a factory (rare)

For dynamic cases — build N tools from config, or hand back an existing
LangChain `BaseTool` — define a factory instead of (or alongside) a class:

```python
from langchain_core.tools import StructuredTool

def build_tools(cfg) -> list:        # or build_tool(cfg) -> single BaseTool
    async def echo(text: str) -> str:
        """Echo the input back."""
        return text
    return [StructuredTool.from_function(
        coroutine=echo, name="echo", description="Echo the input back.")]
```

Both `build_tools` and `build_tool` are accepted. A file may use the class path,
the factory path, or both — the loader collects from each.

## How loading works

- **Directory:** `cfg.custom_tools_dir` (default `<repo>/custom_tools`, override
  with `DRA_CUSTOM_TOOLS_DIR`). Absent dir → no custom tools.
- **Skipped files:** anything starting with `_` (e.g. `_template.py`, shared
  helper modules) is never loaded.
- **Wiring:** loaded tools are appended to the agent's tool list (orchestrator
  **and** sub-agents) and wrapped with the same instrumentation as MCP tools —
  so a large result offloads to a sandbox file the `execute` tool reads back.
- **Discovery:** the model sees each tool's own `name` / `description`. Put usage
  and citation guidance there; no prompt edits needed.
- **Isolation of failures:** import, build, `enabled()`, or factory errors are
  logged and skipped — one bad plugin never takes down the agent or the others.

## Reference

- `custom_tools/_template.py` — copy-paste starting point.
- `src/deep_research_agent/tools/custom.py` — the `CustomTool` base class and loader.
