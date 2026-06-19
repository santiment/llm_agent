"""Custom (deployment-specific) tool: current date & time.

Drop-in plugin auto-loaded by ``deep_research_agent.tools.custom``. Gives the
agent a live clock — an LLM has a training cutoff and no sense of "now", so it
can't reliably resolve relative dates ("today", "last 24h", "this quarter") or
stamp a report without one. Pure stdlib, no network, no config, always enabled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from deep_research_agent.tools.custom import CustomTool


class CurrentDateTime(CustomTool):
    """Returns the current date and time. Subclass of CustomTool — auto-discovered
    and exposed to the orchestrator and every research sub-agent."""

    name = "current_datetime"

    description = (
        "Get the current real-world date and time. Use this to anchor any relative "
        "date reasoning ('today', 'yesterday', 'last 24h', 'this year'), to compute "
        "time windows for other tools, and to stamp a report with its as-of date — "
        "do NOT guess the date from training data. Optional `tz` is an IANA timezone "
        "name (e.g. 'America/New_York', 'Europe/London'); omit for UTC. Returns JSON "
        "with ISO-8601 UTC time, the requested timezone's local time, weekday, and a "
        "Unix timestamp. This is a clock, not a research source — no citation needed."
    )

    def run(self, tz: str = "UTC") -> str:
        """Current date/time, in UTC and (optionally) a named timezone.

        Args:
            tz: IANA timezone name (e.g. 'UTC', 'America/New_York', 'Asia/Tokyo').
                An unknown name falls back to UTC and is flagged in ``tz_warning``.
        """
        now_utc = datetime.now(timezone.utc)

        tz_warning = ""
        zone = timezone.utc
        tz_name = "UTC"
        if tz and tz != "UTC":
            try:
                zone = ZoneInfo(tz)
                tz_name = tz
            except (ZoneInfoNotFoundError, ValueError):
                tz_warning = f"unknown timezone {tz!r}; using UTC"

        local = now_utc.astimezone(zone)
        return json.dumps({
            "utc": now_utc.isoformat(timespec="seconds"),
            "timezone": tz_name,
            "local": local.isoformat(timespec="seconds"),
            "date": local.strftime("%Y-%m-%d"),
            "time": local.strftime("%H:%M:%S"),
            "weekday": local.strftime("%A"),
            "unix": int(now_utc.timestamp()),
            **({"tz_warning": tz_warning} if tz_warning else {}),
        })
