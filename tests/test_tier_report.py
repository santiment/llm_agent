"""`./run.sh tiers` — the tier listing and the stream timer behind `--measure`.

Pins (``tier_report.py``): every tier × slot is listed cheapest tier first; the endpoint
summary counts healthy endpoints and spans their prices; the timer stops the first-token
clock on the first text OR reasoning delta (not on keep-alives or role-only chunks), takes
output tokens from the final usage chunk and the provider from the chunks, and an in-band
error is reported; throughput excludes the wait for the first token.

Offline: no network, no API keys.
"""

from __future__ import annotations

import json

from deep_research_agent.config import MODEL_TIERS, ResearchConfig
from deep_research_agent.tier_report import (SLOT_ROLES, Sample, StreamTimer, endpoint_summary,
                                             summarize, tier_rows)


def _data(obj) -> str:
    return "data: " + json.dumps(obj)


def test_every_tier_and_slot_is_listed_cheapest_first() -> None:
    rows = tier_rows()
    assert len(rows) == sum(len(p) for p in MODEL_TIERS.values())
    assert [r[0] for r in rows][:: len(SLOT_ROLES)] == ["extra-low", "low", "mid", "high"]
    assert ("mid", "utility_model", MODEL_TIERS["mid"]["utility_model"]) in rows


def test_endpoint_summary_counts_healthy_and_spans_prices() -> None:
    cfg = ResearchConfig.from_runnable_config({"configurable": {"openai_api_key": "k"}})
    feed = [{"status": 0, "uptime_last_30m": 100, "pricing": {"prompt": "1e-7", "completion": "5e-7"}},
            {"status": -2, "uptime_last_30m": 80, "pricing": {"prompt": "2e-7", "completion": "1e-6"}}]
    assert endpoint_summary(cfg, feed) == "1/2 healthy, in $0.1–0.2 / out $0.5–1"
    assert endpoint_summary(cfg, None) == "no feed"


def test_timer_measures_first_token_tokens_and_provider() -> None:
    t = StreamTimer(started=100.0)
    t.feed(": OPENROUTER PROCESSING", 100.5)                                       # keep-alive
    t.feed(_data({"provider": "Google", "choices": [{"delta": {"role": "assistant"}}]}), 101.0)
    t.feed(_data({"provider": "Google", "choices": [{"delta": {"reasoning": "hm"}}]}), 102.0)
    t.feed(_data({"provider": "Google", "choices": [{"delta": {"content": "Gutenberg"}}]}), 103.0)
    t.feed(_data({"provider": "Google", "choices": [], "usage": {"completion_tokens": 400}}), 106.0)
    t.feed("data: [DONE]", 106.0)
    s = t.sample(106.0)
    assert (s.provider, s.ttft, s.seconds, s.output_tokens, s.error) == ("Google", 2.0, 6.0, 400, None)
    assert s.tokens_per_second == 100.0                                            # 400 / (6 - 2)


def test_timer_reports_an_in_band_error() -> None:
    t = StreamTimer(started=0.0)
    t.feed(_data({"error": {"message": "Provider returned error"},
                  "choices": [{"delta": {}, "finish_reason": "error"}]}), 1.0)
    s = t.sample(1.0)
    assert s.error == "Provider returned error" and s.ttft is None and s.tokens_per_second is None


def test_summary_averages_and_counts_failures() -> None:
    samples = [Sample("Google", 1.0, 5.0, 400), Sample("Google", 3.0, 7.0, 200),
               Sample("?", None, 2.0, None, "HTTP 429: slow down")]
    out = summarize(samples)
    assert "first token 2.0s" in out and "75 tok/s" in out and "via Google×2" in out
    assert "1 failed (HTTP 429: slow down)" in out
