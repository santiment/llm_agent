# Plans — historical design snapshots

Everything in this directory is a **design document written before the work**, kept for the
reasoning and the alternatives that were weighed. **None of it is a description of current
behaviour**, and it is not maintained as the code moves.

That means a plan can name a knob, key or module that never shipped, or that shipped under a
different name — e.g. per-model config keys like `DRA_SUBAGENT_MODEL`, `DRA_UTILITY_MODEL` or
`compression_model`, which the tier-only model selection replaced (models are chosen by tier
name; see the README's *Model tiers*). Read plans as "what we intended at the time".

**For how the agent actually works today, read, in order:**

| Question | Read |
|---|---|
| What does it do, end to end? | [`../HOW_THE_AGENT_WORKS.md`](../HOW_THE_AGENT_WORKS.md) |
| How do I configure/run it? | [`../../README.md`](../../README.md) |
| How do I add a tool? | [`../CUSTOM_TOOLS.md`](../CUSTOM_TOOLS.md) |
| What is the exact contract? | the code — `config.py`, `events.py` (`EVENT_SCHEMAS`), `tests/` |

Each plan states its own status in its opening lines (e.g. `MODEL_TIERING_PLAN.md` records
which sections landed and which stayed deferred). Where a plan and the code disagree, **the
code is right** — and the plan is not worth "fixing", since its value is the snapshot.
