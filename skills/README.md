# Skills

Drop-in capabilities for the deep-research agent. Each skill is a folder with a
`SKILL.md` file; the agent loads them on startup and reads a skill's full
instructions only when a task matches its description (progressive disclosure).

## Layout

```
skills/
└── <skill-name>/
    ├── SKILL.md      # required — YAML frontmatter + markdown instructions
    └── ...           # optional supporting files (reference docs, examples)
```

## SKILL.md format

```markdown
---
name: <skill-name>          # must EXACTLY match the folder name
description: <one paragraph> # what it does AND when to use it (keywords help matching)
---

# Title

## When to use
...

## Workflow
...
```

Rules (per the [Agent Skills spec](https://agentskills.io/specification)):
- `name`: 1–64 chars, lowercase letters/digits/hyphens, must equal the folder name.
- `description`: 1–1024 chars; describe the task and trigger keywords so the agent
  knows when to reach for it.

## How loading works

On each run the agent mounts this directory read-only at the virtual path
`/skills/` and lists every subfolder containing a `SKILL.md`. The names +
descriptions are injected into the system prompt; the agent calls `read_file` on
`/skills/<name>/SKILL.md` to pull the full instructions when it decides a skill
applies.

The directory is configurable via `DRA_SKILLS_DIR` (defaults to this folder).

## Roadmap

Today the loader reads only this local directory. The planned loader will be
user-aware: it will layer **system-wide** skills (shared) and the requesting
user's **personal** skills (from DB/S3), with later sources overriding earlier
ones by name. The mounting mechanism here is the foundation for that.
