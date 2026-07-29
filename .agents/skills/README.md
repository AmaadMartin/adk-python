# ADK Agent Skills

These are development skills for AI coding agents working *on* this repository.
Compatible tools load them automatically from this directory and pull one in
when the task matches its `description`.

They are not the runtime Skills feature — the library for loading skills into an
agent you are building, documented in
[`src/google/adk/skills/README.md`](../../src/google/adk/skills/README.md). Only
the file format is shared.

## Available skills

| Skill | Use when |
| --- | --- |
| [adk-agent-builder](adk-agent-builder/SKILL.md) | Building, testing, or iterating on an ADK agent: creating one, configuring modes (task, single-turn), or building graph-based workflows. |
| [adk-architecture](adk-architecture/SKILL.md) | Understanding or changing ADK internals — graph orchestration, resumption, event flow, node contracts, observability. Triggers on "how does X work", "BaseNode", "NodeRunner". |
| [adk-cross-language-port](adk-cross-language-port/SKILL.md) | Porting a feature, fix, or test from the TypeScript SDK (`adk-js`) into `adk-python`, or working an "adk-js parity" issue. |
| [adk-debug](adk-debug/SKILL.md) | Debugging an agent: inspecting sessions, troubleshooting tool calls, event flow issues, or diagnosing LLM/model problems. |
| [adk-git](adk-git/SKILL.md) | Any git operation (commit, push, rebase, branch, PR, cherry-pick). Provides the commit message format and conventions. |
| [adk-review](adk-review/SKILL.md) | Reviewing local changes for errors, style compliance, unintended outcomes, and the doc/test/sample updates they imply. |
| [adk-sample-creator](adk-sample-creator/SKILL.md) | Authoring a new sample for this repository, including examples under `contributing/`. |
| [adk-setup](adk-setup/SKILL.md) | Setting up a local development environment, installing dependencies, or getting ready to contribute. |
| [adk-style](adk-style/SKILL.md) | Writing or reviewing Python here — imports, typing, Pydantic patterns, formatting, logging, async, file organization, testing rules. |
| [adk-unit-design](adk-unit-design/SKILL.md) | Creating or updating a code unit design document. |
| [adk-unit-guide](adk-unit-guide/SKILL.md) | Creating a detailed code unit guide. |
| [adk-verify-snippets](adk-verify-snippets/SKILL.md) | Checking that the Python code blocks in a Markdown file actually run, and how much of the file they cover. |

## Skill layout

```text
.agents/skills/<skill-name>/
  SKILL.md        # required: YAML frontmatter + Markdown body
  references/     # optional: long-form docs the body links to
  scripts/        # optional: executable helpers (see adk-verify-snippets)
  assets/         # optional: templates, schemas, static material
```

`references/`, `scripts/`, and `assets/` are the only subdirectories the loader
reads. Each is read recursively and keyed by path relative to it, so
`references/interfaces/event.md` is loaded under the key `interfaces/event.md`.

## Adding a new skill

1. Create `.agents/skills/<name>/SKILL.md`. **The directory name must equal the
   frontmatter `name`.** On a mismatch `load_skill_from_dir` raises `ValueError`
   and `list_skills_in_dir` drops the skill from the listing with only a log
   warning, so it disappears silently.
2. `name`: lowercase kebab-case, at most 64 characters.
3. `description`: non-empty, at most 1024 characters. It is the *only* text the
   model sees before deciding whether to load the skill, so say both what the
   skill does and when to use it. Naming concrete trigger phrases, as
   [adk-architecture](adk-architecture/SKILL.md) and
   [adk-style](adk-style/SKILL.md) do, works well.
4. Allowed frontmatter keys are `name`, `description`, `license`,
   `compatibility`, `allowed-tools` (or `allowed_tools`), and `metadata`. Other
   keys still load — `Frontmatter` sets `extra="allow"`, which is what
   `adk-setup`'s `disable-model-invocation` relies on — but they are outside the
   allowlist in
   [`src/google/adk/skills/_utils.py`](../../src/google/adk/skills/_utils.py), so
   do not add new ones.
5. Keep `SKILL.md` short: its body is injected into the agent's prompt every
   time the skill triggers. Put long material in `references/*.md` and link to
   it from the body — [adk-style](adk-style/SKILL.md) and
   [adk-architecture](adk-architecture/SKILL.md) are the pattern to copy.
6. Executable helpers go in `scripts/`. Python and shell scripts there need the
   Apache license header that `addlicense` adds; Markdown files do not.
7. Add a row to the table above in the same change.
8. Verify it loads before sending the PR:

   ```bash
   python -c "from google.adk.skills import list_skills_in_dir; print(sorted(list_skills_in_dir('.agents/skills')))"
   ```
