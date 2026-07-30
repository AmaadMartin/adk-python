# ADK Skills

> [!WARNING]
> This feature is **experimental** and under **active development**. APIs and
> functionality are subject to change without notice.

## Overview

The ADK Skills system enables dynamic loading of agent instructions, resources,
and scripts. This allows agents to be extended with new capabilities at
runtime.

## Client directives

`SKILL.md` frontmatter may carry client directives defined by the wider Agent
Skills ecosystem. ADK honors `disable-model-invocation`:

```yaml
---
name: deploy
description: Deploy the application to production.
disable-model-invocation: true
---
```

A skill marked this way is withheld from every model-facing discovery surface:
the `list_skills` tool output, the auto-injected `<available_skills>` system
instruction, and `search_skills` results from a `SkillRegistry`. It stays fully
loadable when something names it explicitly, so `load_skill`,
`load_skill_resource` and `run_skill_script` keep working. This is a discovery
control, not an authorization boundary: an application (or an agent instruction
naming the skill) can still load it on purpose.

The behavior is opt-in while it soaks: set
`ADK_ENABLE_SKILL_DISABLE_MODEL_INVOCATION=1`, or call
`override_feature_enabled(FeatureName.SKILL_DISABLE_MODEL_INVOCATION, True)`
from `google.adk.features`. With the feature disabled (the default), ADK parses
the directive but does not act on it.
