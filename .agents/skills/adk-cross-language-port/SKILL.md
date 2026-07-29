---
name: adk-cross-language-port
description: Port a feature, fix, or test from the TypeScript SDK (adk-js) into adk-python. Use when mirroring an adk-js change here, closing an "adk-js parity" gap, or translating TypeScript ADK code to Python. Covers where the adk-js file lands in this repository, the TypeScript-to-Python idiom mapping, how to port the TypeScript tests, and the commands that verify the result.
---

# Porting adk-js to adk-python

## Before you start

1. Locate the source file in `adk-js` and read its `*_test.ts` first — the
   TypeScript tests are the behavioral spec you are matching.
2. Check whether `adk-python` already has an equivalent under
   `src/google/adk/<area>/`. If it does, extend it rather than adding a parallel
   module.
3. If the Python public API already names the concept differently, keep the
   Python name. Parity means the same behavior, not the same spelling.

## File layout mapping

| adk-js | adk-python |
| --- | --- |
| `core/src/<area>/<name>.ts` | `src/google/adk/<area>/<name>.py` |
| `core/test/<area>/<name>_test.ts` | `tests/unittests/<area>/test_<name>.py` |

Worked pair to diff: `core/src/tools/function_tool.ts` → `src/google/adk/tools/function_tool.py`.

Areas that do not map one-to-one:

| adk-js | adk-python |
| --- | --- |
| `core/src/runner/runner.ts` | `src/google/adk/runners.py` |
| `core/src/agents/processors/<x>_llm_request_processor.ts` | `src/google/adk/flows/llm_flows/<x>.py` |
| `core/src/context/<x>_context_compactor.ts` | `src/google/adk/apps/compaction.py` and `src/google/adk/flows/llm_flows/compaction.py` |
| the `dev/` workspace (CLI, web server) | `src/google/adk/cli/` |

`src/google/adk/workflow/`, `planners/`, `evaluation/`, `labs/`,
`optimization/`, and `errors/` have no `adk-js` counterpart. If a port seems to
need one, stop and ask before inventing a layout.

## Idiom mapping

| TypeScript (adk-js) | Python (adk-python) |
| --- | --- |
| `interface` or `zod` schema | Pydantic v2 model — see [adk-style pydantic](../adk-style/references/pydantic.md) |
| destructured options object (`{toolContext, llmRequest}`) | keyword-only arguments (`*, tool_context, llm_request`) — see [adk-style typing](../adk-style/references/typing.md) |
| `override` keyword | `@override` from `typing_extensions` |
| `AsyncGenerator<Event>` and `for await` | `AsyncGenerator[Event, None]` and `async for`; close the generator with `Aclosing` from `..utils.context_utils` |
| `throw new Error(...)` | an existing typed error from `src/google/adk/errors/` — `ToolExecutionError` (with a `ToolErrorType`), `NotFoundError`, `InputValidationError`, `AlreadyExistsError`, `SessionNotFoundError` — never a new hierarchy. Use `raise ... from err` when re-raising |
| `import {X} from './y.js'` | `from .y import X`, one symbol per line, isort-ordered |
| `/** @license */` JSDoc header | the header block in [adk-style file organization](../adk-style/references/file-organization.md) |
| camelCase JSON on the wire | keep it: inherit `SerializedBaseModel` from `google.adk.utils._serialized_base_model` |

Write the Python a Python reader expects, not a transliteration — and do not add
a third-party dependency just because the TypeScript side has one.

## Tests

Port every `it(...)` case from the TypeScript test as its own behavior-named
pytest in `tests/unittests/<area>/test_<name>.py`. Cover the error paths, not
only the happy path, and assert with `pytest.raises(SpecificError, match=...)`
rather than a bare `Exception`. The rest of the rules are in
[adk-style testing](../adk-style/references/testing.md).

## Verify

`pytest tests/unittests/<area> -q` while iterating — only the area you touched.
Environment setup and the pre-PR commands are in
[adk-setup](../adk-setup/SKILL.md).

## Do not

- Change `adk-js` in the same pull request. Queue the reverse direction as its
  own task.
- Port private, underscore-prefixed internals unless the public behavior you are
  matching requires them.
