## Task

### User intent with respect to ADK
The user wants to achieve mypy strict typing parity across the `adk-python` codebase by resolving any remaining strict type errors in the `context` modules (under `google/adk/agents`).

### Feature Description
Fix all `mypy --strict` compliance errors in the context-related modules within `adk-python`. Specifically, this involves adding missing type annotations (e.g., `Optional`, `Any`, explicit list/dict inner types), fixing missing return types (e.g., `-> None`), and ensuring that the `context.py`, `readonly_context.py`, `callback_context.py`, `invocation_context.py`, and `context_cache_config.py` files pass strict static analysis without explicit `type: ignore` pragmas for missing signatures.

### Use Cases & Examples
- Running `mypy --strict src/google/adk/agents/context.py` should return `Success: no issues found`.
- Standardizing types guarantees safer downstream consumption of Context attributes within workflows and agents.

## Context

### ADK Context
- Documentation context: `adk-python` is currently rolling out strict typing across its modules. Modules such as `flows`, `evaluation`, `models`, `plugins`, `integrations`, `agents`, `auth`, `tools`, and `cli` have recently seen strict type fixes. This task targets the context logic which appears to have missing type inferences for dynamic inputs and `kwargs`.
- Reference context: Prior commits like `c7420229 chore: fix mypy strict type errors in adk agents` serve as the reference strategy for adding `-> None` to `__init__` and adding standard Collections types.

### Language Specific Context
- Target language: Python
- Target repo: `adk-python`
- General context: Mypy configuration uses `strict = False` globally for now but PRs are progressively enforcing it per-module or fixing the baseline.

## Definition

### Data Models
- Variables of type `Any` or `dict` in context components must be refined where possible, or explicitly typed as `Any` using `typing.Any` to satisfy `disallow_untyped_defs`.

### Inputs
- Source files:
  - `src/google/adk/agents/context.py`
  - `src/google/adk/agents/readonly_context.py`
  - `src/google/adk/agents/callback_context.py`
  - `src/google/adk/agents/invocation_context.py`
  - `src/google/adk/agents/context_cache_config.py`

### Outputs
- Modified `.py` source files containing explicit type annotations satisfying mypy strict mode.

### Side Effects
- This is a non-functional refactor. There are no runtime logic side effects; only static type checking is impacted.

## Constraints

### Invariants
- The existing runtime behavior for agents and workflows utilizing these contexts MUST NOT change.
- Avoid overriding or narrowing types in a way that breaks user-facing APIs.

### Preconditions
- The target repository is checked out at the `main` branch.

### Postconditions
- All context files mentioned must pass local mypy checks.

### Error Handling Protocols
- If a type cannot be strictly inferred due to dynamic Python behavior, explicit `Any` or `cast` may be used sparingly, provided it resolves the strict typing block.

### Breaking Change Analysis
- No breaking changes. Type annotations are purely static.

### Testing

- #### Unit tests with >=95% New Line Coverage
  - Existing tests must pass (`pytest tests/`). Adding types should not alter branch coverage.

- #### Integration tests
  - No new integration tests are required for a static typing change.

- #### Manual e2e test
  - Run `tox -e mypy` or raw `mypy --strict` on the modified files to manually assert 0 errors.
