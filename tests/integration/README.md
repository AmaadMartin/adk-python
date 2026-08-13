# Integration Tests

These tests run end to end against **live** Gemini and Vertex AI endpoints, and
against the real `adk` command line. They are not hermetic, they cost model
quota, and they are **not** part of pull-request CI. Hermetic tests belong in
`tests/unittests` instead.

## Run them locally

1. Copy `.env.example` to `tests/integration/.env` and fill in the values.
1. Install the dependencies. Use `--all-extras`, or at least
   `--extra test --extra oci`: `tests/integration/integrations/oci` imports the
   `oci` SDK at module scope, and `oci` is not part of the `test` extra.

   ```bash
   uv sync --all-extras
   ```

1. Run the subtree by name. Do not run `pytest tests`: a module named
   `test_oci_genai_llm.py` exists under both `tests/integration` and
   `tests/unittests`, and collecting both trees at once trips pytest's import
   file mismatch check.

   ```bash
   uv run pytest tests/integration -q
   ```

`TEST_BACKEND` selects the backend that `conftest.py` parametrizes each test
over. It is one of `GOOGLE_AI_ONLY`, `VERTEX_ONLY` or `BOTH`, and it defaults to
`BOTH`. `BOTH` doubles every test and needs both an API key and a
Vertex-enabled project.

## Where they run automatically

`.github/workflows/integration-tests.yml` runs them nightly at 07:00 UTC, and
on demand through `workflow_dispatch`. The workflow has no `pull_request` or
`push` trigger, so it can never block a contributor's pull request. It no-ops
with a green result until a maintainer configures the `GOOGLE_API_KEY`
repository secret.

## The rule for a test that cannot run

Delete a module that has to call `pytest.skip(allow_module_level=True)`
unconditionally. Do not park it here. A permanently skipped module reads as
coverage and provides none, and its code rots against `src/` because nothing
executes it.
