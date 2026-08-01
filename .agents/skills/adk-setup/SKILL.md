---
name: adk-setup
description: Set up a local development environment for the ADK Python project. Use when the user wants to get started developing, set up their environment, install dependencies, or prepare for contributing.
disable-model-invocation: true
---

Set up the local development environment for ADK Python.

## Prerequisites

Check the following before proceeding:

1. **Python 3.10+**

   ```bash
   python3 --version
   ```

2. **uv package manager** (required — do not use pip/venv directly)
   ```bash
   uv --version
   ```
   If not installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Setup Steps

Run these commands from the project root:

3. **Create and activate a virtual environment:**

   ```bash
   uv venv --python "python3.11" ".venv"
   source .venv/bin/activate
   ```

4. **Install all dependencies for development:**

   ```bash
   uv sync --all-extras
   ```

   The only supported install path: it installs exactly the versions in the committed `uv.lock`, which is what CI uses. Never install a development environment with `uv pip install -e .` or `pip install -e .` — `CONTRIBUTING.md` explains how they silently break `mypy`.

   Verify an existing environment with `uv sync --all-extras --check`, always passing the extras it was installed with. After editing any dependency table, run `uv lock` and commit `uv.lock` in the same change; CI syncs with `--locked` and fails on a stale lock.

5. **Install development tools:**

   ```bash
   uv tool install pre-commit
   uv tool install tox --with tox-uv
   ```

6. **Install addlicense (requires Go):**

   ```bash
   go version && go install github.com/google/addlicense@latest
   ```

   > [!NOTE]
   > If Go is not installed, tell the user:
   > "Go is required for the addlicense tool. Please install Go from https://go.dev/dl/ and then re-run the `adk-setup` skill to complete the setup."

7. **Set up pre-commit hooks:**

   ```bash
   pre-commit install
   ```

8. **Verify everything works by running tests locally:**
   ```bash
   pytest tests/unittests -n auto
   ```

## Key Commands Reference

| Task                                 | Command                                           |
| :----------------------------------- | :------------------------------------------------ |
| Check the env matches `uv.lock`      | `uv sync --all-extras --check`                    |
| Run unit tests (Fast)                | `pytest tests/unittests`                          |
| Run tests across all Python versions | `tox`                                             |
| Format codebase                      | `pre-commit run --all-files`                      |
| Run tests in parallel                | `pytest tests/unittests -n auto`                  |
| Run specific test file               | `pytest tests/unittests/agents/test_llm_agent.py` |
| Launch web UI                        | `adk web path/to/agents_dir`                      |
| Run agent via CLI                    | `adk run path/to/my_agent`                        |
| Build wheel                          | `uv build`                                        |
