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

   This is the only supported install path. It installs exactly the versions in the committed `uv.lock`, which is what CI uses.

   Never use `uv pip install -e .`, `uv pip compile` or `pip install -e .`: they resolve for a single interpreter, select `numpy` 2.5.x on Python 3.12+, and make `mypy` abort with `Type statement is only supported in Python 3.12 and greater` without checking any first-party file.

   After editing `[project.dependencies]` or any `[project.optional-dependencies]` table, run `uv lock` and commit `uv.lock` in the same change — CI syncs with `--locked` and fails on a stale lock.

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
| Check the env matches `uv.lock`      | `uv sync --check`                                 |
| Run unit tests (Fast)                | `pytest tests/unittests`                          |
| Run tests across all Python versions | `tox`                                             |
| Format codebase                      | `pre-commit run --all-files`                      |
| Run tests in parallel                | `pytest tests/unittests -n auto`                  |
| Run specific test file               | `pytest tests/unittests/agents/test_llm_agent.py` |
| Launch web UI                        | `adk web path/to/agents_dir`                      |
| Run agent via CLI                    | `adk run path/to/my_agent`                        |
| Build wheel                          | `uv build`                                        |
