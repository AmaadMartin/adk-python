# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for optional dependencies and lazy loading.

Includes both fast hermetic unit tests (run by default) and high-fidelity
integration tests using a clean venv (skipped by default, run via env var).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from typing import NoReturn
from unittest import mock

import pytest

from .isolated_import_utils import REPO_ROOT as _REPO_ROOT

# Check if we should run integration tests that require network/install
RUN_INTEGRATION = os.environ.get("ADK_TEST_NETWORK") == "1"


@pytest.fixture(scope="session")
def clean_core_venv(tmp_path_factory):
  """Creates a clean venv with only core ADK installed (requires network)."""
  if not RUN_INTEGRATION:
    pytest.skip("Integration tests requiring network are disabled by default.")

  venv_path = tmp_path_factory.mktemp("adk_core_venv")
  python_exe = venv_path / "bin" / "python"

  # Create venv using uv for speed
  subprocess.run(
      ["uv", "venv", str(venv_path), "--python", "3.11"],
      check=True,
      capture_output=True,
  )

  # Install core ADK from local repo (uses current pyproject.toml)
  subprocess.run(
      ["uv", "pip", "install", "--python", str(python_exe), str(_REPO_ROOT)],
      check=True,
      capture_output=True,
  )

  return python_exe


# =============================================================================
# Approach 1: Hermetic Unit Tests (Fast, No Network, Safe for CI/TAP)
# =============================================================================


def test_pydantic_version():
  """Print the installed Pydantic version."""
  import pydantic

  print(f"Pydantic version: {pydantic.__version__}")
  assert True


def test_a2a_remote_agent_config_raises_importerror():
  """Verify that accessing A2aRemoteAgentConfig without extra raises ImportError using mocks."""
  with mock.patch.dict("sys.modules", {"a2a": None}):
    for mod in list(sys.modules):
      if mod.startswith("a2a.") or mod.startswith("google.adk.a2a."):
        sys.modules.pop(mod, None)
    with pytest.raises(ImportError) as exc_info:
      from google.adk.a2a.agent import A2aRemoteAgentConfig
    assert "a2a-sdk" in str(exc_info.value)


def test_vertex_ai_memory_bank_service_fails_on_creation():
  """Verify that creating VertexAiMemoryBankService without extra fails using mocks."""
  try:
    from google.adk.memory import VertexAiMemoryBankService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"vertexai": None}):
    sys.modules.pop("google.adk.memory.vertex_ai_memory_bank_service", None)
    from google.adk.memory import VertexAiMemoryBankService

    with pytest.raises(ImportError) as exc_info:
      VertexAiMemoryBankService(agent_engine_id="123")
    assert "google-cloud-aiplatform" in str(exc_info.value)


def test_database_session_service_fails_on_creation():
  """Verify that creating DatabaseSessionService without extra fails using mocks."""
  try:
    from google.adk.sessions import DatabaseSessionService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"sqlalchemy": None}):
    sys.modules.pop("google.adk.sessions.database_session_service", None)
    with pytest.raises(ImportError) as exc_info:
      from google.adk.sessions import DatabaseSessionService

      DatabaseSessionService(db_url="sqlite+aiosqlite:///:memory:")
    assert "sqlalchemy" in str(exc_info.value)


def test_vertex_ai_session_service_fails_on_creation():
  """Verify that creating VertexAiSessionService without extra fails using mocks."""
  try:
    from google.adk.sessions import VertexAiSessionService
  except KeyError as e:
    if "pydantic.root_model" in str(e):
      pytest.skip(
          "Skipping mock test due to Pydantic/MCP environment conflict"
          " (KeyError: 'pydantic.root_model')."
      )
    raise

  with mock.patch.dict("sys.modules", {"vertexai": None}):
    sys.modules.pop("google.adk.sessions.vertex_ai_session_service", None)
    from google.adk.sessions import VertexAiSessionService

    with pytest.raises(ImportError) as exc_info:
      VertexAiSessionService(agent_engine_id="123")
    assert "google-cloud-aiplatform" in str(exc_info.value)


def test_vertexai_dependency_shim_raises_clear_importerror():
  """Verify that the Vertex AI dependency shim points users to the dependency."""
  module_path = _REPO_ROOT / "dependencies_internal/vertexai.py"
  if not module_path.is_file():
    pytest.skip("Vertex AI dependency shim is not present in this build.")
  with mock.patch.dict("sys.modules", {"google.cloud.aiplatform": None}):
    spec = importlib.util.spec_from_file_location(
        "_test_google_adk_dependencies_vertexai", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    with pytest.raises(ImportError) as exc_info:
      spec.loader.exec_module(module)

    message = str(exc_info.value)
    assert "//third_party/py/google/cloud/aiplatform" in message


# The verbatim text nltk 3.10.1's nltk/inisec.py meta-path finder raises when it
# blocks a module that resolves under the current working directory.
_NLTK_CWD_GUARD_ERROR = (
    "Blocked import of regex from current working directory for security"
    " reasons. Use '-P' or set PYTHONSAFEPATH to prevent Python from searching"
    " the current working directory."
)


def _exec_rouge_scorer_shim():
  """Executes the rouge_score dependency shim under a throwaway module name.

  The shim is executed from its file rather than imported so that the entry for
  ``google.adk.dependencies.rouge_scorer`` in ``sys.modules`` stays untouched.
  """
  module_path = _REPO_ROOT / "src/google/adk/dependencies/rouge_scorer.py"
  spec = importlib.util.spec_from_file_location(
      "_test_adk_rouge_scorer_shim", module_path
  )
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _fake_rouge_score_raising(error: ImportError) -> types.ModuleType:
  """Returns a fake ``rouge_score`` module whose attribute access raises.

  ``from rouge_score import rouge_scorer`` falls back to attribute access on the
  parent module, and the import machinery only converts ``AttributeError`` into
  ``ImportError`` there, so ``error`` propagates unchanged.
  """
  fake = types.ModuleType("rouge_score")

  def _raise(name: str) -> NoReturn:
    raise error

  fake.__getattr__ = _raise
  return fake


def test_rouge_scorer_shim_explains_nltk_cwd_guard():
  """Verify the shim replaces nltk 3.10.1's CWD guard error with a fix."""
  error = ImportError(_NLTK_CWD_GUARD_ERROR)
  with mock.patch.dict(
      "sys.modules", {"rouge_score": _fake_rouge_score_raising(error)}
  ):
    with pytest.raises(ImportError) as exc_info:
      _exec_rouge_scorer_shim()

  message = str(exc_info.value)
  assert "nltk" in message
  assert "3.10.1" in message
  assert 'pip install --upgrade "nltk!=3.10.1"' in message


def test_rouge_scorer_shim_chains_original_nltk_error():
  """Verify the original nltk error survives as the __cause__."""
  error = ImportError(_NLTK_CWD_GUARD_ERROR)
  with mock.patch.dict(
      "sys.modules", {"rouge_score": _fake_rouge_score_raising(error)}
  ):
    with pytest.raises(ImportError) as exc_info:
      _exec_rouge_scorer_shim()

  assert exc_info.value.__cause__ is error


def test_rouge_scorer_shim_reraises_unrelated_import_error():
  """Verify an import failure that is not the nltk guard propagates as is."""
  with mock.patch.dict("sys.modules", {"rouge_score": None}):
    with pytest.raises(ImportError) as exc_info:
      _exec_rouge_scorer_shim()

  message = str(exc_info.value)
  assert "nltk" not in message
  assert exc_info.value.__cause__ is None


def test_rouge_scorer_shim_reexports_when_import_succeeds():
  """Verify the shim re-exports the real rouge_score submodules."""
  fake = types.ModuleType("rouge_score")
  fake_rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
  fake_tokenizers = types.ModuleType("rouge_score.tokenizers")
  fake.rouge_scorer = fake_rouge_scorer
  fake.tokenizers = fake_tokenizers
  with mock.patch.dict("sys.modules", {"rouge_score": fake}):
    module = _exec_rouge_scorer_shim()

  assert module.rouge_scorer is fake_rouge_scorer
  assert module.tokenizers is fake_tokenizers


# =============================================================================
# Approach 2: High-Fidelity Integration Tests (Clean Venv, Skipped by Default)
# =============================================================================


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_critical_imports_subprocess(clean_core_venv):
  """Verify that all critical import paths in core work in a true core-only environment."""
  imports = [
      "import google.adk",
      "from google.adk import Agent",
      "from google.adk import Context",
      "from google.adk import Event",
      "from google.adk import Runner",
      "from google.adk import Workflow",
  ]
  for imp in imports:
    result = subprocess.run(
        [str(clean_core_venv), "-c", imp],
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Failed to import: {imp}\nStderr: {result.stderr}"


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_a2a_remote_agent_config_raises_importerror_subprocess(clean_core_venv):
  """Verify that accessing A2aRemoteAgentConfig without extra raises ImportError in clean environment."""
  code = """
try:
    from google.adk.a2a.agent import A2aRemoteAgentConfig
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "a2a-sdk" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_vertex_ai_memory_bank_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating VertexAiMemoryBankService without extra fails in clean environment."""
  code = """
from google.adk.memory import VertexAiMemoryBankService
try:
    service = VertexAiMemoryBankService(agent_engine_id="123")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "google-cloud-aiplatform" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_database_session_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating DatabaseSessionService without extra fails in clean environment."""
  code = """
try:
    from google.adk.sessions import DatabaseSessionService
    service = DatabaseSessionService(db_url="sqlite+aiosqlite:///:memory:")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "sqlalchemy" in output


@pytest.mark.skipif(not RUN_INTEGRATION, reason="Requires ADK_TEST_NETWORK=1")
def test_vertex_ai_session_service_fails_on_creation_subprocess(
    clean_core_venv,
):
  """Verify that creating VertexAiSessionService without extra fails in clean environment."""
  code = """
from google.adk.sessions import VertexAiSessionService
try:
    service = VertexAiSessionService(agent_engine_id="123")
    print("SUCCESS")
except ImportError as e:
    print(f"CAUGHT_IMPORT_ERROR: {e}")
"""
  result = subprocess.run(
      [str(clean_core_venv), "-c", code],
      capture_output=True,
      text=True,
      check=True,
  )
  output = result.stdout.strip()
  assert "CAUGHT_IMPORT_ERROR" in output
  assert "google-cloud-aiplatform" in output
