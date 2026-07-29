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

"""Tests for the module allowlist that guards YAML agent-config references."""

from __future__ import annotations

from pathlib import Path
import sys

from google.adk.agents import config_agent_utils
from google.adk.agents.common_configs import CodeConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.tool_configs import ToolConfig
import pytest

# Prefix on every package these tests import, so the autouse fixture can
# evict them from the interpreter-global sys.modules again.
_TEST_PACKAGE_PREFIX = 'allowlist_pkg_'


@pytest.fixture(autouse=True)
def _isolate_allowlist_state():
  """Restores the module-level state these tests mutate."""
  saved_pkgs = set(config_agent_utils._allowed_agent_packages)
  saved_modules = set(sys.modules)
  try:
    yield
  finally:
    config_agent_utils._allowed_agent_packages.clear()
    config_agent_utils._allowed_agent_packages.update(saved_pkgs)
    for name in set(sys.modules) - saved_modules:
      if name.startswith(_TEST_PACKAGE_PREFIX):
        del sys.modules[name]


def _write_agent_config(package_dir: Path, tool_ref: str) -> Path:
  """Writes a ``root_agent.yaml`` whose single tool is ``tool_ref``."""
  config_path = package_dir / 'root_agent.yaml'
  config_path.write_text(
      'name: demo\n'
      'model: gemini-2.5-flash\n'
      'instruction: Be brief.\n'
      'tools:\n'
      f'  - name: {tool_ref}\n'
  )
  return config_path


# --- Allowed references ---------------------------------------------------


def test_google_adk_prefix_is_allowed():
  assert (
      config_agent_utils.resolve_fully_qualified_name(
          'google.adk.agents.llm_agent.LlmAgent'
      )
      is LlmAgent
  )


def test_import_failure_after_validation_is_still_wrapped():
  """Hoisting validation out of the try must not drop the wrapper."""
  with pytest.raises(ValueError, match='Invalid fully qualified name'):
    config_agent_utils.resolve_fully_qualified_name(
        'google.adk.agents.llm_agent.NoSuchSymbol'
    )


def test_user_agent_package_is_allowed_after_config_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
  """A config authorizes its own package, and the tool then imports."""
  package = f'{_TEST_PACKAGE_PREFIX}basic'
  package_dir = tmp_path / package
  package_dir.mkdir()
  (package_dir / '__init__.py').write_text('')
  (package_dir / 'tools.py').write_text(
      'def my_tool(query: str) -> str:\n  return query\n'
  )
  config_path = _write_agent_config(package_dir, f'{package}.tools.my_tool')
  monkeypatch.syspath_prepend(str(tmp_path))

  agent = config_agent_utils.from_config(str(config_path))

  assert agent.tools == [sys.modules[f'{package}.tools'].my_tool]


def test_nested_config_registers_the_root_package(tmp_path: Path):
  """A config nested inside the package authorizes the top-level package."""
  package = f'{_TEST_PACKAGE_PREFIX}nested'
  package_dir = tmp_path / package
  sub_agents_dir = package_dir / 'sub_agents'
  sub_agents_dir.mkdir(parents=True)
  (package_dir / '__init__.py').write_text('')
  (sub_agents_dir / '__init__.py').write_text('')
  config_path = _write_agent_config(sub_agents_dir, f'{package}.tools.my_tool')

  config_agent_utils._load_config_from_path(str(config_path))

  assert package in config_agent_utils._allowed_agent_packages
  assert 'sub_agents' not in config_agent_utils._allowed_agent_packages


def test_dotless_builtin_tool_still_resolves():
  """Bare built-in tool names bypass the allowlist via google.adk.tools."""
  resolved = LlmAgent._resolve_tools(
      [ToolConfig(name='google_search')], '/fake/path.yaml'
  )
  assert len(resolved) == 1


# --- Rejected references --------------------------------------------------


@pytest.mark.parametrize(
    'reference',
    [
        # Stdlib the old denylist named.
        'os.system',
        # Stdlib the old denylist never named -- the bug this change fixes.
        'platform.popen',
        # An installed third-party distribution.
        'crewai_tools.SerperDevTool',
        # Prefix boundaries: 'google.adk' must not match by string prefix.
        'google.adk_evil.rce',
        'google.adk',
        # A relative reference has an empty top-level segment.
        '.agent.Feedback',
    ],
)
def test_disallowed_references_are_rejected(reference: str):
  with pytest.raises(ValueError, match='Disallowed module reference'):
    config_agent_utils.resolve_fully_qualified_name(reference)


def test_agent_directory_named_after_a_stdlib_module_is_not_allowlisted(
    tmp_path: Path,
):
  """An agent directory called 'os' must not authorize the stdlib 'os'."""
  package_dir = tmp_path / 'os'
  package_dir.mkdir()
  config_path = _write_agent_config(package_dir, 'os.system')

  with pytest.raises(ValueError, match='Disallowed module reference'):
    config_agent_utils.from_config(str(config_path))


def test_rejection_happens_before_import(monkeypatch: pytest.MonkeyPatch):
  """The security property: a rejected reference imports nothing."""

  def _fail_on_import(*args, **kwargs):
    pytest.fail('import_module must not be called for a rejected reference')

  monkeypatch.setattr(
      config_agent_utils.importlib, 'import_module', _fail_on_import
  )

  with pytest.raises(ValueError, match='Disallowed module reference'):
    config_agent_utils.resolve_fully_qualified_name('os.system')
  with pytest.raises(ValueError, match='Disallowed module reference'):
    config_agent_utils.resolve_code_reference(CodeConfig(name='os.system'))
  with pytest.raises(ValueError, match='Disallowed module reference'):
    config_agent_utils._resolve_agent_code_reference('os.system')
  with pytest.raises(ValueError, match='Disallowed module reference'):
    LlmAgent._resolve_tools([ToolConfig(name='os.system')], '/fake/path.yaml')
