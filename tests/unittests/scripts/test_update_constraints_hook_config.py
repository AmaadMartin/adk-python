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

"""Guard tests for the ``update-constraints`` pre-commit hook.

The Pre-commit Linter CI job runs ``pre-commit run --all-files``. These tests
pin the two conditions that keep that job green without dropping the pin-drift
guard the hook provides.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex

import pytest
import yaml


def _find_repo_root() -> Path | None:
  """Locates the repo root by walking up for the pre-commit config.

  Returns None outside a source checkout, because the repo config files are
  not shipped in the wheel. The test tree may be symlinked, so the walk avoids
  ``.resolve()``.
  """
  start = Path(__file__).parent
  for candidate in [start, *start.parents]:
    if (candidate / '.pre-commit-config.yaml').is_file():
      return candidate
  return None


_REPO_ROOT = _find_repo_root()

pytestmark = pytest.mark.skipif(
    _REPO_ROOT is None,
    reason='.pre-commit-config.yaml is only available in a source checkout.',
)

_PYTHON_VERSIONS_PATTERN = re.compile(
    r'^PYTHON_VERSIONS=\(([^)]*)\)', re.MULTILINE
)


def _update_constraints_hook() -> dict[str, object]:
  """Returns the single ``update-constraints`` hook from the config."""
  assert _REPO_ROOT is not None
  config = yaml.safe_load(
      (_REPO_ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
  )
  hooks = [
      hook
      for repo in config['repos']
      for hook in repo['hooks']
      if hook['id'] == 'update-constraints'
  ]
  assert len(hooks) == 1, (
      f'Expected exactly one update-constraints hook, found {len(hooks)}.'
  )
  return hooks[0]


def test_update_constraints_hook_runs_in_check_mode() -> None:
  argv = shlex.split(str(_update_constraints_hook()['entry']))
  assert argv[0] == './scripts/update_constraints.sh'
  assert '--check' in argv[1:], (
      'The update-constraints hook must pass --check. Without it the script '
      'runs in update mode, which rewrites the constraints-*.txt files and '
      'exits non-zero whenever it rewrites one, so the hook can never pass in '
      'the Pre-commit Linter job.'
  )


def test_every_constrained_python_version_has_a_committed_file() -> None:
  assert _REPO_ROOT is not None
  script = (_REPO_ROOT / 'scripts' / 'update_constraints.sh').read_text(
      encoding='utf-8'
  )
  match = _PYTHON_VERSIONS_PATTERN.search(script)
  assert match is not None, (
      'Could not read the PYTHON_VERSIONS array out of '
      'scripts/update_constraints.sh.'
  )
  versions = shlex.split(match.group(1))
  assert versions, 'scripts/update_constraints.sh lists no Python versions.'
  missing = [
      version
      for version in versions
      if not (_REPO_ROOT / f'constraints-{version}.txt').is_file()
  ]
  assert not missing, (
      f'No committed constraints file for Python {missing}. Check mode fails '
      'with "constraints-<version>.txt is missing!", which reds the '
      'Pre-commit Linter job.'
  )
