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

"""Guard tests for the Continuous Integration workflow's ``paths`` filters.

Both triggers of the workflow are restricted by a ``paths`` allow-list, and
GitHub queues the workflow only when a changed file matches one of the
patterns. Two file sets must stay inside that allow-list:

* ``constraints-*.txt`` are published artifacts. README.md tells users to
  download one and pass it to ``pip install -c``.
* ``scripts/`` holds ``update_constraints.sh``, which generates those
  artifacts, plus the release and lint helpers.

The ``update-constraints`` pre-commit hook is the only check that the
constraints files are still reproducible from ``pyproject.toml``, and it runs
inside the Pre-commit Linter job of this workflow alone. Drop either pattern
and a change to the files the hook protects queues no jobs at all.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPO_ROOT / '.github' / 'workflows' / 'continuous-integration.yml'

pytestmark = pytest.mark.skipif(
    not _WORKFLOW.exists(),
    reason=f'{_WORKFLOW} is absent outside the GitHub source layout.',
)

_CONSTRAINTS_FILES = (
    'constraints-3.10.txt',
    'constraints-3.11.txt',
    'constraints-3.12.txt',
    'constraints-3.13.txt',
    'constraints-3.14.txt',
)

_GATED_PATHS = _CONSTRAINTS_FILES + (
    'scripts/update_constraints.sh',
    'scripts/check_new_py_files.sh',
    'scripts/db_migration.sh',
    'scripts/release_import_allowlist.txt',
)


def _trigger_paths() -> dict[str, list[str]]:
  """Returns the ``paths`` filter of each trigger, keyed by event name."""
  document = yaml.safe_load(_WORKFLOW.read_text())
  # PyYAML implements YAML 1.1, where the bare key `on` resolves to the
  # boolean True. pyproject.toml pins `pyyaml>=6.0.2,<7`, so the string key
  # 'on' never appears.
  triggers = document[True]
  return {event: triggers[event]['paths'] for event in ('push', 'pull_request')}


def _matches(pattern: str, path: str) -> bool:
  """Applies GitHub's filter rules: ``*`` stops at ``/``, ``**`` does not."""
  parts = [
      re.escape(part).replace(r'\*', '[^/]*') for part in pattern.split('**')
  ]
  return re.fullmatch('.*'.join(parts), path) is not None


@pytest.mark.parametrize('path', _GATED_PATHS)
def test_gated_path_matches_both_triggers(path: str) -> None:
  """Every published constraints file and script must queue the workflow."""
  for event, patterns in _trigger_paths().items():
    assert any(_matches(pattern, path) for pattern in patterns), (
        f"on.{event}.paths matches no pattern for '{path}'. A pull request"
        ' that changes only that file queues no job, so the Pre-commit'
        ' Linter never runs the update-constraints drift check. Add the'
        ' pattern that covers it to both trigger lists.'
    )


@pytest.mark.parametrize('path', ['CHANGELOG.md', 'docs/guides/README.md'])
def test_ungated_path_matches_neither_trigger(path: str) -> None:
  """The filter must stay an allow-list rather than become a catch-all."""
  for event, patterns in _trigger_paths().items():
    assert not any(_matches(pattern, path) for pattern in patterns), (
        f"on.{event}.paths matches '{path}', so the filter now selects files"
        ' it was never meant to gate. Replace the over-broad pattern with one'
        ' that names the paths the jobs actually depend on.'
    )


def test_push_and_pull_request_filters_stay_identical() -> None:
  """A pre-merge gate and a post-merge gate must cover the same files."""
  paths = _trigger_paths()
  assert paths['push'] == paths['pull_request'], (
      'on.push.paths and on.pull_request.paths are hand-duplicated and have'
      ' drifted. The repository would then check one file set before a merge'
      ' and a different one after it.'
  )


@pytest.mark.parametrize('name', _CONSTRAINTS_FILES)
def test_constraints_file_exists(name: str) -> None:
  """The gated constraints files must still live at the repository root."""
  assert (_REPO_ROOT / name).is_file(), (
      f"'{name}' is gone from the repository root, so the pattern that gates"
      ' it protects nothing. Update both trigger lists and this test together'
      ' with the rename.'
  )
