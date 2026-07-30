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

The CI workflow is the merge gate, and both of its triggers are restricted by a
``paths`` allow-list. If that allow-list does not cover ``.github/workflows/``,
the gate stops gating itself: a pull request that only edits workflow files
matches no pattern, so the whole workflow is skipped and none of the lint,
type-check or unit-test jobs ever report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOWS_GLOB = '.github/workflows/**'
_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / '.github'
    / 'workflows'
    / 'continuous-integration.yml'
)

pytestmark = pytest.mark.skipif(
    not _WORKFLOW_PATH.is_file(),
    reason=f'{_WORKFLOW_PATH} is not available outside a source checkout.',
)


@pytest.fixture(scope='module')
def triggers() -> dict[str, Any]:
  """Returns the CI workflow's ``on:`` block, parsed exactly once."""
  with _WORKFLOW_PATH.open('rb') as fh:
    document = yaml.safe_load(fh)
  # PyYAML implements YAML 1.1, where the bare key `on` is the boolean True.
  return document[True]


def test_ci_paths_include_workflow_directory(triggers: dict[str, Any]) -> None:
  """Both triggers must fire on any change under ``.github/workflows/``."""
  for event in ('push', 'pull_request'):
    assert _WORKFLOWS_GLOB in triggers[event]['paths'], (
        f"on.{event}.paths must contain '{_WORKFLOWS_GLOB}'. Without it, a"
        ' change that only touches workflow files matches no pattern, so the'
        ' Continuous Integration workflow is skipped entirely and the lint,'
        ' mypy and unit-test jobs never run on the change least protected'
        ' from breaking CI: an edit to the CI definition itself.'
    )


def test_push_and_pull_request_paths_stay_in_sync(
    triggers: dict[str, Any],
) -> None:
  """Pre-submit and post-submit must gate exactly the same file set."""
  assert triggers['push']['paths'] == triggers['pull_request']['paths'], (
      'on.push.paths and on.pull_request.paths must be identical. They are'
      ' hand-duplicated, so drift between them would leave the repository'
      ' gating one file set before merge and a different one after.'
  )
