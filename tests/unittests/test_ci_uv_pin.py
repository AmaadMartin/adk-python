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

"""Guard tests for the uv version pin in the Continuous Integration workflow.

The Pre-commit Linter job regenerates the committed ``constraints-3.*.txt``
files with ``uv pip compile`` and compares the output byte for byte. With a
floating uv, an Astral release that changed the resolver or the output
formatting would fail that comparison on every open pull request. These tests
pin the invariant that removes the risk: the workflow declares one exact uv
version, and every ``astral-sh/setup-uv`` step installs that version.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml

_WORKFLOW_PATH = (
    Path(__file__).parent.parent.parent
    / '.github'
    / 'workflows'
    / 'continuous-integration.yml'
)
_WORKFLOW = yaml.safe_load(_WORKFLOW_PATH.read_text())

_SETUP_UV_PREFIX = 'astral-sh/setup-uv@'
_PIN_REFERENCE = '${{ env.UV_PINNED_VERSION }}'


def test_ci_workflow_pins_an_exact_uv_version() -> None:
  version = _WORKFLOW.get('env', {}).get('UV_PINNED_VERSION')

  assert version is not None, (
      f'{_WORKFLOW_PATH.name} must declare a top-level env.UV_PINNED_VERSION'
      ' holding the uv version that every job installs.'
  )
  assert re.fullmatch(r'\d+\.\d+\.\d+', version), (
      f'env.UV_PINNED_VERSION is {version!r}, which is not an exact uv version'
      " such as '0.12.3'. A floating value lets an Astral release change the"
      ' constraints check on pull requests that never touched dependencies.'
  )


def test_every_setup_uv_step_uses_the_pinned_version() -> None:
  installed_versions = {
      f'{job_name} / {step.get("name")}': step.get('with', {}).get('version')
      for job_name, job in _WORKFLOW['jobs'].items()
      for step in job.get('steps', [])
      if step.get('uses', '').startswith(_SETUP_UV_PREFIX)
  }

  assert installed_versions, (
      f'{_WORKFLOW_PATH.name} installs uv with no {_SETUP_UV_PREFIX} step, so'
      ' this test would pass without checking anything.'
  )
  assert set(installed_versions.values()) == {_PIN_REFERENCE}, (
      f'every {_SETUP_UV_PREFIX} step must pass'
      f" version: '{_PIN_REFERENCE}' so the jobs share one uv; the steps"
      f' install {installed_versions}.'
  )
