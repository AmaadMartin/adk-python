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

"""Guard test: the click floor CI pins must be the one pyproject.toml declares.

``pyproject.toml`` declares a floored click range and the repository checks in
no lock file, so every job that just runs ``uv sync`` gets whatever the newest
click happens to be that day. The ``unit-test-cli-click-floor`` job is the only
thing that exercises the other end, and it does so from a version hard-coded in
the workflow. Raising the declared floor without updating that pin leaves the
job installing a version the project no longer supports while still reporting a
green check.
"""

from __future__ import annotations

from pathlib import Path

try:
  import tomllib
except ImportError:
  import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT_PATH = _REPO_ROOT / 'pyproject.toml'
_WORKFLOW_PATH = (
    _REPO_ROOT / '.github' / 'workflows' / 'continuous-integration.yml'
)
_JOB_ID = 'unit-test-cli-click-floor'


@pytest.fixture(scope='module')
def declared_click_floor() -> str:
  """Returns the lowest click version ``pyproject.toml`` declares support for."""
  with _PYPROJECT_PATH.open('rb') as fh:
    dependencies = tomllib.load(fh)['project']['dependencies']
  floors = [
      specifier.version
      for requirement in map(Requirement, dependencies)
      if canonicalize_name(requirement.name) == 'click'
      for specifier in requirement.specifier
      if specifier.operator == '>='
  ]
  assert (
      len(floors) == 1
  ), f'pyproject.toml must declare exactly one click >= floor, got {floors}.'
  return floors[0]


@pytest.fixture(scope='module')
def pinned_click_floor() -> str:
  """Returns the click version the CI floor job installs."""
  workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
  return workflow['jobs'][_JOB_ID]['env']['CLICK_FLOOR']


def test_ci_pins_the_declared_click_floor(
    declared_click_floor: str, pinned_click_floor: str
) -> None:
  """The floor job installs exactly the floor pyproject.toml declares."""
  assert pinned_click_floor == declared_click_floor, (
      f'The {_JOB_ID} job installs click=={pinned_click_floor} but '
      f'pyproject.toml declares a floor of {declared_click_floor}. Set '
      f"CLICK_FLOOR to '{declared_click_floor}', or the job reports green "
      'while testing a version the project no longer supports.'
  )
