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

"""Guard tests for the suite a bare ``pytest`` collects.

``testpaths`` in ``pyproject.toml`` decides what ``pytest`` collects when a
contributor passes no path. It must name the one hermetic tree,
``tests/unittests``, and it must agree with the suite ``tox`` runs.
``tests/integration`` and ``tests/remote`` need live cloud credentials and
extras outside ``optional-dependencies.test``, so a default run cannot collect
them.
"""

from __future__ import annotations

import configparser
import shlex

try:
  import tomllib
except ImportError:
  import tomli as tomllib

import pytest

from .isolated_import_utils import REPO_ROOT

_PYPROJECT_PATH = REPO_ROOT / 'pyproject.toml'
_TOX_PATH = REPO_ROOT / 'tox.ini'


@pytest.fixture(scope='module')
def testpaths() -> list[str]:
  """Returns ``tool.pytest.ini_options.testpaths`` from pyproject.toml."""
  with _PYPROJECT_PATH.open('rb') as fh:
    pyproject = tomllib.load(fh)
  return pyproject['tool']['pytest']['ini_options']['testpaths']


def test_testpaths_is_the_unit_suite(testpaths: list[str]) -> None:
  """A bare ``pytest`` must collect the unit suite and nothing else."""
  assert testpaths == ['tests/unittests'], (
      'testpaths in pyproject.toml is the suite a bare `pytest` advertises, so '
      'it must stay ["tests/unittests"]. tests/integration and tests/remote '
      'need live cloud credentials and extras that '
      'optional-dependencies.test does not install, and a default run must not '
      f'collect them. Found {testpaths!r}.'
  )
  assert (REPO_ROOT / testpaths[0]).is_dir(), (
      f'testpaths in pyproject.toml names {testpaths[0]!r}, which is not a '
      f'directory under {REPO_ROOT}. A bare `pytest` would collect nothing.'
  )


def test_tox_runs_the_testpaths_suite(testpaths: list[str]) -> None:
  """``tox`` and a bare ``pytest`` must run the same suite."""
  parser = configparser.ConfigParser()
  parser.read(_TOX_PATH, encoding='utf-8')
  command = shlex.split(parser['testenv']['commands'])
  assert command[:1] == ['pytest'], (
      f'[testenv] commands in {_TOX_PATH.name} must run pytest, so that the '
      f'suite tox runs stays comparable with testpaths. Found {command!r}.'
  )
  tox_paths = [word for word in command[1:] if not word.startswith('-')]
  assert tox_paths == testpaths, (
      f'{_TOX_PATH.name} and pyproject.toml must name one suite, so that `tox` '
      'and a bare `pytest` run the same tests. Update [testenv] commands and '
      f'testpaths together. tox runs {tox_paths!r}, testpaths is {testpaths!r}.'
  )
