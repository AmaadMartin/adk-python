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

"""Guard tests for the build backend the release jobs use.

The PEP 517 build backend writes the wheel's METADATA, its file layout and its
RECORD, so the backend version is a release input. ``uv build`` resolves
``[build-system] requires`` fresh on every run, which makes that input
whatever the index serves on release morning. The repository therefore names
one exact version in ``build-constraints.txt``, and both release workflows
pass that file to ``uv build``.

These tests fail when the pin is dropped, loosened into a range, or drifts
outside the range ``pyproject.toml`` declares. The last case is the valuable
one: uv rejects a constraint outside the declared range with "No solution
found when resolving build-system.requires", and that failure would otherwise
first appear during a release.

The checks are static. They read files and never run uv, build a
distribution, or reach the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
  import tomllib
except ImportError:
  import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest
import yaml

from .isolated_import_utils import REPO_ROOT

_CONSTRAINTS_FILENAME = 'build-constraints.txt'
_CONSTRAINTS_PATH = REPO_ROOT / _CONSTRAINTS_FILENAME
_WORKFLOWS_DIR = REPO_ROOT / '.github' / 'workflows'
_RELEASE_WORKFLOWS = ('release-artifact-check.yml', 'release-publish.yml')

# The command both release workflows must run. Spelling it once here is what
# makes the two jobs provably build under the same backend.
_BUILD_COMMAND = f'uv build --build-constraints {_CONSTRAINTS_FILENAME}'

_PYPROJECT: dict[str, Any] = tomllib.loads(
    (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
)

# flit-core releases that cannot build this project across the supported
# interpreter range: they call ast.Str, which Python 3.14 removed, and the
# unit-test matrix runs through 3.14.
_UNUSABLE_BACKENDS = ('3.8.0', '3.9.0')

# Released flit-core versions that do build this project. The declared range
# must keep admitting more than one of them, or a downstream source build has
# to fetch one specific backend.
_USABLE_BACKENDS = ('3.10.0', '3.12.0')


def _uv_build_steps(workflow_name: str) -> list[dict[str, Any]]:
  """Returns the steps of one release workflow that build with uv.

  Args:
    workflow_name: File name of a workflow in ``.github/workflows``.

  Returns:
    Every step whose ``run`` block invokes ``uv build``, in file order.
  """
  workflow = yaml.safe_load(
      (_WORKFLOWS_DIR / workflow_name).read_text(encoding='utf-8')
  )
  return [
      step
      for job in workflow['jobs'].values()
      for step in job['steps']
      if 'uv build' in str(step.get('run', ''))
  ]


def _parse_constraints(path: Path) -> dict[str, Requirement]:
  """Returns the pins in a constraints file, keyed by distribution name.

  Args:
    path: The constraints file to read.

  Returns:
    One requirement per distribution, under its canonical name. Blank lines
    and comments are skipped.
  """
  pins: dict[str, Requirement] = {}
  for line in path.read_text(encoding='utf-8').splitlines():
    entry = line.split('#', 1)[0].strip()
    if not entry:
      continue
    requirement = Requirement(entry)
    pins[canonicalize_name(requirement.name)] = requirement
  return pins


def _declared_build_requirements() -> dict[str, Requirement]:
  """Returns ``[build-system] requires``, keyed by distribution name."""
  requirements = (
      Requirement(entry) for entry in _PYPROJECT['build-system']['requires']
  )
  return {canonicalize_name(req.name): req for req in requirements}


@pytest.fixture(scope='module')
def pins() -> dict[str, Requirement]:
  """Parses build-constraints.txt exactly once for the module."""
  return _parse_constraints(_CONSTRAINTS_PATH)


def test_build_constraints_file_exists_and_pins_exactly() -> None:
  """Every entry names one exact version, so the file cannot decay."""
  assert _CONSTRAINTS_PATH.is_file(), (
      f'{_CONSTRAINTS_FILENAME} is missing. Both release workflows pass it to'
      ' `uv build`, so every release build would fail on an unreadable path.'
  )
  pins = _parse_constraints(_CONSTRAINTS_PATH)
  assert pins, f'{_CONSTRAINTS_FILENAME} pins nothing.'
  for name, requirement in sorted(pins.items()):
    clauses = list(requirement.specifier)
    assert len(clauses) == 1, (
        f'{_CONSTRAINTS_FILENAME} constrains {name} with '
        f'{str(requirement.specifier) or "no specifier"}, so the version is '
        'still chosen at build time. Use a single == clause.'
    )
    assert clauses[0].operator == '==' and '*' not in clauses[0].version, (
        f'{_CONSTRAINTS_FILENAME} constrains {name} with '
        f'{clauses[0]}, which admits more than one release. Name one '
        'concrete version.'
    )


def test_build_constraints_cover_every_build_requirement(
    pins: dict[str, Requirement],
) -> None:
  """Every declared build requirement is pinned, not just today's backend."""
  missing = sorted(set(_declared_build_requirements()) - set(pins))
  assert not missing, (
      f'pyproject.toml requires {", ".join(missing)} to build, but '
      f'{_CONSTRAINTS_FILENAME} does not pin it. The release jobs would '
      'resolve it fresh and publish a wheel built by an unrecorded version.'
  )


def test_pinned_backend_satisfies_declared_range(
    pins: dict[str, Requirement],
) -> None:
  """The pin is inside the range pyproject.toml declares.

  A pin outside that range makes uv fail the release build with "No solution
  found when resolving build-system.requires". This test turns that into a
  local failure.
  """
  for name, declared in sorted(_declared_build_requirements().items()):
    version = next(iter(pins[name].specifier)).version
    assert declared.specifier.contains(version), (
        f'{_CONSTRAINTS_FILENAME} pins {name}=={version}, which '
        f'pyproject.toml excludes with {declared.specifier}. uv cannot '
        'resolve a build environment from two requirements that disagree.'
    )


@pytest.mark.parametrize('workflow_name', _RELEASE_WORKFLOWS)
def test_release_workflow_builds_under_the_pin(workflow_name: str) -> None:
  """The release build still runs, and runs under the pinned backend.

  Asserting the build step exists is what stops the pin check from passing
  over nothing when someone deletes the `uv build` line rather than its flag.
  """
  steps = _uv_build_steps(workflow_name)
  assert steps, (
      f'{workflow_name} no longer runs `uv build`. Either the release build '
      'moved, and this guard must follow it, or the workflow lost its build '
      'step.'
  )
  for step in steps:
    assert _BUILD_COMMAND in step['run'], (
        f'{workflow_name} step {step.get("name")!r} runs '
        f'{step["run"].strip()!r}, not `{_BUILD_COMMAND}`. Without the pin uv '
        'resolves the PEP 517 backend fresh, so the published wheel carries '
        'whatever metadata the index served that day.'
    )


def test_declared_range_excludes_backends_that_cannot_build() -> None:
  """The declared range advertises only backends that work, and stays a range."""
  declared = _declared_build_requirements()['flit-core'].specifier
  for version in _UNUSABLE_BACKENDS:
    assert not declared.contains(version), (
        f'pyproject.toml advertises flit-core {version}, which calls '
        'ast.Str. Python 3.14 removed ast.Str, so that backend raises '
        'AttributeError in get_docstring_and_version_via_ast and cannot '
        'build this project across the supported interpreter range.'
    )
  admitted = [v for v in _USABLE_BACKENDS if declared.contains(v)]
  assert len(admitted) > 1, (
      f'pyproject.toml admits only {admitted} of the flit-core releases that '
      'build this project. [build-system] requires must stay a range, or a '
      'downstream source build has to fetch one specific backend. Freeze '
      f'our own releases in {_CONSTRAINTS_FILENAME} instead.'
  )


def test_build_constraints_not_shipped_in_sdist() -> None:
  """The constraints file stays out of the sdist, on purpose."""
  include = _PYPROJECT['tool']['flit']['sdist']['include']
  assert _CONSTRAINTS_FILENAME not in include, (
      f'{_CONSTRAINTS_FILENAME} is shipped in the sdist. A downstream builder'
      ' does not apply our constraints, so shipping the file only enlarges '
      'the sdist and changes its contents.'
  )
