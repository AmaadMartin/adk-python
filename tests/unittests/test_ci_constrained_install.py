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

"""Guard tests for the constrained dependency installs in CI.

``constraints-3.10.txt`` … ``constraints-3.14.txt`` only pin anything if the
jobs that consume them actually pass them to the installer. Every failure mode
here is silent -- CI stays green while quietly testing whatever PyPI served
that morning -- so each is pinned separately:

* The workflow must trigger on ``constraints-*.txt``. Without it a
  constraints-only change, including every scheduled refresh, runs no jobs at
  all and the new pins are merged untested.
* ``type-check`` and ``unit-test`` must install with
  ``-c constraints-<matrix version>.txt``.
* Neither may reach for ``uv sync`` or ``uv run``. ``uv sync`` accepts no
  ``--constraint`` at all, and ``uv run`` syncs the project environment before
  executing, re-resolving from scratch and discarding an already-pinned
  install. Both leave a working, wholly unpinned job behind.
* Every interpreter in those matrices needs a committed constraints file, or
  its install step fails on a path that does not exist.
* ``constraints-check.yml`` must trigger on itself. It diffs uv's output
  byte-for-byte against the committed files, so its pinned uv ``version:`` has
  to move together with a regeneration; a change to that pin alone would
  otherwise merge unchecked and fail on an unrelated pull request later.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest
import yaml

from tests.unittests.test_release_dependencies import _find_pyproject

_REPO_ROOT = _find_pyproject().parent
_WORKFLOWS_DIR = _REPO_ROOT / '.github' / 'workflows'
_WORKFLOW_PATH = _WORKFLOWS_DIR / 'continuous-integration.yml'
_CHECK_WORKFLOW_PATH = _WORKFLOWS_DIR / 'constraints-check.yml'

if not (_WORKFLOW_PATH.is_file() and _CHECK_WORKFLOW_PATH.is_file()):
  pytest.skip(
      'Not a full source checkout: the CI workflows are absent.',
      allow_module_level=True,
  )


def _triggers(path: Path) -> dict:
  """Returns a workflow's ``on:`` block.

  PyYAML applies the YAML 1.1 resolver, which reads the bare ``on:`` key as the
  boolean ``True`` rather than the string ``'on'``.
  """
  workflow = yaml.safe_load(path.read_text())
  return workflow.get('on', workflow.get(True))


_WORKFLOW = yaml.safe_load(_WORKFLOW_PATH.read_text())
_TRIGGERS = _triggers(_WORKFLOW_PATH)

# The jobs whose dependency install this pins. `unit-test-a2a-v0-3` is
# deliberately excluded: it force-reinstalls a2a-sdk 0.3.x over the pinned
# 1.x line, so constraining it needs its own reasoning.
_PINNED_JOBS = ('type-check', 'unit-test')
_CONSTRAINT_FLAG = '-c constraints-${{ matrix.python-version }}.txt'
# uv invocations that resolve dependencies themselves, ignoring or replacing
# whatever the constrained install just produced.
_UNPINNED_COMMANDS = ('uv sync', 'uv run')


def _run_script(job: str) -> str:
  """Returns the executed shell of every ``run:`` step in ``job``.

  Whole-line ``#`` comments are dropped: these guards are about the commands
  CI runs, and the workflow's own prose explains the very commands they forbid.
  """
  lines = '\n'.join(
      step['run'] for step in _WORKFLOW['jobs'][job]['steps'] if 'run' in step
  ).splitlines()
  return '\n'.join(line for line in lines if not line.lstrip().startswith('#'))


def _matrix_versions(job: str) -> list[str]:
  return _WORKFLOW['jobs'][job]['strategy']['matrix']['python-version']


@pytest.mark.parametrize('event', ['push', 'pull_request'])
def test_ci_triggers_on_constraints_changes(event: str) -> None:
  assert 'constraints-*.txt' in _TRIGGERS[event]['paths'], (
      f"The {event} trigger's paths filter does not list 'constraints-*.txt',"
      ' so a change to the pinned dependency versions runs no CI at all --'
      ' including the scheduled refresh, whose whole purpose is to surface a'
      ' dependency upgrade as a tested, reviewable diff.'
  )


@pytest.mark.parametrize('job', _PINNED_JOBS)
def test_job_installs_against_its_matrix_constraints_file(job: str) -> None:
  assert _CONSTRAINT_FLAG in _run_script(job), (
      f'The {job} job does not install with {_CONSTRAINT_FLAG!r}, so it'
      ' resolves the newest compatible release of every dependency on every'
      ' run and the committed constraints files pin nothing.'
  )


@pytest.mark.parametrize('command', _UNPINNED_COMMANDS)
@pytest.mark.parametrize('job', _PINNED_JOBS)
def test_job_avoids_uv_commands_that_re_resolve(job: str, command: str) -> None:
  assert command not in _run_script(job), (
      f'The {job} job runs `{command}`, which resolves dependencies itself:'
      ' `uv sync` has no --constraint option, and `uv run` syncs the project'
      ' environment before executing. Either one silently discards the pins'
      ' from constraints-*.txt while leaving the job green.'
  )


@pytest.mark.parametrize('event', ['push', 'pull_request'])
def test_constraints_check_workflow_triggers_on_itself(event: str) -> None:
  paths = _triggers(_CHECK_WORKFLOW_PATH)[event]['paths']
  assert f'.github/workflows/{_CHECK_WORKFLOW_PATH.name}' in paths, (
      f"{_CHECK_WORKFLOW_PATH.name}'s {event} trigger does not list the"
      ' workflow itself, so a pull request that only bumps its pinned uv'
      ' version runs no check. That pin has to be bumped together with a'
      ' regeneration, and the mismatch would instead surface as someone'
      " else's unrelated pull request going red."
  )


@pytest.mark.parametrize('job', _PINNED_JOBS)
def test_matrix_versions_all_have_a_constraints_file(job: str) -> None:
  missing = [
      version
      for version in _matrix_versions(job)
      if not (_REPO_ROOT / f'constraints-{version}.txt').is_file()
  ]
  assert not missing, (
      f'The {job} matrix tests Python {missing}, but no constraints file is'
      ' committed for them. The install step interpolates the matrix version'
      ' into the -c path, so it would fail on a nonexistent file.'
  )


def test_mypy_baseline_uses_the_pull_request_constraints() -> None:
  """The baseline install must take the PR's pins, not the base branch's.

  ``type-check`` compares mypy output on the base branch against the pull
  request. It already copies the PR's ``pyproject.toml`` over the base branch
  checkout; without the constraints files too, the two runs would differ in
  their dependency versions as well as in source, and a third-party stub
  change would be misattributed to the pull request.
  """
  script = _run_script('type-check')
  assert re.search(
      r"git checkout .+ -- pyproject\.toml 'constraints-\*\.txt'", script
  ), (
      "The type-check baseline does not restore the PR's constraints files"
      ' alongside its pyproject.toml, so the baseline and PR mypy runs can'
      ' resolve different dependency versions and the comparison stops'
      ' isolating source changes.'
  )
