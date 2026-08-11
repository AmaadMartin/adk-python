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

"""Guard tests for the clean-checkout step of the unit-test CI job.

The step fails the build when the test run leaves the working tree modified.
Its failure path never runs in a healthy build, so these tests take the script
out of the workflow file and run it against a throwaway git repository. That is
the only way to know the step still fails when it should.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import TypedDict

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

_WORKFLOW = _REPO_ROOT / '.github' / 'workflows' / 'continuous-integration.yml'
_PYTEST_STEP = 'Run unit tests with pytest'
_GUARD_STEP = 'Fail if the test run dirtied the checkout'

# Isolates the temporary repository from the developer's own git configuration,
# which may otherwise sign commits or impose hooks.
_GIT_ENV = {
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_SYSTEM': os.devnull,
    'GIT_AUTHOR_NAME': 'ADK Test',
    'GIT_AUTHOR_EMAIL': 'adk-test@example.com',
    'GIT_COMMITTER_NAME': 'ADK Test',
    'GIT_COMMITTER_EMAIL': 'adk-test@example.com',
}


class _Step(TypedDict, total=False):
  """The workflow-step fields these tests read."""

  name: str
  run: str


def _unit_test_steps() -> list[_Step]:
  """Returns the steps of the workflow's ``unit-test`` job."""
  workflow = yaml.safe_load(_WORKFLOW.read_text())
  steps: list[_Step] = workflow['jobs']['unit-test']['steps']
  return steps


def _git(repo: Path, *args: str) -> None:
  """Runs a git command in ``repo`` and raises if it fails."""
  subprocess.run(
      ('git', *args),
      cwd=repo,
      check=True,
      capture_output=True,
      env={**os.environ, **_GIT_ENV},
  )


def _run_guard(script: str, repo: Path) -> subprocess.CompletedProcess[str]:
  """Runs the guard script over ``repo`` the way a Linux runner does.

  GitHub Actions writes the ``run`` block to a file and executes it with
  ``bash -e``, so a failing command aborts the step.

  Args:
    script: The shell script taken from the workflow step.
    repo: The git repository the script inspects.

  Returns:
    The completed process, with stdout and stderr captured as text.
  """
  # Kept outside the repository so the script does not dirty what it reads.
  script_path = repo.parent / 'guard.sh'
  script_path.write_text(script)
  return subprocess.run(
      ('bash', '-e', str(script_path)),
      cwd=repo,
      capture_output=True,
      text=True,
      env={**os.environ, **_GIT_ENV},
  )


@pytest.fixture(scope='module')
def guard_script() -> str:
  """Returns the shell script that the guard step runs."""
  for step in _unit_test_steps():
    if step.get('name') == _GUARD_STEP and (script := step.get('run')):
      return script
  pytest.fail(f'The unit-test job runs no {_GUARD_STEP!r} script.')


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
  """Returns a throwaway git repository holding one committed asset."""
  repo = tmp_path / 'checkout'
  (repo / 'src').mkdir(parents=True)
  (repo / 'src' / 'runtime-config.json').write_text('{"backendUrl": ""}')
  _git(repo, 'init', '-b', 'main')
  _git(repo, 'add', '.')
  _git(repo, 'commit', '-m', 'Add the packaged asset.')
  return repo


def test_guard_passes_on_a_clean_checkout(
    guard_script: str, checkout: Path
) -> None:
  result = _run_guard(guard_script, checkout)

  assert result.returncode == 0
  assert 'Working tree is clean after the test run.' in result.stdout


def test_guard_fails_and_names_a_modified_tracked_file(
    guard_script: str, checkout: Path
) -> None:
  (checkout / 'src' / 'runtime-config.json').write_text(
      '{"backendUrl": "", "telemetry": null}\n'
  )

  result = _run_guard(guard_script, checkout)

  assert result.returncode == 1
  assert '::error::' in result.stdout
  assert 'src/runtime-config.json' in result.stdout


def test_guard_fails_and_names_an_untracked_file(
    guard_script: str, checkout: Path
) -> None:
  """``git status --porcelain`` is used so a stray new file also fails."""
  (checkout / 'stray-test-output.txt').write_text('written by a test\n')

  result = _run_guard(guard_script, checkout)

  assert result.returncode == 1
  assert '::error::' in result.stdout
  assert 'stray-test-output.txt' in result.stdout


def test_guard_runs_directly_after_the_unit_tests() -> None:
  """Placed any earlier, the guard reads the pre-test tree and never fires."""
  names = [step.get('name') for step in _unit_test_steps()]

  assert _GUARD_STEP in names
  assert names.index(_GUARD_STEP) == names.index(_PYTEST_STEP) + 1


def test_the_artifacts_the_unit_test_job_creates_are_gitignored() -> None:
  """The job writes these into the checkout, so the guard must ignore them."""
  artifacts = (
      '.venv/pyvenv.cfg',
      'uv.lock',
      '.pytest_cache/CACHEDIR.TAG',
      'src/google_adk.egg-info/PKG-INFO',
  )

  result = subprocess.run(
      ('git', 'check-ignore', *artifacts),
      cwd=_REPO_ROOT,
      capture_output=True,
      text=True,
      env={**os.environ, **_GIT_ENV},
  )

  assert set(result.stdout.split()) == set(artifacts)
