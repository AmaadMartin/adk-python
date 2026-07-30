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

"""Behaviour tests for the ``Update Dependency Constraints`` workflow.

``scripts/update_constraints.sh`` exits ``1`` both when it rewrites a
constraints file -- the normal outcome of a scheduled run, because
``--exclude-newer`` moves every time -- and when a PyPI resolution fails. The
workflow therefore cannot trust the exit code and verifies the artifacts
itself. That verification is the only thing standing between a total
resolution failure and a bot pull request that publishes empty supply-chain
pins, so these tests execute the step's real shell body, extracted from the
workflow file, against a stubbed generator.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from tests.unittests.test_release_dependencies import _find_pyproject

_WORKFLOW_PATH = (
    _find_pyproject().parent / '.github/workflows/update-constraints.yml'
)
_STEP_NAME = 'Regenerate constraints files'
_ALL_VERSIONS = ('3.10', '3.11', '3.12', '3.13', '3.14')


def _regeneration_step_body() -> str:
  """Returns the ``run:`` body of the workflow's regeneration step."""
  workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
  (job,) = workflow['jobs'].values()
  (step,) = [s for s in job['steps'] if s.get('name') == _STEP_NAME]
  return step['run']


def _run_step(
    tmp_path: Path, generated: dict[str, str], script_exit: int
) -> subprocess.CompletedProcess[str]:
  """Runs the regeneration step against a stub generator in ``tmp_path``.

  Args:
    tmp_path: Working directory standing in for the repository checkout.
    generated: Constraints file name to contents the stub generator writes.
    script_exit: Status the stub generator exits with.

  Returns:
    The completed ``bash`` process for the step body.
  """
  scripts_dir = tmp_path / 'scripts'
  scripts_dir.mkdir()
  stub = scripts_dir / 'update_constraints.sh'
  writes = '\n'.join(
      f'printf %s {contents!r} > {name!r}'
      for name, contents in generated.items()
  )
  stub.write_text(f'#!/bin/bash\n{writes}\nexit {script_exit}\n')
  stub.chmod(0o755)

  step = tmp_path / 'step.sh'
  step.write_text(_regeneration_step_body())
  # GitHub Actions runs a `run:` block as `bash -e {0}`.
  return subprocess.run(
      ['bash', '-e', str(step)],
      cwd=tmp_path,
      capture_output=True,
      text=True,
      check=False,
  )


def test_step_tolerates_the_exit_code_the_script_uses_for_a_rewrite(tmp_path):
  """A scheduled run always rewrites the files, so exit 1 must not fail it."""
  result = _run_step(
      tmp_path,
      {f'constraints-{ver}.txt': f'pinned-{ver}\n' for ver in _ALL_VERSIONS},
      script_exit=1,
  )

  assert result.returncode == 0, result.stdout + result.stderr


def test_step_fails_when_the_resolution_produced_no_files(tmp_path):
  """A failed resolution must not reach the create-pull-request step."""
  result = _run_step(tmp_path, {}, script_exit=1)

  assert result.returncode == 1
  assert '::error::constraints-3.10.txt was not generated' in result.stdout


@pytest.mark.parametrize('missing_version', _ALL_VERSIONS)
def test_step_fails_when_any_file_is_empty(tmp_path, missing_version):
  """An empty file is a failed resolution too, hence ``-s`` and not ``-f``."""
  generated = {
      f'constraints-{ver}.txt': (
          '' if ver == missing_version else f'pinned-{ver}\n'
      )
      for ver in _ALL_VERSIONS
  }

  result = _run_step(tmp_path, generated, script_exit=1)

  assert result.returncode == 1
  assert (
      f'::error::constraints-{missing_version}.txt was not generated'
      in result.stdout
  )
