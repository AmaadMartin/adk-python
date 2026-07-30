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

"""Tests for the ``Update Dependency Constraints`` regeneration contract.

The scheduled workflow cannot tell from the files on disk whether a resolution
failed: ``scripts/update_constraints.sh`` leaves every ``$TARGET_FILE``
untouched when uv cannot resolve, so a partial failure looks exactly like a
successful run that mixes fresh pins with stale ones. The script therefore
reserves exit ``2`` for that case, and the workflow tolerates only exit ``1``
(files rewritten, the normal outcome of a scheduled run).

These tests pin both halves of that contract: the script's exit codes, driven
by a stub ``uv``, and the workflow step's response to them, driven by a stub
script. Both halves run hermetically -- no network and no real resolution.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from tests.unittests.test_release_dependencies import _find_pyproject

_REPO_ROOT = _find_pyproject().parent
_SCRIPT_PATH = _REPO_ROOT / 'scripts' / 'update_constraints.sh'
_WORKFLOW_PATH = _REPO_ROOT / '.github/workflows/update-constraints.yml'
_STEP_NAME = 'Regenerate constraints files'
_ALL_VERSIONS = ('3.10', '3.11', '3.12', '3.13', '3.14')

# Writes the resolution to the -o path the script passes last, unless that
# path belongs to a version the test wants uv to fail on.
_UV_STUB = """#!/bin/bash
out=${@: -1}
for failed in sentinel %(failed)s; do
  if [ "$out" = "constraints-$failed.txt.new.tmp" ]; then
    echo "stub uv: no solution for $out" >&2
    exit 1
  fi
done
printf '# stub header\\n# stub command\\nstub-package==1.0.0\\n' > "$out"
"""


def _run_script(tmp_path: Path, failed: tuple[str, ...]) -> int:
  """Runs the real update script against a stub uv, returning its exit code.

  Args:
    tmp_path: Working directory standing in for the repository checkout.
    failed: Python versions the stub uv refuses to resolve.

  Returns:
    The script's exit status.
  """
  bin_dir = tmp_path / 'bin'
  bin_dir.mkdir()
  stub = bin_dir / 'uv'
  stub.write_text(_UV_STUB % {'failed': ' '.join(failed)})
  stub.chmod(0o755)

  (tmp_path / 'pyproject.toml').write_text('[project]\nname = "stub"\n')
  for version in _ALL_VERSIONS:
    (tmp_path / f'constraints-{version}.txt').write_text('committed==0.0.1\n')

  return subprocess.run(
      ['bash', str(_SCRIPT_PATH)],
      cwd=tmp_path,
      # The script prepends $HOME/.local/bin to PATH, so HOME must point
      # somewhere without a real uv for the stub to win.
      env={'PATH': f'{bin_dir}:/usr/bin:/bin', 'HOME': str(tmp_path)},
      capture_output=True,
      text=True,
      check=False,
  ).returncode


@pytest.mark.parametrize(
    'failed',
    [
        pytest.param(_ALL_VERSIONS, id='every-resolution-fails'),
        pytest.param(('3.12',), id='one-resolution-fails'),
    ],
)
def test_script_exits_2_when_a_resolution_fails(tmp_path, failed):
  """A single failed version must win over the versions that succeeded."""
  assert _run_script(tmp_path, failed) == 2


def test_script_exits_1_when_it_rewrites_the_files(tmp_path):
  """The successful path keeps its historical exit code."""
  assert _run_script(tmp_path, ()) == 1


def _run_workflow_step(tmp_path: Path, script_exit: int) -> int:
  """Runs the workflow's regeneration step against a stub update script.

  Args:
    tmp_path: Working directory standing in for the repository checkout.
    script_exit: Status the stub update script exits with.

  Returns:
    The step's exit status.
  """
  scripts_dir = tmp_path / 'scripts'
  scripts_dir.mkdir()
  stub = scripts_dir / 'update_constraints.sh'
  stub.write_text(f'#!/bin/bash\nexit {script_exit}\n')
  stub.chmod(0o755)

  workflow = yaml.safe_load(_WORKFLOW_PATH.read_text())
  (job,) = workflow['jobs'].values()
  (step,) = [s for s in job['steps'] if s.get('name') == _STEP_NAME]
  step_path = tmp_path / 'step.sh'
  step_path.write_text(step['run'])

  return subprocess.run(
      # GitHub Actions runs a `run:` block as `bash -e {0}`.
      ['bash', '-e', str(step_path)],
      cwd=tmp_path,
      capture_output=True,
      text=True,
      check=False,
  ).returncode


@pytest.mark.parametrize('script_exit', [0, 1])
def test_workflow_step_accepts_nothing_to_do_and_rewritten(
    tmp_path, script_exit
):
  """Rewriting the files is the normal outcome and must not fail the job."""
  assert _run_workflow_step(tmp_path, script_exit) == 0


def test_workflow_step_rejects_a_failed_resolution(tmp_path):
  """Exit 2 must stop the job before it opens a pull request."""
  assert _run_workflow_step(tmp_path, 2) != 0
