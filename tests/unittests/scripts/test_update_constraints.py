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

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform.startswith('win'), reason='POSIX shell script.'
)

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / 'scripts'
    / 'update_constraints.sh'
)
_PYTHON_VERSIONS = ('3.10', '3.11', '3.12', '3.13', '3.14')
# A stub `uv` that writes the file named by `-o` instead of resolving against
# PyPI, so the success path can be exercised without network access. The
# script keeps only `tail -n +3` of this output, hence the two header lines.
_UV_STUB = """#!/bin/bash
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    printf '# header\\n# command\\ntyping-extensions==4.12.2\\n' > "$2"
    exit 0
  fi
  shift
done
exit 1
"""


def _run(tmp_path: pathlib.Path, path: str) -> subprocess.CompletedProcess[str]:
  """Runs the script in an isolated CWD with an explicit PATH and HOME.

  Setting HOME is mandatory: the script prepends `$HOME/.local/bin` to PATH,
  so leaving the developer's real HOME in place would make the missing-`uv`
  test pass for the wrong reason.
  """
  (tmp_path / 'pyproject.toml').write_text(
      '[project]\nname = "x"\nversion = "0.1"\n'
  )
  return subprocess.run(
      ['bash', str(_SCRIPT)],
      cwd=tmp_path,
      env={'PATH': path, 'HOME': str(tmp_path)},
      capture_output=True,
      text=True,
      check=False,
  )


def test_fails_fast_with_actionable_message_when_uv_missing(
    tmp_path: pathlib.Path,
) -> None:
  result = _run(tmp_path, '/usr/bin:/bin')

  output = result.stdout + result.stderr
  assert result.returncode == 1
  assert 'uv was not found on PATH' in output
  assert 'https://docs.astral.sh/uv/getting-started/installation/' in output
  # The pre-existing failure mode: a missing tool was reported as a phantom
  # dependency conflict, sending readers off to debug the wrong problem.
  assert 'Resolution failed' not in output
  assert not list(tmp_path.glob('constraints-*.txt'))


def test_regenerates_constraints_when_uv_available(
    tmp_path: pathlib.Path,
) -> None:
  stub_dir = tmp_path / 'bin'
  stub_dir.mkdir()
  stub = stub_dir / 'uv'
  stub.write_text(_UV_STUB)
  stub.chmod(0o755)

  result = _run(tmp_path, f'{stub_dir}:/usr/bin:/bin')

  output = result.stdout + result.stderr
  assert 'uv was not found on PATH' not in output
  for version in _PYTHON_VERSIONS:
    assert (tmp_path / f'constraints-{version}.txt').exists()
  # The script is a fixer, not a check: it rewrites missing or stale targets
  # and exits non-zero whenever it wrote. That is why it cannot serve as a CI
  # gate, and why the hook belongs on the `manual` stage.
  assert result.returncode == 1
