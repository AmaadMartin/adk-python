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

"""Tests for the addlicense pre-commit hook and its wrapper script."""

from __future__ import annotations

import os
import pathlib
import subprocess

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / 'scripts' / 'addlicense.sh'
_CONFIG = _REPO_ROOT / '.pre-commit-config.yaml'

_MISSING_BINARY_WARNING = 'Warning: addlicense not installed, skipping'


def _write_stub(directory: pathlib.Path, body: str) -> None:
  """Writes an executable `addlicense` stub into `directory`."""
  stub = directory / 'addlicense'
  stub.write_text(f'#!/bin/bash\n{body}\n')
  stub.chmod(0o755)


def _run(
    fake_bin: pathlib.Path, *args: str
) -> subprocess.CompletedProcess[str]:
  """Runs the wrapper with `fake_bin` as the whole PATH."""
  return subprocess.run(
      [str(_SCRIPT), *args],
      cwd=_REPO_ROOT,
      env={'PATH': str(fake_bin)},
      capture_output=True,
      text=True,
      check=False,
  )


def _addlicense_hook() -> dict[str, object]:
  """Returns the `addlicense` hook mapping from the pre-commit config."""
  config = yaml.safe_load(_CONFIG.read_text())
  hooks = [
      hook
      for repo in config['repos']
      for hook in repo['hooks']
      if hook['id'] == 'addlicense'
  ]
  assert len(hooks) == 1
  return hooks[0]


def test_warns_and_exits_zero_when_addlicense_missing(
    tmp_path: pathlib.Path,
) -> None:
  result = _run(tmp_path, 'a.py')

  assert result.returncode == 0
  assert _MISSING_BINARY_WARNING in result.stdout


def test_forwards_expected_arguments_to_addlicense(
    tmp_path: pathlib.Path,
) -> None:
  argv_dump = tmp_path / 'argv.txt'
  _write_stub(tmp_path, f'printf "%s\\n" "$@" > {argv_dump}')

  result = _run(tmp_path, 'a.py', 'b.sh')

  assert result.returncode == 0
  assert _MISSING_BINARY_WARNING not in result.stdout
  assert argv_dump.read_text().splitlines() == [
      '-c',
      'Google LLC',
      '-l',
      'apache',
      'a.py',
      'b.sh',
  ]


def test_propagates_addlicense_exit_code(tmp_path: pathlib.Path) -> None:
  _write_stub(tmp_path, 'exit 3')

  assert _run(tmp_path, 'a.py').returncode == 3


def test_hook_entry_points_at_the_script() -> None:
  hook = _addlicense_hook()

  assert hook['name'] == 'addlicense'
  assert hook['entry'] == 'scripts/addlicense.sh'
  assert hook['language'] == 'script'
  assert hook['files'] == r'\.(py|sh)$'
  # Filenames must keep reaching the script, so the default must stand.
  assert 'pass_filenames' not in hook


def test_script_is_executable() -> None:
  assert os.access(_SCRIPT, os.X_OK)


def test_script_has_apache_header() -> None:
  lines = _SCRIPT.read_text().splitlines()

  assert lines[0] == '#!/bin/bash'
  # The hook matches its own wrapper, so the header must already be in place
  # or addlicense rewrites the file on every run.
  header = '\n'.join(lines[:20])
  assert 'Copyright' in header
  assert 'Licensed under the Apache License, Version 2.0' in header
