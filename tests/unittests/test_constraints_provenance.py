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

"""Guard tests for the provenance comments in the published constraints files.

``README.md`` tells users to download ``constraints-<ver>.txt`` from this
repository and pass it to ``pip install -c``, so each file is an end-user
artifact rather than a build intermediate.

``scripts/update_constraints.sh`` stabilizes a regeneration by passing a copy
of the committed file to ``uv`` as ``--constraint``. ``uv`` recorded that copy
as a source for every pin, so the published file named a scratch path that the
script's own cleanup trap had already deleted. The script now filters those
lines out of its output.

These tests read the real committed files. Test 1 fails if the scratch path
comes back. Test 2 fails if the filter ever strips the only source out of an
annotation block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# tests/unittests/<this file> -> the repository root. Plain path arithmetic,
# not ``.resolve()``, because the test tree may be symlinked.
_REPO_ROOT = Path(__file__).parent.parent.parent

# The interpreters in the PYTHON_VERSIONS array of
# scripts/update_constraints.sh, which is what decides how many files exist.
_PYTHON_VERSIONS = ('3.10', '3.11', '3.12', '3.13', '3.14')

# The suffix of the stabilization copy that scripts/update_constraints.sh
# feeds to uv. That copy lives only while the script runs.
_TEMPORARY_CONSTRAINT_SUFFIX = '.stable.tmp'

# uv opens an annotation block with this line, then indents one line per
# source under it.
_VIA_HEADER = '# via'
_VIA_SOURCE_PREFIX = '    #   '


def _read_lines(version: str) -> list[str]:
  """Returns the committed constraints file for ``version``, line by line."""
  path = _REPO_ROOT / f'constraints-{version}.txt'
  assert path.is_file(), (
      f'{path} is missing. README.md tells users to download it, and '
      'scripts/update_constraints.sh generates one file per interpreter in '
      'its PYTHON_VERSIONS array.'
  )
  return path.read_text(encoding='utf-8').splitlines()


def _via_blocks_without_a_source(lines: list[str]) -> list[str]:
  """Describes every ``# via`` block that no source line follows."""
  offenders: list[str] = []
  pin = '<the file header>'
  next_lines = [*lines[1:], '']
  for number, (line, next_line) in enumerate(zip(lines, next_lines), start=1):
    stripped = line.strip()
    if not stripped.startswith('#'):
      pin = stripped
    elif stripped == _VIA_HEADER and not next_line.startswith(
        _VIA_SOURCE_PREFIX
    ):
      offenders.append(f'line {number}, under {pin!r}')
  return offenders


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_no_temporary_constraint_file_is_referenced(version: str) -> None:
  """The published file never names the scratch file that generated it."""
  offenders = [
      (number, line)
      for number, line in enumerate(_read_lines(version), start=1)
      if _TEMPORARY_CONSTRAINT_SUFFIX in line
  ]

  assert not offenders, (
      f'constraints-{version}.txt names a {_TEMPORARY_CONSTRAINT_SUFFIX} '
      f'file on {len(offenders)} line(s), starting at line {offenders[0][0]}: '
      f'{offenders[0][1].strip()!r}. That file exists only while '
      'scripts/update_constraints.sh runs, so the reference is dead for '
      'every user who downloads this file. Regenerate with the current '
      'scripts/update_constraints.sh, which filters these lines out.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_every_via_block_lists_at_least_one_source(version: str) -> None:
  """Dropping the scratch file never empties an annotation block.

  A ``--constraint`` file restricts versions; it never pulls a package into
  the resolution. Every pin therefore keeps a real source. This test fails
  loudly if that ever stops holding, instead of publishing a malformed file.
  """
  offenders = _via_blocks_without_a_source(_read_lines(version))

  assert not offenders, (
      f'constraints-{version}.txt has {len(offenders)} "{_VIA_HEADER}" '
      f'block(s) with no source under them: {offenders}. The provenance '
      'filter in scripts/update_constraints.sh removed the only source of '
      'the block instead of a reference to the stabilization scratch file.'
  )
