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

"""Guard tests for the published ``constraints-<ver>.txt`` files.

README.md tells end users to download these files and pass them to
``pip install google-adk -c constraints-<ver>.txt``. Each file is ~1900 lines
of generated pins, so the properties that make the download worth anything are
invisible in review. These tests pin those properties instead:

* The resolution is **universal**. Without ``--universal`` uv resolves for the
  build machine only, so a macOS or Windows user gets no pin at all for the
  transitive dependencies that exist only on their platform.
* The resolution is **runtime-only**. The files protect ``pip install
  google-adk``, so the contributor-only ``dev``, ``docs`` and ``test`` extras
  do not belong in them.
* The extras list has **not drifted** from ``pyproject.toml``. A new extra that
  nobody adds to ``RUNTIME_EXTRAS`` in ``scripts/update_constraints.sh`` would
  otherwise go unpinned in silence.
* Every duplicated pin is **marker-scoped**. A universal resolution may emit
  one distribution several times; pip ANDs two unconditional pins of the same
  name into an unsatisfiable constraint, so the markers are what keep the file
  usable.

The tests read the committed files and ``pyproject.toml`` only. They run no
subprocess, invoke no resolver and touch no network.
"""

from __future__ import annotations

from pathlib import Path
import re

try:
  import tomllib
except ImportError:
  import tomli as tomllib

from packaging.utils import canonicalize_name
import pytest

_PYTHON_VERSIONS = ('3.10', '3.11', '3.12', '3.13', '3.14')

# Extras that exist for contributors, not for users of the published wheel.
_CONTRIBUTOR_EXTRAS = frozenset({'dev', 'docs', 'test'})

_GENERATOR_SCRIPT = 'scripts/update_constraints.sh'

# Distributions reachable only through the contributor extras. Pinning any of
# them means the generator regressed to --all-extras.
_CONTRIBUTOR_TOOLING = frozenset({
    'astroid',
    'codespell',
    'flit',
    'flit-core',
    'furo',
    'isort',
    'mdformat',
    'mypy',
    'pip',
    'pre-commit',
    'pyink',
    'pylint',
    'pyproject-fmt',
    'pytest',
    'pytest-asyncio',
    'pytest-mock',
    'pytest-xdist',
    'myst-parser',
    'ruff',
    'sphinx',
    'tox',
    'virtualenv',
})

# "name==version" optionally followed by "; <PEP 508 marker>". Requirement
# lines start at column 0; provenance comments are indented.
_PIN_PATTERN = re.compile(
    r'^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)'
    r'==(?P<version>[^\s;]+)'
    r'(?:\s*;\s*(?P<marker>.+))?$'
)

_EXTRA_FLAG_PATTERN = re.compile(
    r'--extra\s+(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)'
)


def _find_repo_root() -> Path:
  """Locates the directory holding pyproject.toml and the constraints files.

  Walks up from this file's directory. The test tree may be symlinked, so the
  walk avoids ``.resolve()``.
  """
  start = Path(__file__).parent
  for candidate in [start, *start.parents]:
    if (candidate / 'pyproject.toml').is_file() and all(
        (candidate / f'constraints-{ver}.txt').is_file()
        for ver in _PYTHON_VERSIONS
    ):
      return candidate
  raise FileNotFoundError(
      'Could not find a directory containing pyproject.toml and every '
      f'constraints-<ver>.txt walking up from {start}.'
  )


_REPO_ROOT = _find_repo_root()


@pytest.fixture(scope='module')
def pyproject() -> dict:
  """Parses the project's pyproject.toml exactly once for the module."""
  with (_REPO_ROOT / 'pyproject.toml').open('rb') as fh:
    return tomllib.load(fh)


def _constraints_path(version: str) -> Path:
  """Returns the path of the committed constraints file for ``version``."""
  return _REPO_ROOT / f'constraints-{version}.txt'


def _read_generation_command(version: str) -> str:
  """Returns the generation command recorded on line 2 of the file header.

  ``scripts/update_constraints.sh`` rewrites this line itself, so it is the
  authoritative record of how the file was produced.
  """
  lines = _constraints_path(version).read_text(encoding='utf-8').splitlines()
  assert len(lines) >= 2, (
      f'constraints-{version}.txt is too short to carry the two-line header '
      f'that {_GENERATOR_SCRIPT} writes.'
  )
  header = lines[1]
  assert header.startswith('#    uv pip compile'), (
      f'Line 2 of constraints-{version}.txt must be the generation command '
      f'written by {_GENERATOR_SCRIPT}, but it is: {header!r}'
  )
  return header


def _parse_pins(version: str) -> list[tuple[str, str, str | None]]:
  """Returns the ``(name, version, marker)`` triples pinned by a file.

  Comment lines, indented provenance annotations and blank lines are skipped.
  ``marker`` is ``None`` for a pin that applies to every environment.
  """
  pins: list[tuple[str, str, str | None]] = []
  for line in (
      _constraints_path(version).read_text(encoding='utf-8').splitlines()
  ):
    if not line or line.startswith(('#', ' ')):
      continue
    match = _PIN_PATTERN.match(line)
    assert (
        match is not None
    ), f'Unparsable requirement line in constraints-{version}.txt: {line!r}'
    pins.append((match['name'], match['version'], match['marker']))
  return pins


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_generation_command_is_universal(version: str) -> None:
  """The header records ``--universal``, so the pins cover every platform."""
  command = _read_generation_command(version)

  assert '--universal' in command, (
      f'constraints-{version}.txt was resolved without --universal, so its '
      'pins only describe the machine that generated it. Users on other '
      'platforms get no pin for their platform-only transitive dependencies. '
      f'Restore --universal in {_GENERATOR_SCRIPT}.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_generation_command_does_not_use_all_extras(version: str) -> None:
  """``--all-extras`` would drag the contributor extras back in."""
  command = _read_generation_command(version)

  assert '--all-extras' not in command, (
      f'constraints-{version}.txt was resolved with --all-extras, which pins '
      'the dev, docs and test tooling in a file that exists to protect a '
      f'runtime install. Use the RUNTIME_EXTRAS list in {_GENERATOR_SCRIPT}.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_generation_command_covers_every_runtime_extra(
    pyproject: dict, version: str
) -> None:
  """The recorded ``--extra`` flags match pyproject minus the dev extras.

  This is what turns "someone added an extra and forgot RUNTIME_EXTRAS" from a
  silent gap in the published pins into a failing test.
  """
  command = _read_generation_command(version)
  recorded = {m['name'] for m in _EXTRA_FLAG_PATTERN.finditer(command)}
  expected = (
      set(pyproject['project']['optional-dependencies']) - _CONTRIBUTOR_EXTRAS
  )

  missing = sorted(expected - recorded)
  unexpected = sorted(recorded - expected)
  assert not missing and not unexpected, (
      f'The extras baked into constraints-{version}.txt have drifted from '
      f'pyproject.toml. Missing: {missing or "none"}. Not declared in '
      f'pyproject.toml: {unexpected or "none"}. Update RUNTIME_EXTRAS in '
      f'{_GENERATOR_SCRIPT} and regenerate.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_generation_command_matches_file_name(version: str) -> None:
  """The header's ``--python-version`` agrees with the file name."""
  command = _read_generation_command(version)

  assert f'--python-version {version}' in command, (
      f'constraints-{version}.txt records a --python-version that does not '
      f'match its name: {command!r}'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_pins_carry_platform_markers(version: str) -> None:
  """At least one pin is scoped to an operating system.

  A resolution done for a single platform emits no markers at all, so this
  fails outright if someone regenerates without ``--universal``.
  """
  platform_scoped = [
      name
      for name, _, marker in _parse_pins(version)
      if marker and 'sys_platform' in marker
  ]

  assert platform_scoped, (
      f'No pin in constraints-{version}.txt carries a sys_platform marker, '
      'so the file cannot describe more than one operating system. '
      f'Regenerate it with {_GENERATOR_SCRIPT}.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_no_contributor_tooling_is_pinned(version: str) -> None:
  """None of the dev, docs or test tooling reaches the published file."""
  pinned = {canonicalize_name(name) for name, _, _ in _parse_pins(version)}
  leaked = sorted(pinned & _CONTRIBUTOR_TOOLING)

  assert not leaked, (
      f'constraints-{version}.txt pins contributor-only tooling: {leaked}. '
      'A user running "pip install google-adk -c constraints-<ver>.txt" never '
      f'installs these. Check the RUNTIME_EXTRAS list in {_GENERATOR_SCRIPT}.'
  )


@pytest.mark.parametrize('version', _PYTHON_VERSIONS)
def test_duplicate_pins_are_marker_scoped(version: str) -> None:
  """Every distribution pinned more than once is scoped by a marker.

  pip skips a constraint whose marker does not match the current environment,
  so disjoint markers are safe. Two unconditional pins of one name are ANDed
  into a constraint no version can satisfy.
  """
  by_name: dict[str, list[tuple[str, str | None]]] = {}
  for name, pinned_version, marker in _parse_pins(version):
    by_name.setdefault(canonicalize_name(name), []).append(
        (pinned_version, marker)
    )

  unscoped = sorted(
      name
      for name, entries in by_name.items()
      if len(entries) > 1 and any(marker is None for _, marker in entries)
  )

  assert not unscoped, (
      f'constraints-{version}.txt pins {unscoped} more than once with at '
      'least one unconditional line. pip ANDs those pins together and the '
      'install fails. Every duplicated pin must carry an environment marker.'
  )
