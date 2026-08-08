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

"""Guards that every async test and async fixture declares its asyncio intent.

``pyproject.toml`` sets ``asyncio_mode = "auto"``, which makes pytest-asyncio
adopt async tests and async fixtures implicitly. pytest-asyncio's own default is
``strict``. Any consumer that runs this suite without our ``ini_options`` -- a
downstream packager, or plain ``pytest -o asyncio_mode=strict`` -- silently skips
or errors on anything that relies on auto mode. Explicit markers behave the same
in both modes, so this module fails the build when an unmarked one appears.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MAX_REPORTED = 20


def _dotted_name(node: ast.expr) -> str:
  """Flattens a decorator expression to a dotted name.

  Args:
    node: The decorator expression, which may be a call such as
      ``@pytest.fixture(params=[1])``.

  Returns:
    The dotted name, for example ``pytest.mark.asyncio``. Returns an empty
    string for an expression that is not a (possibly called) dotted name.
  """
  node = node.func if isinstance(node, ast.Call) else node
  parts: list[str] = []
  while isinstance(node, ast.Attribute):
    parts.append(node.attr)
    node = node.value
  if isinstance(node, ast.Name):
    parts.append(node.id)
  return '.'.join(reversed(parts))


def _is_asyncio_marker(name: str) -> bool:
  """Reports whether a dotted decorator name is the pytest-asyncio marker.

  Accepts both ``pytest.mark.asyncio`` and the ``from pytest import mark``
  spelling ``mark.asyncio``, both of which the suite uses.
  """
  segments = name.split('.')
  return segments[-1] == 'asyncio' and 'mark' in segments


def _assigns_asyncio_pytestmark(statement: ast.stmt) -> bool:
  """Reports whether a statement is a ``pytestmark`` holding the asyncio mark.

  Handles the single-value form and the list/tuple form, either of which pytest
  applies to every test in the enclosing module or class.
  """
  if not isinstance(statement, ast.Assign):
    return False
  if not any(
      isinstance(target, ast.Name) and target.id == 'pytestmark'
      for target in statement.targets
  ):
    return False
  value = statement.value
  marks = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
  return any(_is_asyncio_marker(_dotted_name(mark)) for mark in marks)


def _inherits_test_case(node: ast.ClassDef) -> bool:
  """Reports whether a class derives from a ``unittest`` ``TestCase``.

  ``IsolatedAsyncioTestCase`` awaits its own coroutines, so its async methods
  already run identically in both asyncio modes and need no marker.
  """
  return any(
      _dotted_name(base).split('.')[-1].endswith('TestCase')
      for base in node.bases
  )


def collect_violations(source: str, path: str) -> tuple[list[str], list[str]]:
  """Finds async tests and async fixtures that do not declare asyncio intent.

  Args:
    source: The Python source text of a test module.
    path: The path reported in each violation, for the failure message.

  Returns:
    A tuple of (unmarked async tests, async fixtures not declared with
    ``pytest_asyncio.fixture``). Each entry reads ``path:lineno name``.

  Raises:
    SyntaxError: If ``source`` does not parse. A broken test file is a real
      problem, but not this guard's problem, so the error surfaces as-is.
  """
  tests: list[str] = []
  fixtures: list[str] = []

  def visit(body: list[ast.stmt], marked: bool, in_test_case: bool) -> None:
    marked = marked or any(_assigns_asyncio_pytestmark(stmt) for stmt in body)
    for statement in body:
      if isinstance(statement, ast.ClassDef):
        visit(
            statement.body,
            marked
            or any(
                _is_asyncio_marker(_dotted_name(decorator))
                for decorator in statement.decorator_list
            ),
            in_test_case or _inherits_test_case(statement),
        )
      elif isinstance(statement, ast.AsyncFunctionDef):
        names = [_dotted_name(d) for d in statement.decorator_list]
        location = f'{path}:{statement.lineno} {statement.name}'
        if any(name.split('.')[-1] == 'fixture' for name in names):
          if not any(name.startswith('pytest_asyncio.') for name in names):
            fixtures.append(location)
        elif statement.name.startswith('test') and not in_test_case:
          if not marked and not any(_is_asyncio_marker(n) for n in names):
            tests.append(location)

  visit(ast.parse(source, filename=path).body, False, False)
  return tests, fixtures


def _format_failure(tests: list[str], fixtures: list[str]) -> str:
  """Builds an actionable failure message listing the offending sites."""
  lines = [f'{t}  -> add @pytest.mark.asyncio' for t in tests]
  lines += [f'{f}  -> use @pytest_asyncio.fixture' for f in fixtures]
  hidden = len(lines) - _MAX_REPORTED
  reported = lines[:_MAX_REPORTED]
  if hidden > 0:
    reported.append(f'... and {hidden} more')
  return (
      'Without an explicit marker these are skipped or error out when'
      ' pytest-asyncio runs in its default strict mode.\n'
      + '\n'.join(reported)
  )


def _scan_directory(root: Path) -> tuple[list[str], list[str]]:
  """Collects the violations in every Python file under a directory.

  Only the files that are present are read, so a partial copy of the tree
  reports on what it can see instead of failing on a missing path.
  """
  tests: list[str] = []
  fixtures: list[str] = []
  for path in sorted(root.rglob('*.py')):
    file_tests, file_fixtures = collect_violations(
        path.read_text(encoding='utf-8'),
        str(path.relative_to(root.parent.parent)),
    )
    tests += file_tests
    fixtures += file_fixtures
  return tests, fixtures


def test_no_unmarked_async_tests_or_fixtures() -> None:
  """Every async test and async fixture under this directory is explicit."""
  tests, fixtures = _scan_directory(Path(__file__).parent)
  assert not tests and not fixtures, _format_failure(tests, fixtures)


_MARKED_FUNCTION = """
import pytest

@pytest.mark.asyncio
async def test_x():
  pass
"""

_MARKED_VIA_MARK_IMPORT = """
from pytest import mark

@mark.asyncio
async def test_x():
  pass
"""

_UNMARKED_FUNCTION = """
async def test_x():
  pass
"""

_MARKED_CLASS = """
import pytest

@pytest.mark.asyncio
class TestThing:

  async def test_x(self):
    pass
"""

_MODULE_PYTESTMARK = """
import pytest

pytestmark = pytest.mark.asyncio

async def test_x():
  pass
"""

_MODULE_PYTESTMARK_LIST = """
import pytest

pytestmark = [pytest.mark.asyncio, pytest.mark.slow]

async def test_x():
  pass
"""

_CLASS_PYTESTMARK = """
import pytest

class TestThing:
  pytestmark = pytest.mark.asyncio

  async def test_x(self):
    pass
"""

_DOTTED_TEST_CASE = """
import unittest

class TestThing(unittest.IsolatedAsyncioTestCase):

  async def test_x(self):
    pass
"""

_BARE_TEST_CASE = """
from unittest import IsolatedAsyncioTestCase

class TestThing(IsolatedAsyncioTestCase):

  async def test_x(self):
    pass
"""

_PLAIN_FIXTURE = """
import pytest

@pytest.fixture
async def env():
  yield 1
"""

_PLAIN_FIXTURE_CALL_FORM = """
import pytest

@pytest.fixture(params=[1])
async def env(request):
  yield request.param
"""

_ASYNCIO_FIXTURE = """
import pytest_asyncio

@pytest_asyncio.fixture(name='env')
async def env_fixture():
  yield 1
"""

_PARAMETRIZE_ONLY = """
import pytest

@pytest.mark.parametrize('value', [1])
async def test_x(value):
  pass
"""

_EXOTIC_DECORATOR = """
import pytest

@_DECORATORS['slow']
async def test_x():
  pass
"""

_ASYNC_HELPER = """
async def helper():
  pass
"""

_SYNC_TEST = """
def test_x():
  pass
"""


@pytest.mark.parametrize(
    'source, expected_tests, expected_fixtures',
    [
        (_MARKED_FUNCTION, [], []),
        (_MARKED_VIA_MARK_IMPORT, [], []),
        (_UNMARKED_FUNCTION, ['m.py:2 test_x'], []),
        (_MARKED_CLASS, [], []),
        (_MODULE_PYTESTMARK, [], []),
        (_MODULE_PYTESTMARK_LIST, [], []),
        (_CLASS_PYTESTMARK, [], []),
        (_DOTTED_TEST_CASE, [], []),
        (_BARE_TEST_CASE, [], []),
        (_PLAIN_FIXTURE, [], ['m.py:5 env']),
        (_PLAIN_FIXTURE_CALL_FORM, [], ['m.py:5 env']),
        (_ASYNCIO_FIXTURE, [], []),
        (_PARAMETRIZE_ONLY, ['m.py:5 test_x'], []),
        (_EXOTIC_DECORATOR, ['m.py:5 test_x'], []),
        (_ASYNC_HELPER, [], []),
        (_SYNC_TEST, [], []),
    ],
)
def test_collect_violations(
    source: str, expected_tests: list[str], expected_fixtures: list[str]
) -> None:
  assert collect_violations(source, 'm.py') == (
      expected_tests,
      expected_fixtures,
  )


def test_scan_directory_checks_only_the_files_that_are_present(
    tmp_path: Path,
) -> None:
  root = tmp_path / 'unittests'
  (root / 'sub').mkdir(parents=True)
  (root / 'sub' / 'test_clean.py').write_text(_MARKED_FUNCTION)
  assert _scan_directory(root) == ([], [])

  (root / 'test_dirty.py').write_text(_UNMARKED_FUNCTION)
  assert _scan_directory(root) == (
      [f'{tmp_path.name}/unittests/test_dirty.py:2 test_x'],
      [],
  )


def test_collect_violations_propagates_syntax_error() -> None:
  with pytest.raises(SyntaxError) as excinfo:
    collect_violations('async def (:', 'broken.py')
  assert excinfo.value.filename == 'broken.py'


def test_format_failure_reports_both_kinds_with_the_fix() -> None:
  message = _format_failure(['a.py:1 test_x'], ['a.py:9 env'])
  assert 'strict mode' in message
  assert 'a.py:1 test_x  -> add @pytest.mark.asyncio' in message
  assert 'a.py:9 env  -> use @pytest_asyncio.fixture' in message


def test_format_failure_truncates_a_long_offender_list() -> None:
  message = _format_failure([f'a.py:{i} test_x' for i in range(25)], [])
  assert '... and 5 more' in message
  assert 'a.py:19 test_x' in message
  assert 'a.py:20 test_x' not in message
