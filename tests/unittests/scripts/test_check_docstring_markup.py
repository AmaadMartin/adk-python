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

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
import pathlib
import subprocess
import sys
import tempfile
import textwrap

import pytest
import sphinx.application
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.docutils import docutils_namespace

from scripts import check_docstring_markup as checker

# Fragments of the docutils messages this check exists to surface. Every one
# was observed from the real pipeline.
_STRONG = 'Inline strong start-string without end-string.'
_LITERAL = 'Inline literal start-string without end-string.'
_DEFINITION_LIST = (
    'Definition list ends without a blank line; unexpected unindent.'
)

_BROKEN_DOCSTRING = 'Accepts **kwargs and forwards them.'


def _write(path: pathlib.Path, source: str) -> pathlib.Path:
  """Writes dedented source to a file, creating parent directories."""
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(textwrap.dedent(source), encoding='utf-8')
  return path


# --- extraction -------------------------------------------------------------


def test_iter_docstrings_yields_every_kind_of_definition() -> None:
  source = '''\
      """Module doc."""


      class Outer:
        """Outer doc."""

        def method(self):
          """Method doc."""

        class Inner:
          """Inner doc."""


      async def coroutine():
        """Coroutine doc."""


      def plain():
        """Plain doc."""
      '''
  found = {
      symbol: docstring
      for symbol, _, docstring in checker.iter_docstrings(
          textwrap.dedent(source)
      )
  }
  assert found == {
      '<module>': 'Module doc.',
      'Outer': 'Outer doc.',
      'Outer.method': 'Method doc.',
      'Outer.Inner': 'Inner doc.',
      'coroutine': 'Coroutine doc.',
      'plain': 'Plain doc.',
  }


def test_iter_docstrings_reports_the_owner_line_not_the_string_line() -> None:
  source = '"""Module doc."""\n\n\nclass Klass:\n  """Class doc."""\n'
  linenos = {
      symbol: lineno for symbol, lineno, _ in checker.iter_docstrings(source)
  }
  assert linenos == {'<module>': 1, 'Klass': 4}


def test_iter_docstrings_yields_pydantic_attribute_docstrings() -> None:
  source = '''\
      class Model:
        annotated: int = 1
        """Annotated doc."""

        plain = 2
        """Plain doc."""
      '''
  found = {
      symbol: docstring
      for symbol, _, docstring in checker.iter_docstrings(
          textwrap.dedent(source)
      )
  }
  assert found == {
      'Model.annotated': 'Annotated doc.',
      'Model.plain': 'Plain doc.',
  }


def test_iter_docstrings_yields_module_level_attribute_docstrings() -> None:
  source = 'CONSTANT = 1\n"""Constant doc."""\n'
  assert list(checker.iter_docstrings(source)) == [
      ('CONSTANT', 2, 'Constant doc.')
  ]


def test_iter_docstrings_yields_a_field_declared_without_a_default() -> None:
  # A Pydantic field with no default is still a documented field.
  source = 'class Model:\n  field: int\n  """Field doc."""\n'
  assert list(checker.iter_docstrings(source)) == [
      ('Model.field', 3, 'Field doc.')
  ]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param(
            '"""Orphan string."""\n\n"""Second string."""\n', id='no_assignment'
        ),
        pytest.param('a = b = 1\n"""Two targets."""\n', id='two_targets'),
        pytest.param(
            'obj.attr = 1\n"""Attribute target."""\n', id='not_a_name'
        ),
        pytest.param('x += 1\n"""Augmented assign."""\n', id='augmented'),
        pytest.param('import os\n"""After an import."""\n', id='not_an_assign'),
    ],
)
def test_iter_docstrings_skips_strings_that_document_nothing(
    source: str,
) -> None:
  # The first string of a module is its module docstring, so only the
  # trailing string of each case is under test.
  assert [
      symbol
      for symbol, _, _ in checker.iter_docstrings(source)
      if symbol != '<module>'
  ] == []


def test_iter_docstrings_skips_annotated_attribute_without_a_name_target() -> (
    None
):
  source = 'class Model:\n  obj.attr: int = 1\n  """Doc."""\n'
  assert [symbol for symbol, _, _ in checker.iter_docstrings(source)] == []


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('""""""\n', id='empty_module'),
        pytest.param('"""   \n\n  """\n', id='whitespace_module'),
        pytest.param('def f():\n  """  """\n', id='whitespace_function'),
        pytest.param('X = 1\n"""  """\n', id='whitespace_attribute'),
    ],
)
def test_iter_docstrings_skips_empty_docstrings(source: str) -> None:
  assert list(checker.iter_docstrings(source)) == []


def test_iter_docstrings_ignores_statements_that_are_not_definitions() -> None:
  source = 'import os\n\nif os.name:\n  pass\n\ndef f():\n  """Doc."""\n'
  assert [symbol for symbol, _, _ in checker.iter_docstrings(source)] == ['f']


# --- detection: must report -------------------------------------------------


def test_reports_unterminated_inline_strong() -> None:
  assert checker.check_docstring(_BROKEN_DOCSTRING) == [_STRONG]


def test_reports_unterminated_inline_literal() -> None:
  assert checker.check_docstring('Set ``mode to "auto".') == [_LITERAL]


def test_reports_definition_list_that_does_not_end_with_a_blank_line() -> None:
  docstring = 'Summary.\n\nterm\n  definition\nback at the margin.\n'
  assert _DEFINITION_LIST in checker.check_docstring(docstring)


@pytest.mark.parametrize(
    'docstring',
    [
        pytest.param('Doc.\n\n```python\nx = 1\n```\n', id='column_zero'),
        pytest.param(
            'Doc.\n\nArgs:\n  x: A thing.\n\n    ```python\n    x = 1\n   '
            ' ```\n',
            id='indented',
        ),
    ],
)
def test_reports_markdown_code_fence(docstring: str) -> None:
  # docutils accepts a fence silently, so this check is the only thing that
  # catches one.
  assert checker.check_docstring(docstring) == [checker._FENCE_MESSAGE]


def test_reports_a_misspelled_directive() -> None:
  # Proves the stand-ins are a finite list rather than a blanket accept.
  assert checker.check_docstring(
      'Doc.\n\n.. code-blok:: python\n\n   x = 1\n'
  ) == ['Unknown directive type "code-blok".']


def test_reports_an_unknown_role() -> None:
  assert checker.check_docstring('See :nosuchrole:`x`.') == [
      'Unknown interpreted text role "nosuchrole".'
  ]


def test_reports_broken_markup_in_a_prose_directive_argument() -> None:
  # Sphinx puts the replacement text of a deprecation in the argument.
  docstring = 'Doc.\n\n.. deprecated:: 2.0\n   Accepts **kwargs now.\n'
  assert checker.check_docstring(docstring) == [_STRONG]


def test_reports_broken_markup_in_a_prose_directive_body() -> None:
  docstring = 'Doc.\n\n.. seealso::\n\n   Accepts **kwargs now.\n'
  assert checker.check_docstring(docstring) == [_STRONG]


def test_reports_a_code_block_missing_the_blank_line_before_its_body() -> None:
  docstring = 'Doc.\n\n.. code-block:: python\n   x = 1\n'
  assert checker.check_docstring(docstring) == [
      'Error in "code-block" directive: maximum 1 argument(s) allowed, 4'
      ' supplied.'
  ]


# --- detection: must stay silent --------------------------------------------


def test_accepts_a_clean_google_style_docstring() -> None:
  docstring = """\
      Does one thing.

      Args:
        value: The value to use.

      Returns:
        The result.

      Raises:
        ValueError: If the value is negative.
      """
  assert checker.check_docstring(textwrap.dedent(docstring)) == []


def test_accepts_a_clean_numpy_style_docstring() -> None:
  docstring = """\
      Does one thing.

      Parameters
      ----------
      value : int
          The value to use.

      Returns
      -------
      int
          The result.
      """
  assert checker.check_docstring(textwrap.dedent(docstring)) == []


@pytest.mark.parametrize('role', sorted(checker._ROLES))
@pytest.mark.parametrize('prefix', ['', 'py:'])
def test_accepts_sphinx_cross_reference_roles(role: str, prefix: str) -> None:
  assert checker.check_docstring(f'See :{prefix}{role}:`Target`.') == []


@pytest.mark.parametrize('directive', sorted(checker._PROSE_DIRECTIVES))
def test_accepts_prose_directives(directive: str) -> None:
  docstring = f'Doc.\n\n.. {directive}:: name\n\n   Some prose.\n'
  assert checker.check_docstring(docstring) == []


@pytest.mark.parametrize('directive', sorted(checker._LITERAL_DIRECTIVES))
def test_accepts_literal_directives(directive: str) -> None:
  docstring = f'Doc.\n\n.. {directive}:: python\n\n   x = 1\n'
  assert checker.check_docstring(docstring) == []


def test_accepts_the_attribute_directive_napoleon_emits() -> None:
  # Napoleon turns an `Attributes:` section into `.. attribute::`. Hundreds of
  # docstrings in the tree rely on this being registered.
  docstring = 'Doc.\n\nAttributes:\n  foo (int): The foo.\n  bar: The bar.\n'
  assert '.. attribute:: foo' in checker.docstring_to_rst(docstring)
  assert checker.check_docstring(docstring) == []


def test_a_code_block_is_handled_by_the_stand_in_not_by_docutils() -> None:
  # The fence message sends contributors to `.. code-block::`. docutils aliases
  # that name to its own `code` directive, which runs Pygments and rejects any
  # lexer it does not know, while Sphinx accepts it. The stand-in keeps a lexer
  # name from being reported as a markup error.
  assert '.. code-block::' in checker._FENCE_MESSAGE
  docstring = 'Doc.\n\n.. code-block:: notalanguage\n\n   x = 1\n'
  assert checker.check_docstring(docstring) == []


@pytest.mark.parametrize(
    ('docstring', 'message'),
    [
        pytest.param(
            'Doc.\n\n.. include:: /etc/passwd\n',
            '"include" directive disabled.',
            id='include',
        ),
        pytest.param(
            'Doc.\n\n.. raw:: html\n\n   <b>x</b>\n',
            '"raw" directive disabled.',
            id='raw',
        ),
    ],
)
def test_a_docstring_cannot_make_the_check_read_a_file(
    docstring: str, message: str
) -> None:
  # `file_insertion_enabled` and `raw_enabled` are off, so a hostile docstring
  # is reported rather than obeyed.
  assert checker.check_docstring(docstring) == [message]


def test_a_code_block_body_is_not_parsed_as_prose() -> None:
  # `x = a ** b` is invalid inline markup but perfectly good Python.
  docstring = 'Doc.\n\n.. code-block:: python\n\n   x = a ** b\n   s = "``"\n'
  assert checker.check_docstring(docstring) == []


def test_informational_messages_are_not_reported() -> None:
  # Looking the misspelled name up emits a level 1 note as well as the level 3
  # error. Only the error is a violation.
  messages = checker.check_docstring('Doc.\n\n.. code-blok:: python\n\n   x\n')
  assert not [m for m in messages if 'as canonical directive name' in m]
  assert len(messages) == 1


def test_docstring_to_rst_applies_the_napoleon_transformation() -> None:
  rst = checker.docstring_to_rst('Doc.\n\nArgs:\n  value: The value.\n')
  assert ':param value: The value.' in rst


# --- parity with the real docs build ----------------------------------------


@pytest.fixture(scope='module')
def napoleon_transform() -> Iterator[Callable[[str], str]]:
  """Napoleon's own transformation, driven by a real Sphinx application.

  autodoc hands a docstring to Napoleon through the `autodoc-process-docstring`
  event, so emitting that event on an application configured the way the API
  reference build configures it produces exactly the reStructuredText the docs
  build parses.
  """
  with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / 'conf.py').write_text(
        "extensions = ['sphinx.ext.napoleon']\n", encoding='utf-8'
    )
    (root / 'index.rst').write_text('Title\n=====\n', encoding='utf-8')
    # Building an application registers Sphinx's own directives and roles into
    # the process-wide docutils registry, which displaces the stand-ins the
    # checker installs. `docutils_namespace` puts the registry back. The
    # application stays usable afterwards because the event below only
    # rewrites text.
    with docutils_namespace():
      app = sphinx.application.Sphinx(
          srcdir=str(root),
          confdir=str(root),
          outdir=str(root / 'out'),
          doctreedir=str(root / 'doctrees'),
          buildername='html',
          status=None,
          warning=None,
      )

    def transform(docstring: str) -> str:
      lines = list(prepare_docstring(docstring))
      app.events.emit(
          'autodoc-process-docstring', 'class', 'X', None, None, lines
      )
      return '\n'.join(lines)

    yield transform


@pytest.mark.parametrize(
    'docstring',
    [
        pytest.param('Summary.\n\nNote:\n  First.\n\n  Second.\n', id='note'),
        pytest.param(
            'Summary.\n\nArgs:\n  x: The x.\n\nReturns:\n  The y.\n',
            id='google',
        ),
        pytest.param(
            'Summary.\n\nParameters\n----------\nx : int\n    The x.\n',
            id='numpy',
        ),
        pytest.param(
            'Summary.\n\nAttributes:\n  foo (int): The foo.\n', id='attributes'
        ),
        pytest.param(
            'Summary.\n\nExample:\n  >>> f()\n\nYields:\n  Each item.\n',
            id='example_and_yields',
        ),
    ],
)
def test_docstring_to_rst_matches_what_sphinx_hands_to_docutils(
    docstring: str, napoleon_transform: Callable[[str], str]
) -> None:
  # The `note` case pins the order: Napoleon runs NumPy style before Google
  # style, and the two orders indent the blank line inside the generated
  # `.. note::` differently.
  assert checker.docstring_to_rst(docstring) == napoleon_transform(docstring)


# --- plumbing ---------------------------------------------------------------


def test_check_source_locates_a_violation_in_a_method() -> None:
  source = '''\
      """Module doc."""


      class Klass:
        """Class doc."""

        def method(self):
          """Accepts **kwargs and forwards them."""
      '''
  violations = checker.check_source(textwrap.dedent(source), 'pkg/mod.py')
  assert violations == [
      checker.Violation(
          path='pkg/mod.py',
          lineno=7,
          symbol='Klass.method',
          message=_STRONG,
      )
  ]
  assert str(violations[0]) == f'pkg/mod.py:7 Klass.method: {_STRONG}'


def test_load_allowlist_ignores_comments_and_blank_lines(
    tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'allowlist.txt'
  path.write_text(
      '# a header\n\n  pkg/a.py  \npkg/b.py  # trailing note\n\n',
      encoding='utf-8',
  )
  assert checker.load_allowlist(path) == {'pkg/a.py', 'pkg/b.py'}


def test_load_allowlist_reports_a_missing_file_as_a_harness_failure(
    tmp_path: pathlib.Path,
) -> None:
  missing = tmp_path / 'absent.txt'
  with pytest.raises(checker.HarnessError, match='cannot read the allowlist'):
    checker.load_allowlist(missing)


def test_write_allowlist_sorts_the_entries_under_the_header(
    tmp_path: pathlib.Path,
) -> None:
  path = tmp_path / 'allowlist.txt'
  checker.write_allowlist(path, ['pkg/b.py', 'pkg/a.py'])
  text = path.read_text(encoding='utf-8')
  assert text.startswith('# Files whose docstrings still contain')
  assert text.endswith('pkg/a.py\npkg/b.py\n')
  assert checker.load_allowlist(path) == {'pkg/a.py', 'pkg/b.py'}


def test_report_base_is_the_repository_root_for_a_path_inside_it() -> None:
  assert checker.report_base(checker._SRC_ROOT) == checker._REPO_ROOT


def test_report_base_is_the_scanned_directory_for_a_path_outside_it(
    tmp_path: pathlib.Path,
) -> None:
  assert checker.report_base(tmp_path) == tmp_path.resolve()


def test_display_path_shortens_a_path_inside_the_repository() -> None:
  assert (
      checker.display_path(checker._ALLOWLIST)
      == 'scripts/docstring_markup_allowlist.txt'
  )


def test_display_path_leaves_a_path_outside_the_repository_alone(
    tmp_path: pathlib.Path,
) -> None:
  outside = tmp_path / 'allowlist.txt'
  assert checker.display_path(outside) == str(outside)


def test_check_tree_reports_an_unreadable_file_as_a_harness_failure(
    tmp_path: pathlib.Path,
) -> None:
  (tmp_path / 'binary.py').write_bytes(b'\xff\xfe not utf-8')
  with pytest.raises(checker.HarnessError, match='cannot read'):
    checker.check_tree(tmp_path)


def test_check_tree_reports_an_unparsable_file_as_a_harness_failure(
    tmp_path: pathlib.Path,
) -> None:
  _write(tmp_path / 'broken.py', 'def (:\n')
  with pytest.raises(checker.HarnessError, match='cannot parse'):
    checker.check_tree(tmp_path)


# --- command line -----------------------------------------------------------


@pytest.fixture
def tree(tmp_path: pathlib.Path) -> pathlib.Path:
  """A scratch source tree with one clean and one broken module."""
  root = tmp_path / 'src'
  _write(root / 'clean.py', '"""A clean docstring."""\n')
  _write(root / 'dirty.py', f'"""{_BROKEN_DOCSTRING}"""\n')
  return root


def _run(tree: pathlib.Path, allowlist: pathlib.Path, *flags: str) -> int:
  return checker.main(
      ['--root', str(tree), '--allowlist', str(allowlist), *flags]
  )


def test_main_exits_ok_when_every_violation_is_allowlisted(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, ['dirty.py'])
  assert _run(tree, allowlist) == checker.EXIT_OK
  assert (
      capsys.readouterr().out
      == '0 violation(s) in 0 file(s) outside the allowlist.\n'
  )


def test_main_exits_with_violations_and_prints_them(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, [])
  assert _run(tree, allowlist) == checker.EXIT_VIOLATIONS
  out = capsys.readouterr().out
  assert f'dirty.py:1 <module>: {_STRONG}' in out
  assert '1 violation(s) in 1 file(s) outside the allowlist.' in out


def test_main_fails_on_a_stale_allowlist_entry(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, ['dirty.py', 'clean.py'])
  assert _run(tree, allowlist) == checker.EXIT_VIOLATIONS
  out = capsys.readouterr().out
  assert 'clean.py: allowlisted but clean; delete this line' in out
  assert '1 stale allowlist entry(s).' in out


def test_main_all_reports_allowlisted_violations_without_failing(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, ['dirty.py'])
  assert _run(tree, allowlist, '--all') == checker.EXIT_OK
  out = capsys.readouterr().out
  assert f'dirty.py:1 <module>: {_STRONG}' in out
  assert '1 violation(s) in 1 file(s).' in out


def test_main_all_still_fails_on_a_violation_outside_the_allowlist(
    tree: pathlib.Path, tmp_path: pathlib.Path
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, [])
  assert _run(tree, allowlist, '--all') == checker.EXIT_VIOLATIONS


def test_main_update_allowlist_rewrites_the_file_and_then_passes(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  assert _run(tree, allowlist, '--update-allowlist') == checker.EXIT_OK
  assert 'Wrote 1 entries' in capsys.readouterr().out
  assert checker.load_allowlist(allowlist) == {'dirty.py'}
  assert _run(tree, allowlist) == checker.EXIT_OK


def test_main_reports_a_syntax_error_as_a_harness_failure(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  _write(tree / 'unparsable.py', 'def (:\n')
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, [])
  assert _run(tree, allowlist) == checker.EXIT_HARNESS_FAILURE
  assert 'cannot parse' in capsys.readouterr().err


def test_main_reports_a_missing_allowlist_as_a_harness_failure(
    tree: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
  assert _run(tree, tmp_path / 'absent.txt') == checker.EXIT_HARNESS_FAILURE
  assert 'cannot read the allowlist' in capsys.readouterr().err


def test_main_defaults_to_the_adk_tree_and_the_checked_in_allowlist() -> None:
  # No arguments: the invocation a contributor runs while fixing a docstring.
  assert checker.main([]) == checker.EXIT_OK


# --- end to end, through the real command line ------------------------------


def _script(*flags: str) -> subprocess.CompletedProcess[str]:
  """Runs the checker as a subprocess, the way a contributor runs it."""
  return subprocess.run(
      [sys.executable, str(checker.__file__), *flags],
      capture_output=True,
      text=True,
      check=False,
  )


def test_the_shipped_script_passes_on_this_tree() -> None:
  result = _script()
  assert result.returncode == checker.EXIT_OK, result.stdout + result.stderr
  assert 'outside the allowlist' in result.stdout


def test_the_shipped_script_fails_on_a_broken_docstring(
    tree: pathlib.Path, tmp_path: pathlib.Path
) -> None:
  allowlist = tmp_path / 'allowlist.txt'
  checker.write_allowlist(allowlist, [])
  result = _script('--root', str(tree), '--allowlist', str(allowlist))
  assert result.returncode == checker.EXIT_VIOLATIONS
  assert f'dirty.py:1 <module>: {_STRONG}' in result.stdout


# --- the real tree ----------------------------------------------------------


@pytest.fixture(scope='module')
def adk_violations() -> list[checker.Violation]:
  """Every markup violation in src/google/adk, measured once."""
  return checker.check_tree(checker._SRC_ROOT)


@pytest.fixture(scope='module')
def adk_allowlist() -> set[str]:
  return checker.load_allowlist(checker._ALLOWLIST)


def test_adk_docstrings_have_no_markup_violations(
    adk_violations: list[checker.Violation], adk_allowlist: set[str]
) -> None:
  blocking = [v for v in adk_violations if v.path not in adk_allowlist]
  preview = '\n'.join(str(violation) for violation in blocking[:20])
  assert not blocking, (
      f'{len(blocking)} docstring markup violation(s) outside'
      f' {checker._ALLOWLIST.name}. Fix the markup, or run'
      ' `python scripts/check_docstring_markup.py` to see them all:\n'
      f'{preview}'
  )


def test_allowlist_has_no_stale_entries(
    adk_violations: list[checker.Violation], adk_allowlist: set[str]
) -> None:
  offending = {violation.path for violation in adk_violations}
  stale = sorted(adk_allowlist - offending)
  assert not stale, (
      'These files are clean now; drop them from'
      f' scripts/{checker._ALLOWLIST.name}: {stale}'
  )


def test_allowlist_paths_exist(adk_allowlist: set[str]) -> None:
  missing = sorted(
      entry
      for entry in adk_allowlist
      if not (checker._REPO_ROOT / entry).is_file()
  )
  assert not missing, (
      'These allowlisted files no longer exist; drop them from'
      f' scripts/{checker._ALLOWLIST.name}: {missing}'
  )
