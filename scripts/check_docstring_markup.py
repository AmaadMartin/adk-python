#!/usr/bin/env python3
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

"""Checks that ADK docstrings are valid reStructuredText.

The public Python API reference at
https://google.github.io/adk-docs/api-reference/python/ is generated from
these docstrings by the google/adk-docs repository, which runs Sphinx with
`sphinx.ext.autodoc` and `sphinx.ext.napoleon`. Nothing in this repository
runs Sphinx, so malformed markup lands silently and is only found once it is
published.

This check reproduces the docstring half of that build without Sphinx's
machinery: it extracts every docstring with `ast`, applies the same Napoleon
transformation autodoc applies, parses the result with standalone docutils,
and reports every message of level WARNING or above. It also reports Markdown
code fences, which docutils accepts silently and Sphinx renders as garbage.

Modules are never imported, so no optional dependency has to be installed.

Run it over the whole tree:

  python scripts/check_docstring_markup.py

List everything, including the grandfathered files, to pick burn-down work:

  python scripts/check_docstring_markup.py --all

Exit codes: 0 clean, 1 violations found, 2 the check itself could not run.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import io
import pathlib
import re
import sys
from typing import Any

from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst import roles
from docutils.utils import Reporter

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_HARNESS_FAILURE = 2

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / 'src' / 'google' / 'adk'
_ALLOWLIST = _REPO_ROOT / 'scripts' / 'docstring_markup_allowlist.txt'

# Only WARNING (2) and above. Level 1 is INFO, which docutils emits for
# ordinary things such as an unreferenced hyperlink target.
_MIN_LEVEL = 2

_FENCE_RE = re.compile(r'^\s*```', re.MULTILINE)
_FENCE_MESSAGE = (
    'Markdown code fence (```) in a docstring; use a Sphinx'
    ' ".. code-block::" directive.'
)

_ALLOWLIST_HEADER = """\
# Files whose docstrings still contain reStructuredText that the API reference
# build cannot render. A file belongs here only while it would fail
# scripts/check_docstring_markup.py; once every docstring in it is valid, drop
# its entry so the check applies again.
#
# This list can only shrink. An entry whose file is now clean fails the check,
# and so does an entry naming a file that no longer exists.
#
# Do not add new files. There is no per-line or per-symbol suppression and no
# flag that skips the check, on purpose: a gate that can be waved through from
# a command line stops being a gate.
#
# Regenerate after a burn-down pass:
#   python scripts/check_docstring_markup.py --update-allowlist
#
# Format: one repository-relative POSIX path per line. Blank lines and
# #-comments are ignored.

"""

# Directives that carry prose. Their bodies are parsed, so broken markup
# inside them is still reported.
_PROSE_DIRECTIVES = (
    'attribute',
    'class',
    'data',
    'deprecated',
    'exception',
    'function',
    'method',
    'module',
    'property',
    'seealso',
    'todo',
    'versionadded',
    'versionchanged',
)

# Directives whose body is code. Their content is never parsed as prose.
_LITERAL_DIRECTIVES = (
    'code',
    'code-block',
    'literalinclude',
    'sourcecode',
)

# Cross-reference roles. Sphinx resolves these against its own inventory; here
# they only have to render as inline text.
_ROLES = (
    'attr',
    'class',
    'data',
    'doc',
    'envvar',
    'exc',
    'func',
    'meth',
    'mod',
    'obj',
    'option',
    'ref',
    'term',
)


class HarnessError(RuntimeError):
  """The check could not be completed, so its result means nothing."""


@dataclasses.dataclass(frozen=True)
class Violation:
  """One docstring problem, located precisely enough to fix."""

  path: str
  """Repository-relative POSIX path, e.g. 'src/google/adk/runners.py'."""

  lineno: int
  """1-based line of the docstring's owner in the source file."""

  symbol: str
  """'<module>', 'Klass', 'Klass.method', 'Klass.field' or 'fn'."""

  message: str
  """Single-line, whitespace-collapsed description of the problem."""

  def __str__(self) -> str:
    return f'{self.path}:{self.lineno} {self.symbol}: {self.message}'


class _StandInDirective(Directive):  # type: ignore[misc]
  """Accepts a Sphinx directive that docutils alone does not know.

  docutils ships no type information, so mypy sees `Directive` as `Any` and
  strict mode rejects the subclass. There is no way to register a directive
  without subclassing, so the base class is ignored here and nowhere else.
  """

  has_content = True
  optional_arguments = 1
  final_argument_whitespace = True


class _ProseDirective(_StandInDirective):
  """Stands in for a Sphinx directive whose body is reStructuredText."""

  def run(self) -> list[nodes.Node]:
    container = nodes.container()
    self.state.nested_parse(self.content, self.content_offset, container)
    return [container]


class _LiteralDirective(_StandInDirective):
  """Stands in for a Sphinx directive whose body is code, not prose."""

  def run(self) -> list[nodes.Node]:
    text = '\n'.join(self.content)
    return [nodes.literal_block(text, text)]


def _literal_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Any,
    options: Mapping[str, Any] | None = None,
    content: Sequence[str] | None = None,
) -> tuple[list[nodes.Node], list[nodes.system_message]]:
  """Stands in for a Sphinx cross-reference role."""
  del name, lineno, inliner, options, content  # Unused by the stand-in.
  return [nodes.literal(rawtext, text)], []


def _register_stand_ins() -> None:
  """Teaches docutils the Sphinx directives and roles used in docstrings.

  Without these, every Sphinx construct is reported as an unknown directive or
  role and the check drowns in false positives. Anything outside these lists
  is still reported, which is what keeps a typo such as `.. code-blok::` a
  failure.

  This mutates the process-wide docutils registry. It runs once at import.
  """
  for name in _PROSE_DIRECTIVES:
    directives.register_directive(name, _ProseDirective)
  for name in _LITERAL_DIRECTIVES:
    directives.register_directive(name, _LiteralDirective)
  for name in _ROLES:
    roles.register_local_role(name, _literal_role)
    roles.register_local_role(f'py:{name}', _literal_role)


_register_stand_ins()


def _symbol_name(prefix: str, name: str) -> str:
  return f'{prefix}.{name}' if prefix else name


def _attribute_docstrings(
    body: Sequence[ast.stmt], prefix: str
) -> Iterator[tuple[str, int, str]]:
  """Yields the attribute docstrings in a class or module body.

  An attribute docstring is a bare string expression that directly follows an
  assignment. autodoc-pydantic renders these as the documentation of a
  Pydantic field, so they reach the API reference like any other docstring.

  Args:
    body: The statements of the enclosing class or module.
    prefix: Dotted name of the enclosing scope, or '' at module level.

  Yields:
    (symbol, lineno, docstring) for each attribute docstring.
  """
  previous: ast.stmt | None = None
  for statement in body:
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
        and statement.value.value.strip()
    ):
      target = _assignment_target(previous)
      if target is not None:
        yield (
            _symbol_name(prefix, target),
            statement.lineno,
            statement.value.value,
        )
    previous = statement


def _assignment_target(statement: ast.stmt | None) -> str | None:
  """Returns the attribute name an assignment binds, if it binds exactly one."""
  if isinstance(statement, ast.AnnAssign) and isinstance(
      statement.target, ast.Name
  ):
    return statement.target.id
  if (
      isinstance(statement, ast.Assign)
      and len(statement.targets) == 1
      and isinstance(statement.targets[0], ast.Name)
  ):
    return statement.targets[0].id
  return None


def iter_docstrings(source: str) -> Iterator[tuple[str, int, str]]:
  """Yields every docstring in a Python source file.

  Args:
    source: The full text of a Python module.

  Yields:
    (symbol, lineno, docstring) triples. `symbol` is '<module>' for the module
    docstring and a dotted path such as 'Klass.method' otherwise. `lineno` is
    the 1-based line of the docstring's owner, so an editor jumps to the
    definition rather than to the string.

  Raises:
    SyntaxError: The source does not parse.
  """
  tree = ast.parse(source)
  module_docstring = ast.get_docstring(tree, clean=False)
  if module_docstring and module_docstring.strip():
    yield '<module>', 1, module_docstring
  yield from _attribute_docstrings(tree.body, '')
  yield from _iter_definitions(tree.body, '')


def _iter_definitions(
    body: Sequence[ast.stmt], prefix: str
) -> Iterator[tuple[str, int, str]]:
  """Yields docstrings of the definitions in a body, recursing into them."""
  for statement in body:
    if not isinstance(
        statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
      continue
    symbol = _symbol_name(prefix, statement.name)
    docstring = ast.get_docstring(statement, clean=False)
    if docstring and docstring.strip():
      yield symbol, statement.lineno, docstring
    if isinstance(statement, ast.ClassDef):
      yield from _attribute_docstrings(statement.body, symbol)
    yield from _iter_definitions(statement.body, symbol)


def docstring_to_rst(docstring: str) -> str:
  """Applies the transformation autodoc applies before docutils sees a docstring.

  Mirrors `sphinx.ext.napoleon._process_docstring`: NumPy style first, then
  Google style. Neither is given a config, so both fall back to
  `sphinx.ext.napoleon.Config()`. The docs build sets no `napoleon_*` option,
  so those defaults are exactly the settings it runs with.

  Args:
    docstring: The raw docstring, exactly as written in the source.

  Returns:
    The reStructuredText that Sphinx would hand to docutils.

  Raises:
    HarnessError: Sphinx is not installed.
  """
  # Imported here rather than at module level so that this module still
  # imports without sphinx. That is what lets a missing sphinx exit as a
  # harness failure naming the extra to install, instead of dying with a
  # traceback before `main` runs.
  try:
    from sphinx.ext.napoleon.docstring import GoogleDocstring
    from sphinx.ext.napoleon.docstring import NumpyDocstring
    from sphinx.util.docstrings import prepare_docstring
  except ImportError as err:
    raise HarnessError(
        'sphinx is required to check docstring markup; install it with'
        ' `uv sync --extra test`.'
    ) from err

  lines: list[str] = prepare_docstring(docstring)
  lines = NumpyDocstring(lines).lines()
  lines = GoogleDocstring(lines).lines()
  return '\n'.join(lines)


def check_docstring(docstring: str) -> list[str]:
  """Returns the markup problems in one docstring, in document order.

  Args:
    docstring: The raw docstring, exactly as written in the source.

  Returns:
    Single-line problem descriptions. Empty when the docstring is clean.

  Raises:
    HarnessError: Sphinx is not installed.
  """
  messages: list[str] = []
  if _FENCE_RE.search(docstring):
    messages.append(_FENCE_MESSAGE)

  rst = docstring_to_rst(docstring)
  doctree = publish_doctree(
      rst,
      settings_overrides={
          'report_level': Reporter.INFO_LEVEL,
          # Above SEVERE, so docutils reports every problem as a node instead
          # of raising and ending the scan at the first bad docstring.
          'halt_level': Reporter.SEVERE_LEVEL + 1,
          'warning_stream': io.StringIO(),
          'input_encoding': 'unicode',
          # A docstring must never make this check read a file.
          'file_insertion_enabled': False,
          'raw_enabled': False,
          'docutils_source_link': False,
      },
  )
  for message in doctree.findall(nodes.system_message):
    if int(message['level']) < _MIN_LEVEL:
      continue
    # `message.astext()` prefixes an unhelpful '<string>:NN:'; the first child
    # is the description on its own.
    text: str = message.children[0].astext()
    messages.append(' '.join(text.split()))
  return messages


def check_source(source: str, path: str) -> list[Violation]:
  """Returns every markup violation in one Python source file.

  Args:
    source: The full text of a Python module.
    path: Repository-relative POSIX path, used to locate the violations.

  Returns:
    The violations, in source order.

  Raises:
    SyntaxError: The source does not parse.
    HarnessError: Sphinx is not installed.
  """
  violations: list[Violation] = []
  for symbol, lineno, docstring in iter_docstrings(source):
    for message in check_docstring(docstring):
      violations.append(
          Violation(path=path, lineno=lineno, symbol=symbol, message=message)
      )
  return violations


def report_base(root: pathlib.Path) -> pathlib.Path:
  """Returns the directory that reported paths are relative to.

  Inside this repository that is the repository root, so a violation reads as
  `src/google/adk/runners.py`. Outside it -- a scratch tree in a test -- it is
  the scanned directory itself, because there is no wider root to name.
  """
  resolved = root.resolve()
  return _REPO_ROOT if resolved.is_relative_to(_REPO_ROOT) else resolved


def check_tree(
    root: pathlib.Path, base: pathlib.Path | None = None
) -> list[Violation]:
  """Returns every markup violation under a directory.

  Args:
    root: Directory to walk. Every `*.py` file below it is checked.
    base: Directory the reported paths are made relative to. Defaults to
      `report_base(root)`.

  Returns:
    The violations, ordered by path then source order.

  Raises:
    HarnessError: A file could not be read or parsed. A check that silently
      skipped part of its input must not read as a pass.
  """
  base = report_base(root) if base is None else base
  violations: list[Violation] = []
  for path in sorted(root.rglob('*.py')):
    relative = path.relative_to(base).as_posix()
    try:
      source = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as err:
      raise HarnessError(f'{relative}: cannot read: {err}') from err
    try:
      violations.extend(check_source(source, relative))
    except SyntaxError as err:
      raise HarnessError(f'{relative}: cannot parse: {err}') from err
  return violations


def load_allowlist(path: pathlib.Path) -> set[str]:
  """Reads the allowlisted paths, ignoring comments and blank lines.

  Args:
    path: The allowlist file.

  Returns:
    Repository-relative POSIX paths that are exempt from the check.

  Raises:
    HarnessError: The file is missing or unreadable.
  """
  try:
    text = path.read_text(encoding='utf-8')
  except (OSError, UnicodeDecodeError) as err:
    raise HarnessError(f'cannot read the allowlist {path}: {err}') from err
  entries: set[str] = set()
  for line in text.splitlines():
    entry = line.split('#', 1)[0].strip()
    if entry:
      entries.add(entry)
  return entries


def write_allowlist(path: pathlib.Path, paths: Iterable[str]) -> None:
  """Rewrites the allowlist with the given paths, sorted, under the header."""
  body = ''.join(f'{entry}\n' for entry in sorted(paths))
  path.write_text(f'{_ALLOWLIST_HEADER}{body}', encoding='utf-8')


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--root',
      type=pathlib.Path,
      default=_SRC_ROOT,
      help='Directory to check. Defaults to src/google/adk.',
  )
  parser.add_argument(
      '--allowlist',
      type=pathlib.Path,
      default=_ALLOWLIST,
      help=(
          'Allowlist file. Defaults to scripts/docstring_markup_allowlist.txt.'
      ),
  )
  parser.add_argument(
      '--update-allowlist',
      action='store_true',
      help='Rewrite the allowlist from the violations found now.',
  )
  parser.add_argument(
      '--all',
      action='store_true',
      help='Report every violation, including allowlisted files.',
  )
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  """Runs the check and prints its findings.

  Args:
    argv: Command line arguments, or None to read `sys.argv`.

  Returns:
    EXIT_OK, EXIT_VIOLATIONS or EXIT_HARNESS_FAILURE.
  """
  args = _parse_args(argv)
  try:
    violations = check_tree(args.root)
    allowlist = (
        set() if args.update_allowlist else load_allowlist(args.allowlist)
    )
  except HarnessError as err:
    print(f'error: {err}', file=sys.stderr)
    return EXIT_HARNESS_FAILURE

  offending = {violation.path for violation in violations}
  if args.update_allowlist:
    write_allowlist(args.allowlist, offending)
    print(f'Wrote {len(offending)} entries to {args.allowlist}.')
    return EXIT_OK

  # `--all` widens what is printed. It never widens what fails, so a burn-down
  # listing cannot be mistaken for a broken build.
  blocking = [v for v in violations if v.path not in allowlist]
  reported = violations if args.all else blocking
  for violation in reported:
    print(violation)

  stale = sorted(allowlist - offending)
  for entry in stale:
    print(
        f'{entry}: allowlisted but clean; delete this line from'
        f' {args.allowlist}.'
    )

  files = len({violation.path for violation in reported})
  scope = '' if args.all else ' outside the allowlist'
  print(f'{len(reported)} violation(s) in {files} file(s){scope}.')
  if stale:
    print(f'{len(stale)} stale allowlist entry(s).')
  return EXIT_VIOLATIONS if blocking or stale else EXIT_OK


if __name__ == '__main__':
  sys.exit(main())
