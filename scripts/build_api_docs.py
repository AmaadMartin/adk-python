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

"""Builds the google.adk API reference as a smoke test.

The `docs` optional-dependency group is installed by the google/adk-docs
API-reference generator but is otherwise never run from this repository. This
script builds the reference with the checked-in Sphinx configuration in
docs/api-reference/, so CI notices when a dependency bump or a new module
breaks that build.

The build writes only into a temporary directory. It fails when Sphinx fails,
or when autodoc cannot import a module that is not in
`_ALLOWED_IMPORT_FAILURES`. Warnings are not errors here: the docstrings carry
several hundred reStructuredText warnings that are tracked separately.

Usage:

  python scripts/build_api_docs.py [--source-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import os
import pkgutil
import re
import shutil
import sys
import tempfile
from types import ModuleType

_ROOT_PACKAGE = 'google.adk'

# Package prefixes whose autodoc import failure is known and tracked
# elsewhere. It is empty because every discovered module imports under
# `uv sync --all-extras`. Add a prefix here only for a pre-existing failure
# you cannot fix, and only with a linked follow-up issue. A submodule of a
# listed prefix is allowed too.
_ALLOWED_IMPORT_FAILURES: frozenset[str] = frozenset()

# Emitted by autodoc as, for example:
#   WARNING: autodoc: failed to import module 'agents' from module
#   'google.adk'; the following exception was raised: ...
# The "from module" part is absent when autodoc cannot split the name.
_IMPORT_FAILURE_RE = re.compile(
    r"autodoc: failed to import module '(?P<module>[^']+)'"
    r"(?: from module '(?P<parent>[^']+)')?"
)

_GENERATED_DOCUMENT = 'google-adk'

_UNEXPECTED_FAILURE_HINT = (
    'autodoc could not import the modules above. Fix the module, or, if the'
    ' breakage is pre-existing and out of scope, add it to'
    ' _ALLOWED_IMPORT_FAILURES in scripts/build_api_docs.py together with a'
    ' linked follow-up issue.'
)


def discover_modules(package: ModuleType) -> list[str]:
  """Returns the sorted public submodule names of a package.

  Recurses to any depth. A module is private, and therefore skipped, when any
  component of its name below the package starts with an underscore.

  Args:
    package: The imported root package to walk.

  Returns:
    Fully-qualified module names, sorted. The package itself is not included.
  """
  prefix = f'{package.__name__}.'
  modules = [
      info.name
      for info in pkgutil.walk_packages(package.__path__, prefix)
      if not any(
          part.startswith('_') for part in info.name[len(prefix) :].split('.')
      )
  ]
  return sorted(modules)


def render_rst(modules: Sequence[str]) -> str:
  """Returns a reStructuredText document with one automodule per module."""
  blocks = ['API Reference\n=============\n']
  for module in modules:
    heading = module.replace('_', r'\_')
    blocks.append(
        f'{heading}\n'
        f'{"-" * len(heading)}\n'
        '\n'
        f'.. automodule:: {module}\n'
        '   :members:\n'
        '   :undoc-members:\n'
        '   :show-inheritance:\n'
    )
  return '\n'.join(blocks)


def find_unexpected_import_failures(warning_text: str) -> list[str]:
  """Returns the sorted modules autodoc failed to import unexpectedly.

  Args:
    warning_text: The contents of the Sphinx warnings file.

  Returns:
    Fully-qualified names of the modules that failed to import and are not
    covered by `_ALLOWED_IMPORT_FAILURES`.
  """
  failures: set[str] = set()
  for match in _IMPORT_FAILURE_RE.finditer(warning_text):
    module = match['module']
    parent = match['parent']
    failures.add(f'{parent}.{module}' if parent else module)
  return sorted(f for f in failures if not _is_allowed(f))


def _is_allowed(module: str) -> bool:
  """Returns whether the module is covered by the import-failure allowlist."""
  return any(
      module == allowed or module.startswith(f'{allowed}.')
      for allowed in _ALLOWED_IMPORT_FAILURES
  )


def _run_sphinx(argv: list[str]) -> int:
  """Runs a Sphinx build and returns its exit code.

  Sphinx is imported here rather than at module level so that the helpers
  above, and the unit tests that cover them, import without the `docs` extra.
  The unit-test CI job installs only the `test` extra.
  """
  from sphinx.cmd.build import build_main

  return build_main(argv)


def _default_source_dir() -> str:
  """Returns the checked-in Sphinx source directory."""
  repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  return os.path.join(repo_root, 'docs', 'api-reference')


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--source-dir',
      default=_default_source_dir(),
      help='Directory holding the checked-in conf.py and index.rst.',
  )
  parser.add_argument(
      '--output-dir',
      default=None,
      help='Directory for the generated HTML. Defaults to a temporary one.',
  )
  return parser.parse_args(argv)


def main(argv: list[str]) -> int:
  """Builds the API reference and reports unexpected autodoc failures."""
  args = _parse_args(argv)
  package = importlib.import_module(_ROOT_PACKAGE)

  with tempfile.TemporaryDirectory() as build_dir:
    source_dir = os.path.join(build_dir, 'source')
    shutil.copytree(args.source_dir, source_dir)
    document = os.path.join(source_dir, f'{_GENERATED_DOCUMENT}.rst')
    with open(document, 'w', encoding='utf-8') as f:
      f.write(render_rst(discover_modules(package)))

    warnings_path = os.path.join(build_dir, 'warnings.txt')
    output_dir = args.output_dir or os.path.join(build_dir, 'html')
    # -T prints the full traceback when the build crashes. Sphinx otherwise
    # writes it to a temporary file that a CI runner discards.
    status = _run_sphinx(
        ['-b', 'html', '-T', '-w', warnings_path, source_dir, output_dir]
    )
    if status != 0:
      return status
    with open(warnings_path, encoding='utf-8') as f:
      unexpected = find_unexpected_import_failures(f.read())

  if not unexpected:
    return 0
  print('\n'.join(unexpected), file=sys.stderr)
  print(_UNEXPECTED_FAILURE_HINT, file=sys.stderr)
  return 1


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
