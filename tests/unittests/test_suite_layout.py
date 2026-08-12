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

"""Guards the test tree against duplicate pytest module names.

pytest's default ``prepend`` import mode names a test module after the first
ancestor directory that is not a package. Two same-named test files in two
non-package directories therefore claim one module name, and pytest aborts
collection for the whole session when it sees both. Every automated invocation
scopes the run to a single path, so CI cannot surface that. This module
enforces the invariant directly instead.
"""

from __future__ import annotations

import collections
import pathlib

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _pytest_module_name(path: pathlib.Path) -> str:
  """Returns the module name pytest imports ``path`` under in prepend mode."""
  parts = [path.stem]
  directory = path.parent
  while (directory / "__init__.py").is_file():
    parts.insert(0, directory.name)
    directory = directory.parent
  return ".".join(parts)


def _duplicate_module_names(tests_root: pathlib.Path) -> dict[str, list[str]]:
  """Returns each pytest module name that more than one test file claims.

  Args:
    tests_root: The directory to walk for ``test_*.py`` files.

  Returns:
    A mapping of module name to the paths that claim it, relative to the
    parent of ``tests_root``. Module names claimed by a single file are
    omitted.
  """
  by_module_name = collections.defaultdict(list)
  for path in sorted(tests_root.rglob("test_*.py")):
    module_name = _pytest_module_name(path)
    by_module_name[module_name].append(
        path.relative_to(tests_root.parent).as_posix()
    )
  return {
      name: paths for name, paths in by_module_name.items() if len(paths) > 1
  }


def test_no_duplicate_pytest_module_names() -> None:
  collisions = _duplicate_module_names(_REPO_ROOT / "tests")

  assert not collisions, (
      "These test files import under the same module name, so a single pytest"
      " session that sees both aborts collection. Give the files unique"
      " basenames, or add __init__.py to every directory between them and"
      f" their nearest packaged ancestor: {collisions}"
  )


def test_duplicate_module_names_flags_same_name_in_two_namespace_dirs(
    tmp_path: pathlib.Path,
) -> None:
  """Pins the detector on the layout that broke a bare ``pytest`` run."""
  tests_root = tmp_path / "tests"
  for suite in ("integration", "unittests"):
    leaf = tests_root / suite / "integrations" / "oci"
    leaf.mkdir(parents=True)
    (leaf / "__init__.py").touch()
    (leaf / "test_thing.py").touch()

  assert _duplicate_module_names(tests_root) == {
      "oci.test_thing": [
          "tests/integration/integrations/oci/test_thing.py",
          "tests/unittests/integrations/oci/test_thing.py",
      ]
  }


def test_packaged_directories_produce_distinct_module_names(
    tmp_path: pathlib.Path,
) -> None:
  """An unbroken package chain to ``tests_root`` disambiguates the two files."""
  tests_root = tmp_path / "tests"
  for suite in ("integration", "unittests"):
    leaf = tests_root / suite / "integrations" / "oci"
    leaf.mkdir(parents=True)
    for package in (tests_root / suite, leaf.parent, leaf):
      (package / "__init__.py").touch()
    (leaf / "test_thing.py").touch()

  assert not _duplicate_module_names(tests_root)
