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

"""Validates every SKILL.md that ships inside the google.adk package.

The walk is anchored on the imported package rather than on the repository
root, the same way ``bigquery_skill.py`` resolves its own manifest. CI installs
the project as an editable checkout, so there the walk covers
``src/google/adk/``. Run against a non-editable install it covers
``site-packages/google/adk/`` instead, and then
``test_package_ships_the_expected_skills`` also reports a manifest that the
distribution dropped.
"""

from __future__ import annotations

import pathlib

import google.adk
from google.adk.skills import load_skill_from_dir
from google.adk.skills._utils import _validate_skill_dir
import pytest

_ADK_PACKAGE_DIR = pathlib.Path(google.adk.__file__).resolve().parent

# Shipping a skill to users is a deliberate act, so adding an entry here is
# part of adding a packaged skill.
_EXPECTED_PACKAGED_SKILLS = frozenset({"bigquery-ai-ml", "bigquery-graph"})


def _packaged_skill_dirs() -> list[pathlib.Path]:
  """Returns every skill directory shipped inside the google.adk package."""
  # The loader accepts either spelling of the manifest filename.
  return sorted({
      skill_md.parent
      for pattern in ("SKILL.md", "skill.md")
      for skill_md in _ADK_PACKAGE_DIR.rglob(pattern)
  })


_for_each_packaged_skill = pytest.mark.parametrize(
    "skill_dir",
    _packaged_skill_dirs(),
    ids=lambda path: path.relative_to(_ADK_PACKAGE_DIR).as_posix(),
)


def test_package_ships_the_expected_skills():
  """The distribution contains exactly the skill manifests we intend to ship."""
  discovered = {path.name for path in _packaged_skill_dirs()}

  assert discovered == _EXPECTED_PACKAGED_SKILLS


@_for_each_packaged_skill
def test_packaged_skill_loads(skill_dir: pathlib.Path):
  """Every packaged manifest parses through the public skills loader."""
  skill = load_skill_from_dir(skill_dir)

  assert skill.instructions.strip()


@_for_each_packaged_skill
def test_packaged_skill_reports_no_validator_problems(skill_dir: pathlib.Path):
  """ADK's own validator accepts every packaged manifest."""
  problems = _validate_skill_dir(skill_dir)

  assert not problems, f"{skill_dir.name} validation problems: {problems}"


@_for_each_packaged_skill
def test_packaged_skill_references_are_non_empty(skill_dir: pathlib.Path):
  """Every reference document shipped alongside a skill has content."""
  references = load_skill_from_dir(skill_dir).resources.references

  for name, content in references.items():
    assert content.strip(), f"{skill_dir.name}: {name} is empty"
