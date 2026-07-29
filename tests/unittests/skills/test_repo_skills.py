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

"""Validates the agent skills checked into this repository's .agents/skills/."""

from __future__ import annotations

import pathlib

from google.adk.skills import list_skills_in_dir
from google.adk.skills import load_skill_from_dir
from google.adk.skills._utils import _validate_skill_dir
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SKILLS_DIR = _REPO_ROOT / ".agents" / "skills"

# Frontmatter keys a specific repository skill is allowed to carry even though
# google.adk.skills does not recognize them. `disable-model-invocation` is an
# Agent Skills client directive that stops an agent harness from auto-loading
# the skill; it is deliberate on adk-setup and must not be deleted or moved
# under `metadata:` to satisfy this test. Frontmatter accepts unknown keys as
# extra fields, so the skill still loads; _validate_skill_dir only reports it.
_ALLOWED_EXTRA_FRONTMATTER_KEYS: dict[str, frozenset[str]] = {
    "adk-setup": frozenset({"disable-model-invocation"}),
}


def _repo_skill_dirs() -> list[pathlib.Path]:
  """Returns the skill directories checked into .agents/skills/."""
  return sorted(
      path
      for path in _SKILLS_DIR.glob("*")
      if path.is_dir() and not path.name.startswith((".", "_"))
  )


_for_each_repo_skill = pytest.mark.parametrize(
    "skill_dir", _repo_skill_dirs(), ids=lambda path: path.name
)

pytestmark = pytest.mark.skipif(
    not _SKILLS_DIR.is_dir(),
    reason=(
        ".agents/skills/ is not part of this checkout; repository skill"
        " validation only applies to a full source tree."
    ),
)


def test_repo_ships_at_least_one_skill():
  """The repository's .agents/skills/ directory contains skills to validate."""
  assert _repo_skill_dirs()


@_for_each_repo_skill
def test_repo_skill_loads(skill_dir: pathlib.Path):
  """Every checked-in skill loads through the public loader."""
  skill = load_skill_from_dir(skill_dir)

  assert skill.name == skill_dir.name
  assert skill.description
  assert skill.instructions.strip()


@_for_each_repo_skill
def test_repo_skill_frontmatter_has_no_unrecognized_keys(
    skill_dir: pathlib.Path,
):
  """A skill's frontmatter only uses keys ADK recognizes, plus exceptions."""
  frontmatter = load_skill_from_dir(skill_dir).frontmatter
  extra_keys = set(frontmatter.model_extra or {})
  allowed = _ALLOWED_EXTRA_FRONTMATTER_KEYS.get(skill_dir.name, frozenset())

  assert extra_keys <= allowed, (
      f"{skill_dir.name} has unrecognized frontmatter keys:"
      f" {sorted(extra_keys - allowed)}"
  )


@_for_each_repo_skill
def test_repo_skill_reports_no_unexpected_validator_problems(
    skill_dir: pathlib.Path,
):
  """The validator reports nothing beyond a skill's documented extra keys."""
  problems = _validate_skill_dir(skill_dir)
  allowed = _ALLOWED_EXTRA_FRONTMATTER_KEYS.get(skill_dir.name, frozenset())

  if not allowed:
    assert problems == [], f"{skill_dir.name}: {problems}"
  else:
    # The only tolerated report is the advisory naming the documented keys.
    assert len(problems) == 1, f"{skill_dir.name}: {problems}"
    assert all(
        key in problems[0] for key in allowed
    ), f"{skill_dir.name}: {problems}"


def test_all_repo_skills_are_listed():
  """Listing the skills directory surfaces every skill, none skipped."""
  listed = list_skills_in_dir(_SKILLS_DIR)

  assert set(listed) == {path.name for path in _repo_skill_dirs()}
