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

"""Guard tests for the uv pin in the release workflows.

Neither release workflow runs on an ordinary pull request to ``main``, so
nothing else catches a commit that unpins uv before it reaches a release
branch. These tests fail when someone drops the pin back to ``latest``, drops
the download checksum, or bumps one workflow and forgets the other.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any

import pytest
import yaml

_WORKFLOWS = (
    pathlib.Path(__file__).parent.parent.parent / ".github" / "workflows"
)
_RELEASE_WORKFLOWS = ("release-artifact-check.yml", "release-publish.yml")
_EXACT_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _setup_uv_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
  """Returns the ``with`` mapping of every astral-sh/setup-uv step.

  Args:
    workflow: A parsed GitHub Actions workflow.

  Returns:
    One mapping per setup-uv step, in file order. A step with no ``with``
    block yields an empty mapping, so a caller sees a missing input rather
    than a ``KeyError``.
  """
  return [
      step.get("with", {})
      for job in workflow["jobs"].values()
      for step in job["steps"]
      if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
  ]


_PINS = {
    name: _setup_uv_steps(
        yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))
    )
    for name in _RELEASE_WORKFLOWS
}


@pytest.mark.parametrize("workflow", _RELEASE_WORKFLOWS)
def test_release_workflows_pin_an_exact_uv_version(workflow: str) -> None:
  """The release toolchain is one exact uv version, never ``latest``."""
  steps = _PINS[workflow]
  assert len(steps) == 1, (
      f"{workflow} must install uv exactly once, so there is a single pin to"
      f" review and bump; found {len(steps)} astral-sh/setup-uv steps."
  )

  version = steps[0].get("version", "")
  assert version not in ("", "latest"), (
      f"{workflow} must pin an exact uv version. setup-uv treats a missing or"
      " empty version as 'latest', which installs whatever uv shipped that"
      " day."
  )
  assert _EXACT_VERSION.fullmatch(version), (
      f"{workflow} must pin uv as an exact X.Y.Z version, not {version!r}."
      " setup-uv resolves a range to the newest match, so a range is not a"
      " pin."
  )


@pytest.mark.parametrize("workflow", _RELEASE_WORKFLOWS)
def test_release_workflows_pin_the_uv_download_checksum(workflow: str) -> None:
  """The pinned uv download is verified against a sha256."""
  checksum = _PINS[workflow][0].get("checksum", "")
  assert _SHA256.fullmatch(checksum), (
      f"{workflow} must set the setup-uv checksum input to the sha256 of"
      f" uv-x86_64-unknown-linux-gnu.tar.gz, not {checksum!r}. The pinned"
      " setup-uv commit only bundles checksums up to uv 0.10.10, so a newer"
      " uv is downloaded unverified without it."
  )


def test_release_workflows_agree_on_the_uv_pin() -> None:
  """Both release workflows install the same uv build."""
  pins = {
      name: (steps[0].get("version"), steps[0].get("checksum"))
      for name, steps in _PINS.items()
  }
  assert len(set(pins.values())) == 1, (
      "The release workflows must install the same uv version and checksum,"
      " otherwise the wheel that is checked is not the wheel that is"
      f" published. Found {pins}."
  )


def test_setup_uv_steps_ignores_other_actions() -> None:
  """The helper matches setup-uv only, and tolerates a step with no ``uses``."""
  workflow = {
      "jobs": {
          "build": {
              "steps": [
                  {"uses": "actions/checkout@v6"},
                  {"run": "uv build"},
              ]
          }
      }
  }

  assert _setup_uv_steps(workflow) == []
