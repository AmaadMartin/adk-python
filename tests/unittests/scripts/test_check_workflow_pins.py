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

import pathlib
import subprocess
from typing import Any

import pytest
import yaml

from scripts import check_workflow_pins

_SHA = 'df4cb1c069e1874edd31b4311f1884172cec0e10'
_REPO_ROOT = pathlib.Path(__file__).parents[3]
_WORKFLOWS_DIR = _REPO_ROOT / '.github' / 'workflows'
_SCRIPT = _REPO_ROOT / 'scripts' / 'check_workflow_pins.py'
_CI_WORKFLOW = _WORKFLOWS_DIR / 'continuous-integration.yml'
_WORKFLOWS_PATH_FILTER = '.github/workflows/**'


def test_pinned_ref_with_version_comment_is_accepted() -> None:
  content = (
      f'      - name: Checkout\n        uses: actions/checkout@{_SHA} # v6\n'
  )
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_tag_ref_is_reported_with_the_full_value() -> None:
  content = '        uses: actions/checkout@v6\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, 'actions/checkout@v6')
  ]


def test_branch_ref_is_reported() -> None:
  content = '        uses: actions/checkout@main\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, 'actions/checkout@main')
  ]


@pytest.mark.parametrize(
    'ref',
    [
        'df4cb1c',  # Abbreviated.
        _SHA[:39],  # One character too short.
        _SHA + 'a',  # One character too long.
    ],
)
def test_hex_ref_of_the_wrong_length_is_reported(ref: str) -> None:
  content = f'        uses: actions/checkout@{ref}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, f'actions/checkout@{ref}')
  ]


def test_uppercase_sha_is_accepted() -> None:
  content = f'        uses: actions/checkout@{_SHA.upper()}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_quoted_tag_ref_is_reported_without_its_quotes() -> None:
  content = "        uses: 'google-github-actions/auth@v3'\n"
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, 'google-github-actions/auth@v3')
  ]


def test_quoted_pinned_ref_is_accepted() -> None:
  content = f"        uses: 'google-github-actions/auth@{_SHA}'\n"
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_inline_step_pinned_ref_is_accepted() -> None:
  content = f'      - uses: actions/checkout@{_SHA}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_inline_step_tag_ref_is_reported() -> None:
  content = '      - uses: actions/checkout@v6\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, 'actions/checkout@v6')
  ]


@pytest.mark.parametrize(
    'value',
    ['./.github/actions/foo', './.github/workflows/x.yml'],
)
def test_repo_local_reference_is_accepted(value: str) -> None:
  content = f'        uses: {value}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


@pytest.mark.parametrize(
    'value',
    [
        f'actions/cache/restore@{_SHA}',
        f'owner/repo/.github/workflows/x.yml@{_SHA}',
    ],
)
def test_subdirectory_and_reusable_workflow_pins_are_accepted(
    value: str,
) -> None:
  content = f'        uses: {value}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_tag_whose_name_embeds_a_sha_is_reported() -> None:
  # The ref here is the mutable tag `v1@<sha>`, not the SHA. Taking the text
  # after the last `@` would accept it.
  content = f'        uses: owner/repo@v1@{_SHA}\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, f'owner/repo@v1@{_SHA}')
  ]


def test_reference_without_a_ref_is_reported() -> None:
  content = '        uses: actions/checkout\n'
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (1, 'actions/checkout')
  ]


def test_docker_image_reference_is_accepted() -> None:
  content = '        uses: docker://alpine:3.18\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_commented_out_reference_is_accepted() -> None:
  content = '        # uses: actions/checkout@v6\n'
  assert check_workflow_pins.find_unpinned_uses(content) == []


def test_reported_line_number_is_one_based() -> None:
  content = (
      'name: Example\n'
      'jobs:\n'
      '  build:\n'
      '    steps:\n'
      '      - uses: actions/checkout@v6\n'
  )
  assert check_workflow_pins.find_unpinned_uses(content) == [
      (5, 'actions/checkout@v6')
  ]


def test_main_accepts_a_pinned_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  workflow = tmp_path / 'clean.yml'
  workflow.write_text(
      f'steps:\n  - uses: actions/checkout@{_SHA} # v6\n', encoding='utf-8'
  )

  assert check_workflow_pins.main([str(workflow)]) == 0
  assert capsys.readouterr().out == ''


def test_main_reports_an_unpinned_file(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
  workflow = tmp_path / 'dirty.yml'
  workflow.write_text(
      'steps:\n  - uses: actions/checkout@v6\n', encoding='utf-8'
  )

  assert check_workflow_pins.main([str(workflow)]) == 1
  out = capsys.readouterr().out
  assert f'{workflow}:2:' in out
  assert "'actions/checkout@v6'" in out


@pytest.mark.skipif(
    not _SCRIPT.is_file(),
    reason='scripts/ is absent from this copy of the source tree.',
)
def test_script_runs_as_an_executable_and_exits_non_zero(
    tmp_path: pathlib.Path,
) -> None:
  # pre-commit runs this hook with `language: script`, which executes the file
  # directly. That needs the executable bit and the shebang, neither of which
  # an in-process import exercises.
  workflow = tmp_path / 'dirty.yml'
  workflow.write_text(
      'steps:\n  - uses: actions/checkout@v6\n', encoding='utf-8'
  )

  result = subprocess.run(
      [str(_SCRIPT), str(workflow)], capture_output=True, text=True, check=False
  )

  assert result.returncode == 1
  assert f'{workflow}:2:' in result.stdout


@pytest.mark.skipif(
    not _WORKFLOWS_DIR.is_dir(),
    reason='.github/workflows is absent from this copy of the source tree.',
)
def test_every_repository_workflow_is_pinned() -> None:
  offenders = {
      str(
          path.relative_to(_WORKFLOWS_DIR)
      ): check_workflow_pins.find_unpinned_uses(
          path.read_text(encoding='utf-8')
      )
      for path in sorted(_WORKFLOWS_DIR.iterdir())
      if path.suffix in ('.yml', '.yaml')
  }
  assert {name: found for name, found in offenders.items() if found} == {}


@pytest.mark.skipif(
    not _CI_WORKFLOW.is_file(),
    reason='.github/workflows is absent from this copy of the source tree.',
)
@pytest.mark.parametrize('event', ['push', 'pull_request'])
def test_continuous_integration_triggers_on_workflow_changes(
    event: str,
) -> None:
  # Continuous Integration is the only workflow that runs pre-commit, so it is
  # the only place this check runs server-side. Its triggers are restricted by
  # a paths allow-list; drop `.github/workflows/**` from either one and a pull
  # request that edits only a workflow runs no CI, which leaves the hook
  # enforced solely by contributors who installed it.
  workflow: dict[Any, Any] = yaml.safe_load(
      _CI_WORKFLOW.read_text(encoding='utf-8')
  )
  # PyYAML reads the bare `on` key as the YAML 1.1 boolean `True`.
  triggers = workflow[True]

  assert _WORKFLOWS_PATH_FILTER in triggers[event]['paths']
