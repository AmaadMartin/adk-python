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
from typing import Any

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_PRE_COMMIT_CONFIG = _REPO_ROOT / '.pre-commit-config.yaml'
_WORKFLOW = _REPO_ROOT / '.github' / 'workflows' / 'constraints-check.yml'


def _load(path: pathlib.Path) -> dict[str, Any]:
  return yaml.safe_load(path.read_text(encoding='utf-8'))


def test_pre_commit_declares_no_constraints_hook() -> None:
  """The drift check must stay out of the offline formatting gate.

  The lint job runs `pre-commit run --all-files`, which applies every hook to
  the whole repository regardless of its `files:` filter. A hook here that
  needs `uv` and network access to PyPI therefore runs on every pull request.
  """
  config = _load(_PRE_COMMIT_CONFIG)
  entries = [
      hook.get('entry', '')
      for repo in config['repos']
      for hook in repo.get('hooks', [])
  ]
  assert not [entry for entry in entries if 'update_constraints' in entry]


def test_constraints_workflow_runs_the_script_in_check_mode() -> None:
  """The workflow must pass --check.

  Without it the script runs as a fixer: it rewrites the tracked files inside
  the CI checkout and exits non-zero whenever it wrote, which is the failure
  this workflow exists to replace.
  """
  jobs = _load(_WORKFLOW)['jobs']
  commands = [
      step.get('run', '')
      for job in jobs.values()
      for step in job.get('steps', [])
  ]
  assert [
      command
      for command in commands
      if 'update_constraints.sh --check' in command
  ]
