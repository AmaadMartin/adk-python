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

import pathlib

import yaml

_PRE_COMMIT_CONFIG = (
    pathlib.Path(__file__).resolve().parents[3] / '.pre-commit-config.yaml'
)


def test_update_constraints_hook_is_manual_only() -> None:
  config = yaml.safe_load(_PRE_COMMIT_CONFIG.read_text())
  hooks = [
      hook
      for repo in config['repos']
      for hook in repo['hooks']
      if hook['id'] == 'update-constraints'
  ]
  assert len(hooks) == 1
  # The hook shells out to `uv`, needs the network, and rewrites untracked
  # files, so it must never run in the default `pre-commit` stage -- that
  # stage covers both `git commit` and the `pre-commit run --all-files`
  # invocation used by the Pre-commit Linter CI job.
  assert hooks[0]['stages'] == ['manual']
