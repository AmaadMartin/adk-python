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

"""Guard for the ``ApiServer.__init__`` eval-manager annotations.

``api_server`` uses ``from __future__ import annotations``, so an annotation
that names an unimported class survives until something resolves it. The two
eval-manager annotations must therefore stay backed by real module-level
imports.
"""

from __future__ import annotations

import typing

from google.adk.cli.api_server import ApiServer
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager


def test_api_server_init_type_hints_resolve():
  hints = typing.get_type_hints(ApiServer.__init__)

  assert hints["eval_sets_manager"] is EvalSetsManager
  assert hints["eval_set_results_manager"] is EvalSetResultsManager
