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

"""Guards for the ``ApiServer.__init__`` eval-manager annotations.

``api_server`` uses ``from __future__ import annotations``, so an annotation
that names an unimported class survives until something resolves it. The two
eval-manager annotations must therefore stay backed by real module-level
imports, and those imports must not pull in the optional ``eval`` extra.
"""

from __future__ import annotations

import subprocess
import sys
import typing

from google.adk.cli.api_server import ApiServer
from google.adk.evaluation.eval_set_results_manager import EvalSetResultsManager
from google.adk.evaluation.eval_sets_manager import EvalSetsManager


def test_api_server_init_type_hints_resolve():
  hints = typing.get_type_hints(ApiServer.__init__)

  assert hints["eval_sets_manager"] is EvalSetsManager
  assert hints["eval_set_results_manager"] is EvalSetResultsManager


def test_importing_api_server_does_not_import_eval_extra():
  # Run in a fresh interpreter so the check is not polluted by modules that
  # other tests already imported into sys.modules.
  code = (
      "import google.adk.cli.api_server\n"
      "import sys\n"
      "forbidden = ['gepa', 'nltk', 'rouge_score', 'tabulate']\n"
      "loaded = [name for name in forbidden if name in sys.modules]\n"
      "assert not loaded, loaded\n"
  )

  result = subprocess.run(
      [sys.executable, "-c", code],
      capture_output=True,
      text=True,
      check=False,
  )

  assert result.returncode == 0, result.stderr
