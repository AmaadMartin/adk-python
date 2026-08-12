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

"""Fresh-process checks that the MCP connection params models are complete.

Pydantic rebuilds an incomplete model lazily on first use, so an in-process
assertion passes as soon as an earlier test builds one of these models. Every
check here therefore runs in an interpreter that has just imported the module.
"""

from __future__ import annotations

from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
import pydantic
import pytest

from ... import isolated_import_utils
from ...isolated_import_utils import run_isolated

pytestmark = pytest.mark.skipif(
    not isolated_import_utils.SOURCE_ROOT.is_dir(),
    reason='Import-time model checks need the source checkout layout.',
)

_MODELS = (
    'StdioConnectionParams',
    'SseConnectionParams',
    'StreamableHTTPConnectionParams',
)


@pytest.mark.parametrize('model_name', _MODELS)
def test_connection_params_complete_right_after_import(model_name):
  result = run_isolated(f"""
from google.adk.tools.mcp_tool import mcp_session_manager

model = mcp_session_manager.{model_name}
assert model.__pydantic_complete__ is True, (
    '{model_name} is not fully defined right after import'
)
""")
  assert result.returncode == 0, result.stderr


def test_sse_connection_params_is_subclassable_from_another_module():
  result = run_isolated("""
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams


class CustomSseConnectionParams(SseConnectionParams):
  retries: int = 3


params = CustomSseConnectionParams(url='http://localhost:8080/sse')
assert params.retries == 3
""")
  assert result.returncode == 0, result.stderr


def test_sse_connection_params_rejects_a_non_factory_client_factory():
  """The resolved annotation must still be the is-instance Protocol check."""
  with pytest.raises(pydantic.ValidationError, match='httpx_client_factory'):
    SseConnectionParams(
        url='http://localhost:8080/sse', httpx_client_factory=123
    )
