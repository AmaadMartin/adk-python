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

from unittest.mock import MagicMock
from unittest.mock import patch
import pytest

from google.adk.agents.invocation_context import InvocationContext
from google.adk.flows.llm_flows.base_llm_flow import BaseLlmFlow
from google.adk.cli.plugins.recordings_schema import Recordings


class BaseLlmFlowForTesting(BaseLlmFlow):
  """A concrete implementation of BaseLlmFlow for testing purposes."""
  pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_key",
    [
        "temp:_adk_replay_config",
        "_adk_replay_config",
    ],
)
async def test_base_llm_flow_get_llm_with_replay_config(config_key):
  flow = BaseLlmFlowForTesting()

  mock_agent = MagicMock()
  mock_agent.name = "test_agent"

  mock_session = MagicMock()
  mock_recordings = MagicMock(spec=Recordings)
  mock_recordings.recordings = []

  mock_session.state = {
      config_key: {
          "dir": "/tmp/test",
          "user_message_index": 0,
          "streaming_mode": "none",
          "_adk_replay_recordings": mock_recordings,
      }
  }

  mock_ctx = MagicMock(spec=InvocationContext)
  mock_ctx.agent = mock_agent
  mock_ctx.session = mock_session

  with patch(
      "google.adk.cli.conformance._conformance_test_google_llm._ConformanceTestGemini"
  ) as mock_gemini_class:
    model = flow._BaseLlmFlow__get_llm(mock_ctx)

    mock_gemini_class.assert_called_once()
    called_config = mock_gemini_class.call_args[1]["config"]
    assert called_config["dir"] == "/tmp/test"
    assert called_config["agent_name"] == "test_agent"
