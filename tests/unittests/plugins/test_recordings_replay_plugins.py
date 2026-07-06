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

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.cli.plugins.recordings_plugin import RecordingsPlugin
from google.adk.cli.plugins.recordings_schema import Recording
from google.adk.cli.plugins.recordings_schema import Recordings
from google.adk.cli.plugins.replay_plugin import ReplayConfigError
from google.adk.cli.plugins.replay_plugin import ReplayPlugin
import pytest
import yaml

# --- RecordingsPlugin Tests ---


@pytest.mark.asyncio
async def test_recordings_plugin_record_mode_on_with_temp_prefix(tmp_path):
  plugin = RecordingsPlugin()

  # Mock context with temp: prefixed config
  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.invocation_id = "test_inv"
  mock_ctx.state = {
      "temp:_adk_recordings_config": {
          "dir": str(tmp_path),
          "user_message_index": 0,
          "streaming_mode": "none",
      }
  }

  # Verify mode is detected as ON
  assert plugin._is_record_mode_on(mock_ctx) is True

  # Verify state creation works
  state = plugin._create_invocation_state(mock_ctx)
  assert state.test_case_path == str(tmp_path)
  assert state.user_message_index == 0


@pytest.mark.asyncio
async def test_recordings_plugin_record_mode_on_without_temp_prefix(tmp_path):
  plugin = RecordingsPlugin()

  # Mock context with non-prefixed config (backward compatibility)
  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.invocation_id = "test_inv"
  mock_ctx.state = {
      "_adk_recordings_config": {
          "dir": str(tmp_path),
          "user_message_index": 0,
          "streaming_mode": "none",
      }
  }

  assert plugin._is_record_mode_on(mock_ctx) is True
  state = plugin._create_invocation_state(mock_ctx)
  assert state.test_case_path == str(tmp_path)


@pytest.mark.asyncio
async def test_recordings_plugin_record_mode_off():
  plugin = RecordingsPlugin()
  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.state = {}

  assert plugin._is_record_mode_on(mock_ctx) is False


# --- ReplayPlugin Tests ---


@pytest.mark.asyncio
async def test_replay_plugin_replay_mode_on_with_temp_prefix(tmp_path):
  plugin = ReplayPlugin()

  # Create a dummy recordings file
  recordings_file = tmp_path / "generated-recordings.yaml"
  recordings_data = Recordings(recordings=[])
  with open(recordings_file, "w") as f:
    yaml.dump(recordings_data.model_dump(), f)

  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.invocation_id = "test_inv"
  mock_ctx.state = {
      "temp:_adk_replay_config": {
          "dir": str(tmp_path),
          "user_message_index": 0,
          "streaming_mode": "none",
      }
  }

  assert plugin._is_replay_mode_on(mock_ctx) is True

  state = plugin._load_invocation_state(mock_ctx)
  assert state.test_case_path == str(tmp_path)
  assert state.user_message_index == 0


@pytest.mark.asyncio
async def test_replay_plugin_replay_mode_on_without_temp_prefix(tmp_path):
  plugin = ReplayPlugin()

  recordings_file = tmp_path / "generated-recordings.yaml"
  recordings_data = Recordings(recordings=[])
  with open(recordings_file, "w") as f:
    yaml.dump(recordings_data.model_dump(), f)

  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.invocation_id = "test_inv"
  mock_ctx.state = {
      "_adk_replay_config": {
          "dir": str(tmp_path),
          "user_message_index": 0,
          "streaming_mode": "none",
      }
  }

  assert plugin._is_replay_mode_on(mock_ctx) is True
  state = plugin._load_invocation_state(mock_ctx)
  assert state.test_case_path == str(tmp_path)


@pytest.mark.asyncio
async def test_replay_plugin_replay_mode_off():
  plugin = ReplayPlugin()
  mock_ctx = MagicMock(spec=CallbackContext)
  mock_ctx.state = {}

  assert plugin._is_replay_mode_on(mock_ctx) is False
