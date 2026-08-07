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

"""Tests for WriteFileTool."""

from pathlib import Path
from typing import Any

from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._write_file_tool import WriteFileTool
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(name='env')
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestWriteFileTool:
  """Tests for WriteFileTool behavior."""

  @pytest.mark.asyncio
  async def test_write_file_creates_file(self, env: LocalEnvironment):
    """Writes the content to a new file and reports the path."""
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': 'notes.txt', 'content': 'hello\n'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'message': 'Wrote notes.txt'}
    assert await env.read_file('notes.txt') == b'hello\n'

  @pytest.mark.asyncio
  async def test_write_file_creates_parent_directories(
      self, env: LocalEnvironment, tmp_path: Path
  ):
    """Creates the missing parent directories of the target path."""
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': 'a/b/c.txt', 'content': 'nested'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'message': 'Wrote a/b/c.txt'}
    assert (tmp_path / 'a' / 'b' / 'c.txt').read_text() == 'nested'

  @pytest.mark.asyncio
  async def test_write_file_overwrites_existing_content(
      self, env: LocalEnvironment
  ):
    """Replaces the whole file instead of appending to it."""
    await env.write_file('notes.txt', 'old content')
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': 'notes.txt', 'content': 'new'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'message': 'Wrote notes.txt'}
    assert await env.read_file('notes.txt') == b'new'

  @pytest.mark.asyncio
  async def test_write_file_defaults_missing_content_to_empty_file(
      self, env: LocalEnvironment
  ):
    """Creates an empty file when the caller omits `content`."""
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': 'empty.txt'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'message': 'Wrote empty.txt'}
    assert await env.read_file('empty.txt') == b''

  @pytest.mark.parametrize(
      'args',
      [{}, {'path': '', 'content': 'ignored'}],
      ids=['missing_path', 'empty_path'],
  )
  @pytest.mark.asyncio
  async def test_write_file_requires_path(
      self, env: LocalEnvironment, tmp_path: Path, args: dict[str, str]
  ):
    """Rejects a missing or empty path before touching the filesystem."""
    tool = WriteFileTool(env)

    result = await tool.run_async(args=args, tool_context=None)

    assert result == {'status': 'error', 'error': '`path` is required.'}
    assert list(tmp_path.iterdir()) == []

  @pytest.mark.asyncio
  async def test_write_file_rejects_relative_path_traversal(
      self, env: LocalEnvironment, tmp_path: Path
  ):
    """A relative traversal path writes nothing outside the working dir."""
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': '../escape.txt', 'content': 'owned'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': 'Path escapes working directory: ../escape.txt',
    }
    assert not (tmp_path.parent / 'escape.txt').exists()
    assert list(tmp_path.iterdir()) == []

  @pytest.mark.asyncio
  async def test_write_file_rejects_absolute_path_outside_working_dir(
      self, env: LocalEnvironment, tmp_path: Path
  ):
    """An absolute path outside the working dir writes nothing."""
    outside = tmp_path.parent / 'outside.txt'
    tool = WriteFileTool(env)

    result = await tool.run_async(
        args={'path': str(outside), 'content': 'owned'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': f'Path escapes working directory: {outside}',
    }
    assert not outside.exists()
    assert list(tmp_path.iterdir()) == []

  @pytest.mark.asyncio
  async def test_write_file_surfaces_environment_failure_as_error(self):
    """An environment failure becomes a structured error, not an exception."""
    tool = WriteFileTool(LocalEnvironment())

    result = await tool.run_async(
        args={'path': 'notes.txt', 'content': 'x'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': '`working_dir` is not set. Call initialize() first.',
    }


def test_detect_error_in_response_flags_error_status():
  """The telemetry hook reports an error payload as a tool error."""
  tool = WriteFileTool(LocalEnvironment())

  detected = tool._detect_error_in_response({
      'status': 'error',
      'error': 'boom',
  })

  assert detected == 'TOOL_ERROR'


@pytest.mark.parametrize(
    'response',
    [{'status': 'ok', 'message': 'Wrote a.txt'}, 'plain string', None],
    ids=['ok_payload', 'string', 'none'],
)
def test_detect_error_in_response_ignores_non_error_payloads(response: Any):
  """The telemetry hook reports nothing for payloads without an error status."""
  tool = WriteFileTool(LocalEnvironment())

  assert tool._detect_error_in_response(response) is None
