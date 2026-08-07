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

"""Tests for ExecuteTool."""

from pathlib import Path
from typing import Any
from typing import Optional

from google.adk.environment._base_environment import BaseEnvironment
from google.adk.environment._base_environment import ExecutionResult
from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._execute_tool import ExecuteTool
import pytest
import pytest_asyncio


class _StubEnvironment(BaseEnvironment):
  """Environment double that returns a canned execution result."""

  def __init__(self, execution_result: ExecutionResult):
    self._execution_result = execution_result
    self.execute_calls: list[tuple[str, Optional[float]]] = []

  @property
  def working_dir(self) -> Path:
    return Path('/tmp/adk-test')

  async def execute(
      self,
      command: str,
      *,
      timeout: Optional[float] = None,
  ) -> ExecutionResult:
    self.execute_calls.append((command, timeout))
    return self._execution_result

  async def read_file(self, path: Path) -> bytes:
    del path
    raise AssertionError('ExecuteTool should not invoke read_file().')

  async def write_file(self, path: Path, content: str | bytes) -> None:
    del path, content
    raise AssertionError('ExecuteTool should not invoke write_file().')


@pytest_asyncio.fixture(name='env')
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


class TestExecuteTool:
  """Tests for ExecuteTool behavior."""

  @pytest.mark.asyncio
  async def test_execute_returns_stdout_when_stderr_empty(
      self, env: LocalEnvironment
  ):
    """Omits the `stderr` key when the command writes nothing to stderr."""
    tool = ExecuteTool(env)

    result = await tool.run_async(
        args={'command': 'echo hi'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'stdout': 'hi\n'}

  @pytest.mark.asyncio
  async def test_execute_returns_stderr_when_stdout_empty(
      self, env: LocalEnvironment
  ):
    """Omits the `stdout` key when the command writes nothing to stdout."""
    tool = ExecuteTool(env)

    result = await tool.run_async(
        args={'command': 'echo boom >&2'},
        tool_context=None,
    )

    assert result == {'status': 'ok', 'stderr': 'boom\n'}

  @pytest.mark.asyncio
  async def test_execute_omits_both_streams_when_command_is_silent(
      self, env: LocalEnvironment
  ):
    """Omits both stream keys when the command produces no output."""
    tool = ExecuteTool(env)

    result = await tool.run_async(
        args={'command': 'true'},
        tool_context=None,
    )

    assert result == {'status': 'ok'}

  @pytest.mark.asyncio
  async def test_execute_non_zero_exit_sets_error_status_and_exit_code(
      self, env: LocalEnvironment
  ):
    """Reports a non-zero exit code as an error without an `error` key."""
    tool = ExecuteTool(env)

    result = await tool.run_async(
        args={'command': 'exit 3'},
        tool_context=None,
    )

    assert result == {'status': 'error', 'exit_code': 3}

  @pytest.mark.asyncio
  async def test_execute_non_zero_exit_keeps_stdout_and_stderr(
      self, env: LocalEnvironment
  ):
    """Keeps both streams alongside the exit code of a failing command."""
    tool = ExecuteTool(env)

    result = await tool.run_async(
        args={'command': "printf 'out' && printf 'err' >&2 && exit 7"},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'stdout': 'out',
        'stderr': 'err',
        'exit_code': 7,
    }

  @pytest.mark.parametrize(
      'args',
      [{}, {'command': ''}],
      ids=['missing_command', 'empty_command'],
  )
  @pytest.mark.asyncio
  async def test_execute_requires_command(
      self, env: LocalEnvironment, args: dict[str, str]
  ):
    """Rejects a missing or empty command."""
    tool = ExecuteTool(env)

    result = await tool.run_async(args=args, tool_context=None)

    assert result == {'status': 'error', 'error': '`command` is required.'}

  @pytest.mark.asyncio
  async def test_execute_timeout_reports_timeout_error(self):
    """Reports the timeout message and the exit code of the killed process."""
    environment = _StubEnvironment(
        ExecutionResult(exit_code=-9, stdout='', stderr='', timed_out=True)
    )
    tool = ExecuteTool(environment)

    result = await tool.run_async(
        args={'command': 'sleep 60'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'exit_code': -9,
        'error': 'Command timed out after 30s.',
    }
    assert environment.execute_calls == [('sleep 60', 30)]

  @pytest.mark.asyncio
  async def test_execute_timeout_preserves_captured_output(self):
    """Keeps the output captured before the process was killed."""
    environment = _StubEnvironment(
        ExecutionResult(
            exit_code=-9, stdout='partial', stderr='', timed_out=True
        )
    )
    tool = ExecuteTool(environment)

    result = await tool.run_async(
        args={'command': 'sleep 60'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'stdout': 'partial',
        'exit_code': -9,
        'error': 'Command timed out after 30s.',
    }

  @pytest.mark.asyncio
  async def test_execute_surfaces_environment_failure_as_error(self):
    """An environment failure becomes a structured error, not an exception."""
    tool = ExecuteTool(LocalEnvironment())

    result = await tool.run_async(
        args={'command': 'echo hi'},
        tool_context=None,
    )

    assert result == {
        'status': 'error',
        'error': '`working_dir` is not set. Call initialize() first.',
    }


def test_detect_error_in_response_flags_error_status():
  """The telemetry hook reports an error payload as a tool error."""
  tool = ExecuteTool(LocalEnvironment())

  detected = tool._detect_error_in_response({
      'status': 'error',
      'error': 'boom',
  })

  assert detected == 'TOOL_ERROR'


@pytest.mark.parametrize(
    'response',
    [{'status': 'ok', 'stdout': 'hi\n'}, 'plain string', None],
    ids=['ok_payload', 'string', 'none'],
)
def test_detect_error_in_response_ignores_non_error_payloads(response: Any):
  """The telemetry hook reports nothing for payloads without an error status."""
  tool = ExecuteTool(LocalEnvironment())

  assert tool._detect_error_in_response(response) is None
