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

"""Tests for E2BEnvironment."""

import inspect
import typing
from unittest import mock

import e2b
from e2b import CommandExitException
from e2b import CommandResult
from e2b import FileNotFoundException
from e2b import TimeoutException
from google.adk.integrations.e2b._e2b_environment import E2BEnvironment
import pytest


def _autospec_property(cls: type, name: str) -> mock.MagicMock:
  """Autospecs the object a property returns.

  create_autospec() does not descend into properties -- the attribute comes
  back as a bare MagicMock that accepts anything -- so the handler behind one
  has to be specced from its own declared type.
  """
  hints = typing.get_type_hints(inspect.getattr_static(cls, name).fget)
  return mock.create_autospec(hints['return'], instance=True, spec_set=True)


def _make_sandbox(*, running: bool = True) -> mock.MagicMock:
  """Build an AsyncSandbox double specced against the installed e2b SDK."""
  sandbox = mock.create_autospec(e2b.AsyncSandbox, instance=True, spec_set=True)
  sandbox.commands = _autospec_property(e2b.AsyncSandbox, 'commands')
  sandbox.files = _autospec_property(e2b.AsyncSandbox, 'files')
  # kill and set_timeout are class_method_variant descriptors. create_autospec
  # renders them as non-async mocks that still expect `self`, so awaiting one
  # raises TypeError. spec_set still rejects the name if the SDK drops it.
  sandbox.kill = mock.AsyncMock(return_value=True)
  sandbox.set_timeout = mock.AsyncMock()
  sandbox.is_running.return_value = running
  return sandbox


@pytest.fixture(name='sandbox')
def _sandbox() -> mock.MagicMock:
  return _make_sandbox()


@pytest.fixture(name='create_patch')
def _create_patch(sandbox: mock.MagicMock):
  """Patch AsyncSandbox.create to return the mock sandbox."""
  with mock.patch.object(e2b.AsyncSandbox, 'create', autospec=True) as create:
    create.return_value = sandbox
    yield create


@pytest.mark.asyncio
async def test_initialize_creates_sandbox(create_patch, sandbox):
  env = E2BEnvironment(image='custom', timeout=120, env_vars={'A': '1'})
  assert env.is_initialized is False
  await env.initialize()
  assert env.is_initialized is True

  create_patch.assert_awaited_once()
  _, kwargs = create_patch.call_args
  assert kwargs['template'] == 'custom'
  assert kwargs['timeout'] == 120
  assert kwargs['envs'] == {'A': '1'}
  assert env._sandbox is sandbox


@pytest.mark.asyncio
async def test_initialize_is_idempotent(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()
  await env.initialize()
  create_patch.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_kills_sandbox_and_is_idempotent(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()
  assert env.is_initialized is True
  await env.close()
  sandbox.kill.assert_awaited_once()
  assert env._sandbox is None
  assert env.is_initialized is False
  # Second close is a no-op.
  await env.close()
  sandbox.kill.assert_awaited_once()


@pytest.mark.asyncio
async def test_working_dir_requires_initialize():
  env = E2BEnvironment()
  with pytest.raises(RuntimeError):
    _ = env.working_dir


@pytest.mark.asyncio
async def test_execute_before_initialize_raises():
  env = E2BEnvironment()
  with pytest.raises(RuntimeError):
    await env.execute('echo hi')


@pytest.mark.asyncio
async def test_execute_success(create_patch, sandbox):
  sandbox.commands.run.return_value = CommandResult(
      stdout='out', stderr='err', exit_code=0, error=None
  )
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('echo out')

  assert result.exit_code == 0
  assert result.stdout == 'out'
  assert result.stderr == 'err'
  assert result.timed_out is False
  sandbox.set_timeout.assert_awaited()  # keepalive


@pytest.mark.asyncio
async def test_execute_nonzero_exit_is_normal_result(create_patch, sandbox):
  exc = CommandExitException(
      stdout='partial', stderr='boom', exit_code=2, error='failed'
  )
  sandbox.commands.run.side_effect = exc
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('false')

  assert result.exit_code == 2
  assert result.stdout == 'partial'
  assert result.stderr == 'boom'
  assert result.timed_out is False


@pytest.mark.asyncio
async def test_execute_timeout(create_patch, sandbox):
  sandbox.commands.run.side_effect = TimeoutException('too slow')
  env = E2BEnvironment()
  await env.initialize()

  result = await env.execute('sleep 999')

  assert result.timed_out is True


@pytest.mark.asyncio
async def test_read_file_returns_bytes(create_patch, sandbox):
  sandbox.files.read.return_value = b'data'
  env = E2BEnvironment()
  await env.initialize()

  data = await env.read_file('notes.txt')

  assert data == b'data'
  sandbox.files.read.assert_awaited_once_with(
      '/home/user/notes.txt', format='bytes'
  )


@pytest.mark.asyncio
async def test_read_file_absolute_path_passthrough(create_patch, sandbox):
  sandbox.files.read.return_value = b'x'
  env = E2BEnvironment()
  await env.initialize()

  await env.read_file('/etc/hostname')

  sandbox.files.read.assert_awaited_once_with('/etc/hostname', format='bytes')


@pytest.mark.asyncio
async def test_read_file_missing_raises(create_patch, sandbox):
  sandbox.files.read.side_effect = FileNotFoundException('nope')
  env = E2BEnvironment()
  await env.initialize()

  with pytest.raises(FileNotFoundError):
    await env.read_file('missing.txt')


@pytest.mark.asyncio
async def test_write_file_resolves_relative_path(create_patch, sandbox):
  env = E2BEnvironment()
  await env.initialize()

  await env.write_file('sub/out.txt', 'hello')

  sandbox.files.write.assert_awaited_once_with(
      '/home/user/sub/out.txt', 'hello'
  )


@pytest.mark.asyncio
async def test_keepalive_extends_timeout_when_running(create_patch, sandbox):
  sandbox.files.read.return_value = b'1'
  env = E2BEnvironment(timeout=200)
  await env.initialize()

  await env.read_file('a.txt')

  sandbox.set_timeout.assert_awaited_with(200)


@pytest.mark.asyncio
async def test_lazy_recreate_when_expired(sandbox):
  expired = _make_sandbox(running=False)
  fresh = _make_sandbox(running=True)
  fresh.files.read.return_value = b'fresh'

  with mock.patch.object(e2b.AsyncSandbox, 'create', autospec=True) as create:
    create.side_effect = [expired, fresh]
    env = E2BEnvironment()
    await env.initialize()  # -> expired
    data = await env.read_file('a.txt')  # detects dead, recreates -> fresh

  assert data == b'fresh'
  assert create.await_count == 2
  assert env._sandbox is fresh
  expired.set_timeout.assert_not_awaited()
