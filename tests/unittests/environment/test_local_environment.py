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

"""Tests for LocalEnvironment.execute, read_file and write_file."""

import asyncio
from pathlib import Path
import subprocess
import sys
from unittest import mock

from google.adk.environment._local_environment import LocalEnvironment
import pytest
import pytest_asyncio

# The real Win32 value of CREATE_NO_WINDOW, which POSIX builds do not define.
_CREATE_NO_WINDOW = 0x08000000


@pytest_asyncio.fixture(name="env")
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


def _patch_spawn(monkeypatch: pytest.MonkeyPatch) -> mock.AsyncMock:
  """Replace asyncio.create_subprocess_shell with a stub, return the stub."""
  proc = mock.AsyncMock()
  proc.returncode = 0
  proc.communicate.return_value = (b"out", b"err")
  spawn = mock.AsyncMock(return_value=proc)
  monkeypatch.setattr(asyncio, "create_subprocess_shell", spawn)
  return spawn


class TestReadFileWriteFile:
  """Verify read_file and write_file accept both str and Path arguments."""

  @pytest.mark.asyncio
  async def test_write_and_read_with_str(self, env: LocalEnvironment):
    """Round-trip a file using str paths."""
    await env.write_file("hello.txt", "hello world")
    data = await env.read_file("hello.txt")
    assert data == b"hello world"

  @pytest.mark.asyncio
  async def test_write_and_read_with_path(self, env: LocalEnvironment):
    """Round-trip a file using Path objects."""
    await env.write_file(Path("path_obj.txt"), "path content")
    data = await env.read_file(Path("path_obj.txt"))
    assert data == b"path content"

  @pytest.mark.asyncio
  async def test_write_str_read_path(self, env: LocalEnvironment):
    """Write with str, read with Path."""
    await env.write_file("mixed.txt", "mixed")
    data = await env.read_file(Path("mixed.txt"))
    assert data == b"mixed"

  @pytest.mark.asyncio
  async def test_write_path_read_str(self, env: LocalEnvironment):
    """Write with Path, read with str."""
    await env.write_file(Path("mixed2.txt"), "mixed2")
    data = await env.read_file("mixed2.txt")
    assert data == b"mixed2"

  @pytest.mark.asyncio
  async def test_write_bytes_content(self, env: LocalEnvironment):
    """Write raw bytes and read them back."""
    raw = b"\x00\x01\x02\xff"
    await env.write_file(Path("binary.bin"), raw)
    data = await env.read_file("binary.bin")
    assert data == raw

  @pytest.mark.asyncio
  async def test_write_preserves_explicit_crlf(self, env: LocalEnvironment):
    """Explicit CRLF sequences are written without newline translation."""
    await env.write_file("crlf.txt", "first\r\nsecond\r\n")

    data = await env.read_file("crlf.txt")

    assert data == b"first\r\nsecond\r\n"

  @pytest.mark.asyncio
  async def test_write_creates_parent_dirs(self, env: LocalEnvironment):
    """Parent directories are created automatically."""
    await env.write_file(Path("sub/dir/file.txt"), "nested")
    data = await env.read_file("sub/dir/file.txt")
    assert data == b"nested"

  @pytest.mark.asyncio
  async def test_absolute_path_inside_working_dir(self, env: LocalEnvironment):
    """Absolute paths are accepted when they stay inside the workspace."""
    path = env.working_dir / "absolute.txt"
    await env.write_file(path, "absolute")
    data = await env.read_file(path)
    assert data == b"absolute"

  @pytest.mark.asyncio
  async def test_rejects_relative_path_escape(self, env: LocalEnvironment):
    """Parent traversal cannot escape the workspace."""
    outside = env.working_dir.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.read_file(Path("..") / outside.name)

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.write_file(Path("..") / "write-outside.txt", "nope")

    assert not (env.working_dir.parent / "write-outside.txt").exists()

  @pytest.mark.asyncio
  async def test_rejects_absolute_path_outside_working_dir(
      self, env: LocalEnvironment
  ):
    """Absolute paths outside the workspace are rejected."""
    outside = env.working_dir.parent / "outside-absolute.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes working directory"):
      await env.read_file(outside)

  @pytest.mark.asyncio
  async def test_read_nonexistent_raises(self, env: LocalEnvironment):
    """Reading a missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
      await env.read_file(Path("does_not_exist.txt"))


class TestExecuteConsoleWindow:
  """Verify execute() suppresses the Windows console window."""

  @pytest.mark.asyncio
  async def test_execute_hides_console_window_on_windows(
      self, env: LocalEnvironment, monkeypatch: pytest.MonkeyPatch
  ):
    """On Windows the shell is spawned with CREATE_NO_WINDOW."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW, raising=False
    )
    spawn = _patch_spawn(monkeypatch)

    await env.execute("echo hi")

    assert spawn.call_args.kwargs["creationflags"] == _CREATE_NO_WINDOW

  @pytest.mark.asyncio
  async def test_execute_passes_no_creation_flags_on_posix(
      self, env: LocalEnvironment, monkeypatch: pytest.MonkeyPatch
  ):
    """On POSIX the shell is spawned with the no-op creation flags value."""
    monkeypatch.setattr(sys, "platform", "linux")
    spawn = _patch_spawn(monkeypatch)

    await env.execute("echo hi")

    assert spawn.call_args.kwargs["creationflags"] == 0

  @pytest.mark.asyncio
  async def test_execute_still_pipes_output_into_the_working_directory(
      self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
  ):
    """The pre-existing spawn arguments survive alongside the new flags."""
    environment = LocalEnvironment(
        working_dir=tmp_path, env_vars={"ADK_TEST_VAR": "set"}
    )
    await environment.initialize()
    spawn = _patch_spawn(monkeypatch)

    await environment.execute("echo hi")

    kwargs = spawn.call_args.kwargs
    assert kwargs["cwd"] == tmp_path
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["stderr"] == asyncio.subprocess.PIPE
    assert kwargs["env"]["ADK_TEST_VAR"] == "set"


class TestExecute:
  """Verify execute() runs real commands and reports real failures."""

  @pytest.mark.asyncio
  async def test_execute_runs_a_real_command(self, env: LocalEnvironment):
    """A successful command reports its exit code and captured stdout."""
    result = await env.execute("echo hello")

    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert result.timed_out is False

  @pytest.mark.asyncio
  async def test_execute_reports_a_failing_command(self, env: LocalEnvironment):
    """A failing command reports a non-zero exit code."""
    result = await env.execute(f'"{sys.executable}" -c "raise SystemExit(3)"')

    assert result.exit_code == 3
    assert result.timed_out is False

  @pytest.mark.asyncio
  async def test_execute_times_out_a_slow_command(self, env: LocalEnvironment):
    """A command that outlives its timeout is killed and flagged."""
    slow = f'"{sys.executable}" -c "import time; time.sleep(30)"'

    result = await env.execute(slow, timeout=0.5)

    assert result.timed_out is True

  @pytest.mark.asyncio
  async def test_execute_before_initialize_raises(self):
    """Executing before initialize() reports the missing working directory."""
    environment = LocalEnvironment()

    with pytest.raises(RuntimeError, match="`working_dir` is not set"):
      await environment.execute("echo hi")
