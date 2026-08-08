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

"""Tests for EditFileTool.

Verifies that EditFileTool correctly handles line break differences.
"""

from pathlib import Path

from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._edit_file_tool import EditFileTool
import pytest
import pytest_asyncio


@pytest_asyncio.fixture(name="env")
async def _env(tmp_path: Path):
  """Create and initialize a LocalEnvironment backed by a temp directory."""
  environment = LocalEnvironment(working_dir=tmp_path)
  await environment.initialize()
  yield environment
  await environment.close()


@pytest_asyncio.fixture(name="nested_env")
async def _nested_env(tmp_path: Path):
  """A LocalEnvironment rooted one level below ``tmp_path``.

  Leaves ``tmp_path`` itself outside the working directory, so a test can put
  a file there and try to reach it with an escaping path.
  """
  environment = LocalEnvironment(working_dir=tmp_path / "workspace")
  await environment.initialize()
  yield environment
  await environment.close()


class TestEditFileTool:
  """Tests for EditFileTool behavior."""

  @pytest.mark.asyncio
  async def test_edit_file_handles_line_breaks_linux_file_windows_search(
      self, env: LocalEnvironment
  ):
    """File has \\n, search string has \\r\\n."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\r\nline2",
        "new_string": "line1_replaced\nline2_replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"line1_replaced\nline2_replaced\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_handles_line_breaks_windows_file_linux_search(
      self, env: LocalEnvironment
  ):
    """File has \\r\\n, search string has \\n."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\r\nline2\r\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "line1_replaced\r\nline2_replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"line1_replaced\r\nline2_replaced\r\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_fails_on_multiple_matches(
      self, env: LocalEnvironment
  ):
    """Tool fails if old_string appears multiple times."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline1\nline2")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "error"
    assert "appears 2 times" in result["error"]

  @pytest.mark.asyncio
  async def test_edit_file_exact_match_works(self, env: LocalEnvironment):
    """Exact match works as before."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1\nline2\nline3")

    args = {
        "path": "test.txt",
        "old_string": "line1\nline2",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"replaced\nline3"

  @pytest.mark.asyncio
  async def test_edit_file_handles_special_regex_chars(
      self, env: LocalEnvironment
  ):
    """Special regex characters in old_string are escaped."""
    # Arrange
    tool = EditFileTool(env)
    await env.write_file("test.txt", "line1.content\nline2")

    args = {
        "path": "test.txt",
        "old_string": "line1.content",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "ok"
    data = await env.read_file("test.txt")
    assert data == b"replaced\nline2"

  @pytest.mark.asyncio
  async def test_edit_file_rejects_relative_path_escaping_working_dir(
      self, nested_env: LocalEnvironment, tmp_path: Path
  ):
    """A relative path that climbs out of the working dir is a tool error."""
    # Arrange
    tool = EditFileTool(nested_env)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    args = {
        "path": "../outside.txt",
        "old_string": "secret",
        "new_string": "leaked",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "error"
    assert "escapes working directory" in result["error"]
    assert outside.read_text() == "secret"

  @pytest.mark.asyncio
  async def test_edit_file_rejects_absolute_path_outside_working_dir(
      self, nested_env: LocalEnvironment, tmp_path: Path
  ):
    """An absolute path outside the working dir is a tool error."""
    # Arrange
    tool = EditFileTool(nested_env)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    args = {
        "path": str(outside),
        "old_string": "secret",
        "new_string": "leaked",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "error"
    assert "escapes working directory" in result["error"]
    assert outside.read_text() == "secret"

  @pytest.mark.asyncio
  async def test_edit_file_rejects_symlink_pointing_outside_working_dir(
      self, nested_env: LocalEnvironment, tmp_path: Path
  ):
    """A symlink inside the working dir is resolved before the containment check."""
    # Arrange
    tool = EditFileTool(nested_env)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    (nested_env.working_dir / "link.txt").symlink_to(outside)

    args = {
        "path": "link.txt",
        "old_string": "secret",
        "new_string": "leaked",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result["status"] == "error"
    assert "escapes working directory" in result["error"]
    assert outside.read_text() == "secret"

  @pytest.mark.asyncio
  async def test_edit_file_reports_write_failure_as_error(
      self, env: LocalEnvironment
  ):
    """A failing write is reported as a tool error, and the file is untouched."""

    # Arrange
    class _WriteFailingEnvironment(LocalEnvironment):
      """Reads normally; every write fails."""

      async def write_file(self, path: str | Path, content: str | bytes):
        raise OSError("no space left on device")

    await env.write_file("test.txt", "line1\nline2")
    failing_env = _WriteFailingEnvironment(working_dir=env.working_dir)
    await failing_env.initialize()
    tool = EditFileTool(failing_env)

    args = {
        "path": "test.txt",
        "old_string": "line1",
        "new_string": "replaced",
    }

    # Act
    result = await tool.run_async(args=args, tool_context=None)

    # Assert
    assert result == {"status": "error", "error": "no space left on device"}
    assert await env.read_file("test.txt") == b"line1\nline2"
    await failing_env.close()

  def test_detect_error_in_response_reports_error_status(self):
    """An error result is classified as a tool error for telemetry."""
    # Arrange
    tool = EditFileTool(LocalEnvironment())

    # Act
    detected = tool._detect_error_in_response(
        {"status": "error", "error": "boom"}
    )

    # Assert
    assert detected == "TOOL_ERROR"

  def test_detect_error_in_response_ignores_ok_status(self):
    """A successful result is not classified as a tool error."""
    # Arrange
    tool = EditFileTool(LocalEnvironment())

    # Act
    detected = tool._detect_error_in_response({"status": "ok"})

    # Assert
    assert detected is None

  def test_detect_error_in_response_ignores_non_dict(self):
    """A non-dict response is not classified, even if it reads as an error."""
    # Arrange
    tool = EditFileTool(LocalEnvironment())

    # Act
    detected = tool._detect_error_in_response("error")

    # Assert
    assert detected is None
