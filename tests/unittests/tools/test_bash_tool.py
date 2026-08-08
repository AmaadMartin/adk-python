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

import asyncio
import os
import signal
import sys
from unittest import mock

import pytest

if sys.platform == "win32":
  pytest.skip(
      "bash tool tests require Unix resource module", allow_module_level=True
  )

import resource

from google.adk.tools import bash_tool
from google.adk.tools import tool_context
from google.adk.tools.tool_confirmation import ToolConfirmation


@pytest.fixture
def workspace(tmp_path):
  """Creates a workspace mirroring the anthropics/skills PDF skill layout."""
  # Skill: pdf/
  skill_dir = tmp_path / "pdf"
  skill_dir.mkdir()
  (skill_dir / "SKILL.md").write_text(
      "---\nname: pdf\n"
      "description: Use this skill whenever the user wants to do"
      " anything with PDF files.\n"
      "---\n# PDF Processing Guide\n\n## Overview\n"
      "This guide covers PDF processing operations."
  )
  scripts = skill_dir / "scripts"
  scripts.mkdir()
  (scripts / "extract_form_structure.py").write_text(
      "import sys; print(f'extracting from {sys.argv[1]}')"
  )
  (scripts / "fill_pdf_form_with_annotations.py").write_text(
      "print('filling form')"
  )
  references = skill_dir / "references"
  references.mkdir()
  (references / "REFERENCE.md").write_text("# Reference\nDetailed docs.")
  # A loose file at workspace root (not inside a skill).
  (tmp_path / "sample.pdf").write_bytes(b"%PDF-1.4 fake")
  return tmp_path


@pytest.fixture
def tool_context_no_confirmation():
  """ToolContext with no confirmation (initial call)."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  ctx.tool_confirmation = None
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_confirmed():
  """ToolContext with confirmation approved."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = True
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


@pytest.fixture
def tool_context_rejected():
  """ToolContext with confirmation rejected."""
  ctx = mock.create_autospec(tool_context.ToolContext, instance=True)
  confirmation = mock.create_autospec(ToolConfirmation, instance=True)
  confirmation.confirmed = False
  ctx.tool_confirmation = confirmation
  ctx.actions = mock.MagicMock()
  return ctx


# --- _validate_command tests ---


class TestValidateCommand:

  def test_empty_command(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("", policy) is not None
    assert bash_tool._validate_command("   ", policy) is not None

  def test_default_policy_allows_everything(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("rm -rf /", policy) is None
    assert bash_tool._validate_command("cat /etc/passwd", policy) is None
    assert bash_tool._validate_command("sudo curl", policy) is None
    assert bash_tool._validate_command("echo hello | grep h", policy) is None
    assert bash_tool._validate_command("ls ; rm -rf /", policy) is None

  def test_restricted_policy_allows_prefixes(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("ls -la", policy) is None
    assert bash_tool._validate_command("cat file.txt", policy) is None

  def test_restricted_policy_blocks_others(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls", "cat"))
    assert bash_tool._validate_command("rm -rf .", policy) is not None
    assert bash_tool._validate_command("tree", policy) is not None
    assert "Permitted prefixes are: ls, cat" in bash_tool._validate_command(
        "tree", policy
    )

  def test_blocked_operators_validation(self):
    policy = bash_tool.BashToolPolicy(
        allowed_command_prefixes=("*",),
        blocked_operators=("|", ";", "$(", "`", "&&", "||"),
    )
    assert (
        bash_tool._validate_command("echo hello | grep h", policy)
        == "Command contains blocked operator: |"
    )
    assert (
        bash_tool._validate_command("ls ; rm -rf /", policy)
        == "Command contains blocked operator: ;"
    )


class TestExecuteBashTool:

  @pytest.mark.asyncio
  async def test_requests_confirmation(
      self, workspace, tool_context_no_confirmation
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "requires confirmation" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_called_once()

  @pytest.mark.asyncio
  async def test_rejected(self, workspace, tool_context_rejected):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"}, tool_context=tool_context_rejected
    )
    assert result == {"error": "This tool call is rejected."}

  @pytest.mark.asyncio
  async def test_executes_when_confirmed(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "ls"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert "pdf" in result["stdout"]

  @pytest.mark.asyncio
  async def test_cat_skill_md(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "cat pdf/SKILL.md"},
        tool_context=tool_context_confirmed,
    )
    assert "PDF Processing Guide" in result["stdout"]

  @pytest.mark.asyncio
  async def test_python_script(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={
            "command": "python3 pdf/scripts/extract_form_structure.py test.pdf"
        },
        tool_context=tool_context_confirmed,
    )
    assert "extracting from test.pdf" in result["stdout"]
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_blocks_disallowed_by_policy(
      self, workspace, tool_context_no_confirmation
  ):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={"command": "rm -rf ."},
        tool_context=tool_context_no_confirmation,
    )
    assert "error" in result
    assert "Permitted prefixes are: ls" in result["error"]
    tool_context_no_confirmation.request_confirmation.assert_not_called()

  @pytest.mark.asyncio
  async def test_captures_stderr(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import sys; sys.stderr.write(\"err\")'"},
        tool_context=tool_context_confirmed,
    )
    # An equality check, because a substring check would also pass if the
    # subprocess never ran and the tool reported the failure instead.
    assert "error" not in result
    assert result["stderr"].strip() == "err"
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_spawn_failure_reports_error(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    with mock.patch.object(
        asyncio,
        "create_subprocess_exec",
        autospec=True,
        side_effect=FileNotFoundError(
            2, "No such file or directory", "python3"
        ),
    ):
      result = await tool.run_async(
          args={"command": "python3 -c 'pass'"},
          tool_context=tool_context_confirmed,
      )
    # An attempted execution reports the same keys however it ended, so a
    # caller can read result["returncode"] without first testing the key.
    assert result == {
        "error": "Command not found: python3",
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }

  @pytest.mark.asyncio
  async def test_spawn_failure_permission_denied(
      self, workspace, tool_context_confirmed
  ):
    script = workspace / "noexec.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    os.chmod(script, 0o644)
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": str(script)},
        tool_context=tool_context_confirmed,
    )
    assert result == {
        "error": f"Permission denied: {script}",
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }

  @pytest.mark.asyncio
  async def test_spawn_failure_missing_workspace(
      self, workspace, tool_context_confirmed
  ):
    missing = workspace / "gone"
    tool = bash_tool.ExecuteBashTool(workspace=missing)
    result = await tool.run_async(
        args={"command": "echo hi"},
        tool_context=tool_context_confirmed,
    )
    # The executable exists; only the working directory is missing, so the
    # tool must not report "Command not found: echo".
    assert result == {
        "error": f"Workspace directory is not accessible: {missing}",
        "stdout": "",
        "stderr": "",
        "returncode": None,
    }

  @pytest.mark.asyncio
  async def test_empty_output_is_empty_strings(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'pass'"},
        tool_context=tool_context_confirmed,
    )
    assert result == {"stdout": "", "stderr": "", "returncode": 0}
    # A placeholder such as "<no stderr captured>" would make this fragment
    # match output the process never produced.
    assert "err" not in result["stderr"]

  @pytest.mark.asyncio
  async def test_unparseable_command_keeps_result_shape(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": 'echo "unbalanced'},
        tool_context=tool_context_confirmed,
    )
    assert set(result) == {"error", "stdout", "stderr", "returncode"}
    assert result["error"] == "Execution failed: No closing quotation"
    assert result["returncode"] is None

  @pytest.mark.asyncio
  async def test_undecodable_output_is_replaced(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={
            "command": (
                "python3 -c 'import sys; sys.stdout.buffer.write(b\"\\xff\")'"
            )
        },
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"] == "\ufffd"
    assert result["returncode"] == 0

  @pytest.mark.asyncio
  async def test_nonzero_returncode(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'exit(42)'"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 42

  @pytest.mark.asyncio
  async def test_timeout(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    mock_process = mock.AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = -9
    mock_process.communicate.return_value = (b"", b"")
    with (
        mock.patch.object(
            asyncio,
            "create_subprocess_exec",
            autospec=True,
            return_value=mock_process,
        ),
        mock.patch.object(
            asyncio, "wait_for", autospec=True, side_effect=asyncio.TimeoutError
        ),
        mock.patch("os.killpg") as mock_killpg,
    ):
      result = await tool.run_async(
          args={"command": "python scripts/do_thing.py"},
          tool_context=tool_context_confirmed,
      )
      mock_killpg.assert_called_with(12345, signal.SIGKILL)
    assert set(result) == {"error", "stdout", "stderr", "returncode"}
    assert "timed out" in result["error"].lower()
    assert result["returncode"] == -9

  @pytest.mark.asyncio
  async def test_cwd_is_workspace(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import os; print(os.getcwd())'"},
        tool_context=tool_context_confirmed,
    )
    assert result["stdout"].strip() == str(workspace)

  @pytest.mark.asyncio
  async def test_no_command(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(args={}, tool_context=tool_context_confirmed)
    assert "error" in result
    assert "required" in result["error"].lower()

  @pytest.mark.asyncio
  async def test_resource_limits_set(self, workspace, tool_context_confirmed):
    policy = bash_tool.BashToolPolicy(
        max_memory_bytes=100 * 1024 * 1024,
        max_file_size_bytes=50 * 1024 * 1024,
        max_child_processes=10,
    )
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    mock_process = mock.AsyncMock()
    mock_process.pid = None  # Ensure finally block doesn't try to kill it
    mock_process.communicate.return_value = (b"", b"")
    mock_exec = mock.AsyncMock(return_value=mock_process)

    with mock.patch("asyncio.create_subprocess_exec", mock_exec):
      await tool.run_async(
          args={"command": "ls"},
          tool_context=tool_context_confirmed,
      )
      assert "preexec_fn" in mock_exec.call_args.kwargs
      preexec_fn = mock_exec.call_args.kwargs["preexec_fn"]

      mock_setrlimit = mock.create_autospec(resource.setrlimit, instance=True)
      with mock.patch("resource.setrlimit", mock_setrlimit):
        preexec_fn()
        mock_setrlimit.assert_any_call(resource.RLIMIT_CORE, (0, 0))
        mock_setrlimit.assert_any_call(
            resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024)
        )
        mock_setrlimit.assert_any_call(
            resource.RLIMIT_FSIZE, (50 * 1024 * 1024, 50 * 1024 * 1024)
        )
