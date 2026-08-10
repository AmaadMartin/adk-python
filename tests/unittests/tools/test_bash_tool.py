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
import shlex
import signal
import subprocess
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

  def test_default_policy_allows_plain_commands(self):
    policy = bash_tool.BashToolPolicy()
    assert bash_tool._validate_command("rm -rf /", policy) is None
    assert bash_tool._validate_command("cat /etc/passwd", policy) is None
    assert bash_tool._validate_command("sudo curl", policy) is None

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


# Argument strings the validator accepts, checked against a real shell by
# TestShellSyntaxRejection.test_accepted_arguments_split_exactly_as_a_shell_would.
_FAITHFULLY_SPLIT_ARGUMENTS = [
    "-la src",
    "'|'",
    '"a|b"',
    "'$HOME'",
    "'a > b'",
    '"a; b | c"',
    "a\\|b",
    '"a\\"b"',
    "'it'\\''s'",
    "https://example.com/x?a=b",
    "{}",
    "[abc]",
    "a~b",
    "http://example.com/page#top",
    "'~'",
    "\\~",
    "a\\ ~",
]


class TestShellSyntaxRejection:
  """The tool runs a single program, so unhonoured syntax must be refused."""

  @pytest.mark.parametrize(
      "command, char",
      [
          ("echo hello | grep h", "|"),
          ("ls ; rm -rf /", ";"),
          ("a && b", "&"),
          ("a || b", "|"),
          ("cat f > out.txt", ">"),
          ("cat < f", "<"),
          ("sleep 1 &", "&"),
          ("echo $(whoami)", "$"),
          ("echo `whoami`", "`"),
          ("(cd /tmp)", "("),
          ("ls *.py", "*"),
      ],
  )
  def test_unquoted_shell_syntax_is_rejected(self, command, char):
    error = bash_tool._validate_command(command, bash_tool.BashToolPolicy())
    assert error is not None
    assert f"'{char}' is shell syntax" in error

  def test_newline_is_reported_without_breaking_the_message(self):
    error = bash_tool._validate_command(
        "ls\nrm -rf /", bash_tool.BashToolPolicy()
    )
    assert error is not None
    assert "'\\n' is shell syntax" in error
    assert "\n" not in error

  @pytest.mark.parametrize(
      "command",
      [
          # The exact command used by TestExecuteBashTool.test_captures_stderr.
          "python3 -c 'import sys; sys.stderr.write(\"err\")'",
          "echo '|'",
          'grep "a|b" file',
          "echo '$HOME'",
          "echo 'a > b'",
          # A shell leaves these literal inside double quotes, so we agree.
          'echo "a; b | c"',
          # A shell also passes a backslash-escaped operator through literally.
          "echo a\\|b",
      ],
  )
  def test_quoted_or_escaped_metacharacters_are_accepted(self, command):
    assert (
        bash_tool._validate_command(command, bash_tool.BashToolPolicy()) is None
    )

  @pytest.mark.parametrize(
      "command, syntax",
      [
          ('echo "$HOME"', "$"),
          ('echo "`id`"', "`"),
          # A shell drops the backslash and still expands; shlex keeps it.
          ('echo "a\\$b"', "\\$"),
          ('echo "a\\`b"', "\\`"),
      ],
  )
  def test_expansion_inside_double_quotes_is_rejected(self, command, syntax):
    error = bash_tool._validate_command(command, bash_tool.BashToolPolicy())
    assert error is not None
    assert f"'{syntax}' is shell syntax" in error

  @pytest.mark.parametrize(
      "command, char",
      [
          ("ls ~/src", "~"),
          ("mkdir ~", "~"),
          ("echo a #comment", "#"),
          ("#comment", "#"),
      ],
  )
  def test_word_start_expansion_is_rejected(self, command, char):
    error = bash_tool._validate_command(command, bash_tool.BashToolPolicy())
    assert error is not None
    assert f"'{char}' is shell syntax" in error

  @pytest.mark.parametrize(
      "command",
      [
          # A shell only acts on ~ and # at the start of a word.
          "grep a~b file",
          "curl https://example.com/page#section",
          "echo '~'",
          'echo "#c"',
          "echo \\~",
          # An escaped space keeps the next character inside the same word.
          "echo a\\ ~",
      ],
  )
  def test_word_start_characters_are_accepted_elsewhere(self, command):
    assert (
        bash_tool._validate_command(command, bash_tool.BashToolPolicy()) is None
    )

  @pytest.mark.parametrize("arguments", _FAITHFULLY_SPLIT_ARGUMENTS)
  def test_accepted_arguments_split_exactly_as_a_shell_would(self, arguments):
    """An accepted command must mean the same to execve and to a shell."""
    command = f"echo {arguments}"
    assert (
        bash_tool._validate_command(command, bash_tool.BashToolPolicy()) is None
    )
    # printf writes each word a real shell parsed, NUL separated.
    shell = subprocess.run(
        ["bash", "-c", f'printf "%s\\0" {arguments}'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert shlex.split(command)[1:] == shell.stdout.split("\0")[:-1]

  @pytest.mark.parametrize(
      "command", ["echo 'unterminated", 'echo "x', "echo a\\"]
  )
  def test_unterminated_quote_or_trailing_backslash_is_rejected(self, command):
    error = bash_tool._validate_command(command, bash_tool.BashToolPolicy())
    assert error is not None
    assert "ends inside a quote or with a trailing backslash" in error

  def test_opt_out_restores_literal_passthrough(self):
    policy = bash_tool.BashToolPolicy(reject_shell_syntax=False)
    assert bash_tool._validate_command("echo hello | grep h", policy) is None
    assert bash_tool._validate_command("ls ; rm -rf /", policy) is None

  def test_positional_construction_defaults_to_rejecting(self):
    policy = bash_tool.BashToolPolicy(("ls",), ("|",), 60)
    assert policy.reject_shell_syntax is True

  def test_blocked_operator_message_wins(self):
    policy = bash_tool.BashToolPolicy(blocked_operators=("|",))
    assert (
        bash_tool._validate_command("echo hello | grep h", policy)
        == "Command contains blocked operator: |"
    )

  def test_shell_syntax_message_wins_over_prefix_check(self):
    policy = bash_tool.BashToolPolicy(allowed_command_prefixes=("ls",))
    error = bash_tool._validate_command("ls | wc -l", policy)
    assert error is not None
    assert "'|' is shell syntax" in error
    assert "Permitted prefixes" not in error


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
  async def test_pipe_is_rejected_without_executing(
      self, workspace, tool_context_confirmed
  ):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    mock_exec = mock.AsyncMock()
    with mock.patch("asyncio.create_subprocess_exec", mock_exec):
      result = await tool.run_async(
          args={"command": "echo hello | grep h"},
          tool_context=tool_context_confirmed,
      )
    mock_exec.assert_not_called()
    assert "without a shell" in result["error"]
    assert "returncode" not in result

  @pytest.mark.asyncio
  async def test_metacharacters_execute_literally_when_opted_out(
      self, workspace, tool_context_confirmed
  ):
    policy = bash_tool.BashToolPolicy(reject_shell_syntax=False)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    result = await tool.run_async(
        args={"command": "echo hello | grep h"},
        tool_context=tool_context_confirmed,
    )
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello | grep h"

  def test_declaration_states_no_shell(self, workspace):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    declaration = tool._get_declaration()
    assert declaration is not None
    description = declaration.description
    assert description is not None
    assert "without a shell" in description
    assert "Shell syntax is not interpreted and is rejected" in description
    assert declaration.parameters_json_schema is not None
    command = declaration.parameters_json_schema["properties"]["command"]
    assert "bash command" not in command["description"]
    assert "not a shell command line" in command["description"]

  def test_declaration_reflects_opt_out(self, workspace):
    policy = bash_tool.BashToolPolicy(reject_shell_syntax=False)
    tool = bash_tool.ExecuteBashTool(workspace=workspace, policy=policy)
    declaration = tool._get_declaration()
    assert declaration is not None
    description = declaration.description
    assert description is not None
    assert "passed to the program as literal arguments" in description
    assert "rejected" not in description

  @pytest.mark.asyncio
  async def test_captures_stderr(self, workspace, tool_context_confirmed):
    tool = bash_tool.ExecuteBashTool(workspace=workspace)
    result = await tool.run_async(
        args={"command": "python3 -c 'import sys; sys.stderr.write(\"err\")'"},
        tool_context=tool_context_confirmed,
    )
    assert "err" in result["stderr"]

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
    assert "error" in result
    assert "timed out" in result["error"].lower()

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
