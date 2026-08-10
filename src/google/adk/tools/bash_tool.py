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

"""Tool to execute a single program directly, without a shell."""

from __future__ import annotations

import asyncio
import dataclasses
import importlib
import logging
import os
import pathlib
import shlex
import signal
from typing import Any
from typing import Optional

from google.genai import types

from .base_tool import BaseTool
from .tool_context import ToolContext

logger = logging.getLogger("google_adk." + __name__)

_resource = importlib.import_module("resource") if os.name == "posix" else None

# Characters a POSIX shell interprets. This tool never runs a shell, so an
# unquoted occurrence would silently become a literal argv entry.
_SHELL_METACHARACTERS = frozenset("|&;<>()$`*\n")
# Inside double quotes a shell still expands these two; the rest go literal.
_DOUBLE_QUOTED_METACHARACTERS = frozenset("$`")
# A shell only expands ~ and only opens a comment at # when the character
# starts a word, so both stay literal elsewhere (a~b, http://example.com#top).
_WORD_START_METACHARACTERS = frozenset("~#")


@dataclasses.dataclass(frozen=True)
class BashToolPolicy:
  """Configuration for allowed bash commands and resource limits.

  Set allowed_command_prefixes to ("*",) to allow all commands (default),
  or explicitly list allowed prefixes.

  Values for max_memory_bytes, max_file_size_bytes, and max_child_processes
  will be enforced upon the spawned subprocess.

  reject_shell_syntax (default True) rejects commands containing unquoted
  shell syntax. The tool executes a single program directly and never invokes
  a shell, so operators such as |, >, && and $VAR would otherwise be passed to
  the program as literal arguments instead of being interpreted. Set it to
  False only if you want that literal passthrough.
  """

  allowed_command_prefixes: tuple[str, ...] = ("*",)
  blocked_operators: tuple[str, ...] = ()
  timeout_seconds: Optional[int] = 30
  max_memory_bytes: Optional[int] = None
  max_file_size_bytes: Optional[int] = None
  max_child_processes: Optional[int] = None
  reject_shell_syntax: bool = True


def _shell_syntax_error(syntax: str) -> str:
  """Builds the rejection message for one unhonoured piece of shell syntax."""
  displayed = "\\n" if syntax == "\n" else syntax
  return (
      f"Command rejected: '{displayed}' is shell syntax, but this tool runs a"
      " single program directly without a shell, so it would be passed to the"
      " program as a literal argument instead of being interpreted. Run one"
      " program per call (no pipes, redirection, &&/||, ;, globs, $VAR"
      " expansion or subshells), or quote the character if you meant it"
      " literally."
  )


def _find_unhonoured_shell_syntax(command: str) -> Optional[str]:
  """Returns an error message if the command relies on shell interpretation.

  Detection is quote-aware: a metacharacter a shell would itself treat as
  literal (single-quoted, double-quoted where applicable, or backslash-escaped)
  is not reported. It is also position-aware for the characters a shell only
  acts on at the start of a word.

  Args:
    command: The raw command string supplied by the model.

  Returns:
    An error message, or None when the command can be executed faithfully as
    argv.
  """
  quote: Optional[str] = None
  escaped = False
  at_word_start = True
  for char in command:
    next_word_start = False
    if escaped:
      escaped = False
      # A shell drops the backslash before an expansion inside double quotes,
      # so "\$HOME" reaches the program as $HOME. shlex keeps the backslash.
      if quote == '"' and char in _DOUBLE_QUOTED_METACHARACTERS:
        return _shell_syntax_error("\\" + char)
    elif quote == "'":
      if char == "'":
        quote = None
    elif char == "\\":
      escaped = True
    elif quote == '"':
      if char == '"':
        quote = None
      elif char in _DOUBLE_QUOTED_METACHARACTERS:
        return _shell_syntax_error(char)
    elif char in ("'", '"'):
      quote = char
    elif char in _SHELL_METACHARACTERS:
      return _shell_syntax_error(char)
    elif char.isspace():
      next_word_start = True
    elif at_word_start and char in _WORD_START_METACHARACTERS:
      return _shell_syntax_error(char)
    at_word_start = next_word_start

  if quote is not None or escaped:
    return (
        "Command rejected: it ends inside a quote or with a trailing"
        " backslash, so it cannot be split into a program and its arguments."
        " Close the quote or drop the trailing backslash."
    )
  return None


def _validate_command(command: str, policy: BashToolPolicy) -> Optional[str]:
  """Validates a bash command against the permitted prefixes."""
  stripped = command.strip()
  if not stripped:
    return "Command is required."

  for op in policy.blocked_operators:
    if op in command:
      return f"Command contains blocked operator: {op}"

  if policy.reject_shell_syntax:
    shell_syntax_error = _find_unhonoured_shell_syntax(command)
    if shell_syntax_error:
      return shell_syntax_error

  if "*" in policy.allowed_command_prefixes:
    return None

  for prefix in policy.allowed_command_prefixes:
    if stripped.startswith(prefix):
      return None

  allowed = ", ".join(policy.allowed_command_prefixes)
  return f"Command blocked. Permitted prefixes are: {allowed}"


def _set_resource_limits(policy: BashToolPolicy) -> None:
  """Sets resource limits for the subprocess based on the provided policy."""
  if _resource is None:
    return
  try:
    _resource.setrlimit(_resource.RLIMIT_CORE, (0, 0))
    if policy.max_memory_bytes:
      _resource.setrlimit(
          _resource.RLIMIT_AS,
          (policy.max_memory_bytes, policy.max_memory_bytes),
      )
    if policy.max_file_size_bytes:
      _resource.setrlimit(
          _resource.RLIMIT_FSIZE,
          (policy.max_file_size_bytes, policy.max_file_size_bytes),
      )
    if policy.max_child_processes:
      _resource.setrlimit(
          _resource.RLIMIT_NPROC,
          (policy.max_child_processes, policy.max_child_processes),
      )
  except (ValueError, OSError) as e:
    logger.warning("Failed to set resource limits: %s", e)


class ExecuteBashTool(BaseTool):
  """Tool to execute a validated command within a workspace directory.

  The command is executed directly, without a shell, so shell syntax such as
  pipes, redirection and $VAR expansion is not interpreted.
  """

  def __init__(
      self,
      *,
      workspace: pathlib.Path | None = None,
      policy: Optional[BashToolPolicy] = None,
  ):
    if workspace is None:
      workspace = pathlib.Path.cwd()
    policy = policy or BashToolPolicy()
    allowed_hint = (
        "any command"
        if "*" in policy.allowed_command_prefixes
        else (
            "commands matching prefixes:"
            f" {', '.join(policy.allowed_command_prefixes)}"
        )
    )
    shell_hint = (
        "Shell syntax is not interpreted and is rejected: no pipes,"
        " redirection, &&/||, ;, globs, $VAR expansion or subshells"
        if policy.reject_shell_syntax
        else (
            "Shell syntax is not interpreted; operators such as | and > are"
            " passed to the program as literal arguments"
        )
    )
    super().__init__(
        name="execute_bash",
        description=(
            "Executes a single program directly, without a shell, with the"
            " working directory set to the workspace. The command string is"
            f" split into an executable and its arguments. {shell_hint}."
            f" Allowed: {allowed_hint}. All commands require user"
            " confirmation."
        ),
    )
    self._workspace = workspace
    self._policy = policy

  def _get_declaration(self) -> Optional[types.FunctionDeclaration]:
    return types.FunctionDeclaration(
        name=self.name,
        description=self.description,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "The program to run and its arguments, for example"
                        " 'ls -la src'. This is not a shell command line."
                    ),
                },
            },
            "required": ["command"],
        },
    )

  async def run_async(
      self, *, args: dict[str, Any], tool_context: ToolContext
  ) -> Any:
    command = args.get("command")
    if not command:
      return {"error": "Command is required."}

    # Static validation.
    error = _validate_command(command, self._policy)
    if error:
      return {"error": error}

    # Always request user confirmation.
    if not tool_context.tool_confirmation:
      tool_context.request_confirmation(
          hint=f"Please approve or reject the bash command: {command}",
      )
      tool_context.actions.skip_summarization = True
      return {
          "error": (
              "This tool call requires confirmation, please approve or reject."
          )
      }
    elif not tool_context.tool_confirmation.confirmed:
      return {"error": "This tool call is rejected."}

    if os.name != "posix":
      return {"error": "ExecuteBashTool is only supported on POSIX systems."}

    stdout = None
    stderr = None
    try:
      process = await asyncio.create_subprocess_exec(
          *shlex.split(command),
          cwd=str(self._workspace),
          stdout=asyncio.subprocess.PIPE,
          stderr=asyncio.subprocess.PIPE,
          start_new_session=True,
          preexec_fn=lambda: _set_resource_limits(self._policy),
      )

      try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=self._policy.timeout_seconds
        )
      except asyncio.TimeoutError:
        try:
          if process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
        stdout, stderr = await process.communicate()
        return {
            "error": (
                f"Command timed out after {self._policy.timeout_seconds}"
                " seconds."
            ),
            "stdout": (
                stdout.decode(errors="replace")
                if stdout
                else "<no stdout captured>"
            ),
            "stderr": (
                stderr.decode(errors="replace")
                if stderr
                else "<no stderr captured>"
            ),
            "returncode": process.returncode,
        }
      finally:
        try:
          if process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
          pass
      return {
          "stdout": (
              stdout.decode(errors="replace")
              if stdout
              else "<no stdout captured>"
          ),
          "stderr": (
              stderr.decode(errors="replace")
              if stderr
              else "<no stderr captured>"
          ),
          "returncode": process.returncode,
      }
    except Exception as e:  # pylint: disable=broad-except
      logger.exception("ExecuteBashTool execution failed")

      stdout_res = (
          stdout.decode(errors="replace") if stdout else "<no stdout captured>"
      )
      stderr_res = (
          stderr.decode(errors="replace") if stderr else "<no stderr captured>"
      )

      return {
          "error": f"Execution failed: {str(e)}",
          "stdout": stdout_res,
          "stderr": stderr_res,
      }

  def _detect_error_in_response(self, response: Any) -> Optional[str]:
    """Telemetry hook: returns an error type if the response indicates an error."""
    if isinstance(response, dict) and response.get("error"):
      return "TOOL_ERROR"
    return None
