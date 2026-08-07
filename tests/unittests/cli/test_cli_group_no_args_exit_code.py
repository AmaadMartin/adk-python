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

"""Unit tests pinning the exit code of a CLI group invoked with no subcommand.

click 8.2.0 made a bare group a usage error (exit 2). ADK keeps it a help
request (exit 0) across the whole supported ``click>=8.1.8,<9`` range, so these
tests assert the contract rather than whichever click happens to be resolved.

``result.output`` is asserted throughout, never ``result.stderr``: click 8.1.x's
``CliRunner`` defaults to ``mix_stderr=True`` and raises ``ValueError`` on
``result.stderr``, while 8.2 removed ``mix_stderr`` and split the streams.
``output`` is the only stream assertion portable across the supported range.
"""

from __future__ import annotations

import contextlib
import io
from unittest import mock

import click
from click.testing import CliRunner
from google.adk.cli import cli_tools_click
import pytest

# Every group in the CLI tree, with a subcommand only that group lists. The
# subcommand assertion proves the *correct* group's help was rendered rather
# than the root group's.
_GROUPS = [
    pytest.param([], "deploy", id="main"),
    pytest.param(["telemetry"], "status", id="telemetry"),
    pytest.param(["deploy"], "cloud_run", id="deploy"),
    pytest.param(["conformance"], "record", id="conformance"),
    pytest.param(["eval_set"], "create", id="eval_set"),
    pytest.param(["migrate"], "session", id="migrate"),
]


@pytest.mark.parametrize("argv, subcommand", _GROUPS)
def test_bare_group_prints_help_and_exits_zero(
    argv: list[str], subcommand: str
) -> None:
  """A group invoked with no subcommand prints its help and exits 0."""
  result = CliRunner().invoke(cli_tools_click.main, argv)

  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert result.output.startswith("Usage:")
  assert "Commands:" in result.output
  assert subcommand in result.output


@pytest.mark.parametrize(
    "argv", [[], ["deploy"], ["telemetry"]], ids=["main", "deploy", "telemetry"]
)
def test_explicit_help_flag_exits_zero(argv: list[str]) -> None:
  """``--help`` keeps exiting 0 for the root group and for subgroups."""
  result = CliRunner().invoke(cli_tools_click.main, argv + ["--help"])

  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert "Usage:" in result.output


@pytest.mark.parametrize(
    "argv", [["bogus"], ["deploy", "bogus"]], ids=["main", "deploy"]
)
def test_unknown_subcommand_still_exits_two(argv: list[str]) -> None:
  """An unknown subcommand stays a usage error; ``UsageError`` is not swallowed."""
  result = CliRunner().invoke(cli_tools_click.main, argv)

  assert result.exit_code == 2
  assert "No such command" in result.output


def test_resilient_parsing_does_not_print_help() -> None:
  """Shell completion parses with ``resilient_parsing``; it must not exit."""
  ctx = click.Context(cli_tools_click.main, resilient_parsing=True)
  stdout = io.StringIO()

  with contextlib.redirect_stdout(stdout):
    # Must return normally: raising click.exceptions.Exit here would abort
    # completion, and echoing help would corrupt the completion candidates.
    cli_tools_click.main.parse_args(ctx, [])

  assert stdout.getvalue() == ""


def test_group_with_no_args_is_help_disabled_falls_through() -> None:
  """A group that opts out of ``no_args_is_help`` is not hijacked."""
  group = cli_tools_click._HelpOnNoArgsGroup(
      "probe",
      no_args_is_help=False,
      invoke_without_command=True,
      callback=lambda: click.echo("callback ran"),
  )

  result = CliRunner().invoke(group, [])

  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert "callback ran" in result.output


def test_all_groups_use_help_on_no_args_group() -> None:
  """Every group in the tree, root included, inherits the exit-0 behaviour.

  This is the regression guard: a group added later without
  ``cls=_HelpOnNoArgsGroup`` reintroduces the click >= 8.2 exit code 2.
  """
  found: list[str] = []

  def walk(command: click.Command, path: str) -> None:
    if isinstance(command, click.Group):
      found.append(path)
      assert isinstance(command, cli_tools_click._HelpOnNoArgsGroup), (
          f"group {path!r} is a {type(command).__name__} and will exit 2 when"
          " invoked with no subcommand on click >= 8.2; pass"
          " cls=_HelpOnNoArgsGroup"
      )
      for name, child in command.commands.items():
        walk(child, f"{path} {name}".strip())

  walk(cli_tools_click.main, "adk")

  # Guards against a vacuous pass if the walk ever stops finding groups.
  assert len(found) >= 6, found


def test_subcommand_dispatch_unaffected() -> None:
  """The ``parse_args`` override does not disturb normal subcommand dispatch."""
  result = CliRunner().invoke(cli_tools_click.main, ["telemetry", "status"])

  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert "Telemetry collection is" in result.output


def test_bare_invocation_short_circuits_before_telemetry() -> None:
  """A bare ``adk`` never enters ``TelemetryGroup.invoke`` and records nothing.

  Asserting ``MetricsCollector`` was not constructed is not enough on its own:
  the recording block is also guarded by ``ctx.invoked_subcommand is not None``,
  which is independently false on the no-args path, so that assertion alone
  would still hold if the short-circuit moved out of ``parse_args``. Pinning
  that ``invoke`` is never entered is what actually locks in the structural
  property. Consent is forced on so nothing else can mask a regression.
  """
  entered: list[click.Context] = []
  original_invoke = cli_tools_click.TelemetryGroup.invoke

  def spy(group: cli_tools_click.TelemetryGroup, ctx: click.Context) -> object:
    entered.append(ctx)
    return original_invoke(group, ctx)

  with (
      mock.patch.object(cli_tools_click.TelemetryGroup, "invoke", spy),
      mock.patch.object(cli_tools_click, "MetricsCollector") as mock_collector,
      mock.patch.object(
          cli_tools_click, "read_telemetry_consent", return_value=True
      ),
  ):
    result = CliRunner().invoke(cli_tools_click.main, [])

  assert result.exit_code == 0, (result.output, repr(result.exception))
  assert entered == []
  mock_collector.assert_not_called()


def test_telemetry_group_inherits_help_on_no_args_group() -> None:
  """``TelemetryGroup`` keeps its telemetry behaviour and gains exit 0."""
  assert issubclass(
      cli_tools_click.TelemetryGroup, cli_tools_click._HelpOnNoArgsGroup
  )
