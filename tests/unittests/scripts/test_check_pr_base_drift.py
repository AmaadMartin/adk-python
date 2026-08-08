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

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
import json
import os
import pathlib
import runpy
import subprocess
import sys
from typing import Any

import pytest

from scripts import check_pr_base_drift

_REPO = 'OWNER/NAME'
_BASE_SHA = '352d11d3aed42214b6ce7fdfbc16bb37c0121a2b'
_MERGE_BASE_SHA = 'c5672031f0d94a0e6e6d1f7a4f2b9c8d1e3a5b7c'
_HEAD_SHA = 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'

_PULLS_PAGE_1 = f'repos/{_REPO}/pulls?state=open&per_page=100&page=1'
_PULLS_PAGE_2 = f'repos/{_REPO}/pulls?state=open&per_page=100&page=2'
_PULL_123 = f'repos/{_REPO}/pulls/123'
_COMPARE_123 = f'repos/{_REPO}/compare/main...{_HEAD_SHA}'

_SCRIPT_PATH = str(pathlib.Path(check_pr_base_drift.__file__).resolve())

# A fake `gh` that answers from a routing table baked in at write time. It
# exists so the command line can be exercised as a real process.
_FAKE_GH_SOURCE = """\
#!/usr/bin/env python3
import json
import sys

ROUTES = json.loads({routes!r})
path = sys.argv[-1]
if path not in ROUTES:
  sys.stderr.write('HTTP 404: Not Found (' + path + ')')
  sys.exit(1)
sys.stdout.write(json.dumps(ROUTES[path]))
"""


class _FakeGh:
  """Replays canned `gh api` responses and records every command it is given."""

  def __init__(self, routes: Mapping[str, object]) -> None:
    self._routes = dict(routes)
    self.commands: list[list[str]] = []

  @property
  def paths(self) -> list[str]:
    return [command[-1] for command in self.commands]

  def __call__(
      self, command: Sequence[str], **kwargs: Any
  ) -> subprocess.CompletedProcess[str]:
    self.commands.append(list(command))
    path = command[-1]
    if path not in self._routes:
      return subprocess.CompletedProcess(
          args=list(command),
          returncode=1,
          stdout='',
          stderr=f'HTTP 404: Not Found ({path})',
      )
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout=json.dumps(self._routes[path]),
        stderr='',
    )


def _pull_payload(
    number: int = 123,
    *,
    base_ref: str = 'main',
    base_sha: str = _BASE_SHA,
    head_sha: str = _HEAD_SHA,
    changed_files: int = 3,
    commits: int = 2,
) -> dict[str, object]:
  return {
      'number': number,
      'title': 'feat(agents): add a thing',
      'html_url': f'https://github.com/{_REPO}/pull/{number}',
      'base': {'ref': base_ref, 'sha': base_sha},
      'head': {'sha': head_sha},
      'changed_files': changed_files,
      'commits': commits,
  }


def _compare_payload(
    *,
    merge_base_sha: str = _MERGE_BASE_SHA,
    ahead_by: int = 1,
    total_commits: int = 2,
    files: int = 3,
) -> dict[str, object]:
  return {
      'merge_base_commit': {'sha': merge_base_sha},
      'ahead_by': ahead_by,
      'total_commits': total_commits,
      'files': [
          {'filename': f'pkg/module_{index}.py'} for index in range(files)
      ],
  }


def _pull(
    *,
    rendered_files: int = 3,
    rendered_commits: int = 2,
    base_sha: str = _BASE_SHA,
) -> check_pr_base_drift.PullRequest:
  return check_pr_base_drift.PullRequest(
      number=123,
      title='feat(agents): add a thing',
      html_url=f'https://github.com/{_REPO}/pull/123',
      base_ref='main',
      base_sha=base_sha,
      head_sha=_HEAD_SHA,
      rendered_files=rendered_files,
      rendered_commits=rendered_commits,
  )


def _comparison(
    *,
    true_files: int = 3,
    true_commits: int = 2,
    ahead_by: int = 1,
    files_truncated: bool = False,
    merge_base_sha: str = _MERGE_BASE_SHA,
) -> check_pr_base_drift.Comparison:
  return check_pr_base_drift.Comparison(
      merge_base_sha=merge_base_sha,
      ahead_by=ahead_by,
      true_commits=true_commits,
      true_files=true_files,
      files_truncated=files_truncated,
  )


def _install_fake_gh(
    tmp_path: pathlib.Path, routes: Mapping[str, object]
) -> dict[str, str]:
  """Writes a fake `gh` into a fresh directory and puts it first on PATH."""
  bin_dir = tmp_path / 'bin'
  bin_dir.mkdir()
  executable = bin_dir / 'gh'
  executable.write_text(
      _FAKE_GH_SOURCE.format(routes=json.dumps(routes)), encoding='utf-8'
  )
  executable.chmod(0o755)
  env = dict(os.environ)
  env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
  return env


# --- evaluate ---------------------------------------------------------------


def test_evaluate_passes_a_healthy_pull_request() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=3, rendered_commits=2),
      _comparison(true_files=3, true_commits=2),
  )
  assert finding is None


def test_evaluate_reports_a_pull_request_inflated_in_files_and_commits() -> (
    None
):
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=412, rendered_commits=87),
      _comparison(true_files=3, true_commits=1),
  )
  assert finding is not None
  assert finding.by_files is True
  assert finding.by_commits is True


def test_evaluate_reports_drift_that_moves_only_the_commit_count() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=5, rendered_commits=40),
      _comparison(true_files=5, true_commits=2),
  )
  assert finding is not None
  assert finding.by_files is False
  assert finding.by_commits is True


def test_evaluate_ignores_a_recorded_base_below_the_merge_base() -> None:
  """A recorded base sha that is not the merge base is not, on its own, drift.

  This is the naive detector. It fired on 15 of 40 open pull requests of
  google/adk-python and every one of those rendered a correct diff, so it must
  never produce a finding by itself.
  """
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=3, rendered_commits=2, base_sha=_BASE_SHA),
      _comparison(true_files=3, true_commits=2, merge_base_sha=_MERGE_BASE_SHA),
  )
  assert finding is None


def test_evaluate_passes_a_large_healthy_pull_request() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=350, rendered_commits=4),
      _comparison(true_files=300, true_commits=4, files_truncated=True),
  )
  assert finding is None


def test_evaluate_reports_a_large_drifted_pull_request_by_commit_count() -> (
    None
):
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=350, rendered_commits=90),
      _comparison(true_files=300, true_commits=2, files_truncated=True),
  )
  assert finding is not None
  assert finding.by_files is False
  assert finding.by_commits is True


def test_evaluate_treats_exactly_300_compare_files_as_truncated() -> None:
  comparison = check_pr_base_drift.parse_comparison(
      _compare_payload(total_commits=4, files=300)
  )
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=317, rendered_commits=4), comparison
  )
  assert comparison.files_truncated is True
  assert finding is None


def test_evaluate_skips_a_head_already_contained_in_the_base() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=9, rendered_commits=9),
      _comparison(true_files=0, true_commits=0, ahead_by=0),
  )
  assert finding is None


# --- parsing ----------------------------------------------------------------


def test_parse_pull_reads_every_field() -> None:
  pull = check_pr_base_drift.parse_pull(
      _pull_payload(number=7, base_ref='release', changed_files=11, commits=4)
  )
  assert pull == check_pr_base_drift.PullRequest(
      number=7,
      title='feat(agents): add a thing',
      html_url=f'https://github.com/{_REPO}/pull/7',
      base_ref='release',
      base_sha=_BASE_SHA,
      head_sha=_HEAD_SHA,
      rendered_files=11,
      rendered_commits=4,
  )


def test_parse_comparison_without_a_files_key() -> None:
  payload = _compare_payload()
  del payload['files']
  comparison = check_pr_base_drift.parse_comparison(payload)
  assert comparison.true_files == 0
  assert comparison.files_truncated is False


def test_parse_comparison_reads_the_recomputed_merge_base() -> None:
  comparison = check_pr_base_drift.parse_comparison(
      _compare_payload(ahead_by=6, total_commits=6, files=4)
  )
  assert comparison == check_pr_base_drift.Comparison(
      merge_base_sha=_MERGE_BASE_SHA,
      ahead_by=6,
      true_commits=6,
      true_files=4,
      files_truncated=False,
  )
  assert comparison.degenerate is False


def test_parse_comparison_with_300_files_is_truncated() -> None:
  comparison = check_pr_base_drift.parse_comparison(_compare_payload(files=300))
  assert comparison.true_files == 300
  assert comparison.files_truncated is True


# --- gh_api -----------------------------------------------------------------


def test_gh_api_decodes_the_document_and_only_issues_a_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake = _FakeGh({_PULL_123: _pull_payload()})
  monkeypatch.setattr(subprocess, 'run', fake)

  payload = check_pr_base_drift.gh_api(_PULL_123)

  assert payload['number'] == 123
  assert fake.commands == [['gh', 'api', _PULL_123]]


def test_gh_api_raises_on_a_non_zero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(subprocess, 'run', _FakeGh({}))
  with pytest.raises(check_pr_base_drift.GhError, match='HTTP 404'):
    check_pr_base_drift.gh_api(_PULL_123)


def test_gh_api_raises_on_an_unparseable_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def run(
      command: Sequence[str], **kwargs: Any
  ) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=list(command), returncode=0, stdout='not json', stderr=''
    )

  monkeypatch.setattr(subprocess, 'run', run)
  with pytest.raises(check_pr_base_drift.GhError, match='unparseable JSON'):
    check_pr_base_drift.gh_api(_PULL_123)


def test_gh_api_raises_when_gh_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def run(command: Sequence[str], **kwargs: Any) -> None:
    raise FileNotFoundError(2, 'No such file or directory', 'gh')

  monkeypatch.setattr(subprocess, 'run', run)
  with pytest.raises(check_pr_base_drift.GhError, match='could not run'):
    check_pr_base_drift.gh_api(_PULL_123)


def test_gh_api_raises_when_the_subprocess_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  def run(command: Sequence[str], **kwargs: Any) -> None:
    raise subprocess.SubprocessError('the pipe broke')

  monkeypatch.setattr(subprocess, 'run', run)
  with pytest.raises(check_pr_base_drift.GhError, match='the pipe broke'):
    check_pr_base_drift.gh_api(_PULL_123)


# --- listing ----------------------------------------------------------------


def test_list_open_pull_numbers_pages_until_a_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake = _FakeGh({
      _PULLS_PAGE_1: [{'number': index} for index in range(100)],
      _PULLS_PAGE_2: [{'number': 100 + index} for index in range(7)],
  })
  monkeypatch.setattr(subprocess, 'run', fake)

  numbers = check_pr_base_drift.list_open_pull_numbers(_REPO)

  assert len(numbers) == 107
  assert numbers[-1] == 106
  assert fake.paths == [_PULLS_PAGE_1, _PULLS_PAGE_2]


def test_list_open_pull_numbers_stops_at_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake = _FakeGh({
      _PULLS_PAGE_1: [{'number': index} for index in range(100)],
      _PULLS_PAGE_2: [{'number': 100 + index} for index in range(7)],
  })
  monkeypatch.setattr(subprocess, 'run', fake)

  numbers = check_pr_base_drift.list_open_pull_numbers(_REPO, limit=5)

  assert numbers == [0, 1, 2, 3, 4]
  assert fake.paths == [_PULLS_PAGE_1]


# --- scan -------------------------------------------------------------------


def test_scan_collects_findings_and_never_issues_a_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake = _FakeGh({
      _PULLS_PAGE_1: [{'number': 123}, {'number': 124}],
      _PULL_123: _pull_payload(changed_files=412, commits=87),
      _COMPARE_123: _compare_payload(ahead_by=1, total_commits=1, files=3),
      f'repos/{_REPO}/pulls/124': _pull_payload(number=124, head_sha='b' * 40),
      f'repos/{_REPO}/compare/main...{"b" * 40}': _compare_payload(
          ahead_by=2, total_commits=2, files=3
      ),
  })
  monkeypatch.setattr(subprocess, 'run', fake)

  result = check_pr_base_drift.scan(_REPO)

  assert result.scanned == 2
  assert result.skipped == 0
  assert [finding.pull.number for finding in result.findings] == [123]
  assert all(
      command[:2] == ['gh', 'api'] and len(command) == 3
      for command in fake.commands
  )


def test_scan_counts_a_contained_head_as_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  fake = _FakeGh({
      _PULLS_PAGE_1: [{'number': 123}],
      _PULL_123: _pull_payload(changed_files=9, commits=9),
      _COMPARE_123: _compare_payload(ahead_by=0, total_commits=0, files=0),
  })
  monkeypatch.setattr(subprocess, 'run', fake)

  result = check_pr_base_drift.scan(_REPO)

  assert result == check_pr_base_drift.Scan(scanned=0, skipped=1, findings=())


# --- report -----------------------------------------------------------------


def test_render_report_names_both_measurements_for_a_finding() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=412, rendered_commits=87),
      _comparison(true_files=3, true_commits=1),
  )
  assert finding is not None

  report = check_pr_base_drift.render_report(
      check_pr_base_drift.Scan(scanned=40, skipped=0, findings=(finding,))
  )

  assert f'https://github.com/{_REPO}/pull/123' in report
  assert f'recorded base   main @ {_BASE_SHA[:8]}' in report
  assert f'true merge base       {_MERGE_BASE_SHA[:8]}' in report
  assert 'files    412 rendered vs 3 real   (+409 folded in)' in report
  assert (
      'commits   87 rendered vs 1 real   (+86 base-branch commits folded in)'
      in report
  )
  assert 'tripped by: file count, commit count' in report
  assert '1 of 40 open pull requests renders an inflated diff.' in report
  assert 'Re-push the head branch' in report


def test_render_report_marks_a_truncated_comparison_as_not_comparable() -> None:
  finding = check_pr_base_drift.evaluate(
      _pull(rendered_files=350, rendered_commits=90),
      _comparison(true_files=300, true_commits=2, files_truncated=True),
  )
  assert finding is not None

  report = check_pr_base_drift.render_report(
      check_pr_base_drift.Scan(scanned=1, skipped=0, findings=(finding,))
  )

  assert 'files    n/a (>300 files, not comparable)' in report
  assert 'rendered vs 300 real' not in report
  assert 'tripped by: commit count' in report


def test_render_report_summarises_a_clean_sweep() -> None:
  report = check_pr_base_drift.render_report(
      check_pr_base_drift.Scan(scanned=40, skipped=0, findings=())
  )
  assert report == 'Scanned 40 open pull requests; no base drift found.\n'


def test_render_report_counts_skipped_pull_requests() -> None:
  report = check_pr_base_drift.render_report(
      check_pr_base_drift.Scan(scanned=38, skipped=2, findings=())
  )
  assert 'Scanned 38 open pull requests; no base drift found.' in report
  assert (
      'Skipped 2 of 40 open pull requests: the head is already contained in'
      ' the base branch.'
      in report
  )


# --- main -------------------------------------------------------------------


def test_main_returns_zero_for_a_clean_repository(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(
      subprocess,
      'run',
      _FakeGh({
          _PULLS_PAGE_1: [{'number': 123}],
          _PULL_123: _pull_payload(changed_files=3, commits=2),
          _COMPARE_123: _compare_payload(ahead_by=2, total_commits=2, files=3),
      }),
  )

  code = check_pr_base_drift.main(['--repo', _REPO])

  assert code == check_pr_base_drift.EXIT_OK
  assert (
      capsys.readouterr().out
      == 'Scanned 1 open pull requests; no base drift found.\n'
  )


def test_main_returns_one_and_prints_the_finding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(
      subprocess,
      'run',
      _FakeGh({
          _PULLS_PAGE_1: [{'number': 123}],
          _PULL_123: _pull_payload(changed_files=412, commits=87),
          _COMPARE_123: _compare_payload(ahead_by=1, total_commits=1, files=3),
      }),
  )

  code = check_pr_base_drift.main(['--repo', _REPO, '--limit', '1'])

  captured = capsys.readouterr()
  assert code == check_pr_base_drift.EXIT_DRIFT
  assert 'PR #123  feat(agents): add a thing' in captured.out
  assert '1 of 1 open pull requests renders an inflated diff.' in captured.out
  assert captured.err == ''


def test_main_returns_two_when_gh_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
  monkeypatch.setattr(subprocess, 'run', _FakeGh({}))

  code = check_pr_base_drift.main(['--repo', _REPO])

  captured = capsys.readouterr()
  assert code == check_pr_base_drift.EXIT_SETUP_ERROR
  assert captured.out == ''
  assert 'Base drift check could not run' in captured.err
  assert 'HTTP 404' in captured.err


@pytest.mark.parametrize(
    'repo', ['OWNER', 'OWNER/NAME/extra', 'OWNER/', '/NAME', 'OWNER/NA ME']
)
def test_main_rejects_a_repository_that_is_not_owner_slash_name(
    repo: str, capsys: pytest.CaptureFixture[str]
) -> None:
  with pytest.raises(SystemExit) as excinfo:
    check_pr_base_drift.main(['--repo', repo])
  assert excinfo.value.code == check_pr_base_drift.EXIT_SETUP_ERROR
  assert 'expected OWNER/NAME' in capsys.readouterr().err


@pytest.mark.parametrize('limit', ['-1', 'many'])
def test_main_rejects_an_unusable_limit(
    limit: str, capsys: pytest.CaptureFixture[str]
) -> None:
  with pytest.raises(SystemExit) as excinfo:
    check_pr_base_drift.main(['--repo', _REPO, '--limit', limit])
  assert excinfo.value.code == check_pr_base_drift.EXIT_SETUP_ERROR
  assert '--limit' in capsys.readouterr().err


def test_module_entry_point_exits_with_the_code_from_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(subprocess, 'run', _FakeGh({_PULLS_PAGE_1: []}))
  monkeypatch.setattr(sys, 'argv', ['check_pr_base_drift.py', '--repo', _REPO])

  with pytest.raises(SystemExit) as excinfo:
    runpy.run_path(_SCRIPT_PATH, run_name='__main__')

  assert excinfo.value.code == check_pr_base_drift.EXIT_OK


# --- the script as a real command -------------------------------------------
#
# These drive the installed file through a real interpreter, a real PATH
# lookup and a real `gh` executable, so they prove the wiring the unit tests
# above stub out. No network: the `gh` they find is a local fake.


def test_script_run_as_a_command_exits_zero_on_a_clean_repository(
    tmp_path: pathlib.Path,
) -> None:
  env = _install_fake_gh(
      tmp_path,
      {
          _PULLS_PAGE_1: [{'number': 123}],
          _PULL_123: _pull_payload(changed_files=3, commits=2),
          _COMPARE_123: _compare_payload(ahead_by=2, total_commits=2, files=3),
      },
  )

  completed = subprocess.run(
      [sys.executable, _SCRIPT_PATH, '--repo', _REPO],
      capture_output=True,
      text=True,
      env=env,
      check=False,
  )

  assert completed.returncode == 0
  assert completed.stdout == (
      'Scanned 1 open pull requests; no base drift found.\n'
  )


def test_script_run_as_a_command_exits_one_on_a_drifted_repository(
    tmp_path: pathlib.Path,
) -> None:
  env = _install_fake_gh(
      tmp_path,
      {
          _PULLS_PAGE_1: [{'number': 123}],
          _PULL_123: _pull_payload(changed_files=412, commits=87),
          _COMPARE_123: _compare_payload(ahead_by=1, total_commits=1, files=3),
      },
  )

  completed = subprocess.run(
      [sys.executable, _SCRIPT_PATH, '--repo', _REPO],
      capture_output=True,
      text=True,
      env=env,
      check=False,
  )

  assert completed.returncode == 1
  assert 'tripped by: file count, commit count' in completed.stdout


def test_script_run_as_a_command_exits_two_without_gh(
    tmp_path: pathlib.Path,
) -> None:
  empty_bin = tmp_path / 'empty'
  empty_bin.mkdir()
  env = dict(os.environ)
  env['PATH'] = str(empty_bin)

  completed = subprocess.run(
      [sys.executable, _SCRIPT_PATH, '--repo', _REPO],
      capture_output=True,
      text=True,
      env=env,
      check=False,
  )

  assert completed.returncode == 2
  assert completed.stdout == ''
  assert 'Base drift check could not run' in completed.stderr
