#!/usr/bin/env python3
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

"""Reports open pull requests whose rendered diff is inflated by a stale base.

GitHub freezes `pull.base.sha` when the head branch is pushed. If the head was
cut from a base-branch state that the base repository had not received yet, the
recorded sha is a strict ancestor of the branch's real fork point. GitHub keeps
rendering the three-dot diff from that frozen sha, so every base-branch commit
between the recorded base and the real fork point is folded into the "Files
changed" view.

This walks the open pull requests of one repository and compares what GitHub
renders against `base.ref...head.sha` recomputed now:

  python scripts/check_pr_base_drift.py --repo google/adk-python

The check lags the defect by construction. While the base branch tip is still
at the stale sha, the recomputed merge base equals the recorded one and nothing
is visible. The drift appears once the base branch advances past the branch's
true fork point, so run this as a periodic sweep rather than a gate.

The script is read only. Every request is `gh api <path>`, an HTTP GET; it
builds no `--method`/`-X`/`-f`/`-F` argument, touches no working tree and runs
no git command.

Exit codes: 0 = ok, 1 = violation(s) found, 2 = usage/setup error.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import json
import subprocess
import sys
from typing import Any

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_SETUP_ERROR = 2

# The compare endpoint returns at most 300 changed files for the whole
# comparison, and paging it pages the commits rather than the files, so a
# larger comparison is not measurable by file count at all.
_COMPARE_FILE_CAP = 300

_PAGE_SIZE = 100
_SHA_LENGTH = 8
_MAX_CAPTURED_STDERR = 2000

_NOT_COMPARABLE = f'n/a (>{_COMPARE_FILE_CAP} files, not comparable)'


class GhError(RuntimeError):
  """The check could not be completed, so its result means nothing."""


@dataclasses.dataclass(frozen=True)
class PullRequest:
  """An open pull request as GitHub currently renders it."""

  number: int
  title: str
  html_url: str
  base_ref: str
  base_sha: str
  head_sha: str
  rendered_files: int
  rendered_commits: int


@dataclasses.dataclass(frozen=True)
class Comparison:
  """`base.ref...head.sha`, with the merge base recomputed now."""

  merge_base_sha: str
  ahead_by: int
  # Exact even when the returned commit array is capped at 250.
  true_commits: int
  # Meaningless when files_truncated; the file list is capped, not the diff.
  true_files: int
  files_truncated: bool

  @property
  def degenerate(self) -> bool:
    """The head adds nothing to the base, so there is nothing to compare."""
    return self.ahead_by == 0


@dataclasses.dataclass(frozen=True)
class Finding:
  """One pull request whose rendered diff does not match the recomputed one."""

  pull: PullRequest
  comparison: Comparison
  by_files: bool
  by_commits: bool


@dataclasses.dataclass(frozen=True)
class Scan:
  """The outcome of one sweep over a repository's open pull requests."""

  scanned: int
  skipped: int
  findings: tuple[Finding, ...]


# --- the GitHub boundary ----------------------------------------------------


def gh_api(path: str) -> Any:
  """Runs `gh api <path>` and decodes the JSON document it prints.

  Args:
    path: REST path, for example `repos/google/adk-python/pulls/1`.

  Returns:
    The decoded JSON document. Callers narrow it immediately.

  Raises:
    GhError: `gh` is missing, failed, or printed something that is not JSON.
  """
  command = ['gh', 'api', path]
  rendered = ' '.join(command)
  try:
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
  except (OSError, subprocess.SubprocessError) as err:
    raise GhError(f'`{rendered}` could not run: {err}') from err
  if completed.returncode != 0:
    detail = completed.stderr.strip()[:_MAX_CAPTURED_STDERR]
    raise GhError(f'`{rendered}` exited {completed.returncode}: {detail}')
  try:
    return json.loads(completed.stdout)
  except json.JSONDecodeError as err:
    raise GhError(f'`{rendered}` printed unparseable JSON: {err}') from err


def parse_pull(payload: Any) -> PullRequest:
  """Narrows a `pulls/{number}` payload into a PullRequest."""
  base = payload['base']
  head = payload['head']
  return PullRequest(
      number=int(payload['number']),
      title=str(payload['title']),
      html_url=str(payload['html_url']),
      base_ref=str(base['ref']),
      base_sha=str(base['sha']),
      head_sha=str(head['sha']),
      rendered_files=int(payload['changed_files']),
      rendered_commits=int(payload['commits']),
  )


def parse_comparison(payload: Any) -> Comparison:
  """Narrows a `compare/{basehead}` payload into a Comparison."""
  # `files` is absent when the two refs are identical.
  files = payload.get('files') or []
  return Comparison(
      merge_base_sha=str(payload['merge_base_commit']['sha']),
      ahead_by=int(payload['ahead_by']),
      true_commits=int(payload['total_commits']),
      true_files=len(files),
      files_truncated=len(files) >= _COMPARE_FILE_CAP,
  )


# --- the invariant ----------------------------------------------------------


def evaluate(pull: PullRequest, comparison: Comparison) -> Finding | None:
  """Decides whether GitHub renders this pull request's diff correctly.

  Two independent measurements, either of which trips a finding. The file
  count is the symptom a reviewer sees, but it is unusable once the compare
  file list truncates. The commit count is exact at any size, and it also
  catches folded-in commits that only touch files the branch already touches.

  `base.sha != merge_base_commit.sha` is deliberately not a measurement. It
  fires on the benign mirror case, where GitHub recorded a base sha newer than
  the fork point, which leaves the rendered diff correct.

  Args:
    pull: The pull request as GitHub renders it.
    comparison: `base.ref...head.sha`, recomputed now.

  Returns:
    A Finding, or None when the rendered diff is correct.
  """
  if comparison.degenerate:
    return None
  by_files = (
      not comparison.files_truncated
      and pull.rendered_files != comparison.true_files
  )
  by_commits = pull.rendered_commits != comparison.true_commits
  if not by_files and not by_commits:
    return None
  return Finding(
      pull=pull,
      comparison=comparison,
      by_files=by_files,
      by_commits=by_commits,
  )


# --- the sweep --------------------------------------------------------------


def list_open_pull_numbers(repo: str, *, limit: int = 0) -> list[int]:
  """Lists open pull request numbers, newest first, paging by hand.

  `gh api --paginate` emits one JSON document per page, which `json.loads`
  cannot read, so the page cursor is explicit here.

  Args:
    repo: `OWNER/NAME`.
    limit: Stop after this many pull requests; 0 means all of them.

  Returns:
    The pull request numbers to scan.
  """
  numbers: list[int] = []
  page = 1
  while True:
    payload = gh_api(
        f'repos/{repo}/pulls?state=open&per_page={_PAGE_SIZE}&page={page}'
    )
    numbers.extend(int(item['number']) for item in payload)
    if len(payload) < _PAGE_SIZE or (limit and len(numbers) >= limit):
      break
    page += 1
  return numbers[:limit] if limit else numbers


def scan(repo: str, *, limit: int = 0) -> Scan:
  """Sweeps a repository's open pull requests.

  Args:
    repo: `OWNER/NAME`.
    limit: Stop after this many pull requests; 0 means all of them.

  Returns:
    What was scanned, what was skipped, and every finding.

  Raises:
    GhError: Any request failed. A partial sweep that reports no drift is
      worse than no sweep, so one failure ends the run.
  """
  findings: list[Finding] = []
  skipped = 0
  numbers = list_open_pull_numbers(repo, limit=limit)
  for number in numbers:
    pull = parse_pull(gh_api(f'repos/{repo}/pulls/{number}'))
    # Pinned to head.sha, not to the head branch name, so a push landing
    # mid-sweep cannot contradict the counts read from the payload above.
    comparison = parse_comparison(
        gh_api(f'repos/{repo}/compare/{pull.base_ref}...{pull.head_sha}')
    )
    if comparison.degenerate:
      skipped += 1
      continue
    finding = evaluate(pull, comparison)
    if finding is not None:
      findings.append(finding)
  return Scan(
      scanned=len(numbers) - skipped,
      skipped=skipped,
      findings=tuple(findings),
  )


# --- the report -------------------------------------------------------------


def _render_finding(finding: Finding) -> list[str]:
  """Renders one finding, naming both measurements and which tripped."""
  pull = finding.pull
  comparison = finding.comparison
  if comparison.files_truncated:
    files = f'  files    {_NOT_COMPARABLE}'
  else:
    folded = pull.rendered_files - comparison.true_files
    files = (
        f'  files    {pull.rendered_files} rendered vs'
        f' {comparison.true_files} real   ({folded:+d} folded in)'
    )
  tripped = [
      name
      for name, hit in (
          ('file count', finding.by_files),
          ('commit count', finding.by_commits),
      )
      if hit
  ]
  folded_commits = pull.rendered_commits - comparison.true_commits
  return [
      f'PR #{pull.number}  {pull.title}',
      f'  {pull.html_url}',
      (
          f'  recorded base   {pull.base_ref} @'
          f' {pull.base_sha[:_SHA_LENGTH]}   (frozen by GitHub when the head'
          ' was pushed)'
      ),
      f'  true merge base       {comparison.merge_base_sha[:_SHA_LENGTH]}',
      files,
      (
          f'  commits   {pull.rendered_commits} rendered vs'
          f' {comparison.true_commits} real   ({folded_commits:+d}'
          ' base-branch commits folded in)'
      ),
      f'  tripped by: {", ".join(tripped)}',
      '',
  ]


def render_report(result: Scan) -> str:
  """Builds the whole report, findings first and then the summary."""
  lines: list[str] = []
  for finding in result.findings:
    lines.extend(_render_finding(finding))
  if result.findings:
    lines.append(
        f'{len(result.findings)} of {result.scanned} open pull requests'
        ' renders an inflated diff.'
    )
    lines.append(
        'Re-push the head branch after syncing the base branch so GitHub'
        ' records a fresh base sha.'
    )
  else:
    lines.append(
        f'Scanned {result.scanned} open pull requests; no base drift found.'
    )
  if result.skipped:
    total = result.scanned + result.skipped
    lines.append(
        f'Skipped {result.skipped} of {total} open pull requests: the head is'
        ' already contained in the base branch.'
    )
  return '\n'.join(lines) + '\n'


# --- entry point ------------------------------------------------------------


def _repo_slug(value: str) -> str:
  """argparse type: a repository as `OWNER/NAME`.

  The value is interpolated into a REST path, so it is checked here rather
  than handed to `gh` as-is.
  """
  parts = value.split('/')
  if len(parts) != 2 or not all(
      part and all(c.isalnum() or c in '-._' for c in part) for part in parts
  ):
    raise argparse.ArgumentTypeError(f'expected OWNER/NAME, got {value!r}')
  return value


def _pull_count(value: str) -> int:
  """argparse type: a non-negative pull request count."""
  count = int(value)
  if count < 0:
    raise argparse.ArgumentTypeError(f'must not be negative, got {count}')
  return count


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Builds the command line and parses it."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--repo',
      required=True,
      type=_repo_slug,
      metavar='OWNER/NAME',
      help='Repository whose open pull requests are scanned.',
  )
  parser.add_argument(
      '--limit',
      type=_pull_count,
      default=0,
      metavar='N',
      help='Stop after N open pull requests (default: all of them).',
  )
  return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
  """Runs the sweep and returns the process exit code."""
  args = _parse_args(argv)
  try:
    result = scan(args.repo, limit=args.limit)
  except GhError as err:
    # Fail closed. A sweep that could not run must never read as a pass.
    print(f'Base drift check could not run: {err}', file=sys.stderr)
    return EXIT_SETUP_ERROR
  print(render_report(result), end='')
  return EXIT_DRIFT if result.findings else EXIT_OK


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
