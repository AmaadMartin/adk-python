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

"""Requires every GitHub Actions `uses:` reference to name a commit SHA.

A tag or a branch is a mutable ref. Whoever owns the action repository can
re-point `v6` at other code, and that code then runs in jobs that hold this
repository's release secrets. A full 40-character commit SHA cannot move, so it
is the only ref this check accepts.

The scan is line-oriented rather than a YAML parse because pre-commit runs this
hook with `language: script`, which executes the file with the system
interpreter and installs no dependencies. PyYAML is therefore unavailable. One
consequence: a `uses:`-looking line inside a `run:` block is reported as a
violation, because a line-oriented scan cannot see the block it belongs to.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# Applied with `match`, so it is anchored at the start of the line: a
# commented-out `# uses: ...` is not a step. The optional `- ` covers the
# inline step form, and `\S+` stops before any trailing `# vN` comment.
_USES_RE = re.compile(r'\s*(?:-\s*)?uses\s*:\s*(?P<ref>\S+)')
_SHA_RE = re.compile(r'[0-9a-fA-F]{40}')


def find_unpinned_uses(content: str) -> list[tuple[int, str]]:
  """Finds `uses:` references that are not pinned to a commit SHA.

  Args:
    content: The full text of a GitHub Actions workflow file.

  Returns:
    A `(line_number, uses_value)` pair for every unpinned reference, where
    `line_number` is 1-based and `uses_value` is the reference as written,
    without surrounding quotes.
  """
  violations: list[tuple[int, str]] = []
  for line_number, line in enumerate(content.splitlines(), start=1):
    match = _USES_RE.match(line)
    if not match:
      continue
    value = match.group('ref').strip('\'"')
    # A repo-local action or reusable workflow ships with this commit, and a
    # Docker image reference is not a git ref. Neither has a SHA to pin.
    if value.startswith('./') or value.startswith('docker://'):
      continue
    # A `uses:` value is `owner/repo[/path]@ref`, so the ref is everything
    # after the first `@`. Subdirectory actions (`actions/cache/restore@<sha>`)
    # and reusable workflows (`owner/repo/.github/workflows/x.yml@<sha>`) put
    # the extra path before that `@`, so they still resolve. Splitting here
    # rather than on the last `@` also rejects a tag whose own name contains
    # an `@` and ends in 40 hex characters. A value with no `@` at all yields
    # an empty ref, which the SHA test rejects: an omitted ref resolves to the
    # action's default branch, which is mutable.
    ref = value.partition('@')[2]
    if not _SHA_RE.fullmatch(ref):
      violations.append((line_number, value))
  return violations


def main(argv: list[str]) -> int:
  """Reports unpinned `uses:` references in the given workflow files.

  Args:
    argv: Command-line arguments, excluding the program name.

  Returns:
    `1` if any unpinned reference was found, `0` otherwise.
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('files', nargs='*', help='Workflow files to check')
  args = parser.parse_args(argv)

  failed = False
  for path in args.files:
    content = pathlib.Path(path).read_text(encoding='utf-8')
    for line_number, value in find_unpinned_uses(content):
      print(
          f'{path}:{line_number}: unpinned action reference {value!r}; pin it'
          ' to a full 40-character commit SHA (e.g.'
          ' actions/checkout@<sha> # v6)'
      )
      failed = True
  return 1 if failed else 0


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
