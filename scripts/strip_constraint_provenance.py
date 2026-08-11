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

"""Removes a temporary constraint file from uv's `# via` provenance comments.

`update_constraints.sh` stabilizes each resolution by passing the committed
constraints file back to `uv pip compile --constraint` under a scratch name.
uv records every constraint source in the `# via` comment of every pinned
package, so that scratch path lands in each published `constraints-<ver>.txt`.
README.md tells users to download those files, and the path never exists on
their machine.

Removing the entry is not enough on its own. uv writes a single source on one
line and two or more as an indented list, so this filter also restores the
shape uv would have emitted for the remaining sources. That keeps the script's
constrained and unconstrained resolution paths byte-identical.

Reads a uv-generated constraints file on stdin and writes the cleaned file to
stdout; see `scripts/update_constraints.sh` for the invocation.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from collections.abc import Sequence
import sys

# uv indents a `# via` comment by four spaces, and the sources it lists under
# a multi-source header by four spaces plus two more.
_VIA_HEADER = '    # via'
_SOURCE_INDENT = '    #   '


def _render_via_block(sources: Sequence[str]) -> list[str]:
  """Renders a `# via` block in the shape uv itself would emit.

  Args:
    sources: The remaining sources of the block, in their original order.

  Returns:
    No lines for an empty block, the one-line form for a single source, and
    the header plus an indented list for two or more.
  """
  if not sources:
    return []
  if len(sources) == 1:
    return [f'{_VIA_HEADER} {sources[0]}']
  return [_VIA_HEADER, *(f'{_SOURCE_INDENT}{source}' for source in sources)]


def strip_provenance(lines: Iterable[str], constraint_file: str) -> list[str]:
  """Drops one constraint file from every `# via` block.

  Args:
    lines: The lines of a uv-generated constraints file, without line endings.
    constraint_file: The constraint path as uv spells it, i.e. the value passed
      to `uv pip compile --constraint`.

  Returns:
    The same lines in the same order, with the `-c <constraint_file>` source
    removed and every block it touched restored to uv's own shape. A source
    naming any other file is left alone.
  """
  dropped_source = f'-c {constraint_file}'
  dropped_inline = f'{_VIA_HEADER} {dropped_source}'
  result: list[str] = []
  # The sources collected so far for the block being read, or None when the
  # current line is outside a block.
  sources: list[str] | None = None

  for line in lines:
    if sources is not None:
      if line.startswith(_SOURCE_INDENT):
        source = line[len(_SOURCE_INDENT) :]
        if source != dropped_source:
          sources.append(source)
        continue
      result.extend(_render_via_block(sources))
      sources = None
    if line == _VIA_HEADER:
      sources = []
    elif line != dropped_inline:
      result.append(line)

  if sources is not None:
    result.extend(_render_via_block(sources))
  return result


def main(argv: Sequence[str] | None = None) -> int:
  """Filters stdin to stdout.

  Args:
    argv: The command-line arguments, or None to read `sys.argv`.

  Returns:
    The process exit code.
  """
  parser = argparse.ArgumentParser(
      description='Remove a constraint file from uv `# via` provenance.'
  )
  parser.add_argument(
      '--constraint-file',
      required=True,
      help='The constraint path as uv spells it in the `# via` comments.',
  )
  args = parser.parse_args(argv)
  cleaned = strip_provenance(
      sys.stdin.read().splitlines(), args.constraint_file
  )
  sys.stdout.writelines(f'{line}\n' for line in cleaned)
  return 0


if __name__ == '__main__':
  sys.exit(main())
