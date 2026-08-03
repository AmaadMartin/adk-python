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

"""Guard tests for the files ``README.md`` tells users to download.

The installation section once shipped a ``curl`` of
``github.com/google/adk-python/blob/main/constraints-3.10.txt``, which was
broken twice over and nothing caught either half:

* The ``/blob/`` form is the GitHub web UI, not the file bytes. Without
  ``curl -f`` a 404 still exits 0, so the recipe wrote an HTML error page to
  disk and the following ``pip install -c`` failed parsing ``<!DOCTYPE html>``.
  ``raw.githubusercontent.com`` is the form that serves the file itself.
* No ``constraints-*.txt`` had ever been committed on any ref, so all five
  advertised URLs 404'd regardless of their form.

These tests pin both properties for every future ``curl`` the README grows: the
URL must be the raw form, and its path must be present in the checkout.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re
import sys

import pytest

# Opening or closing fence of a Markdown code block.
_FENCE_RE = re.compile(r'^\s*```')
_URL_RE = re.compile(r'https?://\S+')
# ``curl`` as the command being run, at the start of a line or after a shell
# operator, so a filename that merely contains "curl" does not match.
_CURL_RE = re.compile(r'(?:^\s*|[|&;]\s*)curl\s')
_BLOB_RE = re.compile(r'^https://github\.com/[^/]+/[^/]+/blob/')
_RAW_RE = re.compile(
    r'^https://raw\.githubusercontent\.com/google/adk-python/[^/]+/(?P<path>.+)$'
)

_TEST_DIR = Path(__file__).parent


def _find_readme(start: Path = _TEST_DIR) -> Path | None:
  """Locates the repository ``README.md`` by walking up from ``start``.

  A ``README.md`` only counts when it sits beside ``pyproject.toml``, which
  identifies the repository root rather than a sample or package readme. The
  test tree may be symlinked, so the walk avoids ``.resolve()``. Returns
  ``None`` when the layout does not expose one, which the internal source tree
  does not.
  """
  for candidate in [start, *start.parents]:
    readme = candidate / 'README.md'
    if readme.is_file() and (candidate / 'pyproject.toml').is_file():
      return readme
  return None


def _iter_fetched_urls(markdown: str) -> list[str]:
  """Returns the URLs passed to ``curl`` inside fenced code blocks.

  URLs in prose and URLs on non-``curl`` lines are excluded: this guard is
  about files a copy-pasting reader actually downloads, not about every link
  the document mentions. Every fence language is searched rather than only
  ``bash``, so relabelling a block ```` ```sh ```` cannot silently retire the
  guard. ``_CURL_RE`` matching only in command position is what keeps a
  ``curl`` mentioned inside a ```` ```python ```` block from counting.
  """
  urls: list[str] = []
  in_fence = False
  for line in markdown.splitlines():
    if _FENCE_RE.match(line):
      in_fence = not in_fence
    elif in_fence and _CURL_RE.search(line):
      urls.extend(_URL_RE.findall(line))
  return urls


def _github_blob_urls(urls: Iterable[str]) -> list[str]:
  """Returns the URLs served as the GitHub web UI instead of file bytes."""
  return [url for url in urls if _BLOB_RE.match(url)]


def _repo_relative_paths(urls: Iterable[str]) -> list[str]:
  """Returns the in-repo paths that raw URLs for this repository resolve to.

  URLs for another host or another repository are ignored; this guard can only
  speak for files that live in this checkout.
  """
  matches = (_RAW_RE.match(url) for url in urls)
  return [match.group('path') for match in matches if match]


def _readme_or_skip() -> Path:
  """Returns the repository ``README.md``, skipping when there is none."""
  readme = _find_readme()
  if readme is None:
    pytest.skip(
        'Not a full source checkout: no README.md beside pyproject.toml when '
        f'walking up from {_TEST_DIR}.'
    )
  return readme


def test_iter_fetched_urls_extracts_curl_targets() -> None:
  markdown = '\n'.join([
      'Prose mentioning https://example.com/prose.txt inline.',
      '',
      '```bash',
      'curl -o f.txt https://example.com/flagged.txt',
      'curl https://example.com/bare.txt',
      # Not a curl invocation, even though the filename contains "curl".
      'pip install google-adk -c https://example.com/named-curl.txt',
      '```',
      '',
      # Relabelling the fence must not retire the guard.
      '```sh',
      'curl https://example.com/other-fence.txt',
      '```',
      '',
      # "curl" outside command position is a mention, not an invocation.
      '```python',
      'urlopen("https://example.com/not-a-command.txt")  # curl',
      '```',
      '',
      'Trailing prose https://example.com/after.txt.',
      # Excluded only if the closing fence was honoured: a tracker that never
      # leaves the block would read this sentence as a curl invocation.
      'curl is discussed at https://example.com/prose-curl.txt in prose.',
  ])

  assert _iter_fetched_urls(markdown) == [
      'https://example.com/flagged.txt',
      'https://example.com/bare.txt',
      'https://example.com/other-fence.txt',
  ]


def test_github_blob_urls_flags_blob_and_allows_raw() -> None:
  blob = 'https://github.com/google/adk-python/blob/main/constraints-3.10.txt'
  raw = (
      'https://raw.githubusercontent.com/google/adk-python/main/'
      'constraints-3.10.txt'
  )

  assert _github_blob_urls([blob, raw]) == [blob]


def test_repo_relative_paths_maps_raw_urls() -> None:
  urls = [
      'https://raw.githubusercontent.com/google/adk-python/main/README.md',
      'https://raw.githubusercontent.com/other/repo/main/README.md',
      'https://github.com/google/adk-python/blob/main/README.md',
  ]

  assert _repo_relative_paths(urls) == ['README.md']


def test_find_readme_returns_none_when_absent(tmp_path: Path) -> None:
  # A README.md with no pyproject.toml beside it is not a repository root, so
  # the walk must pass over it and run out of parents.
  orphan = tmp_path / 'nested'
  orphan.mkdir()
  (orphan / 'README.md').write_text('# not a repository root\n')

  assert _find_readme(orphan) is None


def test_readme_or_skip_skips_outside_a_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # The internal source tree exposes no README.md beside pyproject.toml, so the
  # two tests below have to skip there rather than fail on a missing file.
  monkeypatch.setattr(
      sys.modules[__name__], '_find_readme', lambda start=_TEST_DIR: None
  )

  with pytest.raises(pytest.skip.Exception, match='Not a full source checkout'):
    _readme_or_skip()


def test_readme_curl_targets_use_raw_urls() -> None:
  readme = _readme_or_skip()

  blob_urls = _github_blob_urls(_iter_fetched_urls(readme.read_text()))

  assert not blob_urls, (
      f'{readme} tells users to curl the GitHub blob view: {blob_urls}. That '
      'URL serves the web UI as text/html, so curl writes an HTML page to the '
      'output file instead of the file bytes. Use the raw form, '
      'https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>.'
  )


def test_readme_curl_targets_exist_in_repo() -> None:
  readme = _readme_or_skip()
  repo_root = readme.parent

  paths = _repo_relative_paths(_iter_fetched_urls(readme.read_text()))
  # simplicity: presence in the checkout stands in for "tracked by git". CI
  # runs on a clean checkout, where the two agree; upgrade to `git ls-files`
  # only if this needs to hold on a dirty working tree too.
  missing = [path for path in paths if not (repo_root / path).exists()]

  assert not missing, (
      f'{readme} tells users to curl {missing}, which the repository does not '
      'publish, so the download 404s. Commit the file or drop the instruction.'
  )
