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

"""Guard tests for ``README.md``'s references to files in this repository.

The installation section once shipped a ``curl`` of
``github.com/google/adk-python/blob/main/constraints-3.10.txt``, which was
broken twice over and nothing caught either half:

* No ``constraints-*.txt`` had ever been committed on any ref, so the URL 404'd.
* ``/blob/`` is the GitHub web UI, not the file bytes. Without ``curl -f`` a 404
  still exits 0, so the recipe wrote an HTML error page to disk and the
  following ``pip install -c`` failed parsing ``<!DOCTYPE html>``.

``test_readme_repo_links_resolve`` covers the first and more general half: any
README link into this repository -- today the logo, tomorrow a restored ``curl``
target -- must name a path that exists.
``test_readme_does_not_curl_the_blob_view`` covers the second.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest

from tests.unittests.test_release_dependencies import _find_pyproject

# A README link into this repository, capturing the path it points at. Both
# forms carry a ``<ref>/`` segment between the repository and the path.
_REPO_LINK_RE = re.compile(
    r'https://(?:raw\.githubusercontent\.com/google/adk-python'
    r'|github\.com/google/adk-python/blob)'
    r'/[^/\s]+/([^\s"\')>]+)'
)

# A ``curl`` command whose target is a GitHub blob URL, for any repository.
_CURL_BLOB_RE = re.compile(
    r'curl\b[^\n]*?(https://github\.com/[^\s/]+/[^\s/]+/blob/\S+)'
)


def _find_readme() -> Path | None:
  """Locates the repository ``README.md``, or ``None`` if there is none.

  Defers the repository-root walk to ``_find_pyproject``, which already handles
  the layouts this project is checked out in. Returns ``None`` rather than
  raising so callers can skip: the internal source tree exposes a
  ``pyproject.toml`` with no ``README.md`` beside it.
  """
  readme = _find_pyproject().parent / 'README.md'
  return readme if readme.is_file() else None


def _readme_or_skip() -> Path:
  """Returns the repository ``README.md``, skipping when there is none."""
  readme = _find_readme()
  if readme is None:
    pytest.skip('Not a full source checkout: README.md is absent.')
  return readme


def test_repo_link_re_captures_paths_into_this_repository() -> None:
  markdown = '\n'.join([
      (
          '<img src="https://raw.githubusercontent.com/google/adk-python/main/'
          'assets/logo.png">'
      ),
      'curl https://github.com/google/adk-python/blob/main/pinned.txt',
      # Another repository's files are not ours to verify.
      'https://raw.githubusercontent.com/other/repo/main/elsewhere.txt',
      'https://github.com/other/repo/blob/main/elsewhere.txt',
      # Not a link into a repository tree at all.
      'https://github.com/google/adk-python/issues/1',
      'https://example.com/main/unrelated.txt',
  ])

  assert _REPO_LINK_RE.findall(markdown) == [
      'assets/logo.png',
      'pinned.txt',
  ]


def test_curl_blob_re_matches_only_blob_downloads() -> None:
  markdown = '\n'.join([
      'curl -o p.txt https://github.com/google/adk-python/blob/main/p.txt',
      # The raw form is the correct way to download a file.
      (
          'curl -o p.txt https://raw.githubusercontent.com/google/adk-python/'
          'main/p.txt'
      ),
      # A blob URL a human is meant to click is fine; only curl is wrong.
      'See https://github.com/google/adk-python/blob/main/CONTRIBUTING.md.',
  ])

  assert _CURL_BLOB_RE.findall(markdown) == [
      'https://github.com/google/adk-python/blob/main/p.txt',
  ]


def test_find_readme_returns_none_when_readme_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  # The internal source tree exposes pyproject.toml with no README beside it.
  monkeypatch.setattr(
      sys.modules[__name__],
      '_find_pyproject',
      lambda: tmp_path / 'pyproject.toml',
  )

  assert _find_readme() is None


def test_readme_or_skip_skips_outside_a_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setattr(sys.modules[__name__], '_find_readme', lambda: None)

  with pytest.raises(pytest.skip.Exception, match='Not a full source checkout'):
    _readme_or_skip()


def test_readme_repo_links_resolve() -> None:
  readme = _readme_or_skip()
  repo_root = readme.parent

  paths = _REPO_LINK_RE.findall(readme.read_text())
  # simplicity: presence in the checkout stands in for "tracked by git". CI
  # runs on a clean checkout, where the two agree; upgrade to `git ls-files`
  # only if this needs to hold on a dirty working tree too.
  missing = [path for path in paths if not (repo_root / path).exists()]

  assert paths, f'No links into this repository were found in {readme}.'
  assert not missing, (
      f'{readme} links to {missing}, which this repository does not publish, '
      'so the link 404s. Commit the file or drop the reference.'
  )


def test_readme_does_not_curl_the_blob_view() -> None:
  readme = _readme_or_skip()

  blob_downloads = _CURL_BLOB_RE.findall(readme.read_text())

  assert not blob_downloads, (
      f'{readme} tells users to curl the GitHub blob view: {blob_downloads}. '
      'That URL serves the web UI as text/html, so curl writes an HTML page '
      'to the output file instead of the file bytes. Use the raw form, '
      'https://raw.githubusercontent.com/<owner>/<repo>/<ref>/<path>.'
  )
