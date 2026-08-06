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

from collections.abc import Callable
from collections.abc import Sequence
import importlib
import os
import pathlib
import sys
from types import ModuleType

import pytest

from scripts import build_api_docs


def _make_package(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    relpaths: Sequence[str],
) -> ModuleType:
  """Creates an importable package tree under tmp_path and imports it."""
  root = tmp_path / name
  for relpath in ('__init__.py', *relpaths):
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('', encoding='utf-8')
  monkeypatch.syspath_prepend(str(tmp_path))
  importlib.invalidate_caches()
  return importlib.import_module(name)


def _make_source_dir(tmp_path: pathlib.Path) -> pathlib.Path:
  """Creates a stand-in for the checked-in Sphinx source directory."""
  source_dir = tmp_path / 'source'
  source_dir.mkdir()
  (source_dir / 'conf.py').write_text('', encoding='utf-8')
  (source_dir / 'index.rst').write_text('', encoding='utf-8')
  return source_dir


def _fake_sphinx_run(
    status: int, warning_text: str, recorder: dict[str, str]
) -> Callable[[list[str]], int]:
  """Returns a `_run_sphinx` stand-in that records the build it was asked for."""

  def build(argv: list[str]) -> int:
    source_dir = pathlib.Path(argv[-2])
    recorder['argv'] = ' '.join(argv)
    recorder['source_dir'] = str(source_dir)
    recorder['output_dir'] = argv[-1]
    recorder['document'] = (source_dir / 'google-adk.rst').read_text(
        encoding='utf-8'
    )
    pathlib.Path(argv[argv.index('-w') + 1]).write_text(
        warning_text, encoding='utf-8'
    )
    return status

  return build


@pytest.fixture(name='allowlisted_package')
def _allowlisted_package(monkeypatch: pytest.MonkeyPatch) -> None:
  """Puts google.adk.integrations.agent_identity on the allowlist."""
  monkeypatch.setattr(
      build_api_docs,
      '_ALLOWED_IMPORT_FAILURES',
      frozenset({'google.adk.integrations.agent_identity'}),
  )


def test_discover_modules_returns_sorted_public_submodules(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(
      tmp_path, monkeypatch, 'pkg_public', ('zeta.py', 'alpha.py')
  )

  assert build_api_docs.discover_modules(package) == [
      'pkg_public.alpha',
      'pkg_public.zeta',
  ]


def test_discover_modules_skips_private_components(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(
      tmp_path,
      monkeypatch,
      'pkg_private',
      (
          'foo/__init__.py',
          'foo/public.py',
          'foo/_private.py',
          '_internal/__init__.py',
          '_internal/bar.py',
      ),
  )

  assert build_api_docs.discover_modules(package) == [
      'pkg_private.foo',
      'pkg_private.foo.public',
  ]


def test_discover_modules_recurses_to_arbitrary_depth(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(
      tmp_path,
      monkeypatch,
      'pkg_deep',
      ('one/__init__.py', 'one/two/__init__.py', 'one/two/three.py'),
  )

  assert 'pkg_deep.one.two.three' in build_api_docs.discover_modules(package)


def test_discover_modules_returns_empty_for_empty_package(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_empty', ())

  assert build_api_docs.discover_modules(package) == []


def test_render_rst_emits_one_block_per_module() -> None:
  document = build_api_docs.render_rst(['a.b', 'a.c'])

  assert document.count('.. automodule::') == 2
  assert '.. automodule:: a.b\n' in document
  assert '.. automodule:: a.c\n' in document
  assert document.count('   :members:\n') == 2
  assert document.count('   :undoc-members:\n') == 2
  assert document.count('   :show-inheritance:\n') == 2


def test_render_rst_escapes_underscores_in_headings() -> None:
  document = build_api_docs.render_rst(['a.my_module'])

  assert r'a.my\_module' in document
  assert '.. automodule:: a.my_module\n' in document


def test_render_rst_of_no_modules_is_a_header_only_document() -> None:
  document = build_api_docs.render_rst([])

  assert document == 'API Reference\n=============\n'


def test_find_unexpected_import_failures_of_empty_text() -> None:
  assert build_api_docs.find_unexpected_import_failures('') == []


@pytest.mark.usefixtures('allowlisted_package')
def test_find_unexpected_import_failures_ignores_allowlisted_module() -> None:
  warnings = (
      "WARNING: autodoc: failed to import module 'integrations.agent_identity'"
      " from module 'google.adk'; the following exception was raised: boom\n"
  )

  assert build_api_docs.find_unexpected_import_failures(warnings) == []


def test_find_unexpected_import_failures_reports_other_module() -> None:
  warnings = (
      "WARNING: autodoc: failed to import module 'agents' from module"
      " 'google.adk'; the following exception was raised: boom\n"
  )

  assert build_api_docs.find_unexpected_import_failures(warnings) == [
      'google.adk.agents'
  ]


@pytest.mark.usefixtures('allowlisted_package')
def test_find_unexpected_import_failures_ignores_allowlisted_submodule() -> (
    None
):
  warnings = (
      'WARNING: autodoc: failed to import module'
      " 'integrations.agent_identity.gcp_auth_provider' from module"
      " 'google.adk'; the following exception was raised: boom\n"
  )

  assert build_api_docs.find_unexpected_import_failures(warnings) == []


@pytest.mark.usefixtures('allowlisted_package')
def test_find_unexpected_import_failures_reports_only_unknown_modules() -> None:
  warnings = (
      "WARNING: autodoc: failed to import module 'integrations.agent_identity'"
      " from module 'google.adk'; the following exception was raised: boom\n"
      "WARNING: autodoc: failed to import module 'tools' from module"
      " 'google.adk'; the following exception was raised: boom\n"
      "WARNING: autodoc: failed to import module 'google.adk.plugins'; the"
      ' following exception was raised: boom\n'
  )

  assert build_api_docs.find_unexpected_import_failures(warnings) == [
      'google.adk.plugins',
      'google.adk.tools',
  ]


def test_find_unexpected_import_failures_ignores_other_warnings() -> None:
  warnings = (
      'docs/x.rst:12: WARNING: Unexpected indentation.\n'
      "WARNING: autosummary: failed to import module 'google.adk.agents'.\n"
  )

  assert build_api_docs.find_unexpected_import_failures(warnings) == []


def test_run_sphinx_delegates_to_the_sphinx_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  recorded: list[list[str]] = []
  fake_module = ModuleType('sphinx.cmd.build')

  def build_main(argv: list[str]) -> int:
    recorded.append(argv)
    return 7

  fake_module.build_main = build_main
  monkeypatch.setitem(sys.modules, 'sphinx.cmd.build', fake_module)

  assert build_api_docs._run_sphinx(['-b', 'html']) == 7
  assert recorded == [['-b', 'html']]


def test_default_source_dir_holds_the_checked_in_sphinx_config() -> None:
  source_dir = build_api_docs._default_source_dir()

  assert os.path.basename(source_dir) == 'api-reference'
  assert os.path.isfile(os.path.join(source_dir, 'conf.py'))
  assert os.path.isfile(os.path.join(source_dir, 'index.rst'))


@pytest.mark.usefixtures('allowlisted_package')
def test_main_succeeds_when_sphinx_reports_no_unknown_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_ok', ('alpha.py',))
  monkeypatch.setattr(build_api_docs, '_ROOT_PACKAGE', package.__name__)
  recorder: dict[str, str] = {}
  monkeypatch.setattr(
      build_api_docs,
      '_run_sphinx',
      _fake_sphinx_run(
          0,
          "WARNING: autodoc: failed to import module 'integrations."
          "agent_identity' from module 'google.adk'; raised: boom\n",
          recorder,
      ),
  )
  source_dir = _make_source_dir(tmp_path)

  assert build_api_docs.main(['--source-dir', str(source_dir)]) == 0
  assert capsys.readouterr().err == ''
  assert '.. automodule:: pkg_ok.alpha\n' in recorder['document']
  assert '-T' in recorder['argv'].split()


def test_main_builds_outside_the_source_directory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_clean', ('alpha.py',))
  monkeypatch.setattr(build_api_docs, '_ROOT_PACKAGE', package.__name__)
  recorder: dict[str, str] = {}
  monkeypatch.setattr(
      build_api_docs, '_run_sphinx', _fake_sphinx_run(0, '', recorder)
  )
  source_dir = _make_source_dir(tmp_path)

  assert build_api_docs.main(['--source-dir', str(source_dir)]) == 0
  assert recorder['source_dir'] != str(source_dir)
  assert not (source_dir / 'google-adk.rst').exists()
  assert sorted(p.name for p in source_dir.iterdir()) == [
      'conf.py',
      'index.rst',
  ]


def test_main_writes_html_to_the_requested_output_dir(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_out', ('alpha.py',))
  monkeypatch.setattr(build_api_docs, '_ROOT_PACKAGE', package.__name__)
  recorder: dict[str, str] = {}
  monkeypatch.setattr(
      build_api_docs, '_run_sphinx', _fake_sphinx_run(0, '', recorder)
  )
  source_dir = _make_source_dir(tmp_path)
  output_dir = tmp_path / 'html'

  assert (
      build_api_docs.main([
          '--source-dir',
          str(source_dir),
          '--output-dir',
          str(output_dir),
      ])
      == 0
  )
  assert recorder['output_dir'] == str(output_dir)


def test_main_returns_the_sphinx_exit_code(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_fail', ('alpha.py',))
  monkeypatch.setattr(build_api_docs, '_ROOT_PACKAGE', package.__name__)
  recorder: dict[str, str] = {}
  monkeypatch.setattr(
      build_api_docs,
      '_run_sphinx',
      _fake_sphinx_run(
          2,
          "WARNING: autodoc: failed to import module 'agents' from module"
          " 'google.adk'; raised: boom\n",
          recorder,
      ),
  )
  source_dir = _make_source_dir(tmp_path)

  assert build_api_docs.main(['--source-dir', str(source_dir)]) == 2
  assert capsys.readouterr().err == ''


def test_main_fails_on_an_unexpected_autodoc_import_failure(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
  package = _make_package(tmp_path, monkeypatch, 'pkg_broken', ('alpha.py',))
  monkeypatch.setattr(build_api_docs, '_ROOT_PACKAGE', package.__name__)
  recorder: dict[str, str] = {}
  monkeypatch.setattr(
      build_api_docs,
      '_run_sphinx',
      _fake_sphinx_run(
          0,
          "WARNING: autodoc: failed to import module 'agents' from module"
          " 'google.adk'; raised: boom\n",
          recorder,
      ),
  )
  source_dir = _make_source_dir(tmp_path)

  assert build_api_docs.main(['--source-dir', str(source_dir)]) == 1
  stderr = capsys.readouterr().err
  assert 'google.adk.agents' in stderr
  assert '_ALLOWED_IMPORT_FAILURES' in stderr


def test_no_import_failure_is_allowlisted_today() -> None:
  assert build_api_docs._ALLOWED_IMPORT_FAILURES == frozenset()
