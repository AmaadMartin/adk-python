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

"""Tests that a lazy provider whose module fails to import is explained."""

import logging
import re
import sys

from google.adk import models
from google.adk.models import registry
from google.adk.models.base_llm import BaseLlm
from google.adk.models.google_llm import Gemini
from google.adk.models.registry import LLMRegistry
import pytest

_MISSING_MODULE = 'google.adk.models._nonexistent_provider'
_MISSING_MODULE_ERROR = f"No module named '{_MISSING_MODULE}'"
_DROP_MESSAGE = 'Dropping lazy LLM registry entry'


@pytest.fixture(autouse=True)
def restore_registry_state():
  """Keeps registry mutations from leaking between tests."""
  registry_snapshot = dict(registry._llm_registry_dict)
  failed_snapshot = dict(registry._failed_lazy_imports)
  registry._failed_lazy_imports.clear()
  LLMRegistry.resolve.cache_clear()
  try:
    yield
  finally:
    registry._llm_registry_dict.clear()
    registry._llm_registry_dict.update(registry_snapshot)
    registry._failed_lazy_imports.clear()
    registry._failed_lazy_imports.update(failed_snapshot)
    LLMRegistry.resolve.cache_clear()


def _register_missing_provider(regex: str = r'fake-provider-.*') -> None:
  """Registers a lazy entry whose module genuinely does not exist."""
  LLMRegistry._register_lazy([regex], _MISSING_MODULE, 'FakeLlm')


def _drop_records(caplog: pytest.LogCaptureFixture) -> list[str]:
  return [r.getMessage() for r in caplog.records if _DROP_MESSAGE in r.message]


def test_dropped_lazy_entry_is_logged(caplog):
  _register_missing_provider()

  with caplog.at_level(logging.DEBUG, logger=registry.logger.name):
    with pytest.raises(ValueError):
      LLMRegistry.resolve('fake-provider-1')

  messages = _drop_records(caplog)
  assert len(messages) == 1
  assert 'fake-provider-.*' in messages[0]
  assert _MISSING_MODULE in messages[0]
  assert 'FakeLlm' in messages[0]
  assert _MISSING_MODULE_ERROR in messages[0]


def test_dropped_lazy_entry_is_removed_from_registry():
  _register_missing_provider()

  with pytest.raises(ValueError):
    LLMRegistry.resolve('fake-provider-1')

  assert 'fake-provider-.*' not in registry._llm_registry_dict
  assert registry._failed_lazy_imports['fake-provider-.*'][0] == _MISSING_MODULE


def test_error_names_the_provider_module_that_failed():
  _register_missing_provider()

  with pytest.raises(
      ValueError,
      match=(
          r'(?s)Model fake-provider-1 not found\.'
          rf'.*{re.escape(_MISSING_MODULE)}'
      ),
  ):
    LLMRegistry.resolve('fake-provider-1')


def test_error_still_names_provider_after_entry_was_dropped():
  _register_missing_provider()

  with pytest.raises(ValueError):
    LLMRegistry.resolve('fake-provider-1')

  with pytest.raises(
      ValueError,
      match=(
          r'(?s)Model fake-provider-2 not found\.'
          rf'.*{re.escape(_MISSING_MODULE)}'
      ),
  ):
    LLMRegistry.resolve('fake-provider-2')


def test_generic_hint_still_appended_for_provider_style_model():
  _register_missing_provider(r'fakeprov/.*')

  with pytest.raises(ValueError) as exc_info:
    LLMRegistry.resolve('fakeprov/x')

  error_msg = str(exc_info.value)
  assert _MISSING_MODULE in error_msg
  assert 'litellm package' in error_msg
  assert 'Provider-style models' in error_msg


def test_unaffected_model_error_is_unchanged():
  _register_missing_provider()
  with pytest.raises(ValueError):
    LLMRegistry.resolve('fake-provider-1')

  with pytest.raises(ValueError) as exc_info:
    LLMRegistry.resolve('totally-unrelated-model')

  assert str(exc_info.value) == 'Model totally-unrelated-model not found.'


def test_successful_lazy_import_is_not_logged(caplog):
  with caplog.at_level(logging.DEBUG, logger=registry.logger.name):
    assert LLMRegistry.resolve('gemini-2.5-flash') is Gemini

  assert not _drop_records(caplog)
  assert not registry._failed_lazy_imports


def test_prefix_resolution_still_returns_the_lazily_imported_class():
  LLMRegistry._register_lazy(
      [r'zzz-unused-.*'], 'google.adk.models.base_llm', 'BaseLlm'
  )

  assert LLMRegistry.resolve('base:some-model') is BaseLlm


def test_unmatched_prefix_falls_through_to_the_not_found_error():
  with pytest.raises(ValueError, match='Model nosuchprefix:x not found'):
    LLMRegistry.resolve('nosuchprefix:x')


def test_prefix_resolution_still_raises_import_error():
  _register_missing_provider()

  with pytest.raises(ImportError, match=re.escape(_MISSING_MODULE)):
    LLMRegistry.resolve('fake:some-model')

  assert 'fake-provider-.*' in registry._llm_registry_dict


def test_litellm_unavailable_is_logged(caplog, monkeypatch):
  # A None entry in sys.modules makes `import litellm` raise ImportError.
  monkeypatch.setitem(sys.modules, 'litellm', None)

  with caplog.at_level(logging.DEBUG, logger=registry.logger.name):
    with pytest.raises(ValueError):
      LLMRegistry.resolve('unknown-provider/some-model')

  assert any(
      'LiteLLM is not available for provider unknown-provider' in message
      for message in caplog.messages
  )


def test_getattr_import_error_names_module_and_chains(monkeypatch):
  monkeypatch.setitem(
      models._OTHER_LAZY_IMPORTS, 'FakeLlm', '_nonexistent_provider'
  )

  with pytest.raises(
      ImportError, match=re.escape(f'importing `{_MISSING_MODULE}`')
  ) as exc_info:
    getattr(models, 'FakeLlm')

  assert _MISSING_MODULE_ERROR in str(exc_info.value)
  assert isinstance(exc_info.value.__cause__, ImportError)


def test_getattr_import_error_for_absolute_module_path(monkeypatch):
  monkeypatch.setitem(
      models._OTHER_LAZY_IMPORTS,
      'FakeLlm',
      'google.adk._nonexistent_provider',
  )

  with pytest.raises(
      ImportError,
      match=re.escape('importing `google.adk._nonexistent_provider`'),
  ):
    getattr(models, 'FakeLlm')


def test_getattr_returns_the_lazily_imported_class():
  assert getattr(models, 'Gemini') is Gemini


def test_getattr_unknown_name_still_raises_attribute_error():
  with pytest.raises(AttributeError, match='has no attribute'):
    getattr(models, 'NoSuchAttribute')
