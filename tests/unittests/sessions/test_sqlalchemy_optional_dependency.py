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

"""Fresh-process checks for the sessions stack without the ``db`` extra.

Every test hides sqlalchemy in a child interpreter. Imports are process-global
and irreversible, so hiding it in-process would leak into the rest of the suite:
re-importing ``sessions.schemas`` builds a second set of ORM classes on a fresh
``MetaData`` while sibling tests still hold the first set.
"""

from __future__ import annotations

import pytest

from .. import isolated_import_utils
from ..isolated_import_utils import run_isolated

pytestmark = pytest.mark.skipif(
    not isolated_import_utils.SOURCE_ROOT.is_dir(),
    reason='Import-loading checks need the source checkout layout.',
)

# Binding ``None`` in sys.modules makes both ``from sqlalchemy import X`` and
# ``from sqlalchemy.engine import Y`` raise ModuleNotFoundError, which is what
# an install without the ``db`` extra does.
_HIDE_SQLALCHEMY = """
import sys

sys.modules['sqlalchemy'] = None
"""


def _run_without_sqlalchemy(source: str) -> None:
  """Runs source in a fresh interpreter that cannot import sqlalchemy."""
  result = run_isolated(_HIDE_SQLALCHEMY + source)

  assert result.returncode == 0, result.stderr


def test_database_session_service_lookup_reports_the_db_extra():
  """The lazy accessor names the extra to install, and never a NameError."""
  _run_without_sqlalchemy("""
import google.adk.sessions as sessions

try:
  sessions.DatabaseSessionService
except ImportError as error:
  assert 'google-adk[db]' in str(error), error
else:
  raise AssertionError('DatabaseSessionService resolved without sqlalchemy.')
""")


def test_schema_check_utils_import_fails_instead_of_deferring_a_name_error():
  """_schema_check_utils refuses to import half-initialised."""
  _run_without_sqlalchemy("""
import importlib

try:
  module = importlib.import_module(
      'google.adk.sessions.migration._schema_check_utils'
  )
except ImportError as error:
  assert 'sqlalchemy' in str(error), error
else:
  try:
    module.get_db_schema_version('sqlite:///unused.db')
  except Exception as deferred:
    raise AssertionError(
        '_schema_check_utils imported without sqlalchemy and deferred'
        f' {type(deferred).__name__}: {deferred}'
    ) from deferred
  raise AssertionError('_schema_check_utils imported without sqlalchemy.')
""")


def test_database_session_service_module_import_fails_with_an_import_error():
  """The module itself still raises ImportError, not a deferred failure."""
  _run_without_sqlalchemy("""
import importlib

try:
  importlib.import_module('google.adk.sessions.database_session_service')
except ImportError as error:
  assert 'sqlalchemy' in str(error), error
else:
  raise AssertionError(
      'database_session_service imported without sqlalchemy.'
  )
""")


def test_sessions_package_imports_without_sqlalchemy():
  """A base install keeps the non-DB session services."""
  _run_without_sqlalchemy("""
import google.adk.sessions as sessions

assert sessions.InMemorySessionService
""")


def _assert_cli_session_uri_reports_the_db_extra(uri: str) -> None:
  """Asserts the CLI names the ``db`` extra for a database session URI."""
  _run_without_sqlalchemy(f"""
from google.adk.cli.utils.service_factory import (
    create_session_service_from_options,
)

try:
  create_session_service_from_options(
      base_dir='.', session_service_uri={uri!r}
  )
except ImportError as error:
  assert 'google-adk[db]' in str(error), error
else:
  raise AssertionError('The CLI built a session service without sqlalchemy.')
""")


def test_cli_registered_database_uri_reports_the_db_extra():
  """A registered database scheme routes through the translating accessor."""
  _assert_cli_session_uri_reports_the_db_extra('postgresql://user@host/db')


def test_cli_unregistered_database_uri_reports_the_db_extra():
  """The unregistered-scheme fallback routes through the same accessor."""
  _assert_cli_session_uri_reports_the_db_extra('cockroachdb://user@host/db')
