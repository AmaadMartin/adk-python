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

"""Helpers for speccing third-party client doubles against the installed SDK.

A bare MagicMock accepts any attribute and any call signature, so an upstream
rename leaves the tests that use one green. These helpers build doubles with
`create_autospec(..., spec_set=True)` instead, which fails on both.
"""

from __future__ import annotations

import inspect
import typing
from unittest import mock

import vertexai

# The SDK exposes agent_engines and sandboxes as properties, and create_autospec
# does not descend into a property. A probe client resolves the three classes so
# each one can be specced, without deep-importing private modules. They are read
# here, at import time, because the tests patch vertexai.Client itself.
_PROBE_CLIENT = vertexai.Client(project='test-project', location='us-central1')
_CLIENT_CLS = type(_PROBE_CLIENT)
_AGENT_ENGINES_CLS = type(_PROBE_CLIENT.agent_engines)
_SANDBOXES_CLS = type(_PROBE_CLIENT.agent_engines.sandboxes)


def autospec_property(cls: type, name: str) -> mock.MagicMock:
  """Autospecs the object a property returns.

  create_autospec() does not descend into properties -- the attribute comes
  back as a bare MagicMock that accepts anything -- so the handler behind one
  has to be specced from its own declared type.

  Args:
    cls: The class that declares the property.
    name: The property name.

  Returns:
    A double specced against the property's declared return type.
  """
  hints = typing.get_type_hints(inspect.getattr_static(cls, name).fget)
  return mock.create_autospec(hints['return'], instance=True, spec_set=True)


def make_vertexai_client() -> mock.MagicMock:
  """Builds a vertexai.Client double specced against the installed SDK.

  autospec_property() cannot resolve Client.agent_engines: its annotation names
  a module alias that does not exist at runtime, so get_type_hints() raises
  NameError. The classes come from the probe client above instead.

  Returns:
    A client double whose agent_engines and sandboxes handlers are specced too.
  """
  client = mock.create_autospec(_CLIENT_CLS, instance=True, spec_set=True)
  client.agent_engines = mock.create_autospec(
      _AGENT_ENGINES_CLS, instance=True, spec_set=True
  )
  client.agent_engines.sandboxes = mock.create_autospec(
      _SANDBOXES_CLS, instance=True, spec_set=True
  )
  return client
