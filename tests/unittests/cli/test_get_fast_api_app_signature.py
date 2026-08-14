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

"""Unit tests pinning the return annotation of `get_fast_api_app`."""

from __future__ import annotations

import inspect
import typing

from fastapi import FastAPI
from google.adk.cli.api_server import ApiServer
from google.adk.cli.dev_server import DevServer
from google.adk.cli.fast_api import get_fast_api_app
import pytest


def test_api_server_get_fast_api_app_returns_fastapi_annotation():
  hints = typing.get_type_hints(ApiServer.get_fast_api_app)

  assert hints["return"] is FastAPI


def test_dev_server_get_fast_api_app_returns_fastapi_annotation():
  hints = typing.get_type_hints(DevServer.get_fast_api_app)

  assert hints["return"] is FastAPI


def test_dev_server_get_fast_api_app_is_fully_annotated():
  """Guards the partial-annotation state that mypy strict rejects."""
  parameters = inspect.signature(DevServer.get_fast_api_app).parameters

  unannotated = [
      name
      for name, parameter in parameters.items()
      if name != "self" and parameter.annotation is inspect.Parameter.empty
  ]
  assert not unannotated


@pytest.mark.parametrize("web", [False, True])
def test_get_fast_api_app_annotation_matches_runtime_type(tmp_path, web):
  app = get_fast_api_app(
      agents_dir=str(tmp_path),
      web=web,
      session_service_uri="",
      artifact_service_uri="",
      memory_service_uri="",
      allow_origins=["*"],
      a2a=False,
      host="127.0.0.1",
      port=8000,
  )

  assert isinstance(app, FastAPI)
  # web=True must reach DevServer.get_fast_api_app, not the ApiServer fallback.
  has_dev_routes = any(
      getattr(route, "path", "").startswith("/dev/") for route in app.routes
  )
  assert has_dev_routes is web
