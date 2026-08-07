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

"""Sphinx configuration for the google.adk API reference smoke build.

This configuration exists so that CI exercises the `docs` optional-dependency
group: `scripts/build_api_docs.py` copies this directory into a temporary
source tree and builds the whole `google.adk` package with it. Nothing here is
published.

The rendered API reference on https://google.github.io/adk-docs/ is built by
the google/adk-docs repository, which keeps its own copy of this
configuration. Keep the extension list below in sync with that generator, and
keep both in sync with the `docs` extra in pyproject.toml.
"""

from __future__ import annotations

from google.adk.version import __version__

project = 'google-adk'
author = 'Google LLC'
version = __version__
release = __version__

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.autodoc_pydantic',
]

html_theme = 'furo'

autoclass_content = 'both'
autodoc_pydantic_model_show_config_summary = False

# Several ADK models hold fields that pydantic cannot put in a JSON schema,
# such as httpx.Client. autodoc-pydantic then rebuilds the model in its own
# module namespace, where the annotations no longer resolve, and the build
# aborts. The smoke gate therefore renders no JSON schemas.
autodoc_pydantic_model_show_json = False
