# mypy: ignore-errors
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


from google.adk.integrations.eventarc import EventarcToolConfig
from pydantic import ValidationError
import pytest


class TestEventarcToolConfig:

  def test_valid_config(self):
    config = EventarcToolConfig(project_id="my-project")
    assert config.project_id == "my-project"

    config2 = EventarcToolConfig()
    assert config2.project_id is None

  def test_invalid_config(self):
    with pytest.raises(ValidationError):
      EventarcToolConfig(project_id=123)
