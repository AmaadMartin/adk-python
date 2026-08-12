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


import base64
import datetime
import json
from unittest import mock
import uuid

from google.adk.integrations.eventarc import _config as config
from google.adk.integrations.eventarc import _message_tool as message_tool
import google.oauth2.credentials
import pytest


class TestMessageTool:

  @pytest.fixture(autouse=True)
  def _patch_eventarc_dependencies(self):
    with (
        mock.patch.object(
            message_tool, "eventarc_client", autospec=True
        ) as mock_client_module,
        mock.patch.object(
            message_tool, "eventarc_publishing_v1", autospec=True
        ) as mock_eventarc_v1,
    ):
      self.mock_client_module = mock_client_module
      self.mock_publisher_client = mock.MagicMock(spec=["publish"])
      self.mock_publisher_client.publish = mock.AsyncMock()
      mock_client_module.get_publisher_client = mock.AsyncMock(
          return_value=self.mock_publisher_client
      )
      mock_client_module.remove_publisher_client = mock.AsyncMock()
      self.mock_eventarc_v1 = mock_eventarc_v1

      self.settings = config.EventarcToolConfig(project_id="test-project")
      self.credentials = google.oauth2.credentials.Credentials(token="fake")
      yield

  @pytest.mark.asyncio
  async def test_publish_message_success_text(self):
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello world",
    )
    assert res["status"] == "SUCCESS"
    assert "message_id" in res

    # Verify get_publisher_client was called
    self.mock_client_module.get_publisher_client.assert_called_once_with(
        credentials=self.credentials, project_id="test-project"
    )

  @pytest.mark.asyncio
  async def test_publish_message_custom_timeout(self):
    custom_settings = config.EventarcToolConfig(
        project_id="test-project", publish_timeout=30.0
    )
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=custom_settings,
        data="hello world",
    )
    assert res["status"] == "SUCCESS"
    self.mock_publisher_client.publish.assert_called_once()
    call_kwargs = self.mock_publisher_client.publish.call_args.kwargs
    assert call_kwargs.get("timeout") == 30.0

  @pytest.mark.asyncio
  async def test_publish_message_success_json(self):
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": "bar"},
    )
    assert res["status"] == "SUCCESS"

  @pytest.mark.asyncio
  async def test_publish_message_base64_encoded(self):
    encoded_data = base64.b64encode(b"binary data").decode("utf-8")
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data=encoded_data,
        is_base64_encoded=True,
    )
    assert res["status"] == "SUCCESS"

  @pytest.mark.asyncio
  async def test_publish_message_invalid_base64(self):
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="not-base64-!@#",
        is_base64_encoded=True,
    )
    assert res["status"] == "ERROR"
    assert "Invalid base64" in res["error_details"]

  @pytest.mark.asyncio
  async def test_publish_message_unserializable_json(self):
    class CustomClass:
      pass

    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": CustomClass()},
    )
    assert res["status"] == "ERROR"
    assert "Failed to serialize data" in res["error_details"]

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      ("update_kwargs", "expected_error"),
      [
          pytest.param(
              {"type": ""},
              "type must be a non-empty string",
              id="invalid_type",
          ),
          pytest.param(
              {"source": ""},
              "source must be a non-empty string",
              id="invalid_source",
          ),
          pytest.param(
              {"id": "   "},
              "id, if provided, must be a non-empty string",
              id="invalid_id",
          ),
          pytest.param(
              {"data": 123, "is_base64_encoded": True},
              "data must be a string when is_base64_encoded is True",
              id="invalid_base64_data_type",
          ),
          pytest.param(
              {"custom_attributes": "not a dict"},
              "custom_attributes must be a dict",
              id="invalid_custom_attributes_type",
          ),
          pytest.param(
              {"custom_attributes": {"InvalidKey!": "val"}},
              "Invalid custom attribute key",
              id="invalid_custom_attributes_keys",
          ),
          pytest.param(
              {"time": 12345},
              "time must be a string",
              id="invalid_time_type",
          ),
          pytest.param(
              {"time": "invalid-time"},
              "Invalid RFC 3339",
              id="invalid_time_format",
          ),
      ],
  )
  async def test_publish_message_invalid_inputs(
      self, update_kwargs, expected_error
  ):
    kwargs = {
        "bus": "bus",
        "type": "type",
        "source": "source",
        "credentials": self.credentials,
        "settings": self.settings,
    }
    kwargs.update(update_kwargs)
    res = await message_tool.publish_message(**kwargs)
    assert res["status"] == "ERROR"
    assert expected_error in res["error_details"]

  @pytest.mark.asyncio
  @pytest.mark.parametrize(
      "valid_time",
      [
          "2026-06-03T12:00:00Z",
          "2026-06-03T12:00:00.123456Z",
          "2026-06-03T12:00:00+00:00",
          "2026-06-03T12:00:00-07:00",
          "2026-06-03T12:00:00.123+02:00",
      ],
  )
  async def test_publish_message_time_valid_rfc3339(self, valid_time):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        time=valid_time,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    attributes = event_kwargs.get("attributes", {})
    assert "time" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string=valid_time
    )

  @pytest.mark.asyncio
  async def test_publish_message_exception_eviction(self):
    self.mock_publisher_client.publish.side_effect = RuntimeError("API failed")
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
    )
    assert res["status"] == "ERROR"
    assert "API failed" in res["error_details"]

    # Verify remove_publisher_client was called
    self.mock_client_module.remove_publisher_client.assert_called_once_with(
        credentials=self.credentials, project_id="test-project"
    )

  @pytest.mark.asyncio
  @mock.patch.object(message_tool, "opentelemetry", autospec=True)
  async def test_publish_message_tracing(self, mock_opentelemetry):
    def inject_mock(carrier):
      carrier["traceparent"] = "00-testtrace-testid-01"
      carrier["tracestate"] = "teststate=1"

    mock_opentelemetry.propagate.get_global_textmap.return_value.inject = (
        inject_mock
    )

    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        include_tracing_extension=True,
    )
    assert res["status"] == "SUCCESS"

    # Verify custom attributes are appended to CloudEvent
    self.mock_eventarc_v1.types.CloudEvent.assert_called_once()
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    attributes = event_kwargs.get("attributes", {})
    assert "traceparent" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="00-testtrace-testid-01"
    )

  @pytest.mark.asyncio
  async def test_publish_message_empty_string_data(self):
    # Act
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="",
    )
    # Assert
    assert res["status"] == "SUCCESS"

  @pytest.mark.asyncio
  async def test_publish_message_empty_dict_data(self):
    # Act
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data={},
    )
    # Assert
    assert res["status"] == "SUCCESS"

  @pytest.mark.asyncio
  async def test_publish_message_missing_library(self):
    with mock.patch.object(message_tool, "eventarc_publishing_v1", None):
      res = await message_tool.publish_message(
          bus="bus",
          type="type",
          source="source",
          credentials=self.credentials,
          settings=self.settings,
      )
      assert res["status"] == "ERROR"
      assert "not installed" in res["error_details"]

  @pytest.mark.asyncio
  async def test_publish_message_time_empty_string(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        time="",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert "time" not in event_kwargs.get("attributes", {})

  @pytest.mark.asyncio
  async def test_publish_message_explicit_datacontenttype(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="<xml/>",
        datacontenttype="application/xml",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == "<xml/>"
    attributes = event_kwargs.get("attributes", {})
    assert "datacontenttype" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/xml"
    )

  @pytest.mark.asyncio
  async def test_publish_message_image_payload(self):
    # Simulate an agent sending an image
    # "iVBORw0KGgo=" is a valid base64 snippet (e.g. PNG header)
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="iVBORw0KGgo=",
        is_base64_encoded=True,
        datacontenttype="image/png",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("binary_data") == b"\x89PNG\r\n\x1a\n"
    attributes = event_kwargs.get("attributes", {})
    assert "datacontenttype" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="image/png"
    )

  @pytest.mark.asyncio
  async def test_publish_message_explicit_datacontenttype_json_with_binary_data(
      self,
  ):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="e30=",  # base64 for {}
        is_base64_encoded=True,
        datacontenttype="application/json",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("binary_data") == b"{}"
    attributes = event_kwargs.get("attributes", {})
    assert "datacontenttype" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_explicit_datacontenttype_xml_with_dict_data(
      self,
  ):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data={"foo": "bar"},
        datacontenttype="application/xml",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == '{"foo": "bar"}'
    attributes = event_kwargs.get("attributes", {})
    assert "datacontenttype" in attributes
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/xml"
    )

  @pytest.mark.asyncio
  async def test_publish_message_empty_datacontenttype(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello",
        datacontenttype="",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert "datacontenttype" not in event_kwargs.get("attributes", {})

  @pytest.mark.asyncio
  async def test_publish_message_with_subject(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello",
        subject="test-subject",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert "subject" not in event_kwargs
    assert "subject" in event_kwargs.get("attributes", {})
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="test-subject"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_integer(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=12345,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == "12345"
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_boolean(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=True,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == "true"
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_list_of_dicts(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=[{"a": 1}, {"b": 2}],
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == '[{"a": 1}, {"b": 2}]'
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_unicode(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="Hello 🌍!",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == "Hello 🌍!"
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="text/plain"
    )

  @pytest.mark.asyncio
  async def test_publish_message_custom_attributes_type_casting(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        custom_attributes={"isvalid": True, "count": 42},
    )
    assert res["status"] == "SUCCESS"
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="True"
    )
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="42"
    )

  @pytest.mark.asyncio
  async def test_publish_message_explicit_specversion(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        specversion="1.1",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("spec_version") == "1.1"

  @pytest.mark.asyncio
  async def test_publish_message_explicit_id(self):
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        id="custom-event-id-99",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("id") == "custom-event-id-99"
    assert res["message_id"] == "custom-event-id-99"

  @pytest.mark.asyncio
  async def test_publish_message_base64_without_datacontenttype(self):
    # Simulate an agent sending base64 but forgetting the datacontenttype
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data="YmluYXJ5",  # 'binary'
        is_base64_encoded=True,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("binary_data") == b"binary"
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/octet-stream"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_deeply_nested_dict(self):
    nested_data = {
        "user": {
            "id": 101,
            "profile": {
                "name": "Alice",
                "preferences": {
                    "notifications": {"email": True, "sms": False},
                    "tags": ["premium", "beta-tester"],
                },
            },
            "history": [
                {"action": "login", "timestamp": "2026-06-04T00:00:00Z"},
                {
                    "action": "purchase",
                    "details": {"item_id": 999, "amount": 42.5},
                },
            ],
        },
        "metadata": {
            "source": "mobile-app",
            "version": [1, 2, {"build": "rc1"}],
        },
    }
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=nested_data,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == json.dumps(nested_data)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_data_deeply_nested_list(self):
    nested_list = [
        [1, 2, [3, 4, [5, {"six": 6}]]],
        {"seven": [8, 9]},
        "ten",
        True,
        None,
        [{"eleven": {"twelve": [13, 14]}}],
    ]
    res = await message_tool.publish_message(
        bus="bus",
        type="type",
        source="source",
        credentials=self.credentials,
        settings=self.settings,
        data=nested_list,
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs
    assert event_kwargs.get("text_data") == json.dumps(nested_list)
    self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.assert_any_call(
        ce_string="application/json"
    )

  @pytest.mark.asyncio
  async def test_publish_message_auto_generated_attributes(self):
    res = await message_tool.publish_message(
        bus="projects/test/locations/global/messageBuses/my-bus",
        type="com.example.test",
        source="//test/source",
        credentials=self.credentials,
        settings=self.settings,
        data="hello world",
    )
    assert res["status"] == "SUCCESS"
    event_kwargs = self.mock_eventarc_v1.types.CloudEvent.call_args.kwargs

    # Assert ID is a valid UUIDv4
    generated_id = event_kwargs.get("id")
    assert generated_id is not None
    uuid_obj = uuid.UUID(generated_id, version=4)
    assert str(uuid_obj) == generated_id

    # Assert Time is auto-generated and valid RFC 3339
    attributes = event_kwargs.get("attributes", {})
    assert "time" in attributes

    # We need to find the specific CloudEventAttributeValue mock call that corresponds to the time attribute.
    # The actual implementation in message_tool.py populates it in custom_attr["time"] = time_attr
    # Let's inspect the attributes dictionary passed to CloudEvent.
    # We just need to check if ANY of the calls to CloudEventAttributeValue contain a valid RFC 3339 string
    # that could be the time. A simpler approach is to check if it parses via fromisoformat after replacing Z.

    time_val = None
    for (
        call
    ) in (
        self.mock_eventarc_v1.types.CloudEvent.CloudEventAttributeValue.mock_calls
    ):
      ce_string = call.kwargs.get("ce_string")
      if ce_string and (
          "T" in ce_string
          and ("Z" in ce_string or "+" in ce_string or "-" in ce_string)
      ):
        # Attempt to parse it
        try:
          dt = datetime.datetime.fromisoformat(ce_string.replace("Z", "+00:00"))
          time_val = ce_string
          break
        except ValueError:
          continue

    assert time_val is not None, (
        "Failed to find a valid RFC 3339 auto-generated time string in the"
        " attributes."
    )
