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

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.integrations import api_registry
from google.adk.integrations.api_registry import ApiRegistry
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
import pytest
import requests

_PROJECT_ID = "test-project"
_LOCATION = "global"

MOCK_MCP_SERVERS_LIST = {
    "mcpServers": [
        {
            "name": "test-mcp-server-1",
            "urls": ["mcp.server1.com"],
        },
        {
            "name": "test-mcp-server-2",
            "urls": ["mcp.server2.com"],
        },
        {
            "name": "test-mcp-server-no-url",
        },
        {
            "name": "test-mcp-server-http",
            "urls": ["http://mcp.server_http.com"],
        },
        {
            "name": "test-mcp-server-https",
            "urls": ["https://mcp.server_https.com"],
        },
    ]
}


@pytest.fixture
def mock_credentials():
  credentials = MagicMock()
  credentials.token = "mock_token"
  credentials.refresh = MagicMock()
  credentials.quota_project_id = None
  with patch(
      "google.auth.default",
      return_value=(credentials, None),
      autospec=True,
  ):
    yield credentials


@pytest.fixture
def mock_session(mock_credentials):
  with (
      patch(
          "google.auth.transport.requests.AuthorizedSession",
          autospec=True,
      ) as mock_session_class,
      patch(
          "google.adk.integrations.api_registry.api_registry._mtls_utils.use_client_cert_effective",
          return_value=False,
      ),
  ):
    session = mock_session_class.return_value
    session.__enter__.return_value = session
    yield session


class TestApiRegistry:
  """Unit tests for ApiRegistry."""

  def test_deprecation_warning(self, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    with pytest.warns(DeprecationWarning, match="ApiRegistry is deprecated"):
      ApiRegistry(api_registry_project_id=_PROJECT_ID, location=_LOCATION)

  def test_init_success(self, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    assert len(api_registry_instance._mcp_servers) == 5
    assert "test-mcp-server-1" in api_registry_instance._mcp_servers
    assert "test-mcp-server-2" in api_registry_instance._mcp_servers
    assert "test-mcp-server-no-url" in api_registry_instance._mcp_servers
    assert "test-mcp-server-http" in api_registry_instance._mcp_servers
    assert "test-mcp-server-https" in api_registry_instance._mcp_servers
    mock_session.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{_PROJECT_ID}/locations/{_LOCATION}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false"},
    )

  def test_init_with_quota_project_id_success(
      self, mock_credentials, mock_session
  ):
    mock_credentials.quota_project_id = "quota-project"
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    assert len(api_registry_instance._mcp_servers) == 5
    mock_session.get.assert_called_once_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{_PROJECT_ID}/locations/{_LOCATION}/mcpServers",
        headers={
            "Content-Type": "application/json",
            "x-goog-user-project": "quota-project",
        },
        params={"filter": "enabled=false"},
    )

  def test_init_with_pagination_success(self, mock_session):
    mock_response1 = MagicMock()
    mock_response1.json.return_value = {
        "mcpServers": [
            {
                "name": "test-mcp-server-1",
                "urls": ["mcp.server1.com"],
            },
            {
                "name": "test-mcp-server-2",
                "urls": ["mcp.server2.com"],
            },
        ],
        "nextPageToken": "next_page_token",
    }
    mock_response2 = MagicMock()
    mock_response2.json.return_value = {
        "mcpServers": [
            {
                "name": "test-mcp-server-no-url",
            },
            {
                "name": "test-mcp-server-http",
                "urls": ["http://mcp.server_http.com"],
            },
            {
                "name": "test-mcp-server-https",
                "urls": ["https://mcp.server_https.com"],
            },
        ]
    }
    mock_session.get.side_effect = [mock_response1, mock_response2]

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    assert len(api_registry_instance._mcp_servers) == 5
    assert mock_session.get.call_count == 2
    mock_session.get.assert_any_call(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{_PROJECT_ID}/locations/{_LOCATION}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false"},
    )
    mock_session.get.assert_called_with(
        f"https://cloudapiregistry.googleapis.com/v1beta/projects/{_PROJECT_ID}/locations/{_LOCATION}/mcpServers",
        headers={
            "Content-Type": "application/json",
        },
        params={"filter": "enabled=false", "pageToken": "next_page_token"},
    )

  def test_init_http_error(self, mock_session):
    mock_session.get.side_effect = requests.exceptions.RequestException(
        "Connection failed"
    )

    with pytest.raises(RuntimeError, match="Error fetching MCP servers"):
      ApiRegistry(api_registry_project_id=_PROJECT_ID, location=_LOCATION)

  def test_init_bad_response(self, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock(
        side_effect=requests.exceptions.HTTPError(
            "Not Found", request=MagicMock(), response=MagicMock()
        )
    )
    mock_session.get.return_value = mock_response

    with pytest.raises(RuntimeError, match="Error fetching MCP servers"):
      ApiRegistry(api_registry_project_id=_PROJECT_ID, location=_LOCATION)
    mock_response.raise_for_status.assert_called_once()

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_success(self, MockMcpToolset, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    toolset = api_registry_instance.get_toolset("test-mcp-server-1")

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={"Authorization": "Bearer mock_token"},
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    assert toolset == MockMcpToolset.return_value

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_with_quota_project_id_success(
      self, MockMcpToolset, mock_credentials, mock_session
  ):
    mock_credentials.quota_project_id = "quota-project"
    mock_response = MagicMock()
    mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    toolset = api_registry_instance.get_toolset("test-mcp-server-1")

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={
                "Authorization": "Bearer mock_token",
                "x-goog-user-project": "quota-project",
            },
        ),
        tool_filter=None,
        tool_name_prefix=None,
        header_provider=None,
    )
    assert toolset == MockMcpToolset.return_value

  @patch(
      "google.adk.integrations.api_registry.api_registry.McpToolset",
      autospec=True,
  )
  def test_get_toolset_with_filter_and_prefix(
      self, MockMcpToolset, mock_session
  ):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )
    tool_filter = ["tool1"]
    tool_name_prefix = "prefix_"
    toolset = api_registry_instance.get_toolset(
        "test-mcp-server-1",
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
    )

    MockMcpToolset.assert_called_once_with(
        connection_params=StreamableHTTPConnectionParams(
            url="https://mcp.server1.com",
            headers={"Authorization": "Bearer mock_token"},
        ),
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=None,
    )
    assert toolset == MockMcpToolset.return_value

  def test_get_toolset_url_scheme(self, mock_session):
    params = [
        ("test-mcp-server-http", "http://mcp.server_http.com"),
        ("test-mcp-server-https", "https://mcp.server_https.com"),
    ]
    for mock_server_name, mock_url in params:
      with (
          patch.object(
              api_registry.api_registry, "McpToolset", autospec=True
          ) as MockMcpToolset,
      ):
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
        mock_session.get.return_value = mock_response

        api_registry_instance = ApiRegistry(
            api_registry_project_id=_PROJECT_ID, location=_LOCATION
        )

        api_registry_instance.get_toolset(mock_server_name)

        MockMcpToolset.assert_called_once_with(
            connection_params=StreamableHTTPConnectionParams(
                url=mock_url,
                headers={"Authorization": "Bearer mock_token"},
            ),
            tool_filter=None,
            tool_name_prefix=None,
            header_provider=None,
        )

  def test_get_toolset_server_not_found(self, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    with pytest.raises(ValueError, match="not found in API Registry"):
      api_registry_instance.get_toolset("non-existent-server")

  def test_get_toolset_server_no_url(self, mock_session):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value=MOCK_MCP_SERVERS_LIST)
    mock_session.get.return_value = mock_response

    api_registry_instance = ApiRegistry(
        api_registry_project_id=_PROJECT_ID, location=_LOCATION
    )

    with pytest.raises(ValueError, match="has no URLs"):
      api_registry_instance.get_toolset("test-mcp-server-no-url")


class TestApiRegistryMtls:

  @patch(
      "google.auth.transport.mtls.has_default_client_cert_source",
      return_value=True,
  )
  @patch("google.auth.transport.mtls.default_client_cert_source")
  @patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"})
  def test_init_configures_mtls(
      self, mock_cert_source, _mock_has_cert, mock_credentials
  ):
    mock_cert_source.return_value = lambda: (b"cert", b"key")
    with (
        patch(
            "google.adk.integrations.api_registry.api_registry._mtls_utils.use_client_cert_effective",
            return_value=True,
        ),
        patch(
            "google.auth.transport.requests.AuthorizedSession",
            autospec=True,
        ) as mock_session_class,
    ):
      mock_response = MagicMock()
      mock_response.raise_for_status = MagicMock()
      mock_response.json.return_value = MOCK_MCP_SERVERS_LIST
      mock_session = mock_session_class.return_value
      mock_session.__enter__.return_value = mock_session
      mock_session.get.return_value = mock_response

      _ = ApiRegistry(api_registry_project_id=_PROJECT_ID, location=_LOCATION)

      mock_session.configure_mtls_channel.assert_called_once()
      args, _ = mock_session.get.call_args
      assert "cloudapiregistry.mtls.googleapis.com" in args[0]
