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
import logging
from typing import Any
from typing import Dict
from typing import Optional
from unittest import mock

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.auth.auth_credential import AuthCredential
from google.adk.auth.auth_credential import AuthCredentialTypes
from google.adk.auth.auth_credential import OAuth2Auth
from google.adk.auth.auth_credential import ServiceAccount
from google.adk.auth.auth_credential import ServiceAccountCredential
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.sessions.state import State
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.google_api_tool.google_api_tool import GoogleApiTool
from google.adk.tools.google_api_tool.google_api_toolset import GoogleApiToolset
from google.adk.tools.google_api_tool.google_api_toolsets import BigQueryToolset
from google.adk.tools.google_api_tool.google_api_toolsets import CalendarToolset
from google.adk.tools.google_api_tool.google_api_toolsets import DocsToolset
from google.adk.tools.google_api_tool.google_api_toolsets import GmailToolset
from google.adk.tools.google_api_tool.google_api_toolsets import SheetsToolset
from google.adk.tools.google_api_tool.google_api_toolsets import SlidesToolset
from google.adk.tools.google_api_tool.google_api_toolsets import YoutubeToolset
from google.adk.tools.google_api_tool.googleapi_to_openapi_converter import GoogleApiToOpenApiConverter
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset
from google.adk.tools.openapi_tool.openapi_spec_parser.rest_api_tool import RestApiTool
from google.adk.tools.tool_context import ToolContext
import httpx
import pytest

TEST_API_NAME = "calendar"
TEST_API_VERSION = "v3"
DEFAULT_SCOPE = "https://www.googleapis.com/auth/calendar"
TOOLSET_MODULE = "google.adk.tools.google_api_tool.google_api_toolset"


@pytest.fixture
def mock_rest_api_tool():
  """Fixture for a mock RestApiTool."""
  mock_tool = mock.MagicMock(spec=RestApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  return mock_tool


@pytest.fixture
def mock_google_api_tool_instance(
    mock_rest_api_tool,
):  # Renamed from mock_google_api_tool
  """Fixture for a mock GoogleApiTool instance."""
  mock_tool = mock.MagicMock(spec=GoogleApiTool)
  mock_tool.name = "test_tool"
  mock_tool.description = "Test Tool Description"
  mock_tool.rest_api_tool = mock_rest_api_tool
  return mock_tool


@pytest.fixture
def mock_rest_api_tools():
  """Fixture for a list of mock RestApiTools."""
  tools = []
  for i in range(3):
    mock_tool = mock.MagicMock(
        spec=RestApiTool, description=f"Test Tool Description {i}"
    )
    mock_tool.name = f"test_tool_{i}"
    tools.append(mock_tool)
  return tools


@pytest.fixture
def mock_openapi_toolset_instance():  # Renamed from mock_openapi_toolset
  """Fixture for a mock OpenAPIToolset instance."""
  mock_toolset = mock.MagicMock(spec=OpenAPIToolset)
  # Mock async methods if they are called
  mock_toolset.get_tools = mock.AsyncMock(return_value=[])
  mock_toolset.close = mock.AsyncMock()
  return mock_toolset


@pytest.fixture
def mock_converter_instance():  # Renamed from mock_converter
  """Fixture for a mock GoogleApiToOpenApiConverter instance."""
  mock_conv = mock.MagicMock(spec=GoogleApiToOpenApiConverter)
  mock_conv.convert.return_value = {
      "components": {
          "securitySchemes": {
              "oauth2": {
                  "flows": {
                      "authorizationCode": {
                          "scopes": {
                              DEFAULT_SCOPE: "Full access to Google Calendar"
                          }
                      }
                  }
              }
          }
      }
  }
  return mock_conv


@pytest.fixture
def mock_readonly_context():
  """Fixture for a mock ReadonlyContext."""
  return mock.MagicMock(spec=ReadonlyContext)


class TestGoogleApiToolset:
  """Test suite for the GoogleApiToolset class."""

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    client_id = "test_client_id"
    client_secret = "test_client_secret"
    additional_headers = {"developer-token": "abc123"}

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id=client_id,
        client_secret=client_secret,
        additional_headers=additional_headers,
    )

    assert tool_set.api_name == TEST_API_NAME
    assert tool_set.api_version == TEST_API_VERSION
    assert tool_set._client_id == client_id
    assert tool_set._client_secret == client_secret
    assert tool_set._service_account is None
    assert tool_set.tool_filter is None
    assert tool_set._openapi_toolset == mock_openapi_toolset_instance
    assert tool_set._additional_headers == additional_headers

    mock_converter_class.assert_called_once_with(
        TEST_API_NAME, TEST_API_VERSION, discovery_url=None
    )
    mock_converter_instance.convert.assert_called_once()
    spec_dict = mock_converter_instance.convert.return_value

    mock_openapi_toolset_class.assert_called_once()
    _, kwargs = mock_openapi_toolset_class.call_args
    assert kwargs["spec_dict"] == spec_dict
    assert kwargs["spec_str_type"] == "yaml"
    assert isinstance(kwargs["auth_scheme"], OpenIdConnectWithConfig)
    assert kwargs["auth_scheme"].scopes == [DEFAULT_SCOPE]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init_with_additional_scopes(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization with additional scopes."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    extra_scopes = [
        DEFAULT_SCOPE,
        "https://www.googleapis.com/auth/calendar.readonly",
    ]
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        additional_scopes=extra_scopes,
    )

    mock_openapi_toolset_class.assert_called_once()
    _, kwargs = mock_openapi_toolset_class.call_args
    assert isinstance(kwargs["auth_scheme"], OpenIdConnectWithConfig)
    assert kwargs["auth_scheme"].scopes == [
        DEFAULT_SCOPE,
        "https://www.googleapis.com/auth/calendar.readonly",
    ]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init_with_discovery_url(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization with custom discovery URL."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    discovery_url = "https://example.com/discovery"
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        discovery_url=discovery_url,
    )

    mock_converter_class.assert_called_once_with(
        TEST_API_NAME, TEST_API_VERSION, discovery_url=discovery_url
    )

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiTool"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_google_api_tool_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,
      mock_readonly_context,
  ):
    """Test get_tools method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    # Setup mock GoogleApiTool instances to be returned by the constructor
    mock_google_api_tool_instances = [
        mock.MagicMock(spec=GoogleApiTool, name=f"google_tool_{i}")
        for i in range(len(mock_rest_api_tools))
    ]
    mock_google_api_tool_class.side_effect = mock_google_api_tool_instances

    client_id = "cid"
    client_secret = "csecret"
    sa_mock = mock.MagicMock(spec=ServiceAccount)
    additional_headers = {"developer-token": "token"}

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id=client_id,
        client_secret=client_secret,
        service_account=sa_mock,
        additional_headers=additional_headers,
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == len(mock_rest_api_tools)
    mock_openapi_toolset_instance.get_tools.assert_called_once_with(
        mock_readonly_context
    )

    for i, rest_tool in enumerate(mock_rest_api_tools):
      mock_google_api_tool_class.assert_any_call(
          rest_tool,
          client_id,
          client_secret,
          sa_mock,
          additional_headers=additional_headers,
      )
      assert tools[i] is mock_google_api_tool_instances[i]

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools_with_filter_list(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,  # Has test_tool_0, test_tool_1, test_tool_2
      mock_readonly_context,
      mock_converter_instance,
  ):
    """Test get_tools method with a list filter."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    tool_filter = ["test_tool_0", "test_tool_2"]
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_filter=tool_filter,
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == 2
    assert tools[0].name == "test_tool_0"
    assert tools[1].name == "test_tool_2"

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_get_tools_with_filter_predicate(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      mock_rest_api_tools,  # Has test_tool_0, test_tool_1, test_tool_2
      mock_readonly_context,
  ):
    """Test get_tools method with a predicate filter."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_openapi_toolset_instance.get_tools = mock.AsyncMock(
        return_value=mock_rest_api_tools
    )

    class MyPredicate(ToolPredicate):

      def __call__(
          self,
          tool: BaseTool,
          readonly_context: Optional[ReadonlyContext] = None,
      ) -> bool:
        return tool.name == "test_tool_1"

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_filter=MyPredicate(),
    )

    tools = await tool_set.get_tools(mock_readonly_context)

    assert len(tools) == 1
    assert tools[0].name == "test_tool_1"

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_configure_auth(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test configure_auth method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    client_id = "test_client_id"
    client_secret = "test_client_secret"

    tool_set.configure_auth(client_id, client_secret)

    assert tool_set._client_id == client_id
    assert tool_set._client_secret == client_secret

    # To verify its effect, we would ideally call get_tools and check
    # how GoogleApiTool is instantiated. This is covered in test_get_tools.

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_configure_sa_auth(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test configure_sa_auth method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    service_account = ServiceAccount(
        service_account_credential=ServiceAccountCredential(
            type="service_account",
            project_id="project_id",
            private_key_id="private_key_id",
            private_key=(
                "-----BEGIN PRIVATE KEY-----\nprivate_key\n-----END PRIVATE"
                " KEY-----\n"
            ),
            client_email="client_email",
            client_id="client_id",
            auth_uri="auth_uri",
            token_uri="token_uri",
            auth_provider_x509_cert_url="auth_provider_x509_cert_url",
            client_x509_cert_url="client_x509_cert_url",
            universe_domain="universe_domain",
        ),
        scopes=["scope1", "scope2"],
    )

    tool_set.configure_sa_auth(service_account)
    assert tool_set._service_account == service_account
    # Effect verification is covered in test_get_tools.

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  async def test_close(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test close method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )
    await tool_set.close()

    mock_openapi_toolset_instance.close.assert_called_once()

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_set_tool_filter(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test set_tool_filter method."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )

    assert tool_set.tool_filter is None

    new_filter_list = ["tool1", "tool2"]
    tool_set.set_tool_filter(new_filter_list)
    assert tool_set.tool_filter == new_filter_list

    def new_filter_predicate(
        tool_name: str,
        tool: RestApiTool,
        readonly_context: Optional[ReadonlyContext] = None,
    ) -> bool:
      return True

    tool_set.set_tool_filter(new_filter_predicate)
    assert tool_set.tool_filter == new_filter_predicate

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_init_with_tool_name_prefix(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test GoogleApiToolset initialization with tool_name_prefix."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_name_prefix = "test_prefix"
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        tool_name_prefix=tool_name_prefix,
    )

    assert tool_set.tool_name_prefix == tool_name_prefix

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.MtlsClientCerts"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.use_client_cert_effective"
  )
  async def test_mtls_cleanup_on_close(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
  ):
    """Test that mTLS temp files are cleaned up on close."""
    mock_converter_class.return_value = mock.MagicMock()
    mock_openapi_toolset_instance = mock.MagicMock()
    mock_openapi_toolset_instance.close = mock.AsyncMock()
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    mock_use_client_cert.return_value = True
    mock_mtls_certs_instance = mock.MagicMock()
    mock_mtls_certs_instance.get_certs.return_value = ("cert", "key", b"pass")
    mock_mtls_certs_class.return_value = mock_mtls_certs_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )

    assert tool_set._httpx_client_factory is not None

    await tool_set.close()

    mock_openapi_toolset_instance.close.assert_called_once()
    mock_mtls_certs_instance.close.assert_called_once()

  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.httpx.AsyncClient"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.MtlsClientCerts"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.use_client_cert_effective"
  )
  async def test_mtls_no_passphrase(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_async_client_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """Test that mTLS is configured even if key passphrase is None."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    mock_use_client_cert.return_value = True
    mock_mtls_certs_instance = mock.MagicMock()
    mock_mtls_certs_instance.get_certs.return_value = ("cert", "key", None)
    mock_mtls_certs_class.return_value = mock_mtls_certs_instance

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME, api_version=TEST_API_VERSION
    )

    assert tool_set._httpx_client_factory is not None

    client = tool_set._httpx_client_factory()
    assert client is not None
    mock_async_client_class.assert_called_once_with(cert=("cert", "key"))


def custom_factory() -> httpx.AsyncClient:
  """A caller-supplied factory: a fresh client per call, as the contract asks."""
  return httpx.AsyncClient()


@mock.patch(f"{TOOLSET_MODULE}.OpenAPIToolset")
@mock.patch(f"{TOOLSET_MODULE}.GoogleApiToOpenApiConverter")
@mock.patch(f"{TOOLSET_MODULE}.MtlsClientCerts")
@mock.patch(f"{TOOLSET_MODULE}.use_client_cert_effective")
class TestGoogleApiToolsetHttpxClientFactory:
  """Tests for the httpx_client_factory parameter and its mTLS precedence.

  `use_client_cert_effective` is patched in every test: unpatched it reads the
  ambient GOOGLE_API_USE_CLIENT_CERTIFICATE setting and the results stop being
  deterministic.
  """

  def test_no_factory_and_no_certs_leaves_factory_unset(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      caplog,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = False

    with caplog.at_level(logging.WARNING):
      tool_set = GoogleApiToolset(
          api_name=TEST_API_NAME, api_version=TEST_API_VERSION
      )

    assert tool_set._httpx_client_factory is None
    assert (
        mock_openapi_toolset_class.call_args.kwargs["httpx_client_factory"]
        is None
    )
    mock_mtls_certs_class.assert_not_called()
    assert caplog.text == ""

  def test_no_factory_with_certs_builds_cert_backed_factory(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      caplog,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = True
    mock_mtls_certs_class.return_value.get_certs.return_value = (
        "cert",
        "key",
        b"pass",
    )

    with caplog.at_level(logging.WARNING):
      tool_set = GoogleApiToolset(
          api_name=TEST_API_NAME, api_version=TEST_API_VERSION
      )

    factory = tool_set._httpx_client_factory
    assert factory is not None
    assert (
        mock_openapi_toolset_class.call_args.kwargs["httpx_client_factory"]
        is factory
    )
    with mock.patch(f"{TOOLSET_MODULE}.httpx.AsyncClient") as mock_client_class:
      factory()
    mock_client_class.assert_called_once_with(cert=("cert", "key", b"pass"))
    assert caplog.text == ""

  def test_supplied_factory_without_certs_is_forwarded(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      caplog,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = False

    with caplog.at_level(logging.WARNING):
      tool_set = GoogleApiToolset(
          api_name=TEST_API_NAME,
          api_version=TEST_API_VERSION,
          httpx_client_factory=custom_factory,
      )

    assert tool_set._httpx_client_factory is custom_factory
    assert (
        mock_openapi_toolset_class.call_args.kwargs["httpx_client_factory"]
        is custom_factory
    )
    mock_mtls_certs_class.assert_not_called()
    assert caplog.text == ""

  def test_supplied_factory_takes_precedence_over_certs_and_warns(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
      caplog,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = True

    with caplog.at_level(logging.WARNING):
      tool_set = GoogleApiToolset(
          api_name=TEST_API_NAME,
          api_version=TEST_API_VERSION,
          httpx_client_factory=custom_factory,
      )

    assert tool_set._httpx_client_factory is custom_factory
    assert (
        mock_openapi_toolset_class.call_args.kwargs["httpx_client_factory"]
        is custom_factory
    )
    # The certificates must not even be extracted: httpx cannot attach a client
    # certificate to a client the caller already built.
    mock_mtls_certs_class.assert_not_called()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "httpx_client_factory" in message
    assert "mTLS client certificate" in message
    assert TEST_API_NAME in message

  async def test_close_with_supplied_factory_does_not_raise(
      self,
      mock_use_client_cert,
      mock_mtls_certs_class,
      mock_converter_class,
      mock_openapi_toolset_class,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    """close() must cope with _mtls_certs never having been assigned."""
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = True

    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        httpx_client_factory=custom_factory,
    )

    await tool_set.close()

    mock_openapi_toolset_instance.close.assert_called_once()
    mock_mtls_certs_class.return_value.close.assert_not_called()


def _discovery_spec() -> Dict[str, Any]:
  """A minimal converted discovery document with one GET operation."""
  return {
      "openapi": "3.0.0",
      "info": {"title": "Calendar API", "version": TEST_API_VERSION},
      # A reserved .test host (RFC 6761): if the transport under test is
      # ever bypassed, the request fails to resolve instead of leaving the
      # machine.
      "servers": [{"url": "https://calendar.test/calendar/v3"}],
      "paths": {
          "/calendars/primary": {
              "get": {
                  "operationId": "get_primary_calendar",
                  "description": "Gets the primary calendar.",
                  "responses": {"200": {"description": "Successful response."}},
              }
          }
      },
      "components": {
          "securitySchemes": {
              "oauth2": {
                  "flows": {
                      "authorizationCode": {
                          "scopes": {DEFAULT_SCOPE: "Full access"}
                      }
                  }
              }
          }
      },
  }


def _authorized_tool_context() -> ToolContext:
  """A tool context that hands back an already-exchanged access token."""
  tool_context = mock.MagicMock(spec=ToolContext)
  tool_context.state = State({}, {})
  tool_context.get_auth_response.return_value = AuthCredential(
      auth_type=AuthCredentialTypes.OPEN_ID_CONNECT,
      oauth2=OAuth2Auth(
          client_id="test_client_id",
          client_secret="test_client_secret",
          access_token="test_access_token",
      ),
  )
  return tool_context


async def test_supplied_factory_issues_the_request(
    mock_converter_instance,
):
  """The factory's client, not ADK's default one, carries the API call.

  Only the discovery fetch is mocked. OpenAPIToolset, RestApiTool and the
  request helper are all real, so this exercises the whole path from the
  constructor parameter down to the wire.
  """
  seen_requests = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen_requests.append(request)
    return httpx.Response(200, json={"id": "primary"})

  def mock_transport_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))

  mock_converter_instance.convert.return_value = _discovery_spec()

  with (
      mock.patch(f"{TOOLSET_MODULE}.GoogleApiToOpenApiConverter") as converter,
      mock.patch(
          f"{TOOLSET_MODULE}.use_client_cert_effective", return_value=False
      ),
  ):
    converter.return_value = mock_converter_instance
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id="test_client_id",
        client_secret="test_client_secret",
        httpx_client_factory=mock_transport_factory,
    )

  tools = await tool_set.get_tools()
  assert [tool.name for tool in tools] == ["get_primary_calendar"]

  result = await tools[0].run_async(
      args={}, tool_context=_authorized_tool_context()
  )

  assert result == {"id": "primary"}
  assert len(seen_requests) == 1
  assert seen_requests[0].url == httpx.URL(
      "https://calendar.test/calendar/v3/calendars/primary"
  )
  await tool_set.close()


async def test_no_factory_issues_the_request_without_one(
    mock_converter_instance,
):
  """Without a factory the default client still issues the call."""
  seen_requests = []

  def handler(request: httpx.Request) -> httpx.Response:
    seen_requests.append(request)
    return httpx.Response(200, json={"id": "primary"})

  mock_converter_instance.convert.return_value = _discovery_spec()

  with (
      mock.patch(f"{TOOLSET_MODULE}.GoogleApiToOpenApiConverter") as converter,
      mock.patch(
          f"{TOOLSET_MODULE}.use_client_cert_effective", return_value=False
      ),
  ):
    converter.return_value = mock_converter_instance
    tool_set = GoogleApiToolset(
        api_name=TEST_API_NAME,
        api_version=TEST_API_VERSION,
        client_id="test_client_id",
        client_secret="test_client_secret",
    )

  tools = await tool_set.get_tools()
  original_init = httpx.AsyncClient.__init__

  def init_with_mock_transport(self, **kwargs) -> None:
    original_init(self, transport=httpx.MockTransport(handler), **kwargs)

  with mock.patch.object(
      httpx.AsyncClient, "__init__", init_with_mock_transport
  ):
    result = await tools[0].run_async(
        args={}, tool_context=_authorized_tool_context()
    )

  assert result == {"id": "primary"}
  assert len(seen_requests) == 1
  await tool_set.close()


# The (api_name, api_version) pair each prebuilt toolset is documented to
# target. The pair decides which discovery document gets fetched, so a
# copy-paste slip between these near-identical subclasses points the toolset at
# the wrong API.
PREBUILT_TOOLSETS = [
    (BigQueryToolset, "bigquery", "v2"),
    (CalendarToolset, "calendar", "v3"),
    (GmailToolset, "gmail", "v1"),
    (YoutubeToolset, "youtube", "v3"),
    (SlidesToolset, "slides", "v1"),
    (SheetsToolset, "sheets", "v4"),
    (DocsToolset, "docs", "v1"),
]


class TestPrebuiltGoogleApiToolsets:
  """Test suite for the prebuilt per-API GoogleApiToolset subclasses."""

  @pytest.mark.parametrize(
      "toolset_class, api_name, api_version", PREBUILT_TOOLSETS
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_prebuilt_toolset_targets_its_documented_api_and_version(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      toolset_class,
      api_name,
      api_version,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    tool_set = toolset_class()

    assert tool_set.api_name == api_name
    assert tool_set.api_version == api_version
    mock_converter_class.assert_called_once_with(
        api_name, api_version, discovery_url=None
    )

  @pytest.mark.parametrize(
      "toolset_class, api_name, api_version", PREBUILT_TOOLSETS
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.OpenAPIToolset"
  )
  @mock.patch(
      "google.adk.tools.google_api_tool.google_api_toolset.GoogleApiToOpenApiConverter"
  )
  def test_prebuilt_toolset_forwards_constructor_arguments(
      self,
      mock_converter_class,
      mock_openapi_toolset_class,
      toolset_class,
      api_name,
      api_version,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    # The subclasses forward these positionally, so an argument in the wrong
    # slot would silently swap, say, the client id and the client secret.
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance

    service_account = ServiceAccount(use_default_credential=True)

    tool_set = toolset_class(
        client_id="test_client_id",
        client_secret="test_client_secret",
        tool_filter=["only_this_tool"],
        service_account=service_account,
        tool_name_prefix="test_prefix",
    )

    assert tool_set._client_id == "test_client_id"
    assert tool_set._client_secret == "test_client_secret"
    assert tool_set.tool_filter == ["only_this_tool"]
    assert tool_set._service_account is service_account
    assert tool_set.tool_name_prefix == "test_prefix"

  @pytest.mark.parametrize(
      "toolset_class, api_name, api_version", PREBUILT_TOOLSETS
  )
  @mock.patch(f"{TOOLSET_MODULE}.OpenAPIToolset")
  @mock.patch(f"{TOOLSET_MODULE}.GoogleApiToOpenApiConverter")
  @mock.patch(f"{TOOLSET_MODULE}.use_client_cert_effective")
  def test_prebuilt_toolset_forwards_httpx_client_factory(
      self,
      mock_use_client_cert,
      mock_converter_class,
      mock_openapi_toolset_class,
      toolset_class,
      api_name,
      api_version,
      mock_converter_instance,
      mock_openapi_toolset_instance,
  ):
    mock_converter_class.return_value = mock_converter_instance
    mock_openapi_toolset_class.return_value = mock_openapi_toolset_instance
    mock_use_client_cert.return_value = False

    tool_set = toolset_class(httpx_client_factory=custom_factory)

    assert tool_set._httpx_client_factory is custom_factory
    assert (
        mock_openapi_toolset_class.call_args.kwargs["httpx_client_factory"]
        is custom_factory
    )

  @pytest.mark.parametrize(
      "toolset_class, _api_name, _api_version", PREBUILT_TOOLSETS
  )
  def test_prebuilt_toolset_rejects_positional_httpx_client_factory(
      self, toolset_class, _api_name, _api_version
  ):
    # The parameter is keyword-only, so it can never be mistaken for one of the
    # positional slots the subclasses forward.
    with pytest.raises(TypeError, match="positional argument"):
      toolset_class(None, None, None, None, None, custom_factory)
