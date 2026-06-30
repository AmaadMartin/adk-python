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

"""Tests for OAuth2CredentialExchanger."""

import copy
import time
from unittest.mock import MagicMock

from google.adk.auth.auth_credential import AuthCredential, AuthCredentialTypes, HttpAuth, HttpCredentials, OAuth2Auth
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.tools.openapi_tool.auth.credential_exchangers import OAuth2CredentialExchanger
from google.adk.tools.openapi_tool.auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
import pytest


@pytest.fixture
def oauth2_exchanger():
  return OAuth2CredentialExchanger()


@pytest.fixture
def auth_scheme():
  openid_config = OpenIdConnectWithConfig(
      type_=AuthSchemeType.openIdConnect,
      authorization_endpoint="https://example.com/auth",
      token_endpoint="https://example.com/token",
      scopes=["openid", "profile"],
  )
  return openid_config


def _create_oauth2_credential(
    client_id="test_client",
    client_secret="test_secret",
    redirect_uri="http://localhost:8080",
    access_token=None,
    expires_at=None,
    refresh_token=None,
    auth_response_uri=None,
):
  return AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
      oauth2=OAuth2Auth(
          client_id=client_id,
          client_secret=client_secret,
          redirect_uri=redirect_uri,
          access_token=access_token,
          expires_at=expires_at,
          refresh_token=refresh_token,
          auth_response_uri=auth_response_uri,
      ),
  )


def test_check_scheme_credential_type_success(oauth2_exchanger, auth_scheme):
  auth_credential = _create_oauth2_credential()
  # Check that the method does not raise an exception
  oauth2_exchanger._check_scheme_credential_type(auth_scheme, auth_credential)


def test_check_scheme_credential_type_missing_credential(
    oauth2_exchanger, auth_scheme
):
  # Test case: auth_credential is None
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(auth_scheme, None)
  assert "auth_credential is empty" in str(exc_info.value)


def test_check_scheme_credential_type_invalid_scheme_type(
    oauth2_exchanger, auth_scheme: OpenIdConnectWithConfig
):
  """Test case: Invalid AuthSchemeType."""
  # Test case: Invalid AuthSchemeType
  invalid_scheme = copy.deepcopy(auth_scheme)
  invalid_scheme.type_ = AuthSchemeType.apiKey
  auth_credential = _create_oauth2_credential()
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(
        invalid_scheme, auth_credential
    )
  assert "Invalid security scheme" in str(exc_info.value)


def test_check_scheme_credential_type_missing_openid_connect(
    oauth2_exchanger, auth_scheme
):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.OAUTH2,
  )
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger._check_scheme_credential_type(auth_scheme, auth_credential)
  assert "auth_credential is not configured with oauth2" in str(exc_info.value)


def test_generate_auth_token_success(
    oauth2_exchanger, auth_scheme, monkeypatch
):
  """Test case: Successful generation of access token."""
  # Test case: Successful generation of access token
  auth_credential = _create_oauth2_credential(
      access_token="test_access_token",
      auth_response_uri="https://example.com/callback?code=test_code",
  )
  updated_credential = oauth2_exchanger.generate_auth_token(auth_credential)

  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "test_access_token"


def test_exchange_credential_generate_auth_token(
    oauth2_exchanger, auth_scheme, monkeypatch
):
  """Test exchange_credential when auth_response_uri is present."""
  auth_credential = _create_oauth2_credential(
      access_token="test_access_token",
      auth_response_uri="https://example.com/callback?code=test_code",
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert updated_credential.auth_type == AuthCredentialTypes.HTTP
  assert updated_credential.http.scheme == "bearer"
  assert updated_credential.http.credentials.token == "test_access_token"


def test_exchange_credential_auth_missing(oauth2_exchanger, auth_scheme):
  """Test exchange_credential when auth_credential is missing."""
  with pytest.raises(ValueError) as exc_info:
    oauth2_exchanger.exchange_credential(auth_scheme, None)
  assert "auth_credential is empty. Please create AuthCredential using" in str(
      exc_info.value
  )


@pytest.mark.parametrize(
    "access_token, expires_at, refresh_token",
    [
        ("test_access_token", int(time.time()) + 3600, "test_refresh_token"), # Not expired
        ("expired_access_token", int(time.time()) - 10, None), # Expired, no refresh token
    ]
)
def test_exchange_credential_no_refresh_attempted(
    oauth2_exchanger, auth_scheme, mocker, access_token, expires_at, refresh_token
):
  auth_credential = _create_oauth2_credential(
      access_token=access_token,
      expires_at=expires_at,
      refresh_token=refresh_token,
  )

  mock_create_session = mocker.patch(
      "google.adk.tools.openapi_tool.auth.credential_exchangers.oauth2_exchanger.create_oauth2_session"
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  mock_create_session.assert_not_called()
  assert updated_credential.http.credentials.token == access_token


@pytest.mark.parametrize(
    "initial_access_token, initial_expires_at",
    [
        ("expired_access_token", int(time.time()) - 10),
        (None, None),
    ]
)
def test_exchange_credential_refresh_success(
    oauth2_exchanger, auth_scheme, mocker, initial_access_token, initial_expires_at
):
  auth_credential = _create_oauth2_credential(
      access_token=initial_access_token,
      expires_at=initial_expires_at,
      refresh_token="test_refresh_token",
  )

  mock_client = MagicMock()
  mock_client.refresh_token.return_value = {
      "access_token": "new_access_token",
      "refresh_token": "new_refresh_token",
      "expires_at": int(time.time()) + 3600,
  }

  mock_create_session = mocker.patch(
      "google.adk.tools.openapi_tool.auth.credential_exchangers.oauth2_exchanger.create_oauth2_session",
      return_value=(mock_client, "https://example.com/token"),
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  mock_create_session.assert_called_once_with(auth_scheme, auth_credential)
  mock_client.refresh_token.assert_called_once_with(
      url="https://example.com/token",
      refresh_token="test_refresh_token",
  )

  # Check mutation of input credential
  assert auth_credential.oauth2.access_token == "new_access_token"
  assert auth_credential.oauth2.refresh_token == "new_refresh_token"

  # Check returned credential
  assert updated_credential.http.credentials.token == "new_access_token"


@pytest.mark.parametrize(
    "initial_access_token, initial_expires_at, expected_final_token, expected_credential_returned",
    [
        ("expired_access_token", int(time.time()) - 10, "expired_access_token", True),
        (None, None, None, False),
    ]
)
def test_exchange_credential_refresh_failure(
    oauth2_exchanger, auth_scheme, mocker, initial_access_token, initial_expires_at, expected_final_token, expected_credential_returned
):
  auth_credential = _create_oauth2_credential(
      access_token=initial_access_token,
      expires_at=initial_expires_at,
      refresh_token="test_refresh_token",
  )

  mock_client = MagicMock()
  mock_client.refresh_token.side_effect = Exception("Refresh failed")

  mock_create_session = mocker.patch(
      "google.adk.tools.openapi_tool.auth.credential_exchangers.oauth2_exchanger.create_oauth2_session",
      return_value=(mock_client, "https://example.com/token"),
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  mock_create_session.assert_called_once_with(auth_scheme, auth_credential)
  
  assert auth_credential.oauth2.access_token == expected_final_token
  
  if expected_credential_returned:
    assert updated_credential.http.credentials.token == expected_final_token
  else:
    assert updated_credential is None


def test_generate_auth_token_missing_access_token(oauth2_exchanger):
  auth_credential = _create_oauth2_credential()
  updated_credential = oauth2_exchanger.generate_auth_token(auth_credential)
  assert updated_credential == auth_credential


def test_exchange_credential_already_http(oauth2_exchanger, auth_scheme):
  auth_credential = AuthCredential(
      auth_type=AuthCredentialTypes.HTTP,
      http=HttpAuth(
          scheme="bearer",
          credentials=HttpCredentials(token="existing_token"),
      ),
  )
  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )
  assert updated_credential == auth_credential


def test_exchange_credential_refresh_session_creation_failure(
    oauth2_exchanger, auth_scheme, mocker
):
  auth_credential = _create_oauth2_credential(
      client_id=None,
      client_secret=None,
      access_token="expired_access_token",
      expires_at=int(time.time()) - 10,
      refresh_token="test_refresh_token",
  )

  updated_credential = oauth2_exchanger.exchange_credential(
      auth_scheme, auth_credential
  )

  assert updated_credential.http.credentials.token == "expired_access_token"

