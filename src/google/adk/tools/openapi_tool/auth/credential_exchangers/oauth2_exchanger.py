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

from __future__ import annotations

"""Credential fetcher for OpenID Connect."""

from typing import Optional

from .....auth.auth_credential import AuthCredential
from .....auth.auth_credential import AuthCredentialTypes
from .....auth.auth_credential import HttpAuth
from .....auth.auth_credential import HttpCredentials
from .....auth.auth_schemes import AuthScheme
from .....auth.auth_schemes import AuthSchemeType
from .base_credential_exchanger import BaseAuthCredentialExchanger


class OAuth2CredentialExchanger(BaseAuthCredentialExchanger):
  """Fetches credentials for OAuth2 and OpenID Connect."""

  def _check_scheme_credential_type(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> None:
    if not auth_credential:
      raise ValueError(
          "auth_credential is empty. Please create AuthCredential using"
          " OAuth2Auth."
      )

    if auth_scheme.type_ not in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      raise ValueError(
          "Invalid security scheme, expect AuthSchemeType.openIdConnect or "
          f"AuthSchemeType.oauth2 auth scheme, but got {auth_scheme.type_}"
      )

    if not auth_credential.oauth2 and not auth_credential.http:
      raise ValueError(
          "auth_credential is not configured with oauth2. Please"
          " create AuthCredential and set OAuth2Auth."
      )

  def generate_auth_token(
      self,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Generates an auth token from the authorization response.

    Args:
        auth_scheme: The OpenID Connect or OAuth2 auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP bearer access token. If the
        HTTP bearer token cannot be generated, return the original credential.
    """

    assert auth_credential is not None
    if not auth_credential.oauth2 or not auth_credential.oauth2.access_token:
      return auth_credential

    # Return the access token as a bearer token.
    updated_credential = AuthCredential(
        auth_type=AuthCredentialTypes.HTTP,  # Store as a bearer token
        http=HttpAuth(
            scheme="bearer",
            credentials=HttpCredentials(
                token=auth_credential.oauth2.access_token
            ),
        ),
    )
    return updated_credential

  def exchange_credential(
      self,
      auth_scheme: AuthScheme,
      auth_credential: Optional[AuthCredential] = None,
  ) -> AuthCredential:
    """Exchanges the OpenID Connect auth credential for an access token or an auth URI.

    Args:
        auth_scheme: The auth scheme.
        auth_credential: The auth credential.

    Returns:
        An AuthCredential object containing the HTTP Bearer access token.

    Raises:
        ValueError: If the auth scheme or auth credential is invalid.
    """
    self._check_scheme_credential_type(auth_scheme, auth_credential)
    assert auth_credential is not None

    # If token is already HTTPBearer token, do nothing assuming that this token
    #  is valid.
    if auth_credential.http:
      return auth_credential

    if (
        auth_credential.oauth2
        and auth_credential.oauth2.refresh_token
        and not auth_credential.oauth2.access_token
    ):
      token_endpoint = getattr(auth_scheme, "token_endpoint", None)
      if token_endpoint is None:
        flows = getattr(auth_scheme, "flows", None)
        if flows:
          if flows.authorizationCode and flows.authorizationCode.tokenUrl:
            token_endpoint = flows.authorizationCode.tokenUrl
          elif flows.password and flows.password.tokenUrl:
            token_endpoint = flows.password.tokenUrl

      if not token_endpoint:
        raise ValueError(
            "Could not resolve token_endpoint from auth_scheme for refresh"
            " token flow."
        )

      import json
      from urllib.error import HTTPError
      import urllib.parse
      import urllib.request

      data = {
          "grant_type": "refresh_token",
          "refresh_token": auth_credential.oauth2.refresh_token,
      }
      if auth_credential.oauth2.client_id:
        data["client_id"] = auth_credential.oauth2.client_id
      if auth_credential.oauth2.client_secret:
        data["client_secret"] = auth_credential.oauth2.client_secret

      encoded_data = urllib.parse.urlencode(data).encode("utf-8")
      req = urllib.request.Request(
          token_endpoint, data=encoded_data, method="POST"
      )
      req.add_header("Content-Type", "application/x-www-form-urlencoded")

      try:
        with urllib.request.urlopen(req) as response:
          response_data = json.loads(response.read().decode("utf-8"))
          if "access_token" in response_data:
            auth_credential.oauth2.access_token = response_data["access_token"]
          if "expires_in" in response_data:
            auth_credential.oauth2.expires_in = response_data["expires_in"]
      except HTTPError as e:
        error_msg = e.read().decode("utf-8")
        raise ValueError(
            f"Failed to refresh token: HTTP {e.code} API Error: {error_msg}"
        )
      except Exception as e:
        raise ValueError(f"Failed to refresh token: {e}")

    # If access token is exchanged, exchange a HTTPBearer token.
    if auth_credential.oauth2 and auth_credential.oauth2.access_token:
      return self.generate_auth_token(auth_credential)

    return None  # type: ignore[return-value]
