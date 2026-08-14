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

"""Tests for auth scheme helpers."""

from __future__ import annotations

from fastapi.openapi.models import APIKey
from fastapi.openapi.models import APIKeyIn
from fastapi.openapi.models import HTTPBase
from fastapi.openapi.models import HTTPBearer
from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlowAuthorizationCode
from fastapi.openapi.models import OAuthFlowClientCredentials
from fastapi.openapi.models import OAuthFlowImplicit
from fastapi.openapi.models import OAuthFlowPassword
from fastapi.openapi.models import OAuthFlows
from fastapi.openapi.models import OpenIdConnect
from fastapi.openapi.models import SecuritySchemeType
from google.adk.auth import AuthConfig
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import AuthSchemeType
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
import pytest

_TOKEN_URL = 'https://example.com/token'
_AUTH_URL = 'https://example.com/authorize'
_OIDC_URL = 'https://example.com/.well-known/openid-configuration'

_JSON_DUMP_ARGS = {'by_alias': True, 'exclude_none': True, 'mode': 'json'}


def _oauth2_scheme() -> OAuth2:
  return OAuth2(
      flows=OAuthFlows(
          authorizationCode=OAuthFlowAuthorizationCode(
              authorizationUrl=_AUTH_URL,
              tokenUrl=_TOKEN_URL,
              scopes={'documents.read': 'Read your documents'},
          )
      )
  )


def _open_id_connect_with_config_scheme() -> OpenIdConnectWithConfig:
  return OpenIdConnectWithConfig(
      authorization_endpoint=_AUTH_URL, token_endpoint=_TOKEN_URL
  )


@pytest.mark.parametrize(
    ('flows', 'expected'),
    [
        pytest.param(
            OAuthFlows(
                clientCredentials=OAuthFlowClientCredentials(
                    tokenUrl=_TOKEN_URL, scopes={}
                )
            ),
            OAuthGrantType.CLIENT_CREDENTIALS,
            id='client-credentials',
        ),
        pytest.param(
            OAuthFlows(
                authorizationCode=OAuthFlowAuthorizationCode(
                    authorizationUrl=_AUTH_URL, tokenUrl=_TOKEN_URL, scopes={}
                )
            ),
            OAuthGrantType.AUTHORIZATION_CODE,
            id='authorization-code',
        ),
        pytest.param(
            OAuthFlows(
                implicit=OAuthFlowImplicit(
                    authorizationUrl=_AUTH_URL, scopes={}
                )
            ),
            OAuthGrantType.IMPLICIT,
            id='implicit',
        ),
        pytest.param(
            OAuthFlows(
                password=OAuthFlowPassword(tokenUrl=_TOKEN_URL, scopes={})
            ),
            OAuthGrantType.PASSWORD,
            id='password',
        ),
    ],
)
def test_from_flow_maps_each_configured_flow_to_its_grant_type(flows, expected):
  assert OAuthGrantType.from_flow(flows) == expected


def test_from_flow_without_any_configured_flow_returns_none():
  """An OAuth2 scheme declaring no flow has no grant type to exchange with."""
  assert OAuthGrantType.from_flow(OAuthFlows()) is None


def test_grant_type_values_are_the_oauth2_wire_names():
  # These strings go on the wire as the OAuth2 `grant_type` parameter, so
  # they must stay exactly as the spec names them.
  assert OAuthGrantType.CLIENT_CREDENTIALS.value == 'client_credentials'
  assert OAuthGrantType.AUTHORIZATION_CODE.value == 'authorization_code'
  assert OAuthGrantType.IMPLICIT.value == 'implicit'
  assert OAuthGrantType.PASSWORD.value == 'password'


# The tests below pin the parts of the public API that come from FastAPI. ADK
# re-exports FastAPI's security-scheme models rather than defining its own, so
# a user holds the same classes and the same enum on both sides of an
# `AuthConfig`. Redefining those models inside ADK would break every assertion
# here without raising anything, so each test is a guard against that.


def test_auth_scheme_type_is_fastapi_security_scheme_type():
  """User code compares `scheme.type_` against FastAPI's enum members.

  `SecuritySchemeType` is a plain `Enum`, so a look-alike ADK enum would
  never compare equal to it and every such comparison would go silently
  false.
  """
  assert AuthSchemeType is SecuritySchemeType


def test_auth_config_preserves_a_fastapi_scheme_instance():
  """The scheme a caller hands to `AuthConfig` comes back out unchanged."""
  scheme = _oauth2_scheme()

  config = AuthConfig(auth_scheme=scheme)

  assert isinstance(config.auth_scheme, OAuth2)
  assert config.auth_scheme is scheme
  assert config.auth_scheme.type_ == SecuritySchemeType.oauth2


def test_auth_scheme_union_accepts_isinstance_checks():
  """`AuthScheme` is a union a caller can use in an `isinstance` check."""
  assert isinstance(_oauth2_scheme(), AuthScheme)


@pytest.mark.parametrize(
    ('scheme', 'expected_type'),
    [
        pytest.param(_oauth2_scheme(), OAuth2, id='oauth2'),
        pytest.param(
            APIKey.model_validate({'in': APIKeyIn.header, 'name': 'X-Key'}),
            APIKey,
            id='api-key',
        ),
        pytest.param(
            OpenIdConnect(openIdConnectUrl=_OIDC_URL),
            OpenIdConnect,
            id='open-id-connect',
        ),
        pytest.param(
            _open_id_connect_with_config_scheme(),
            OpenIdConnectWithConfig,
            id='open-id-connect-with-config',
        ),
        # A bare `HTTPBearer` dumps to `{"type": "http", "scheme": "bearer"}`,
        # which `HTTPBase` matches first in FastAPI's union, so it comes back
        # as `HTTPBase`. The ordering is FastAPI's; nothing observable changes,
        # because `credential_key` is derived from the dump, not the class.
        pytest.param(HTTPBearer(), HTTPBase, id='http-bearer-to-http-base'),
    ],
)
def test_auth_config_round_trip_preserves_the_scheme_class(
    scheme, expected_type
):
  """A scheme dumped to JSON and revalidated returns the same class."""
  dumped = scheme.model_dump(**_JSON_DUMP_ARGS)

  config = AuthConfig.model_validate({'auth_scheme': dumped})

  assert type(config.auth_scheme) is expected_type


@pytest.mark.parametrize(
    'scheme',
    [
        pytest.param(_oauth2_scheme(), id='oauth2'),
        pytest.param(
            _open_id_connect_with_config_scheme(),
            id='open-id-connect-with-config',
        ),
    ],
)
def test_credential_key_is_stable_across_a_json_round_trip(scheme):
  """A drifting `credential_key` orphans every credential already stored."""
  before = AuthConfig(auth_scheme=scheme).credential_key
  dumped = scheme.model_dump(**_JSON_DUMP_ARGS)

  after = AuthConfig.model_validate({'auth_scheme': dumped}).credential_key

  assert after == before


def test_open_id_connect_with_config_serialises_snake_case_field_names():
  """`OpenIdConnectWithConfig` extends FastAPI's `SecurityBase`, not ADK's
  `BaseModelWithConfig`, so it has no `to_camel` alias generator.

  Rebasing it on `BaseModelWithConfig` would emit `authorizationEndpoint` and
  change every derived `credential_key`.
  """
  dumped = _open_id_connect_with_config_scheme().model_dump(**_JSON_DUMP_ARGS)

  assert dumped == {
      'type': 'openIdConnect',
      'authorization_endpoint': _AUTH_URL,
      'token_endpoint': _TOKEN_URL,
  }
