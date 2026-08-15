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

from typing import Any

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
from fastapi.openapi.models import SecuritySchemeType
from google.adk.auth.auth_schemes import AuthScheme
from google.adk.auth.auth_schemes import CustomAuthScheme
from google.adk.auth.auth_schemes import OAuthGrantType
from google.adk.auth.auth_schemes import OpenIdConnectWithConfig
from google.adk.auth.auth_tool import AuthConfig
import pydantic
import pytest

_TOKEN_URL = 'https://example.com/token'
_AUTH_URL = 'https://example.com/authorize'
_API_KEY_NAME = 'X-Api-Key'

_ADAPTER = pydantic.TypeAdapter(AuthScheme)


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


# The tests below pin the round-trip of the `AuthScheme` union. The union
# members come from fastapi, which aliases `type_` to `type` and `in_` to `in`
# and does not set `populate_by_name`. ADK serializes by field name, so without
# the alias-restoring validator an `APIKey` reloads as another member and the
# API key is never sent.


def _api_key(location: APIKeyIn) -> APIKey:
  return APIKey(**{'type': 'apiKey', 'in': location, 'name': _API_KEY_NAME})


def _oauth2() -> OAuth2:
  return OAuth2(
      flows=OAuthFlows(
          authorizationCode=OAuthFlowAuthorizationCode(
              authorizationUrl=_AUTH_URL,
              tokenUrl=_TOKEN_URL,
              scopes={'documents.read': 'Read your documents'},
          )
      )
  )


def _restored_from_union(
    scheme: pydantic.BaseModel, **dump_kwargs: Any
) -> list[Any]:
  """Round-trips a scheme through the union, in JSON mode and in python mode.

  Both modes matter: a plain `BaseModel` holding an `AuthScheme` validates the
  nested union straight from JSON, while `AuthConfig` declares its own
  `__init__` and therefore validates it from a python dict.
  """
  return [
      _ADAPTER.validate_json(scheme.model_dump_json(**dump_kwargs)),
      _ADAPTER.validate_python(scheme.model_dump(**dump_kwargs)),
  ]


@pytest.mark.parametrize(
    'location',
    [APIKeyIn.header, APIKeyIn.query, APIKeyIn.cookie],
    ids=lambda location: location.value,
)
def test_api_key_round_trips_through_default_dump(location):
  """The default dump keys by field name, which is the degraded form."""
  scheme = _api_key(location)
  config = AuthConfig(auth_scheme=scheme)

  restored = _restored_from_union(scheme)
  restored.append(
      AuthConfig.model_validate_json(config.model_dump_json()).auth_scheme
  )

  for api_key in restored:
    assert isinstance(api_key, APIKey)
    assert api_key.type_ == SecuritySchemeType.apiKey
    assert api_key.in_ == location
    assert api_key.name == _API_KEY_NAME


@pytest.mark.parametrize(
    'location',
    [APIKeyIn.header, APIKeyIn.query, APIKeyIn.cookie],
    ids=lambda location: location.value,
)
def test_api_key_round_trips_through_dict_dump(location):
  """A dict dump degrades the same way a JSON dump does."""
  scheme = _api_key(location)
  config = AuthConfig(auth_scheme=scheme)

  restored = AuthConfig.model_validate(config.model_dump()).auth_scheme

  assert isinstance(restored, APIKey)
  assert restored.in_ == location
  assert restored.name == _API_KEY_NAME


def test_api_key_round_trips_through_aliased_dump():
  """The aliased dump is the form `_stable_model_digest` already writes."""
  scheme = _api_key(APIKeyIn.header)

  for api_key in _restored_from_union(scheme, by_alias=True, exclude_none=True):
    assert isinstance(api_key, APIKey)
    assert api_key.in_ == APIKeyIn.header


@pytest.mark.parametrize(
    'scheme',
    [
        pytest.param(HTTPBearer(bearerFormat='JWT'), id='http-bearer'),
        pytest.param(_oauth2(), id='oauth2'),
        pytest.param(
            OpenIdConnectWithConfig(
                authorization_endpoint=_AUTH_URL, token_endpoint=_TOKEN_URL
            ),
            id='openid-connect-with-config',
        ),
        pytest.param(
            CustomAuthScheme(**{'type': 'myCustomScheme'}), id='custom'
        ),
    ],
)
@pytest.mark.parametrize(
    'dump_kwargs',
    [
        pytest.param({}, id='default'),
        pytest.param({'by_alias': True, 'exclude_none': True}, id='aliased'),
    ],
)
def test_other_schemes_still_round_trip(scheme, dump_kwargs):
  """Only `APIKey` was broken; the other members must not move."""
  for restored in _restored_from_union(scheme, **dump_kwargs):
    assert type(restored) is type(scheme)


@pytest.mark.parametrize(
    ('payload', 'expected_type'),
    [
        pytest.param(
            {'type': 'apiKey', 'in': 'header', 'name': 'k'},
            APIKey,
            id='api-key',
        ),
        # `HTTPBase`, not `HTTPBearer`, is what the union has always picked
        # here. Pinned so that changing it stays a deliberate act.
        pytest.param(
            {'type': 'http', 'scheme': 'bearer'}, HTTPBase, id='http-bearer'
        ),
        pytest.param(
            {'type': 'customThing', 'extra': 1},
            CustomAuthScheme,
            id='custom',
        ),
    ],
)
def test_raw_openapi_dict_still_validates(payload, expected_type):
  """A raw OpenAPI security scheme is already aliased and must pass through."""
  assert type(_ADAPTER.validate_python(payload)) is expected_type


def test_alias_wins_when_both_key_forms_present():
  """A payload carrying both forms keeps the aliased value."""
  restored = _ADAPTER.validate_python(
      {'type': 'apiKey', 'in': 'header', 'in_': 'query', 'name': 'k'}
  )

  assert isinstance(restored, APIKey)
  assert restored.in_ == APIKeyIn.header


def test_scheme_instance_passes_through_unchanged():
  """An already-constructed scheme is not a dict and must not be rewritten."""
  scheme = _api_key(APIKeyIn.header)

  assert _ADAPTER.validate_python(scheme) is scheme


def test_none_validates_under_optional_auth_scheme():
  """`ParsedOperation.auth_scheme` is optional, so `None` must still validate."""
  adapter = pydantic.TypeAdapter(AuthScheme | None)

  assert adapter.validate_python(None) is None


@pytest.mark.parametrize(
    'payload',
    [
        pytest.param({'type': 123}, id='non-string-type'),
        pytest.param('not-a-scheme', id='not-a-mapping'),
    ],
)
def test_invalid_payload_still_raises(payload):
  """The validator must not absorb a payload no member accepts."""
  with pytest.raises(pydantic.ValidationError):
    _ADAPTER.validate_python(payload)
