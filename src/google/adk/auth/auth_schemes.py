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

from enum import Enum
from typing import Annotated
from typing import Any
from typing import List
from typing import Optional
from typing import Union

from fastapi.openapi.models import OAuth2
from fastapi.openapi.models import OAuthFlows
from fastapi.openapi.models import SecurityBase
from fastapi.openapi.models import SecurityScheme
from fastapi.openapi.models import SecuritySchemeType
from pydantic import BeforeValidator
from pydantic import Field

from ..utils.feature_decorator import experimental
from .auth_credential import BaseModelWithConfig


class OpenIdConnectWithConfig(SecurityBase):
  type_: SecuritySchemeType = Field(
      default=SecuritySchemeType.openIdConnect, alias="type"
  )
  authorization_endpoint: str
  token_endpoint: str
  userinfo_endpoint: Optional[str] = None
  revocation_endpoint: Optional[str] = None
  token_endpoint_auth_methods_supported: Optional[List[str]] = None
  grant_types_supported: Optional[List[str]] = None
  scopes: Optional[List[str]] = None


class CustomAuthScheme(BaseModelWithConfig):
  """A flexible model for custom authentication schemes.

  The subclasses must define a `default` for the `type_` field, if using OAuth2
  user consent flow, to ensure correct rehydration.
  """

  type_: str = Field(alias="type")


_FIELD_NAME_TO_ALIAS = {"type_": "type", "in_": "in"}


def _restore_security_scheme_aliases(value: Any) -> Any:
  """Rewrites field-name keys to the OpenAPI aliases the union expects.

  The OpenAPI security-scheme models re-used from fastapi alias ``type_`` to
  ``type`` and ``in_`` to ``in``, and they do not set ``populate_by_name``, so
  they only validate from the aliased form. ADK models serialize by field name
  by default, which would otherwise leave the union unable to restore an
  ``APIKey``: no member accepts a ``type_`` key, so the payload falls through
  to whichever member absorbs every key as an extra.

  Args:
    value: The value being validated into an ``AuthScheme``.

  Returns:
    ``value`` unchanged unless it is a mapping that carries a field name whose
    alias is absent, in which case a copy with the alias restored.
  """
  if not isinstance(value, dict):
    return value
  restored = None
  for field_name, alias in _FIELD_NAME_TO_ALIAS.items():
    if field_name in value and alias not in value:
      if restored is None:
        restored = dict(value)
      restored[alias] = restored.pop(field_name)
  return value if restored is None else restored


# AuthSchemes contains SecuritySchemes from OpenAPI 3.0, an extra flattened
# OpenIdConnectWithConfig, and supports external schemes
# that subclass CustomAuthScheme.
AuthScheme = Annotated[
    Union[SecurityScheme, OpenIdConnectWithConfig, CustomAuthScheme],
    BeforeValidator(_restore_security_scheme_aliases),
]


class OAuthGrantType(str, Enum):
  """Represents the OAuth2 flow (or grant type)."""

  CLIENT_CREDENTIALS = "client_credentials"
  AUTHORIZATION_CODE = "authorization_code"
  IMPLICIT = "implicit"
  PASSWORD = "password"

  @staticmethod
  def from_flow(flow: OAuthFlows) -> Optional["OAuthGrantType"]:
    """Converts an OAuthFlows object to a OAuthGrantType."""
    if flow.clientCredentials:
      return OAuthGrantType.CLIENT_CREDENTIALS
    if flow.authorizationCode:
      return OAuthGrantType.AUTHORIZATION_CODE
    if flow.implicit:
      return OAuthGrantType.IMPLICIT
    if flow.password:
      return OAuthGrantType.PASSWORD
    return None


# AuthSchemeType re-exports SecuritySchemeType from OpenAPI 3.0.
AuthSchemeType = SecuritySchemeType


@experimental
class ExtendedOAuth2(OAuth2):
  """OAuth2 scheme that incorporates auto-discovery for endpoints."""

  issuer_url: Optional[str] = None  # Used for endpoint-discovery
