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

import hashlib
import logging
from typing import Literal
from typing import Optional

from pydantic import BaseModel

from ....auth.auth_credential import AuthCredential
from ....auth.auth_credential import AuthCredentialTypes
from ....auth.auth_schemes import AuthScheme
from ....auth.auth_schemes import AuthSchemeType
from ....auth.auth_tool import _stable_model_digest
from ....auth.auth_tool import AuthConfig
from ....auth.refresher.oauth2_credential_refresher import OAuth2CredentialRefresher
from ...tool_context import ToolContext
from ..auth.credential_exchangers.auto_auth_credential_exchanger import AutoAuthCredentialExchanger
from ..auth.credential_exchangers.base_credential_exchanger import AuthCredentialMissingError
from ..auth.credential_exchangers.base_credential_exchanger import BaseAuthCredentialExchanger

logger = logging.getLogger("google_adk." + __name__)

AuthPreparationState = Literal["pending", "done"]

# Suffix that keeps the tool's exchanged-credential cache out of the slots that
# `AuthConfig.credential_key` names. Those slots hold a ready-to-use credential
# owned by the credential service or by the application; this cache holds a
# credential that still needs conversion.
_EXCHANGED_CREDENTIAL_KEY_SUFFIX = "_existing_exchanged_credential"


class AuthPreparationResult(BaseModel):
  """Result of the credential preparation process."""

  state: AuthPreparationState
  auth_scheme: Optional[AuthScheme] = None
  auth_credential: Optional[AuthCredential] = None


def _resolve_credential_key(
    credential_key: Optional[str],
    auth_credential: Optional[AuthCredential],
    auth_scheme: Optional[AuthScheme],
) -> Optional[str]:
  """Returns the credential key the developer configured, if any.

  The auth models allow extra fields, so a key can also arrive as a
  `credential_key` or `credentialKey` entry on the credential or the scheme.
  """
  if credential_key:
    return credential_key

  for obj in (auth_credential, auth_scheme):
    if not obj or not obj.model_extra:
      continue
    for key in ("credential_key", "credentialKey"):
      value = obj.model_extra.get(key)
      if isinstance(value, str) and value:
        return value

  return None


class ToolContextCredentialStore:
  """Handles storage and retrieval of credentials within a ToolContext.

  A configured ``credential_key`` selects the cache slot through
  ``_EXCHANGED_CREDENTIAL_KEY_SUFFIX``. Without one, the slot is derived from
  the auth scheme and the auth credential.
  """

  def __init__(
      self,
      tool_context: ToolContext,
      credential_key: Optional[str] = None,
  ):
    self.tool_context = tool_context
    self.credential_key = credential_key

  def _legacy_stable_digest(self, text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

  def _get_legacy_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    if auth_credential and auth_credential.oauth2:
      auth_credential = auth_credential.model_copy(deep=True)
      if auth_credential.oauth2:
        auth_credential.oauth2.auth_uri = None
        auth_credential.oauth2.state = None
        auth_credential.oauth2.auth_response_uri = None
        auth_credential.oauth2.auth_code = None
        auth_credential.oauth2.access_token = None
        auth_credential.oauth2.refresh_token = None
        auth_credential.oauth2.expires_at = None
        auth_credential.oauth2.expires_in = None
        auth_credential.oauth2.redirect_uri = None
    scheme_name = (
        f"{auth_scheme.type_.name}_{self._legacy_stable_digest(auth_scheme.model_dump_json())}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{self._legacy_stable_digest(auth_credential.model_dump_json())}"
        if auth_credential
        else ""
    )
    return f"{scheme_name}_{credential_name}{_EXCHANGED_CREDENTIAL_KEY_SUFFIX}"

  def _get_digest_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    """Derives the cache slot from the auth scheme and the auth credential."""

    if auth_credential and auth_credential.oauth2:
      auth_credential = auth_credential.model_copy(deep=True)
      if auth_credential.oauth2:
        auth_credential.oauth2.auth_uri = None
        auth_credential.oauth2.state = None
        auth_credential.oauth2.auth_response_uri = None
        auth_credential.oauth2.auth_code = None
        auth_credential.oauth2.access_token = None
        auth_credential.oauth2.refresh_token = None
        auth_credential.oauth2.expires_at = None
        auth_credential.oauth2.expires_in = None
        auth_credential.oauth2.redirect_uri = None
    scheme_name = (
        f"{auth_scheme.type_.name}_{_stable_model_digest(auth_scheme)}"
        if auth_scheme
        else ""
    )
    credential_name = (
        f"{auth_credential.auth_type.value}_{_stable_model_digest(auth_credential)}"
        if auth_credential
        else ""
    )
    # no need to prepend temp: namespace, session state is a copy, changes to
    # it won't be persisted , only changes in event_action.state_delta will be
    # persisted. temp: namespace will be cleared after current run. but tool
    # want access token to be there stored across runs

    return f"{scheme_name}_{credential_name}{_EXCHANGED_CREDENTIAL_KEY_SUFFIX}"

  def get_credential_key(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> str:
    """Returns the session state slot that caches the exchanged credential."""

    # A key the developer named selects the slot: it is how they point several
    # tools at one cached credential, or keep two apart.
    if self.credential_key:
      return f"{self.credential_key}{_EXCHANGED_CREDENTIAL_KEY_SUFFIX}"

    return self._get_digest_credential_key(auth_scheme, auth_credential)

  def get_credential(
      self,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
  ) -> Optional[AuthCredential]:
    if not self.tool_context:
      return None

    token_key = self.get_credential_key(auth_scheme, auth_credential)
    # TODO try not to use session state, this looks a hacky way, depend on
    # session implementation, we don't want session to persist the token,
    # meanwhile we want the token shared across runs.
    serialized_credential = self.tool_context.state.get(token_key)
    if serialized_credential:
      return AuthCredential.model_validate(serialized_credential)

    # Slots this credential may already sit in: the digest slot, used while no
    # credential_key was configured, and the pre-SHA256 legacy slot. When
    # several tools share a credential_key, the first one to miss migrates its
    # own cached credential into the shared slot. That sharing is what the
    # developer asked for by naming one key. Without a credential_key the
    # digest slot is the current slot, and re-reading it is a miss again.
    for previous_key in (
        self._get_digest_credential_key(auth_scheme, auth_credential),
        self._get_legacy_credential_key(auth_scheme, auth_credential),
    ):
      serialized_credential = self.tool_context.state.get(previous_key)
      if serialized_credential:
        # Migrate to the current key for future lookups.
        self.tool_context.state[token_key] = serialized_credential
        return AuthCredential.model_validate(serialized_credential)

    return None

  def store_credential(
      self,
      key: str,
      auth_credential: Optional[AuthCredential],
  ):
    if self.tool_context:
      self.tool_context.state[key] = auth_credential.model_dump(
          exclude_none=True
      )

  def remove_credential(self, key: str):
    del self.tool_context.state[key]


class ToolAuthHandler:
  """Handles the preparation and exchange of authentication credentials for tools."""

  def __init__(
      self,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      credential_store: Optional["ToolContextCredentialStore"] = None,
      *,
      credential_key: Optional[str] = None,
  ):
    self.tool_context = tool_context
    self.auth_scheme = (
        auth_scheme.model_copy(deep=True) if auth_scheme else None
    )
    self.auth_credential = (
        auth_credential.model_copy(deep=True) if auth_credential else None
    )
    self._credential_key = _resolve_credential_key(
        credential_key, self.auth_credential, self.auth_scheme
    )
    self.credential_exchanger = (
        credential_exchanger or AutoAuthCredentialExchanger()
    )
    self.credential_store = credential_store
    if credential_store and self._credential_key:
      # The request slot and the cache slot are the same configured slot.
      credential_store.credential_key = self._credential_key
    self.should_store_credential = True

  def _build_auth_config(self) -> AuthConfig:
    return AuthConfig(
        auth_scheme=self.auth_scheme,
        raw_auth_credential=self.auth_credential,
        credential_key=self._credential_key,
    )

  @classmethod
  def from_tool_context(
      cls,
      tool_context: ToolContext,
      auth_scheme: Optional[AuthScheme],
      auth_credential: Optional[AuthCredential],
      credential_exchanger: Optional[BaseAuthCredentialExchanger] = None,
      *,
      credential_key: Optional[str] = None,
  ) -> "ToolAuthHandler":
    """Creates a ToolAuthHandler instance from a ToolContext."""
    credential_store = ToolContextCredentialStore(tool_context)
    return cls(
        tool_context,
        auth_scheme,
        auth_credential,
        credential_key=credential_key,
        credential_exchanger=credential_exchanger,
        credential_store=credential_store,
    )

  async def _get_existing_credential(
      self,
  ) -> Optional[AuthCredential]:
    """Checks for and returns an existing, exchanged credential."""
    if self.credential_store:
      existing_credential = self.credential_store.get_credential(
          self.auth_scheme, self.auth_credential
      )
      if existing_credential:
        if existing_credential.oauth2:
          refresher = OAuth2CredentialRefresher()
          if await refresher.is_refresh_needed(existing_credential):
            existing_credential = await refresher.refresh(
                existing_credential, self.auth_scheme
            )
            # Persist the refreshed credential so the next invocation
            # reads the new tokens instead of the stale pre-refresh ones.
            # Without this, providers that rotate refresh_tokens on each
            # refresh (e.g. Salesforce, many OIDC providers) will fail
            # because the old refresh_token has already been invalidated.
            self._store_credential(existing_credential)
        return existing_credential
    return None

  def _exchange_credential(
      self, auth_credential: AuthCredential
  ) -> Optional[AuthPreparationResult]:
    """Handles an OpenID Connect authorization response."""

    exchanged_credential = None
    try:
      exchanged_credential = self.credential_exchanger.exchange_credential(
          self.auth_scheme, auth_credential
      )
    except Exception as e:
      logger.error("Failed to exchange credential: %s", e)
    return exchanged_credential

  def _store_credential(self, auth_credential: AuthCredential) -> None:
    """stores the auth_credential."""

    if self.credential_store:
      key = self.credential_store.get_credential_key(
          self.auth_scheme, self.auth_credential
      )
      self.credential_store.store_credential(key, auth_credential)

  def _request_credential(self) -> None:
    """Handles the case where an OpenID Connect or OAuth2 authentication request is needed."""
    if self.auth_scheme.type_ in (
        AuthSchemeType.openIdConnect,
        AuthSchemeType.oauth2,
    ):
      if not self.auth_credential or not self.auth_credential.oauth2:
        raise ValueError(
            f"auth_credential is empty for scheme {self.auth_scheme.type_}."
            "Please create AuthCredential using OAuth2Auth."
        )

      if not self.auth_credential.oauth2.client_id:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_id is missing."
        )

      if not self.auth_credential.oauth2.client_secret:
        raise AuthCredentialMissingError(
            "OAuth2 credentials client_secret is missing."
        )

    self.tool_context.request_credential(self._build_auth_config())
    return None

  def _get_auth_response(self) -> AuthCredential:
    return self.tool_context.get_auth_response(self._build_auth_config())

  def _external_exchange_required(self, credential) -> bool:
    return (
        credential.auth_type
        in (
            AuthCredentialTypes.OAUTH2,
            AuthCredentialTypes.OPEN_ID_CONNECT,
        )
        and not credential.oauth2.access_token
    )

  async def prepare_auth_credentials(
      self,
  ) -> AuthPreparationResult:
    """Prepares authentication credentials, handling exchange and user interaction."""

    # no auth is needed
    if not self.auth_scheme:
      return AuthPreparationResult(state="done")

    # Check for existing credential.
    existing_credential = await self._get_existing_credential()

    credential = existing_credential or self.auth_credential
    # fetch credential from adk framework
    # Some auth scheme like OAuth2 AuthCode & OpenIDConnect may require
    # multistep exchange:
    # client_id , client_secret -> auth_uri -> auth_code -> access_token
    # adk framework supports exchange access_token already
    # for other credential, adk can also get back the credential directly
    if not credential or self._external_exchange_required(credential):
      credential = self._get_auth_response()
      # store fetched credential
      if credential:
        self._store_credential(credential)
      else:
        self._request_credential()
        return AuthPreparationResult(
            state="pending",
            auth_scheme=self.auth_scheme,
            auth_credential=self.auth_credential,
        )

    # here exchangers are doing two different thing:
    # for service account the exchanger is doing actual token exchange
    # while for oauth2 it's actually doing the credential conversion
    # from OAuth2 credential to HTTP credentials for setting credential in
    # http header
    # TODO cleanup the logic:
    # 1. service account token exchanger should happen before we store them in
    #    the token store
    # 2. blow line should only do credential conversion

    exchanged_credential = self._exchange_credential(credential)
    return AuthPreparationResult(
        state="done",
        auth_scheme=self.auth_scheme,
        auth_credential=exchanged_credential,
    )
