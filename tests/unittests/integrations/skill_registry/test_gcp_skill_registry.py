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

"""Tests for GCP Skill Registry."""

import contextlib
import datetime
import io
import json
import os
import pathlib
import ssl
import sys
from unittest import mock
import zipfile

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from google.adk.integrations.skill_registry import gcp_skill_registry
from google.adk.utils import _mtls_utils
import google.auth.exceptions
from google.auth.transport import _mtls_helper
from google.auth.transport import mtls
import pytest

# Captured at import time, before the autouse fixture below replaces them, so
# the end-to-end test can run the real mTLS code path.
_REAL_USE_CLIENT_CERT_EFFECTIVE = _mtls_utils.use_client_cert_effective
_REAL_HAS_DEFAULT_CLIENT_CERT_SOURCE = mtls.has_default_client_cert_source


@pytest.fixture(autouse=True)
def mock_env():
  """Fixture to mock environment variables."""
  with mock.patch.dict(
      os.environ,
      {
          "GOOGLE_CLOUD_PROJECT": "test-project",
          "GOOGLE_CLOUD_LOCATION": "us-central1",
      },
  ):
    yield


@pytest.fixture(autouse=True)
def mock_google_auth():
  """Fixture to mock google.auth.default."""
  mock_creds = mock.MagicMock()
  mock_creds.valid = True
  mock_creds.token = "fake-token"
  mock_creds.quota_project_id = None
  with mock.patch(
      "google.auth.default", return_value=(mock_creds, "test-project")
  ):
    yield mock_creds


@pytest.fixture(autouse=True)
def disable_mtls_by_default():
  """Fixture to disable mTLS by default for unit tests."""
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=False,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=False,
      ),
  ):
    yield


def _create_fake_zip_bytes():
  """Creates a fake zip file in memory and returns its bytes."""
  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  return zip_buffer.getvalue()


@pytest.mark.asyncio
async def test_get_skill_success():
  """Verifies that get_skill successfully fetches and loads a skill in memory."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  fake_zip = _create_fake_zip_bytes()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch(
      "httpx.AsyncClient.get", side_effect=mock_get
  ) as mock_get_called:
    skill = await registry.get_skill(name="my-skill")

  assert skill.frontmatter.name == "my-skill"
  assert skill.frontmatter.description == "test"
  assert skill.instructions == "# My Skill"

  mock_get_called.assert_has_calls([
      mock.call(
          "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills/my-skill",
          headers={
              "Authorization": "Bearer fake-token",
              "Content-Type": "application/json",
              "x-goog-user-project": "test-project",
          },
          params=None,
      ),
      mock.call(
          "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123",
          headers={
              "Authorization": "Bearer fake-token",
              "Content-Type": "application/json",
              "x-goog-user-project": "test-project",
          },
          params={"alt": "media"},
      ),
  ])


@pytest.mark.asyncio
async def test_search_skills_success():
  """Verifies that search_skills successfully returns frontmatter list."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {
      "skills": [
          {
              "name": (
                  "projects/test-project/locations/us-central1/skills/skill1"
              ),
              "description": "Description 1",
          },
          {
              "name": (
                  "projects/test-project/locations/us-central1/skills/skill2"
              ),
              "description": "Description 2",
          },
      ]
  }

  with mock.patch(
      "httpx.AsyncClient.get", return_value=mock_response
  ) as mock_get_called:
    results = await registry.search_skills(query="query")

  assert len(results) == 2
  assert results[0].name == "skill1"
  assert results[0].description == "Description 1"
  assert results[1].name == "skill2"
  assert results[1].description == "Description 2"

  mock_get_called.assert_called_once_with(
      "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills:search",
      headers={
          "Authorization": "Bearer fake-token",
          "Content-Type": "application/json",
          "x-goog-user-project": "test-project",
      },
      params={"search_string": "query"},
  )


@pytest.mark.asyncio
async def test_get_skill_raises_on_missing_zip():
  """Verifies that get_skill raises error if zip filesystem is missing."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
  }

  with mock.patch("httpx.AsyncClient.get", return_value=mock_response):
    with pytest.raises(ValueError, match="does not contain default revision"):
      await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_zip_slip():
  """Verifies that get_skill raises error if zip contains dangerous paths."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr("../evil.txt", "malicious content")
    z.writestr(
        "SKILL.md", "---\nname: my-skill\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
    with pytest.raises(ValueError, match="Dangerous zip entry ignored"):
      await registry.get_skill(name="my-skill")


@pytest.mark.asyncio
async def test_get_skill_raises_on_invalid_skill_name():
  """Verifies that get_skill raises error if skill name is invalid."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  zip_buffer = io.BytesIO()
  with zipfile.ZipFile(zip_buffer, "w") as z:
    z.writestr(
        "SKILL.md", "---\nname: ../evil\ndescription: test\n---\n# My Skill\n"
    )
  fake_zip = zip_buffer.getvalue()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with mock.patch("httpx.AsyncClient.get", side_effect=mock_get):
    with pytest.raises(ValueError, match="Invalid skill name in SKILL.md"):
      await registry.get_skill(name="my-skill")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../../../projects/victim/locations/us-central1/skills/secret",
        "my-skill/../other-skill",
        "..%2f..%2fsecret",
        "my-skill?alt=media",
        "my-skill#fragment",
        "my-skill/revisions/rev-123",
        "My-Skill",
        "",
    ],
)
@pytest.mark.asyncio
async def test_get_skill_rejects_unsafe_name_before_any_request(unsafe_name):
  """Verifies that a name that is not a single safe path segment is rejected."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  with mock.patch("httpx.AsyncClient.get") as mock_get_called:
    with pytest.raises(ValueError, match="Invalid skill name"):
      await registry.get_skill(name=unsafe_name)

  mock_get_called.assert_not_called()


@pytest.mark.parametrize("valid_name", ["my-skill", "my_skill", "skill2"])
@pytest.mark.asyncio
async def test_get_skill_builds_expected_url_for_valid_name(valid_name):
  """Verifies that a valid name is still interpolated verbatim into the URL."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": (
          f"projects/test-project/locations/us-central1/skills/{valid_name}"
      ),
      "defaultRevision": (
          f"projects/test-project/locations/us-central1/skills/{valid_name}"
          "/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = _create_fake_zip_bytes()

  async def mock_get(url, *unused_args, **kwargs):
    if kwargs.get("params") and kwargs.get("params").get("alt") == "media":
      return mock_response2
    return mock_response1

  with mock.patch(
      "httpx.AsyncClient.get", side_effect=mock_get
  ) as mock_get_called:
    await registry.get_skill(name=valid_name)

  assert mock_get_called.call_args_list[0].args[0] == (
      "https://agentregistry.googleapis.com/v1alpha/projects/test-project/"
      f"locations/us-central1/skills/{valid_name}"
  )


def test_constructor_configures_base_url():
  """Verifies that constructor configures base URL from environment."""
  # Case 1: Environment variable fallback
  with mock.patch.dict(
      os.environ, {"AGENT_REGISTRY_ENDPOINT": "https://staging.endpoint.com"}
  ):
    registry = gcp_skill_registry.GCPSkillRegistry()
    assert registry.base_url == "https://staging.endpoint.com"

  # Case 2: Default fallback
  registry = gcp_skill_registry.GCPSkillRegistry()
  assert registry.base_url == "https://agentregistry.googleapis.com/v1alpha"


# pylint: disable=protected-access
def test_lazy_load_credentials():
  """Verifies that google.auth.default is not called in constructor."""
  with mock.patch("google.auth.default") as mock_auth:
    registry = gcp_skill_registry.GCPSkillRegistry()
    mock_auth.assert_not_called()
    assert registry._credentials is None


@contextlib.contextmanager
def _client_certs_available(passphrase):
  """Enables mTLS with a default cert source that yields the given passphrase.

  The cert source is mocked at the google-auth boundary, so the real
  MtlsClientCerts helper runs and creates its temporary directory. The mocked
  source never writes the PEM files, so any test that reaches
  _create_httpx_client must also patch httpx.AsyncClient.

  Args:
    passphrase: The passphrase the cert source reports for the private key, or
      None for an unencrypted key.
  """
  cert_source = mock.MagicMock(return_value=(None, None, passphrase))
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.default_client_encrypted_cert_source",
          return_value=cert_source,
      ),
  ):
    yield


def test_constructor_configures_mtls_base_url():
  """Verifies that constructor extracts client certs and uses the mTLS host."""
  with _client_certs_available(b"pass"):
    registry = gcp_skill_registry.GCPSkillRegistry()

  try:
    assert (
        registry.base_url == "https://agentregistry.mtls.googleapis.com/v1alpha"
    )
    certs = registry._mtls_certs
    tempdir = certs._tempdir.name
    assert certs.cert_path is not None
    assert certs.key_path is not None
    assert certs.cert_path.startswith(tempdir)
    assert certs.key_path.startswith(tempdir)
    assert certs.passphrase == b"pass"
  finally:
    registry.close()


@pytest.mark.asyncio
async def test_get_skill_with_mtls():
  """Verifies that get_skill presents the client certificate on every request."""
  fake_zip = _create_fake_zip_bytes()

  mock_response1 = mock.MagicMock()
  mock_response1.status_code = 200
  mock_response1.json.return_value = {
      "name": "projects/test-project/locations/us-central1/skills/my-skill",
      "defaultRevision": (
          "projects/test-project/locations/us-central1/skills/my-skill/revisions/rev-123"
      ),
  }

  mock_response2 = mock.MagicMock()
  mock_response2.status_code = 200
  mock_response2.content = fake_zip

  async def mock_get(url, *unused_args, **kwargs):
    if "alt=media" in str(url) or (
        kwargs.get("params") and kwargs.get("params").get("alt") == "media"
    ):
      return mock_response2
    return mock_response1

  with _client_certs_available(b"pass"):
    registry = gcp_skill_registry.GCPSkillRegistry()

    try:
      with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
        mock_client = mock_client_class.return_value
        mock_client.__aenter__.return_value = mock_client
        mock_client.get = mock.AsyncMock(side_effect=mock_get)

        skill = await registry.get_skill(name="my-skill")

        mock_client_class.assert_called_with(
            cert=(
                registry._mtls_certs.cert_path,
                registry._mtls_certs.key_path,
                b"pass",
            )
        )
    finally:
      registry.close()

  assert skill.frontmatter.name == "my-skill"


def test_constructor_with_no_client_cert_source_uses_plain_client():
  """Verifies that mTLS without a cert source degrades to a plain client."""
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=False,
      ),
  ):
    registry = gcp_skill_registry.GCPSkillRegistry()

  assert registry._mtls_certs.cert_path is None
  assert registry._mtls_certs.key_path is None
  assert registry._mtls_certs.passphrase is None

  with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
    registry._create_httpx_client()

  mock_client_class.assert_called_once_with()


def test_create_httpx_client_omits_passphrase_when_key_is_unencrypted():
  """Verifies that an unencrypted key yields a two-element cert tuple."""
  with _client_certs_available(None):
    registry = gcp_skill_registry.GCPSkillRegistry()

  try:
    assert registry._mtls_certs.passphrase is None
    with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
      registry._create_httpx_client()

    mock_client_class.assert_called_once_with(
        cert=(registry._mtls_certs.cert_path, registry._mtls_certs.key_path)
    )
  finally:
    registry.close()


def test_create_httpx_client_passes_passphrase_for_encrypted_key():
  """Verifies that an encrypted key forwards its passphrase to httpx."""
  with _client_certs_available(b"s3cret"):
    registry = gcp_skill_registry.GCPSkillRegistry()

  try:
    with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
      registry._create_httpx_client()

    mock_client_class.assert_called_once_with(
        cert=(
            registry._mtls_certs.cert_path,
            registry._mtls_certs.key_path,
            b"s3cret",
        )
    )
  finally:
    registry.close()


def test_constructor_raises_when_cert_extraction_fails():
  """Verifies that a broken cert provider fails construction, naming the cause."""
  with (
      mock.patch(
          "google.adk.utils._mtls_utils.use_client_cert_effective",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.has_default_client_cert_source",
          return_value=True,
      ),
      mock.patch(
          "google.auth.transport.mtls.default_client_encrypted_cert_source",
          side_effect=google.auth.exceptions.MutualTLSChannelError(
              "Encrypted private key is not expected"
          ),
      ),
  ):
    with pytest.raises(
        RuntimeError,
        match=(
            "Failed to extract default client certificates for mTLS:"
            " Encrypted private key is not expected"
        ),
    ):
      gcp_skill_registry.GCPSkillRegistry()


def test_close_releases_certs():
  """Verifies that close deletes the certificates and is safe to call twice."""
  with _client_certs_available(b"pass"):
    registry = gcp_skill_registry.GCPSkillRegistry()

  tempdir = registry._mtls_certs._tempdir.name
  assert os.path.exists(tempdir)

  registry.close()

  assert not os.path.exists(tempdir)
  assert registry._mtls_certs is None

  registry.close()

  assert registry._mtls_certs is None

  with mock.patch("httpx.AsyncClient", autospec=True) as mock_client_class:
    registry._create_httpx_client()

  mock_client_class.assert_called_once_with()


def _install_secure_connect_cert_provider(home, passphrase):
  """Installs a SecureConnect cert provider that emits an encrypted key.

  This is the configuration the fix exists for: an administrator whose
  cert_provider_command already carries --with_passphrase.

  Args:
    home: Directory that the test uses as the user's home directory.
    passphrase: Passphrase that protects the generated private key.
  """
  key = ec.generate_private_key(ec.SECP256R1())
  subject = x509.Name(
      [x509.NameAttribute(NameOID.COMMON_NAME, "adk-test-client")]
  )
  now = datetime.datetime.now(datetime.timezone.utc)
  certificate = (
      x509.CertificateBuilder()
      .subject_name(subject)
      .issuer_name(subject)
      .public_key(key.public_key())
      .serial_number(x509.random_serial_number())
      .not_valid_before(now - datetime.timedelta(minutes=1))
      .not_valid_after(now + datetime.timedelta(days=1))
      .sign(key, hashes.SHA256())
  )
  payload_path = home / "cert_provider_output.pem"
  payload_path.write_bytes(
      certificate.public_bytes(serialization.Encoding.PEM)
      + key.private_bytes(
          encoding=serialization.Encoding.PEM,
          format=serialization.PrivateFormat.PKCS8,
          encryption_algorithm=serialization.BestAvailableEncryption(
              passphrase
          ),
      )
      + b"-----BEGIN PASSPHRASE-----\n"
      + passphrase
      + b"\n-----END PASSPHRASE-----\n"
  )

  provider_path = home / "cert_provider.py"
  provider_path.write_text(
      "import sys\n"
      f"sys.stdout.buffer.write(open({str(payload_path)!r}, 'rb').read())\n"
  )

  metadata_path = pathlib.Path(
      _mtls_helper.CONTEXT_AWARE_METADATA_PATH.replace("~", str(home), 1)
  )
  metadata_path.parent.mkdir(parents=True)
  metadata_path.write_text(
      json.dumps({
          "cert_provider_command": [
              sys.executable,
              str(provider_path),
              "--with_passphrase",
          ]
      })
  )


@pytest.mark.asyncio
async def test_encrypted_client_key_configures_a_real_httpx_client(
    tmp_path, monkeypatch
):
  """Verifies the whole mTLS path against a real cert provider, without mocks.

  google-auth runs the provider as a subprocess, MtlsClientCerts writes the
  real PEM files, and httpx decrypts the private key with the passphrase while
  it loads the certificate chain.
  """
  passphrase = b"e2e-passphrase"
  _install_secure_connect_cert_provider(tmp_path, passphrase)
  monkeypatch.setenv("HOME", str(tmp_path))
  monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")
  monkeypatch.delenv("GOOGLE_API_CERTIFICATE_CONFIG", raising=False)
  monkeypatch.delenv(
      "CLOUDSDK_CONTEXT_AWARE_CERTIFICATE_CONFIG_FILE_PATH", raising=False
  )
  monkeypatch.delenv("CLOUDSDK_CONFIG", raising=False)

  with (
      mock.patch.object(
          _mtls_utils,
          "use_client_cert_effective",
          _REAL_USE_CLIENT_CERT_EFFECTIVE,
      ),
      mock.patch.object(
          mtls,
          "has_default_client_cert_source",
          _REAL_HAS_DEFAULT_CLIENT_CERT_SOURCE,
      ),
  ):
    registry = gcp_skill_registry.GCPSkillRegistry()

  certs = registry._mtls_certs
  tempdir = certs._tempdir.name
  try:
    assert (
        registry.base_url == "https://agentregistry.mtls.googleapis.com/v1alpha"
    )
    assert certs.passphrase == passphrase
    assert (
        pathlib.Path(certs.key_path)
        .read_bytes()
        .startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
    )

    # Building the client loads the chain, so it only succeeds if the
    # passphrase decrypts the key.
    client = registry._create_httpx_client()
    await client.aclose()

    certs.passphrase = b"wrong-passphrase"
    with pytest.raises(ssl.SSLError):
      registry._create_httpx_client()
  finally:
    registry.close()

  assert not os.path.exists(tempdir)


@pytest.mark.asyncio
async def test_close_is_safe_without_mtls():
  """Verifies that close on a non-mTLS registry leaves it usable."""
  registry = gcp_skill_registry.GCPSkillRegistry()

  registry.close()

  assert registry._mtls_certs is None

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {"skills": []}

  with mock.patch("httpx.AsyncClient.get", return_value=mock_response):
    assert await registry.search_skills(query="query") == []


# pylint: enable=protected-access


@pytest.mark.asyncio
async def test_use_custom_credentials():
  """Verifies that custom credentials are used when provided."""
  mock_creds = mock.MagicMock()
  mock_creds.valid = True
  mock_creds.token = "custom-token"
  mock_creds.quota_project_id = "custom-quota-project"

  registry = gcp_skill_registry.GCPSkillRegistry(credentials=mock_creds)

  mock_response = mock.MagicMock()
  mock_response.status_code = 200
  mock_response.json.return_value = {"skills": []}

  with mock.patch(
      "httpx.AsyncClient.get", return_value=mock_response
  ) as mock_get_called:
    await registry.search_skills(query="query")

  mock_get_called.assert_called_once_with(
      "https://agentregistry.googleapis.com/v1alpha/projects/test-project/locations/us-central1/skills:search",
      headers={
          "Authorization": "Bearer custom-token",
          "Content-Type": "application/json",
          "x-goog-user-project": "custom-quota-project",
      },
      params={"search_string": "query"},
  )
