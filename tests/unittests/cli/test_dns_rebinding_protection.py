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

"""Tests for DNS-rebinding protection in _OriginCheckMiddleware."""

import re

from google.adk.cli.api_server import _build_allowed_hosts
from google.adk.cli.api_server import _is_loopback_address
from google.adk.cli.api_server import _is_request_origin_allowed
import pytest

_LOOPBACK_HOSTS_8000 = frozenset({
    "localhost",
    "localhost:8000",
    "127.0.0.1",
    "127.0.0.1:8000",
    "[::1]",
    "[::1]:8000",
})


def _make_scope(
    server_host: str = "127.0.0.1",
    host_header: str = "127.0.0.1:8000",
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> dict:
  """Build a minimal ASGI scope for testing."""
  return {
      "type": "http",
      "method": "POST",
      "server": (server_host, 8000),
      "headers": [(b"host", host_header.encode())] + (extra_headers or []),
      "scheme": "http",
  }


class TestIsLoopbackAddress:
  """Unit tests for _is_loopback_address."""

  @pytest.mark.parametrize(
      "host",
      [
          "127.0.0.1",
          "localhost",
          "::1",
          "[::1]",
          "127.0.0.1:8000",
          "localhost:8000",
          "[::1]:8000",
          "127.1.2.3",  # any 127.x.x.x is loopback
      ],
  )
  def test_loopback_hosts(self, host: str):
    assert _is_loopback_address(host), f"{host!r} should be loopback"

  @pytest.mark.parametrize(
      "host",
      [
          "evil.com",
          "127.evil.com",
          "0.0.0.0",
          "192.168.1.1",
          "10.0.0.1",
          "128.0.0.1",
          "",
      ],
  )
  def test_non_loopback_hosts(self, host: str):
    assert not _is_loopback_address(host), f"{host!r} should NOT be loopback"


class TestDnsRebindingProtection:
  """Tests that DNS-rebinding attacks are blocked when server is on loopback."""

  # --- DNS rebinding scenarios (should be BLOCKED) ---

  def test_dns_rebinding_evil_origin_loopback_server_no_configured_origins(
      self,
  ):
    """Attacker page (evil.com) DNS-rebinds to 127.0.0.1 and sends a POST.

    Browser sends Origin: http://evil.com, Host: evil.com.
    Server is bound to 127.0.0.1.
    No explicit allow-origins configured.
    Expected: BLOCKED.
    """
    scope = _make_scope(server_host="127.0.0.1", host_header="evil.com:8000")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert (
        not result
    ), "DNS-rebinding from evil.com should be blocked on loopback server"

  def test_dns_rebinding_127_evil_origin(self):
    """Origin header host starts with '127.' but is a hostname (127.evil.com)."""
    scope = _make_scope(
        server_host="127.0.0.1", host_header="127.evil.com:8000"
    )
    result = _is_request_origin_allowed(
        origin="http://127.evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  def test_dns_rebinding_localhost_server(self):
    """Same attack, server bound as 'localhost'."""
    scope = _make_scope(server_host="localhost", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  def test_dns_rebinding_ipv6_loopback_server(self):
    """Same attack, server bound to ::1."""
    scope = _make_scope(server_host="::1", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert not result

  # --- Legitimate same-origin requests (should be ALLOWED) ---

  def test_same_origin_localhost_allowed(self):
    """Legitimate browser request from localhost UI to localhost server."""
    scope = _make_scope(server_host="127.0.0.1", host_header="127.0.0.1:8000")
    result = _is_request_origin_allowed(
        origin="http://127.0.0.1:8000",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert result, "Same-origin localhost request should be allowed"

  def test_same_origin_localhost_named(self):
    """Browser opens http://localhost:8000 -> requests to localhost:8000."""
    scope = _make_scope(server_host="127.0.0.1", host_header="localhost:8000")
    result = _is_request_origin_allowed(
        origin="http://localhost:8000",
        scope=scope,
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )
    assert result

  # --- Explicit allow-origins configured (allow-list bypasses DNS guard) ---

  def test_explicit_allowlist_overrides_dns_rebinding_guard(self):
    """If the developer explicitly allows evil.com, it should be permitted."""
    scope = _make_scope(server_host="127.0.0.1", host_header="evil.com")
    result = _is_request_origin_allowed(
        origin="http://evil.com",
        scope=scope,
        allowed_literal_origins=["http://evil.com"],
        allowed_origin_regex=None,
        has_configured_allowed_origins=True,
    )
    assert result, "Explicitly allowed origin should still pass"

  def test_request_without_server_or_host_is_rejected(self):
    """No bind address and no Host header leaves nothing to compare against."""
    assert not _is_request_origin_allowed(
        origin="http://evil.com",
        scope={"type": "http", "headers": [], "scheme": "http"},
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )

  def test_unparseable_origin_rejected(self):
    """A malformed Origin makes urlparse raise; the request must be denied."""
    assert not _is_request_origin_allowed(
        origin="http://[::1",
        scope=_make_scope(server_host="127.0.0.1"),
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )

  # --- Non-loopback server: the Host header is the only reference point ---

  def _check_fallback(
      self,
      origin: str,
      host_header: str,
      extra_headers: list[tuple[bytes, bytes]] | None = None,
  ) -> bool:
    """Run the no-declared-bind-address path (same_origin_allowlist is None)."""
    return _is_request_origin_allowed(
        origin=origin,
        scope=_make_scope(
            server_host="0.0.0.0",
            host_header=host_header,
            extra_headers=extra_headers,
        ),
        allowed_literal_origins=[],
        allowed_origin_regex=None,
        has_configured_allowed_origins=False,
    )

  def test_no_declared_bind_address_falls_back_to_host_header(self):
    """This is the wildcard-bind / library-embedding path.

    TestAllowedHostsParameter proves the same input is rejected once a bind
    address has been declared.
    """
    assert self._check_fallback(
        "http://example.com:8000", host_header="example.com:8000"
    )

  def test_fallback_accepts_https_origin_behind_a_tls_terminating_proxy(self):
    """The proxy speaks plaintext to the server, so only the Host matches."""
    assert self._check_fallback(
        "https://example.com", host_header="example.com"
    )

  def test_fallback_ignores_forwarded_host(self):
    assert not self._check_fallback(
        "http://evil.example",
        host_header="example.com:8000",
        extra_headers=[(b"x-forwarded-host", b"evil.example")],
    )

  def test_fallback_rejects_other_hosts(self):
    assert not self._check_fallback(
        "http://evil.example:8000", host_header="example.com:8000"
    )


class TestBuildAllowedHosts:
  """Unit tests for _build_allowed_hosts."""

  @pytest.mark.parametrize(
      "host",
      [
          "127.0.0.1",
          "localhost",
          "LOCALHOST",
          # An IPv6 literal is bracketed, because that is how browsers send it.
          "::1",
          "[::1]",
      ],
  )
  def test_loopback_binds(self, host: str):
    assert _build_allowed_hosts(host, 8000) == _LOOPBACK_HOSTS_8000

  def test_ipv6_bind_is_bracketed(self):
    assert _build_allowed_hosts("fe80::1", 9000) == frozenset({
        "[fe80::1]",
        "[fe80::1]:9000",
        "localhost",
        "localhost:9000",
        "127.0.0.1",
        "127.0.0.1:9000",
        "[::1]",
        "[::1]:9000",
    })

  def test_concrete_non_loopback_bind(self):
    allowed_hosts = _build_allowed_hosts("192.168.1.5", 9000)
    assert allowed_hosts is not None
    assert "192.168.1.5" in allowed_hosts
    assert "192.168.1.5:9000" in allowed_hosts
    assert "192.168.1.6:9000" not in allowed_hosts
    # Loopback aliases stay reachable on a LAN bind.
    assert "localhost:9000" in allowed_hosts

  def test_attacker_host_is_not_allowed(self):
    allowed_hosts = _build_allowed_hosts("127.0.0.1", 8000)
    assert allowed_hosts is not None
    assert "evil.example:8000" not in allowed_hosts

  @pytest.mark.parametrize("host", ["0.0.0.0", "::", "[::]", "", "  "])
  def test_wildcard_binds_disable_validation(self, host: str):
    assert _build_allowed_hosts(host, 8000) is None


class TestAllowedHostsParameter:
  """A supplied allowed_hosts is authoritative."""

  def _check(
      self,
      origin: str,
      host_header: str = "127.0.0.1:8000",
      allowed_literal_origins: list[str] | None = None,
      allowed_origin_regex: re.Pattern[str] | None = None,
      has_configured_allowed_origins: bool = False,
  ) -> bool:
    return _is_request_origin_allowed(
        origin=origin,
        scope=_make_scope(host_header=host_header),
        allowed_literal_origins=allowed_literal_origins or [],
        allowed_origin_regex=allowed_origin_regex,
        has_configured_allowed_origins=has_configured_allowed_origins,
        allowed_hosts=_LOOPBACK_HOSTS_8000,
    )

  def test_rebind_wire_image_rejected(self):
    """Origin and Host both name the attacker, as a rebound browser sends.

    The legacy `origin == recomputed_origin` comparison accepted exactly this;
    see test_no_declared_bind_address_falls_back_to_host_header.
    """
    assert not self._check(
        "http://evil.example:8000", host_header="evil.example:8000"
    )

  def test_configured_literal_origin_wins(self):
    assert self._check(
        "https://example.com",
        allowed_literal_origins=["https://example.com"],
        has_configured_allowed_origins=True,
    )

  def test_configured_regex_origin_wins(self):
    assert self._check(
        "https://app.example.com",
        allowed_origin_regex=re.compile(r"https://.*\.example\.com"),
        has_configured_allowed_origins=True,
    )

  def test_configured_origins_that_do_not_match_are_still_rejected(self):
    assert not self._check(
        "https://evil.example",
        allowed_literal_origins=["https://example.com"],
        has_configured_allowed_origins=True,
    )
