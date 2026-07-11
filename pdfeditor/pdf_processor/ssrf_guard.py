"""SSRF guard for outbound certificate-revocation fetches (pyHanko).

When a PDF is signed with PAdES B-LT, pyHanko fetches the OCSP/CRL/AIA URLs
embedded in the signer's certificate chain so it can stamp the revocation
info into the document. Those URLs come from a user-supplied ``.p12``, so a
crafted certificate could point them at internal services (cloud metadata,
``prometheus:9090``, ``localhost``, RFC1918 hosts …) — a server-side request
forgery vector.

This module wraps pyHanko's requests-based fetchers with a check that refuses
any URL resolving to a non-public address. The hostname is resolved up-front
and the request is blocked if *any* resolved address is private/loopback/
link-local/reserved, which stops both literal-IP and internal-hostname SSRF.

The unauthenticated signature-*verification* path does not fetch at all
(``allow_fetching=False``); this guard backstops the authenticated signing
path, which legitimately needs to reach public CAs.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class BlockedOutboundURL(ValueError):
    """Raised when a fetch target resolves to a non-public address."""


def _ip_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


# Real CA OCSP/CRL/AIA endpoints always live on standard web ports. Pinning
# to these means that even if the public-IP check is ever bypassed, a fetch
# can't reach internal services on their own ports (Redis 6379, Postgres 5432,
# Prometheus 9090, the internal LB, …) — it shrinks the SSRF target set to
# whatever answers on :80/:443.
_ALLOWED_PORTS = frozenset({80, 443})


def hostname_resolves_public(host: str, port: int) -> bool:
    """True iff every address ``host`` resolves to is a public IP.

    Shared with the webhook-URL validator so user-supplied delivery targets get
    the exact same anti-SSRF classification as certificate fetches — one place
    to keep the private/loopback/link-local/metadata rules correct.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    addrs = {str(info[4][0]) for info in infos}
    return bool(addrs) and all(_ip_is_public(ip) for ip in addrs)


def validate_outbound_url(url: str) -> None:
    """Raise :class:`BlockedOutboundURL` unless ``url`` is http(s) on a standard
    web port to a public IP.

    Resolves the hostname and rejects the request if *any* resolved address is
    non-public — blocking internal-network and cloud-metadata SSRF reached via
    crafted certificate URLs — and restricts the port to 80/443.

    Known residual: this resolves-then-checks while pyHanko re-resolves when it
    actually connects, so an attacker controlling a low-TTL domain could DNS-
    rebind between the two calls (public at check time, internal at connect
    time). That path is authenticated, rate-limited, needs a crafted ``.p12``,
    and the port allowlist confines any rebind to :80/:443; the robust complete
    fix is a network egress policy on the worker container.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedOutboundURL(f"blocked non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise BlockedOutboundURL(f"blocked URL without host: {url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in _ALLOWED_PORTS:
        raise BlockedOutboundURL(f"blocked non-web port {port} in {url!r}")
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise BlockedOutboundURL(f"could not resolve {host!r}: {exc}") from exc
    addrs = {str(info[4][0]) for info in infos}
    if not addrs:
        raise BlockedOutboundURL(f"no addresses for {host!r}")
    for ip in addrs:
        if not _ip_is_public(ip):
            raise BlockedOutboundURL(f"blocked non-public address {ip} for {host!r}")


def guarded_fetcher_backend(per_request_timeout: int = 10):
    """Return a pyHanko ``FetcherBackend`` whose every fetch is SSRF-validated.

    Drop-in replacement for the default backend, passed to
    ``ValidationContext(fetcher_backend=...)`` on the signing path.
    """
    from pyhanko_certvalidator.fetchers.requests_fetchers import (
        FetcherBackend,
        Fetchers,
        RequestsCertificateFetcher,
        RequestsCRLFetcher,
        RequestsOCSPFetcher,
    )
    from pyhanko_certvalidator.fetchers.requests_fetchers.util import RequestsFetcherMixin

    # Base on pyHanko's own fetcher mixin so super()._get/_post resolves to the
    # real request logic; we only interpose the SSRF check in front of it.
    class _GuardMixin(RequestsFetcherMixin):
        def _get(self, url, **kwargs):
            validate_outbound_url(url)
            return super()._get(url, **kwargs)

        def _post(self, url, data, **kwargs):
            validate_outbound_url(url)
            return super()._post(url, data, **kwargs)

    class _GuardedOCSP(_GuardMixin, RequestsOCSPFetcher):
        pass

    class _GuardedCRL(_GuardMixin, RequestsCRLFetcher):
        pass

    class _GuardedCert(_GuardMixin, RequestsCertificateFetcher):
        pass

    class _GuardedBackend(FetcherBackend):
        def __init__(self, timeout: int):
            self._timeout = timeout

        def get_fetchers(self) -> Fetchers:
            return Fetchers(
                ocsp_fetcher=_GuardedOCSP(per_request_timeout=self._timeout),
                crl_fetcher=_GuardedCRL(per_request_timeout=self._timeout),
                cert_fetcher=_GuardedCert(per_request_timeout=self._timeout),
            )

        async def close(self):
            return

    return _GuardedBackend(per_request_timeout)
