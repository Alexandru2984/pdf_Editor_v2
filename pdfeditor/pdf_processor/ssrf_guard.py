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


def validate_outbound_url(url: str) -> None:
    """Raise :class:`BlockedOutboundURL` unless ``url`` is http(s) to a public IP.

    Resolves the hostname and rejects the request if *any* resolved address is
    non-public — blocking internal-network and cloud-metadata SSRF reached via
    crafted certificate URLs.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise BlockedOutboundURL(f"blocked non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise BlockedOutboundURL(f"blocked URL without host: {url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
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
