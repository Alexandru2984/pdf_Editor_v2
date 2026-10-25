"""Certificate-revocation fetches use the same public transport as webhooks."""

from __future__ import annotations

from asyncio import to_thread

import requests

# Preserve the existing validation interface for callers.
from ..outbound import (  # noqa: F401
    BlockedOutboundURL,
    _ip_is_public,
    hostname_resolves_public,
    public_request,
    validate_outbound_url,
)


def guarded_fetcher_backend(per_request_timeout: int = 10):
    """Build pyHanko fetchers with pinned DNS, no redirects and bounded bodies."""
    from pyhanko_certvalidator.fetchers.requests_fetchers import (
        FetcherBackend,
        Fetchers,
        RequestsCertificateFetcher,
        RequestsCRLFetcher,
        RequestsOCSPFetcher,
    )
    from pyhanko_certvalidator.fetchers.requests_fetchers.util import RequestsFetcherMixin

    class _GuardMixin(RequestsFetcherMixin):
        async def _http_fetch(self, method, url, acceptable_content_types, *, data=None, content_type=None):
            headers = {"Accept": ",".join(acceptable_content_types), "User-Agent": self.user_agent}
            if content_type:
                headers["Content-Type"] = content_type
            response = await to_thread(
                public_request, method, url, headers=headers, data=data, timeout=self.per_request_timeout
            )
            if response.status_code != 200:
                raise requests.RequestException(f"Certificate fetch status code {response.status_code}")
            return response

        def _get(self, url, *, acceptable_content_types):
            return self._http_fetch("GET", url, acceptable_content_types)

        def _post(self, url, data, *, content_type, acceptable_content_types):
            return self._http_fetch(
                "POST", url, acceptable_content_types, data=data, content_type=content_type
            )

    class _GuardedOCSP(_GuardMixin, RequestsOCSPFetcher):
        pass

    class _GuardedCRL(_GuardMixin, RequestsCRLFetcher):
        pass

    class _GuardedCert(_GuardMixin, RequestsCertificateFetcher):
        pass

    class _GuardedBackend(FetcherBackend):
        def get_fetchers(self) -> Fetchers:
            return Fetchers(
                ocsp_fetcher=_GuardedOCSP(per_request_timeout=per_request_timeout),
                crl_fetcher=_GuardedCRL(per_request_timeout=per_request_timeout),
                cert_fetcher=_GuardedCert(per_request_timeout=per_request_timeout),
            )

        async def close(self):
            return

    return _GuardedBackend()
