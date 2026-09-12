"""Bounded HTTP to untrusted public destinations, with DNS pinned on connect."""

from __future__ import annotations

import ipaddress
import socket
import time
from urllib.parse import urlsplit

import certifi
import requests
import urllib3
from requests.structures import CaseInsensitiveDict

MAX_URL_LENGTH = 2048
MAX_CERTIFICATE_RESPONSE_BYTES = 8 * 1024 * 1024
_TRANSITION_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("64:ff9b::/96", "64:ff9b:1::/48", "2002::/16", "2001::/32")
)


class BlockedOutboundURL(ValueError):
    """The destination is malformed or outside the public web boundary."""


def _ip_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.scope_id or any(addr in network for network in _TRANSITION_NETWORKS):
            return False
        if addr.ipv4_mapped is not None:
            return _ip_is_public(str(addr.ipv4_mapped))
    return addr.is_global and not (addr.is_multicast or addr.is_reserved)


def _public_addresses(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        except OSError as exc:
            raise BlockedOutboundURL("Destination could not be resolved.") from exc
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    else:
        addresses = [str(literal)]
    if not addresses or not all(_ip_is_public(ip) for ip in addresses):
        raise BlockedOutboundURL("Destination must resolve only to public addresses.")
    return addresses


def hostname_resolves_public(host: str, port: int) -> bool:
    try:
        _public_addresses(host, port)
    except BlockedOutboundURL:
        return False
    return True


def _parse_url(url: str, *, https_only: bool = False):
    if not isinstance(url, str) or not url or len(url) > MAX_URL_LENGTH:
        raise BlockedOutboundURL("Invalid destination URL.")
    if "\\" in url or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise BlockedOutboundURL("Invalid characters in destination URL.")
    try:
        parsed = urlsplit(url)
        schemes = ("https",) if https_only else ("http", "https")
        if parsed.scheme not in schemes or not parsed.hostname:
            raise ValueError
        if parsed.username is not None or parsed.password is not None or "%" in parsed.hostname:
            raise ValueError
        expected_port = 443 if parsed.scheme == "https" else 80
        if parsed.port is not None and parsed.port != expected_port:
            raise ValueError
        # URL preparation handles IDNA/path escaping without sessions, netrc
        # credentials or environment proxies.
        prepared = requests.PreparedRequest()
        prepared.prepare_url(url, None)
        canonical = urlsplit(prepared.url)
    except (ValueError, UnicodeError, requests.RequestException) as exc:
        raise BlockedOutboundURL("Use a public HTTP(S) URL on its standard port, without userinfo.") from exc
    return prepared, canonical, expected_port


def validate_outbound_url(url: str, *, https_only: bool = False) -> None:
    _prepared, parsed, port = _parse_url(url, https_only=https_only)
    _public_addresses(parsed.hostname, port)


def public_request(
    method: str,
    url: str,
    *,
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: float = 10,
    max_bytes: int = MAX_CERTIFICATE_RESPONSE_BYTES,
    https_only: bool = False,
) -> requests.Response:
    """Return an already-closed response; never follow redirects or use proxies.

    All resolved addresses must be public. Connect to one of those literal
    addresses while Host, TLS SNI and certificate verification retain the
    original hostname. A later DNS answer cannot change the connection.
    ``max_bytes=0`` reads only status/headers, suitable for webhook delivery.
    """
    if timeout <= 0 or max_bytes < 0:
        raise ValueError("Invalid outbound request budget.")
    prepared, parsed, port = _parse_url(url, https_only=https_only)
    addresses = _public_addresses(parsed.hostname, port)
    request_headers = CaseInsensitiveDict(headers or {})
    request_headers["Host"] = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    request_headers["Accept-Encoding"] = "identity"
    request_headers["Connection"] = "close"
    pool_kwargs = {
        "port": port,
        "timeout": urllib3.Timeout(connect=timeout, read=timeout, total=timeout),
        "retries": False,
    }
    pool: urllib3.HTTPConnectionPool
    if parsed.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            addresses[0],
            server_hostname=parsed.hostname,
            assert_hostname=parsed.hostname,
            cert_reqs="CERT_REQUIRED",
            ca_certs=certifi.where(),
            **pool_kwargs,
        )
    else:
        pool = urllib3.HTTPConnectionPool(addresses[0], **pool_kwargs)
    raw = None
    deadline = time.monotonic() + timeout
    try:
        raw = pool.urlopen(
            method,
            prepared.path_url,
            body=data,
            headers=request_headers,
            redirect=False,
            retries=False,
            preload_content=False,
            decode_content=False,
            assert_same_host=False,
        )
        response = requests.Response()
        response.status_code = raw.status
        response.headers = CaseInsensitiveDict(raw.headers)
        response.url = prepared.url
        content = bytearray()
        if max_bytes and 200 <= raw.status < 300:
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise requests.RequestException("Compressed outbound response refused.")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > max_bytes):
                raise requests.RequestException("Outbound response exceeds size limit.")
            while len(content) <= max_bytes:
                if time.monotonic() > deadline:
                    raise requests.Timeout("Outbound response exceeded time budget.")
                chunk = raw.read1(min(65536, max_bytes + 1 - len(content)), decode_content=False)
                if not chunk:
                    break
                content.extend(chunk)
            if len(content) > max_bytes:
                raise requests.RequestException("Outbound response exceeds size limit.")
        response._content = bytes(content)
        response._content_consumed = True  # type: ignore[attr-defined]  # requests' buffered-response flag
        return response
    except requests.RequestException:
        raise
    except (urllib3.exceptions.HTTPError, OSError) as exc:
        raise requests.RequestException("Outbound request failed.") from exc
    finally:
        if raw is not None:
            raw.close()
        pool.close()
