"""Offline tests of the connection boundary, not just URL validation."""

from __future__ import annotations

import asyncio
import io
import socket
import ssl
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
import urllib3
from django.test import SimpleTestCase

from . import outbound
from .pdf_processor.ssrf_guard import guarded_fetcher_backend


def _answers(*addresses):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 443)) for ip in addresses]


class PublicDestinationTests(SimpleTestCase):
    def test_non_public_and_transition_addresses_are_rejected(self):
        for address in (
            "127.0.0.1",
            "10.0.0.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "::",
            "fc00::1",
            "fe80::1",
            "ff02::1",
            "::ffff:127.0.0.1",
            "::ffff:100.64.0.1",
            "64:ff9b::a00:1",
            "2002:a00:1::",
            "fe80::1%eth0",
        ):
            with self.subTest(address=address):
                self.assertFalse(outbound._ip_is_public(address))

    def test_global_ipv4_and_ipv6(self):
        for address in ("8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"):
            with self.subTest(address=address):
                self.assertTrue(outbound._ip_is_public(address))

    def test_malformed_destinations_never_open_a_connection(self):
        for url in (
            "https://[broken",
            "https://8.8.8.8:bad/",
            "https://8.8.8.8:0/",
            "https://8.8.8.8:65536/",
            "https://user:pw@8.8.8.8/",
            "https://@8.8.8.8/",
            "https://8.8.8.8\\@127.0.0.1/",
            "https://8.8.8.8\r\nHost: internal/",
            "https://%31%32%37.0.0.1/",
            "https://[fe80::1%25eth0]/",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url), patch("pdfeditor.outbound.urllib3.HTTPSConnectionPool") as pool:
                with self.assertRaises(outbound.BlockedOutboundURL):
                    outbound.public_request("GET", url)
                pool.assert_not_called()

    def test_mixed_dns_answers_are_rejected(self):
        with (
            patch("pdfeditor.outbound.socket.getaddrinfo", return_value=_answers("8.8.8.8", "10.0.0.1")),
            patch("pdfeditor.outbound.urllib3.HTTPSConnectionPool") as pool,
        ):
            with self.assertRaises(outbound.BlockedOutboundURL):
                outbound.public_request("GET", "https://mixed.example/test")
            pool.assert_not_called()

    def test_dns_failure_and_empty_answer_fail_closed(self):
        for answer in ([], socket.gaierror("unavailable")):
            with self.subTest(answer=type(answer).__name__):
                kwargs = (
                    {"side_effect": answer} if isinstance(answer, Exception) else {"return_value": answer}
                )
                with (
                    patch("pdfeditor.outbound.socket.getaddrinfo", **kwargs),
                    self.assertRaises(outbound.BlockedOutboundURL),
                ):
                    outbound.validate_outbound_url("https://missing.example/")

    def test_noncanonical_loopback_literals_are_rejected_after_resolution(self):
        # Synthetic libc answers cover decimal, octal and hex forms offline.
        for host in ("2130706433", "0177.0.0.1", "0x7f000001", "127.1"):
            with (
                self.subTest(host=host),
                patch("pdfeditor.outbound.socket.getaddrinfo", return_value=_answers("127.0.0.1")),
                self.assertRaises(outbound.BlockedOutboundURL),
            ):
                outbound.validate_outbound_url(f"https://{host}/")


class PublicTransportTests(SimpleTestCase):
    def setUp(self):
        self.raw = MagicMock(status=200, headers={})
        self.raw.read1.side_effect = [b"certificate", b""]
        patcher = patch("pdfeditor.outbound.urllib3.HTTPSConnectionPool")
        self.factory = patcher.start()
        self.addCleanup(patcher.stop)
        self.pool = self.factory.return_value
        self.pool.urlopen.return_value = self.raw

    def test_dns_is_pinned_while_original_tls_identity_is_retained(self):
        with patch(
            "pdfeditor.outbound.socket.getaddrinfo",
            side_effect=[_answers("8.8.8.8"), _answers("127.0.0.1")],
        ) as dns:
            response = outbound.public_request("GET", "https://rebind.example/ca?x=1")
        dns.assert_called_once()
        args, kwargs = self.factory.call_args
        self.assertEqual(args, ("8.8.8.8",))
        self.assertEqual(kwargs["server_hostname"], "rebind.example")
        self.assertEqual(kwargs["assert_hostname"], "rebind.example")
        self.assertEqual(kwargs["cert_reqs"], "CERT_REQUIRED")
        self.assertTrue(kwargs["ca_certs"])
        args, kwargs = self.pool.urlopen.call_args
        self.assertEqual(args, ("GET", "/ca?x=1"))
        self.assertEqual(kwargs["headers"]["Host"], "rebind.example")
        self.assertFalse(kwargs["redirect"])
        self.assertFalse(kwargs["retries"])
        self.assertEqual(response.content, b"certificate")
        self.raw.close.assert_called_once()
        self.pool.close.assert_called_once()

    def test_ipv6_literal_uses_bracketed_host_header(self):
        outbound.public_request("GET", "https://[2606:4700:4700::1111]/")
        self.assertEqual(self.factory.call_args.args, ("2606:4700:4700::1111",))
        self.assertEqual(self.pool.urlopen.call_args.kwargs["headers"]["Host"], "[2606:4700:4700::1111]")

    def test_redirect_does_not_read_body_or_contact_location(self):
        self.raw.status = 302
        self.raw.headers = {"Location": "http://169.254.169.254/"}
        response = outbound.public_request("GET", "https://8.8.8.8/")
        self.assertEqual(response.status_code, 302)
        self.pool.urlopen.assert_called_once()
        self.raw.read1.assert_not_called()
        self.raw.close.assert_called_once()

    def test_webhook_status_only_never_reads_unbounded_body(self):
        self.raw.headers = {"Content-Length": "9999999999999", "Content-Encoding": "gzip"}
        response = outbound.public_request("POST", "https://8.8.8.8/", data=b"event", max_bytes=0)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"")
        self.raw.read1.assert_not_called()
        self.raw.close.assert_called_once()

    def test_oversize_declared_body_is_rejected_without_read(self):
        self.raw.headers = {"Content-Length": "100"}
        with self.assertRaises(requests.RequestException):
            outbound.public_request("GET", "https://8.8.8.8/", max_bytes=4)
        self.raw.read1.assert_not_called()
        self.raw.close.assert_called_once()
        self.pool.close.assert_called_once()

    def test_undeclared_stream_is_bounded(self):
        # Use an actual urllib3 reader so read limits are exercised, not mocked.
        self.pool.urlopen.return_value = urllib3.HTTPResponse(
            body=io.BytesIO(b"123456789"), status=200, preload_content=False
        )
        with self.assertRaises(requests.RequestException):
            outbound.public_request("GET", "https://8.8.8.8/", max_bytes=4)
        self.assertTrue(self.pool.urlopen.return_value.closed)

    def test_compressed_body_is_refused_before_decompression(self):
        self.raw.headers = {"Content-Encoding": "gzip"}
        with self.assertRaises(requests.RequestException):
            outbound.public_request("GET", "https://8.8.8.8/")
        self.raw.read1.assert_not_called()
        self.raw.close.assert_called_once()

    def test_stream_deadline_is_checked_between_reads(self):
        with (
            patch("pdfeditor.outbound.time.monotonic", side_effect=[0, 0.1, 11]),
            self.assertRaises(requests.Timeout),
        ):
            outbound.public_request("GET", "https://8.8.8.8/", timeout=10)
        self.raw.read1.assert_called_once()
        self.raw.close.assert_called_once()

    def test_connection_failure_closes_pool_and_preserves_error_contract(self):
        self.pool.urlopen.side_effect = urllib3.exceptions.SSLError("certificate mismatch")
        with self.assertRaises(requests.RequestException):
            outbound.public_request("GET", "https://8.8.8.8/")
        self.pool.close.assert_called_once()

    def test_environment_proxy_cannot_redirect_traffic(self):
        with patch.dict("os.environ", {"HTTPS_PROXY": "http://127.0.0.1:9999"}):
            outbound.public_request("GET", "https://8.8.8.8/")
        self.assertEqual(self.factory.call_args.args, ("8.8.8.8",))


class CertificateTransportTests(SimpleTestCase):
    def test_certificate_get_and_ocsp_post_use_guarded_transport(self):
        fetchers = guarded_fetcher_backend().get_fetchers()
        response = requests.Response()
        response.status_code = 200
        response._content = b"test-certificate"
        with patch("pdfeditor.pdf_processor.ssrf_guard.public_request", return_value=response) as send:
            result = asyncio.run(
                fetchers.cert_fetcher._get("https://ca.example/", acceptable_content_types=[])
            )
            self.assertEqual(result.content, b"test-certificate")
            self.assertEqual(send.call_args.args, ("GET", "https://ca.example/"))
            asyncio.run(
                fetchers.ocsp_fetcher._post(
                    "https://ca.example/",
                    b"ocsp",
                    content_type="application/ocsp-request",
                    acceptable_content_types=[],
                )
            )
            self.assertEqual(send.call_args.args, ("POST", "https://ca.example/"))
            self.assertEqual(send.call_args.kwargs["data"], b"ocsp")

    def test_ca_redirect_to_metadata_is_rejected_by_fetcher(self):
        response = requests.Response()
        response.status_code = 302
        response.headers["Location"] = "http://169.254.169.254/"
        fetchers = guarded_fetcher_backend().get_fetchers()
        with patch("pdfeditor.pdf_processor.ssrf_guard.public_request", return_value=response) as send:
            with self.assertRaises(requests.RequestException):
                asyncio.run(fetchers.crl_fetcher._get("https://ca.example/", acceptable_content_types=[]))
            send.assert_called_once()


class TLSConnectionTests(SimpleTestCase):
    """Exercise urllib3 and real TLS over a socket pair, without network egress."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        cls.directory = tempfile.TemporaryDirectory(prefix="outbound-tls-test-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.cert_path = Path(cls.directory.name) / "certificate.pem"
        key_path = Path(cls.directory.name) / "key.pem"
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ca.example")])
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("ca.example")]), critical=False)
            .sign(key, hashes.SHA256())
        )
        cls.cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
        cls.context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        cls.context.load_cert_chain(cls.cert_path, key_path)

    def _exchange(self, host):
        client_socket, server_socket = socket.socketpair()
        client_socket.settimeout(3)
        server_socket.settimeout(3)

        def serve():
            try:
                with self.context.wrap_socket(server_socket, server_side=True) as tls:
                    request = b""
                    while b"\r\n\r\n" not in request and len(request) < 16384:
                        chunk = tls.recv(4096)
                        if not chunk:
                            break
                        request += chunk
                    tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                    return request
            except ssl.SSLError:
                return b""
            finally:
                server_socket.close()

        try:
            with (
                ThreadPoolExecutor(max_workers=1) as executor,
                patch("pdfeditor.outbound.socket.getaddrinfo", return_value=_answers("8.8.8.8")) as dns,
                patch("urllib3.util.connection.create_connection", return_value=client_socket) as connect,
                patch("pdfeditor.outbound.certifi.where", return_value=str(self.cert_path)),
            ):
                serving = executor.submit(serve)
                response = outbound.public_request("GET", f"https://{host}/test", timeout=3)
                self.assertEqual(connect.call_args.args[0], ("8.8.8.8", 443))
                dns.assert_called_once()
                return response, serving.result(timeout=5)
        finally:
            client_socket.close()

    def test_pinned_connection_verifies_certificate_for_original_hostname(self):
        response, request = self._exchange("ca.example")
        self.assertEqual(response.content, b"ok")
        self.assertIn(b"Host: ca.example\r\n", request)

    def test_certificate_for_another_hostname_is_rejected(self):
        with self.assertRaises(requests.RequestException):
            self._exchange("wrong.example")
