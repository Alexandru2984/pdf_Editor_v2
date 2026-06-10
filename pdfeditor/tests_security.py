"""Regression tests for the 2026-06-08 security hardening.

Each class pins one fix so a later refactor can't silently undo it — the
way axes' IP keying silently broke once django-ipware wasn't installed.
Most are DB-free (SimpleTestCase); only the ShareLink cap test needs the DB.
"""

from __future__ import annotations

import re
from datetime import timedelta
from io import BytesIO
from unittest.mock import MagicMock, patch

import pyotp
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.db.models import F
from django.http import HttpResponse
from django.test import RequestFactory, SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from .mfa import consume_backup_code, generate_backup_codes, verify_totp
from .middleware import _VALID_REQUEST_ID, RequestIDMiddleware, SecurityHeadersMiddleware
from .models import MfaDevice, ProcessedPDF, ShareLink, UploadedPDF
from .netutils import client_ip
from .ratelimiting import _compute_key_and_rate
from .scanning import UploadBlocked, scan_fileobj
from .views.metrics import _ip_allowed


class _Anon:
    is_authenticated = False
    pk = None


# --------------------------------------------------------------------------
# Stage 3/4: spoof-safe client IP derivation (netutils.client_ip)
# --------------------------------------------------------------------------
@override_settings(TRUSTED_PROXY_COUNT=3)
class ClientIpTests(SimpleTestCase):
    """Prod chain is Cloudflare → host nginx → docker nginx = 3 hops."""

    def setUp(self):
        self.rf = RequestFactory()

    def _req(self, xff=None, remote="172.18.0.5"):
        extra = {"REMOTE_ADDR": remote}
        if xff is not None:
            extra["HTTP_X_FORWARDED_FOR"] = xff
        return self.rf.get("/", **extra)

    def test_full_chain_returns_real_client(self):
        self.assertEqual(client_ip(self._req("203.0.113.9, 198.51.100.2, 127.0.0.1")), "203.0.113.9")

    def test_spoofed_left_prefix_is_ignored(self):
        # Attacker prepends a fake entry; the real client still resolves.
        r = self._req("1.2.3.4, 203.0.113.9, 198.51.100.2, 127.0.0.1")
        self.assertEqual(client_ip(r), "203.0.113.9")

    def test_no_xff_falls_back_to_remote_addr(self):
        self.assertEqual(client_ip(self._req(None)), "172.18.0.5")

    def test_xff_shorter_than_proxy_depth_uses_remote_addr(self):
        # Don't trust a client-supplied leftmost value when the chain is short.
        self.assertEqual(client_ip(self._req("9.9.9.9")), "172.18.0.5")

    def test_missing_everything_returns_none(self):
        r = self.rf.get("/")
        r.META.pop("REMOTE_ADDR", None)
        self.assertIsNone(client_ip(r))

    @override_settings(TRUSTED_PROXY_COUNT=1)
    def test_single_proxy_is_spoof_safe(self):
        # docker nginx appends the real client; a spoofed left entry is ignored.
        self.assertEqual(client_ip(self._req("1.2.3.4, 203.0.113.9")), "203.0.113.9")

    @override_settings(TRUSTED_PROXY_COUNT=0)
    def test_zero_proxies_uses_remote_addr_only(self):
        # No trusted proxy: XFF is fully untrusted, use REMOTE_ADDR.
        self.assertEqual(client_ip(self._req("203.0.113.9")), "172.18.0.5")


# --------------------------------------------------------------------------
# Stage 3: rate-limit anon keying no longer collapses behind a proxy
# --------------------------------------------------------------------------
@override_settings(TRUSTED_PROXY_COUNT=3)
class RateLimitKeyingTests(SimpleTestCase):
    def setUp(self):
        self.rf = RequestFactory()

    def _anon(self, xff, remote="172.18.0.5"):
        r = self.rf.post("/x/", HTTP_X_FORWARDED_FOR=xff, REMOTE_ADDR=remote)
        r.user = _Anon()
        return r

    def _key(self, xff):
        key, _rate = _compute_key_and_rate(self._anon(xff), anon_rate="5/h", user_rate="100/h")
        return key

    def test_distinct_clients_get_distinct_keys(self):
        # The regression: behind nginx these used to share one REMOTE_ADDR bucket.
        k1 = self._key("203.0.113.9, 198.51.100.2, 127.0.0.1")
        k2 = self._key("203.0.113.77, 198.51.100.2, 127.0.0.1")
        self.assertEqual(k1, "ip:203.0.113.9")
        self.assertNotEqual(k1, k2)

    def test_spoofed_prefix_cannot_change_bucket(self):
        base = self._key("203.0.113.9, 198.51.100.2, 127.0.0.1")
        spoofed = self._key("9.9.9.9, 203.0.113.9, 198.51.100.2, 127.0.0.1")
        self.assertEqual(base, spoofed)


# --------------------------------------------------------------------------
# Stage 4: /metrics allowlist uses the spoof-safe IP
# --------------------------------------------------------------------------
@override_settings(TRUSTED_PROXY_COUNT=3)
class MetricsAllowlistTests(SimpleTestCase):
    ALLOW = {"127.0.0.1", "172.16.0.0/12"}

    def setUp(self):
        self.rf = RequestFactory()

    def test_direct_prometheus_scrape_allowed(self):
        # No XFF on the internal scrape → REMOTE_ADDR (docker CIDR) is allowed.
        r = self.rf.get("/metrics", REMOTE_ADDR="172.18.0.9")
        self.assertTrue(_ip_allowed(client_ip(r) or "", self.ALLOW))

    def test_spoofed_loopback_xff_rejected(self):
        r = self.rf.get(
            "/metrics",
            REMOTE_ADDR="172.18.0.5",
            HTTP_X_FORWARDED_FOR="127.0.0.1, 203.0.113.9, 198.51.100.2, 127.0.0.1",
        )
        ip = client_ip(r)
        self.assertEqual(ip, "203.0.113.9")  # not the spoofed loopback
        self.assertFalse(_ip_allowed(ip or "", self.ALLOW))


# --------------------------------------------------------------------------
# Stage 5: inbound X-Request-Id is sanitised (no log forging / header injection)
# --------------------------------------------------------------------------
class RequestIdSanitizationTests(SimpleTestCase):
    UUID_HEX = r"^[0-9a-f]{32}$"

    def setUp(self):
        self.rf = RequestFactory()

    def _echo(self, incoming=None):
        mw = RequestIDMiddleware(lambda req: HttpResponse("ok"))
        extra = {} if incoming is None else {"HTTP_X_REQUEST_ID": incoming}
        return mw(self.rf.get("/", **extra))["X-Request-Id"]

    def test_valid_id_is_echoed(self):
        self.assertEqual(self._echo("abc-123_OK.9"), "abc-123_OK.9")

    def test_crlf_injection_is_replaced(self):
        rid = self._echo("evil\r\nSet-Cookie: x=y")
        self.assertNotIn("\r", rid)
        self.assertNotIn("\n", rid)
        self.assertRegex(rid, self.UUID_HEX)

    def test_overlong_id_is_replaced(self):
        self.assertRegex(self._echo("a" * 65), self.UUID_HEX)

    def test_absent_header_generates_fresh(self):
        self.assertRegex(self._echo(None), self.UUID_HEX)

    def test_regex_accepts_and_rejects(self):
        self.assertTrue(_VALID_REQUEST_ID.match("good_id-1.2"))
        self.assertFalse(_VALID_REQUEST_ID.match("has space"))
        self.assertFalse(_VALID_REQUEST_ID.match(""))


# --------------------------------------------------------------------------
# Stage 6: deploy-independent Permissions-Policy
# --------------------------------------------------------------------------
class PermissionsPolicyTests(SimpleTestCase):
    def setUp(self):
        self.rf = RequestFactory()

    def test_header_present_with_features_locked(self):
        mw = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))
        pp = mw(self.rf.get("/"))["Permissions-Policy"]
        for directive in ("camera=()", "microphone=()", "geolocation=()", "usb=()", "payment=()"):
            self.assertIn(directive, pp)

    def test_does_not_override_existing_header(self):
        def get_response(req):
            resp = HttpResponse("ok")
            resp["Permissions-Policy"] = "custom=()"
            return resp

        mw = SecurityHeadersMiddleware(get_response)
        self.assertEqual(mw(self.rf.get("/"))["Permissions-Policy"], "custom=()")

    @override_settings(PERMISSIONS_POLICY="")
    def test_empty_setting_omits_header(self):
        mw = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))
        self.assertNotIn("Permissions-Policy", mw(self.rf.get("/")))


# --------------------------------------------------------------------------
# Stage 2: outline titles rendered XSS-safe (json_script, not |safe)
# --------------------------------------------------------------------------
class OutlineXssTests(SimpleTestCase):
    def test_template_uses_json_script_not_safe(self):
        import os

        from django.conf import settings

        path = os.path.join(settings.BASE_DIR, "pdfeditor", "templates", "pdfeditor", "outline.html")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("json_script", src)
        self.assertNotIn("existing_entries_json", src)
        self.assertNotIn("|safe", src)

    def test_json_script_escapes_script_breakout(self):
        from django.template import engines

        t = engines["django"].from_string('{{ entries|json_script:"d" }}')
        out = t.render(
            {"entries": [{"level": 1, "title": "</script><img src=x onerror=alert(1)>", "page": 1}]}
        )
        self.assertNotIn("<img", out)  # injected tag is escaped, never raw
        self.assertIn("\\u003C/script\\u003E", out)  # the title's </script> is escaped
        self.assertEqual(out.count("</script>"), 1)  # only json_script's own closing tag


# --------------------------------------------------------------------------
# Stage 5: ShareLink download cap is enforced atomically (no overrun)
# --------------------------------------------------------------------------
class ShareLinkCounterTests(TestCase):
    def test_conditional_update_enforces_cap(self):
        pdf = ProcessedPDF.objects.create(
            user=None,
            session_key="s",
            kind=ProcessedPDF.KIND_COMPRESS,
            name="x.pdf",
            path="/nonexistent/x.pdf",
            size=1,
        )
        link = ShareLink.objects.create(
            processed_pdf=pdf,
            creator=None,
            session_key="s",
            expires_at=timezone.now() + timedelta(hours=1),
            max_downloads=2,
        )

        def claim():
            return ShareLink.objects.filter(pk=link.pk, download_count__lt=link.max_downloads).update(
                download_count=F("download_count") + 1
            )

        self.assertEqual(claim(), 1)  # 0 -> 1
        self.assertEqual(claim(), 1)  # 1 -> 2
        self.assertEqual(claim(), 0)  # cap reached: no slot left
        link.refresh_from_db()
        self.assertEqual(link.download_count, 2)


# --------------------------------------------------------------------------
# Stage 7: optional ClamAV upload scanning (pdfeditor/scanning.py)
# --------------------------------------------------------------------------
class ClamAvScanTests(SimpleTestCase):
    @override_settings(CLAMAV_ENABLED=False)
    def test_disabled_is_noop_no_clamd_contact(self):
        with patch("pdfeditor.scanning._build_client") as build:
            scan_fileobj(BytesIO(b"%PDF-1.4 data"))
        build.assert_not_called()

    @override_settings(CLAMAV_ENABLED=True)
    def test_clean_file_passes_and_resets_stream(self):
        client = MagicMock()
        client.instream.return_value = {"stream": ("OK", None)}
        buf = BytesIO(b"%PDF-1.4 data")
        with patch("pdfeditor.scanning._build_client", return_value=client):
            scan_fileobj(buf)
        self.assertEqual(buf.tell(), 0)  # rewound so the caller can still save it

    @override_settings(CLAMAV_ENABLED=True)
    def test_infected_file_is_blocked(self):
        client = MagicMock()
        client.instream.return_value = {"stream": ("FOUND", "Eicar-Test-Signature")}
        with (
            patch("pdfeditor.scanning._build_client", return_value=client),
            self.assertRaises(UploadBlocked) as ctx,
        ):
            scan_fileobj(BytesIO(b"infected"))
        self.assertEqual(ctx.exception.reason, "infected")
        self.assertEqual(ctx.exception.detail, "Eicar-Test-Signature")

    @override_settings(CLAMAV_ENABLED=True, CLAMAV_FAIL_OPEN=False)
    def test_scanner_unreachable_fail_closed_rejects(self):
        with (
            patch("pdfeditor.scanning._build_client", side_effect=OSError("connection refused")),
            self.assertRaises(UploadBlocked) as ctx,
        ):
            scan_fileobj(BytesIO(b"%PDF-"))
        self.assertEqual(ctx.exception.reason, "scanner-unavailable")

    @override_settings(CLAMAV_ENABLED=True, CLAMAV_FAIL_OPEN=True)
    def test_scanner_unreachable_fail_open_allows(self):
        with patch("pdfeditor.scanning._build_client", side_effect=OSError("connection refused")):
            scan_fileobj(BytesIO(b"%PDF-"))  # must not raise


class UploadScanIntegrationTests(TestCase):
    def test_infected_web_upload_is_not_stored(self):
        upload = SimpleUploadedFile("doc.pdf", b"%PDF-1.4 pretend", content_type="application/pdf")
        with patch(
            "pdfeditor.views.upload.scan_fileobj",
            side_effect=UploadBlocked("infected", "Eicar-Test-Signature"),
        ):
            self.client.post("/upload/", {"pdf_file": upload})
        self.assertEqual(UploadedPDF.objects.count(), 0)


# --------------------------------------------------------------------------
# Stage 8: optional TOTP MFA (enrol, two-step login, recovery, disable)
# --------------------------------------------------------------------------
_PW = "pw-correct-horse-battery"
_BACKEND = "pdfeditor.auth_backends.CaseInsensitiveModelBackend"


@override_settings(RATELIMIT_ENABLE=False)
class MfaFlowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="alice", password=_PW, is_active=True)

    def _enable_mfa(self, secret: str | None = None) -> str:
        secret = secret or pyotp.random_base32()
        MfaDevice.objects.create(user=self.user, secret=secret, confirmed=True)
        return secret

    def _login(self):
        return self.client.post("/accounts/login/", {"username": "alice", "password": _PW})

    def _logged_in(self) -> bool:
        return "_auth_user_id" in self.client.session

    def test_user_without_mfa_logs_in_one_step(self):
        self._login()
        self.assertTrue(self._logged_in())

    def test_unconfirmed_device_does_not_gate_login(self):
        MfaDevice.objects.create(user=self.user, confirmed=False)
        self._login()
        self.assertTrue(self._logged_in())

    def test_login_with_mfa_requires_code_then_succeeds(self):
        secret = self._enable_mfa()
        resp = self._login()
        self.assertRedirects(resp, "/accounts/mfa/verify/", fetch_redirect_response=False)
        self.assertFalse(self._logged_in())  # password alone is not enough

        self.client.post("/accounts/mfa/verify/", {"code": pyotp.TOTP(secret).now()})
        self.assertTrue(self._logged_in())

    def test_wrong_totp_does_not_authenticate(self):
        self._enable_mfa()
        self._login()
        self.client.post("/accounts/mfa/verify/", {"code": "000000"})
        self.assertFalse(self._logged_in())

    def test_backup_code_works_once_only(self):
        self._enable_mfa()
        device = MfaDevice.objects.get(user=self.user)
        codes = generate_backup_codes(device)

        self._login()
        self.client.post("/accounts/mfa/verify/", {"code": codes[0]})
        self.assertTrue(self._logged_in())

        # The same code must not work a second time.
        self.client.logout()
        self._login()
        self.client.post("/accounts/mfa/verify/", {"code": codes[0]})
        self.assertFalse(self._logged_in())

    def test_setup_confirm_enables_and_issues_backup_codes(self):
        self.client.force_login(self.user, backend=_BACKEND)
        self.assertEqual(self.client.get("/accounts/mfa/setup/").status_code, 200)
        device = MfaDevice.objects.get(user=self.user)
        self.assertFalse(device.confirmed)

        self.client.post("/accounts/mfa/setup/", {"code": pyotp.TOTP(device.secret).now()})
        device.refresh_from_db()
        self.assertTrue(device.confirmed)
        self.assertEqual(device.backup_codes.count(), 10)

    def test_disable_requires_correct_password(self):
        self._enable_mfa()
        self.client.force_login(self.user, backend=_BACKEND)

        self.client.post("/accounts/mfa/disable/", {"password": "wrong"})
        self.assertTrue(MfaDevice.objects.filter(user=self.user).exists())

        self.client.post("/accounts/mfa/disable/", {"password": _PW})
        self.assertFalse(MfaDevice.objects.filter(user=self.user).exists())


# --------------------------------------------------------------------------
# 2026-06-10 audit: TOTP replay protection + backup-code hardening
# --------------------------------------------------------------------------
class TotpReplayTests(TestCase):
    def setUp(self):
        user = get_user_model().objects.create_user(username="bob", password="x", is_active=True)
        self.device = MfaDevice.objects.create(user=user, secret=pyotp.random_base32(), confirmed=True)

    def test_accepted_code_cannot_be_replayed(self):
        code = pyotp.TOTP(self.device.secret).now()
        self.assertTrue(verify_totp(self.device, code))
        self.assertFalse(verify_totp(self.device, code))  # same window, must be burned

    def test_next_step_code_works_after_previous_one(self):
        import time as _time

        totp = pyotp.TOTP(self.device.secret)
        self.assertTrue(verify_totp(self.device, totp.at(_time.time() - 30)))
        self.assertTrue(verify_totp(self.device, totp.now()))  # newer step, accepted

    def test_backup_codes_carry_80_bits(self):
        codes = generate_backup_codes(self.device)
        for code in codes:
            self.assertRegex(code, r"^[0-9a-f]{10}-[0-9a-f]{10}$")

    def test_backup_code_consume_is_single_shot(self):
        codes = generate_backup_codes(self.device)
        self.assertTrue(consume_backup_code(self.device, codes[0]))
        self.assertFalse(consume_backup_code(self.device, codes[0]))


# --------------------------------------------------------------------------
# 2026-06-10 audit: AuditLog keys on the spoof-safe client IP
# --------------------------------------------------------------------------
@override_settings(TRUSTED_PROXY_COUNT=3)
class AuditLogIpTests(SimpleTestCase):
    def test_record_output_ip_ignores_spoofed_left_prefix(self):
        from .views._common import _client_ip

        req = RequestFactory().get(
            "/",
            HTTP_X_FORWARDED_FOR="6.6.6.6, 198.51.100.7, 10.0.0.2, 172.18.0.3",
            REMOTE_ADDR="172.18.0.3",
        )
        # 3 trusted proxies → real client is the 3rd from the right, not the
        # attacker-chosen leftmost entry.
        self.assertEqual(_client_ip(req), "198.51.100.7")


# --------------------------------------------------------------------------
# 2026-06-10 audit: share tokens use secrets.token_urlsafe; public download
# endpoint is rate limited
# --------------------------------------------------------------------------
class ShareTokenTests(TestCase):
    def test_default_token_is_long_urlsafe(self):
        from .models import _default_share_token

        token = _default_share_token()
        self.assertGreaterEqual(len(token), 43)  # 32 bytes of entropy, base64url
        self.assertRegex(token, r"^[A-Za-z0-9_-]+$")

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_public_download_is_rate_limited(self):
        from django.core.cache import cache

        cache.clear()
        self.addCleanup(cache.clear)  # don't leak a full bucket into later tests
        last = None
        for _ in range(61):  # anon rate is 60/h per IP
            last = self.client.get("/s/no-such-token/")
        self.assertEqual(last.status_code, 403)  # Ratelimited → PermissionDenied


# --------------------------------------------------------------------------
# CSP violation reporting endpoint (views/csp.py)
# --------------------------------------------------------------------------
class CspReportEndpointTests(TestCase):
    _LEGACY = {
        "csp-report": {
            "effective-directive": "script-src-elem",
            "blocked-uri": "https://evil.example/x.js",
            "document-uri": "https://pdf.micutu.com/upload/",
        }
    }

    def test_policy_carries_report_uri(self):
        csp = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))(RequestFactory().get("/"))[
            "Content-Security-Policy"
        ]
        self.assertIn("report-uri /csp-report/", csp)

    def test_legacy_report_is_counted(self):
        import json as _json

        from prometheus_client import REGISTRY

        before = (
            REGISTRY.get_sample_value("pdfeditor_csp_violation_total", {"directive": "script-src-elem"}) or 0
        )
        resp = self.client.post(
            "/csp-report/", _json.dumps(self._LEGACY), content_type="application/csp-report"
        )
        self.assertEqual(resp.status_code, 204)
        after = REGISTRY.get_sample_value("pdfeditor_csp_violation_total", {"directive": "script-src-elem"})
        self.assertEqual(after, before + 1)

    def test_reporting_api_format_is_counted(self):
        import json as _json

        payload = [{"type": "csp-violation", "body": {"effectiveDirective": "img-src", "blockedURL": "x"}}]
        resp = self.client.post("/csp-report/", _json.dumps(payload), content_type="application/reports+json")
        self.assertEqual(resp.status_code, 204)

    def test_garbage_and_oversized_bodies_are_swallowed(self):
        self.assertEqual(
            self.client.post("/csp-report/", "not json", content_type="application/csp-report").status_code,
            204,
        )
        big = "x" * (17 * 1024)
        self.assertEqual(
            self.client.post("/csp-report/", big, content_type="application/csp-report").status_code,
            204,
        )

    def test_get_is_rejected(self):
        self.assertEqual(self.client.get("/csp-report/").status_code, 405)


# --------------------------------------------------------------------------
# Stage 9: strict nonce-based Content-Security-Policy
# --------------------------------------------------------------------------
class CspMiddlewareTests(SimpleTestCase):
    def setUp(self):
        self.rf = RequestFactory()

    def _csp(self, path="/"):
        mw = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))
        req = self.rf.get(path)
        return mw(req), req

    def test_app_page_csp_carries_request_nonce(self):
        resp, req = self._csp("/")
        self.assertIn(f"nonce-{req.csp_nonce}", resp["Content-Security-Policy"])

    def test_script_src_drops_unsafe_inline_and_eval(self):
        csp = self._csp("/")[0]["Content-Security-Policy"]
        script_src = csp.split("script-src-elem")[0]  # the script-src directive
        self.assertNotIn("'unsafe-inline'", script_src)  # inline scripts only via nonce
        self.assertNotIn("'unsafe-eval'", script_src)  # PDF.js is self-hosted with isEvalSupported:false
        self.assertNotIn("unpkg.com", csp)  # no third-party CDN scripts (supply chain)
        self.assertIn("style-src 'self' 'unsafe-inline'", csp)  # styles intentionally keep it

    def test_admin_and_api_are_skipped(self):
        for path in ("/admin/", "/admin/health/", "/api/v1/docs/"):
            resp, _ = self._csp(path)
            self.assertFalse(resp.has_header("Content-Security-Policy"), path)

    def test_nonce_is_unique_per_request(self):
        _, r1 = self._csp("/")
        _, r2 = self._csp("/")
        self.assertNotEqual(r1.csp_nonce, r2.csp_nonce)

    @override_settings(SECURE_SSL=True)
    def test_upgrade_insecure_requests_only_on_https(self):
        mw = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))
        self.assertIn("upgrade-insecure-requests", mw(self.rf.get("/"))["Content-Security-Policy"])

    @override_settings(SECURE_SSL=False)
    def test_no_upgrade_insecure_requests_on_http(self):
        mw = SecurityHeadersMiddleware(lambda req: HttpResponse("ok"))
        self.assertNotIn("upgrade-insecure-requests", mw(self.rf.get("/"))["Content-Security-Policy"])


class CspRenderTests(TestCase):
    def test_login_page_inline_scripts_carry_matching_nonce(self):
        """End-to-end: the nonce in the CSP header matches every inline
        <script> the template rendered (proves base.html is wired up)."""
        resp = self.client.get("/accounts/login/")
        match = re.search(r"nonce-([A-Za-z0-9_-]+)", resp["Content-Security-Policy"])
        self.assertIsNotNone(match)
        nonce = match.group(1)

        body = resp.content.decode()
        inline_scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", body)
        self.assertTrue(inline_scripts)  # base.html ships inline scripts
        for tag in inline_scripts:
            if 'type="application/json"' in tag:
                continue  # data island, not executed — no nonce needed
            self.assertIn(f'nonce="{nonce}"', tag)
