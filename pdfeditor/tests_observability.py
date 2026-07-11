"""Tests for the observability stack: request-ID middleware + healthcheck endpoints."""

import logging
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, RequestFactory, TestCase, override_settings
from django.urls import reverse

from .middleware import RequestIDLogFilter, RequestIDMiddleware, get_current_request_id
from .models import ProcessedPDF, UploadedPDF


class RequestIDMiddlewareTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def _wrap(self, response_body=b"ok"):
        from django.http import HttpResponse

        def view(request):
            return HttpResponse(response_body)

        return RequestIDMiddleware(view)

    def test_generates_uuid_when_no_inbound_header(self):
        request = self.factory.get("/")
        response = self._wrap()(request)
        rid = response["X-Request-Id"]
        # 32-char hex (uuid4().hex)
        self.assertEqual(len(rid), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in rid))

    def test_honours_inbound_header(self):
        request = self.factory.get("/", HTTP_X_REQUEST_ID="trace-from-proxy-123")
        response = self._wrap()(request)
        self.assertEqual(response["X-Request-Id"], "trace-from-proxy-123")

    def test_attaches_id_to_request(self):
        captured = {}

        def view(request):
            from django.http import HttpResponse

            captured["rid"] = request.request_id
            return HttpResponse()

        RequestIDMiddleware(view)(self.factory.get("/"))
        self.assertIn("rid", captured)
        self.assertTrue(captured["rid"])

    def test_contextvar_resets_after_request(self):
        # Outside any request the contextvar default ("-") should apply.
        self.assertEqual(get_current_request_id(), "-")
        RequestIDMiddleware(lambda r: __import__("django.http").http.HttpResponse())(self.factory.get("/"))
        self.assertEqual(get_current_request_id(), "-")

    def test_log_filter_injects_request_id(self):
        record = logging.LogRecord(
            name="x",
            level=logging.INFO,
            pathname="x",
            lineno=1,
            msg="hi",
            args=(),
            exc_info=None,
        )
        RequestIDLogFilter().filter(record)
        self.assertTrue(hasattr(record, "request_id"))


class HealthEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_healthz_returns_200(self):
        resp = self.client.get(reverse("healthz"))
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_healthz_emits_request_id_header(self):
        resp = self.client.get(reverse("healthz"))
        self.assertIn("X-Request-Id", resp)
        self.assertEqual(len(resp["X-Request-Id"]), 32)

    def test_readyz_returns_200_when_db_ok(self):
        resp = self.client.get(reverse("readyz"))
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["checks"]["database"], "ok")
        self.assertEqual(body["checks"]["redis"], "ok")

    def test_readyz_returns_503_when_db_fails(self):
        from django.db import OperationalError

        with patch("pdfeditor.views.health.connection") as mock_conn:
            mock_conn.cursor.side_effect = OperationalError("connection refused")
            resp = self.client.get(reverse("readyz"))

        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "degraded")
        self.assertIn("connection refused", body["checks"]["database"])

    def test_readyz_returns_503_when_redis_fails(self):
        with patch("pdfeditor.views.health.cache") as mock_cache:
            mock_cache.set.side_effect = ConnectionError("redis down")
            resp = self.client.get(reverse("readyz"))

        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "degraded")
        self.assertEqual(body["checks"]["database"], "ok")
        self.assertIn("redis down", body["checks"]["redis"])


class AdminHealthDashboardTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.staff = User.objects.create_user(
            username="boss", password="pw1234567x", email="boss@example.com", is_staff=True
        )
        self.regular = User.objects.create_user(
            username="alice", password="pw1234567x", email="alice@example.com"
        )

    def test_anonymous_redirected_to_login(self):
        resp = self.client.get(reverse("admin_health"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_non_staff_redirected_to_login(self):
        self.client.login(username="alice", password="pw1234567x")
        resp = self.client.get(reverse("admin_health"))
        self.assertEqual(resp.status_code, 302)

    def test_staff_sees_dashboard(self):
        UploadedPDF.objects.create(user=self.regular, name="a.pdf", path="/x", size=1024)
        ProcessedPDF.objects.create(
            user=self.regular, name="a-out.pdf", path="/y", size=512, kind=ProcessedPDF.KIND_SPLIT
        )
        self.client.login(username="boss", password="pw1234567x")
        resp = self.client.get(reverse("admin_health"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Platform health")
        # Counts include the seeded rows
        self.assertContains(resp, "1")  # at least one upload + one processed
        # Kind label rendered
        self.assertContains(resp, "Split")


@override_settings(DEBUG=False, ALLOWED_HOSTS=["testserver"])
class CustomErrorPageTests(TestCase):
    def test_404_uses_custom_template(self):
        resp = self.client.get("/this-path-does-not-exist-anywhere/")
        self.assertEqual(resp.status_code, 404)
        # Our custom template renders the "Page not found" string
        self.assertContains(resp, "Page not found", status_code=404)


class MetricsAllowlistTests(TestCase):
    def test_exact_ip_match(self):
        from pdfeditor.views.metrics import _ip_allowed

        self.assertTrue(_ip_allowed("127.0.0.1", {"127.0.0.1"}))
        self.assertFalse(_ip_allowed("127.0.0.2", {"127.0.0.1"}))

    def test_cidr_match(self):
        from pdfeditor.views.metrics import _ip_allowed

        self.assertTrue(_ip_allowed("172.18.0.7", {"172.16.0.0/12"}))
        self.assertTrue(_ip_allowed("10.0.0.5", {"10.0.0.0/8"}))
        self.assertFalse(_ip_allowed("192.168.1.1", {"172.16.0.0/12"}))

    def test_mixed_entries(self):
        from pdfeditor.views.metrics import _ip_allowed

        allow = {"127.0.0.1", "172.16.0.0/12"}
        self.assertTrue(_ip_allowed("127.0.0.1", allow))
        self.assertTrue(_ip_allowed("172.19.5.5", allow))
        self.assertFalse(_ip_allowed("8.8.8.8", allow))

    def test_invalid_ip_rejected(self):
        from pdfeditor.views.metrics import _ip_allowed

        self.assertFalse(_ip_allowed("", {"127.0.0.1"}))
        self.assertFalse(_ip_allowed("not-an-ip", {"127.0.0.1"}))

    def test_garbage_allowlist_entry_skipped(self):
        from pdfeditor.views.metrics import _ip_allowed

        # A bad entry must not crash the matcher; it should just be ignored
        # so the rest of the allowlist still works.
        self.assertTrue(_ip_allowed("127.0.0.1", {"not-a-cidr", "127.0.0.1"}))
