"""Tests for the observability stack: request-ID middleware + healthcheck endpoints."""

import logging
from unittest.mock import patch

from django.test import Client, RequestFactory, TestCase
from django.urls import reverse

from .middleware import RequestIDLogFilter, RequestIDMiddleware, get_current_request_id


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
            name="x", level=logging.INFO, pathname="x", lineno=1,
            msg="hi", args=(), exc_info=None,
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

    def test_readyz_returns_503_when_db_fails(self):
        from django.db import OperationalError

        with patch("pdfeditor.views.health.connection") as mock_conn:
            mock_conn.cursor.side_effect = OperationalError("connection refused")
            resp = self.client.get(reverse("readyz"))

        self.assertEqual(resp.status_code, 503)
        body = resp.json()
        self.assertEqual(body["status"], "degraded")
        self.assertIn("connection refused", body["checks"]["database"])
