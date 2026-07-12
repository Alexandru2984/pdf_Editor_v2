"""Tests for outbound webhooks: SSRF validation, signing, delivery, dispatch, UI.

Network-free: URL validation is exercised with literal public/private IPs (no
DNS), and the delivery task's ``requests.post`` is always mocked.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase
from django.urls import reverse

from . import tasks, webhooks
from .models import Job, ProcessedPDF, Webhook

User = get_user_model()


class WebhookValidationTests(SimpleTestCase):
    def test_internal_and_non_https_targets_are_blocked(self):
        for url in (
            "https://127.0.0.1/hook",
            "https://10.0.0.5/hook",
            "https://192.168.1.10/hook",
            "http://8.8.8.8/hook",  # not https
            "ftp://8.8.8.8/hook",  # not https
            "https:///nohost",
        ):
            with self.assertRaises(webhooks.InvalidWebhookURL):
                webhooks.validate_webhook_url(url)

    def test_public_https_target_is_allowed(self):
        webhooks.validate_webhook_url("https://8.8.8.8/hook")  # literal public IP, no DNS


class WebhookSigningTests(SimpleTestCase):
    def test_signature_matches_hmac_sha256_and_is_deterministic(self):
        body = b'{"event":"job.completed"}'
        sig = webhooks.sign_payload("s3cr3t", body)
        expected = hmac.new(b"s3cr3t", body, hashlib.sha256).hexdigest()
        self.assertEqual(sig, expected)
        self.assertEqual(sig, webhooks.sign_payload("s3cr3t", body))
        self.assertNotEqual(sig, webhooks.sign_payload("other", body))


class WebhookDeliveryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("deliverer", password="x")

    def _hook(self, url="https://8.8.8.8/hook", **kw):
        return Webhook.objects.create(user=self.user, url=url, **kw)

    def test_2xx_marks_success_and_resets_failures(self):
        hook = self._hook(failure_count=3)
        resp = MagicMock(status_code=200)
        with patch("pdfeditor.webhooks.requests.post", return_value=resp) as post:
            tasks.deliver_webhook.apply(args=[str(hook.id), "job.completed", {"event": "job.completed"}])
        hook.refresh_from_db()
        self.assertEqual(hook.last_status, "200")
        self.assertEqual(hook.failure_count, 0)
        self.assertIsNotNone(hook.last_triggered_at)
        # Signature header present and correct over the exact body sent.
        _, kwargs = post.call_args
        sent_body = kwargs["data"]
        expected = "sha256=" + webhooks.sign_payload(hook.secret, sent_body)
        self.assertEqual(kwargs["headers"][webhooks.SIGNATURE_HEADER], expected)
        self.assertFalse(kwargs["allow_redirects"])

    def test_inactive_webhook_is_not_delivered(self):
        hook = self._hook(is_active=False)
        with patch("pdfeditor.webhooks.requests.post") as post:
            tasks.deliver_webhook.apply(args=[str(hook.id), "job.completed", {}])
        post.assert_not_called()

    def test_url_that_became_internal_is_blocked_and_disabled(self):
        # Simulates DNS rebinding: the row holds an internal target at delivery.
        hook = self._hook(url="https://127.0.0.1/hook")
        with patch("pdfeditor.webhooks.requests.post") as post:
            tasks.deliver_webhook.apply(args=[str(hook.id), "job.completed", {}])
        post.assert_not_called()
        hook.refresh_from_db()
        self.assertFalse(hook.is_active)
        self.assertIn("blocked", hook.last_status)

    def test_non_2xx_triggers_a_retry(self):
        from celery.exceptions import Retry

        hook = self._hook()
        resp = MagicMock(status_code=500)
        # retries not yet exhausted → the task schedules a retry (raises Retry).
        with patch("pdfeditor.webhooks.requests.post", return_value=resp), self.assertRaises(Retry):
            tasks.deliver_webhook.apply(args=[str(hook.id), "job.completed", {}], throw=True)
        hook.refresh_from_db()
        self.assertEqual(hook.last_status, "500")

    def test_disable_after_retries_exhausted_and_max_failures(self):
        hook = self._hook(failure_count=webhooks.MAX_CONSECUTIVE_FAILURES - 1)
        resp = MagicMock(status_code=500)
        with patch("pdfeditor.webhooks.requests.post", return_value=resp):
            # retries == max_retries → skip retry, take the final-failure branch.
            tasks.deliver_webhook.apply(
                args=[str(hook.id), "job.completed", {}],
                retries=tasks.deliver_webhook.max_retries,
                throw=True,
            )
        hook.refresh_from_db()
        self.assertGreaterEqual(hook.failure_count, webhooks.MAX_CONSECUTIVE_FAILURES)
        self.assertFalse(hook.is_active)


class WebhookDispatchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("owner", password="x")

    def test_terminal_job_dispatches_to_active_webhook(self):
        hook = Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        job = Job.objects.create(user=self.user, kind=ProcessedPDF.KIND_PDFA, status=Job.STATUS_DONE)
        with patch("pdfeditor.tasks.deliver_webhook.delay") as delay:
            tasks._trigger_webhooks(job)
        delay.assert_called_once()
        args = delay.call_args.args
        self.assertEqual(args[0], str(hook.id))
        self.assertEqual(args[1], "job.completed")

    def test_failed_job_uses_failed_event(self):
        Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        job = Job.objects.create(user=self.user, kind="pdfa", status=Job.STATUS_FAILED)
        with patch("pdfeditor.tasks.deliver_webhook.delay") as delay:
            tasks._trigger_webhooks(job)
        self.assertEqual(delay.call_args.args[1], "job.failed")

    def test_anonymous_job_never_dispatches(self):
        job = Job.objects.create(user=None, session_key="abc", kind="pdfa", status=Job.STATUS_DONE)
        with patch("pdfeditor.tasks.deliver_webhook.delay") as delay:
            tasks._trigger_webhooks(job)
        delay.assert_not_called()

    def test_user_without_webhooks_does_not_dispatch(self):
        job = Job.objects.create(user=self.user, kind="pdfa", status=Job.STATUS_DONE)
        with patch("pdfeditor.tasks.deliver_webhook.delay") as delay:
            tasks._trigger_webhooks(job)
        delay.assert_not_called()

    def test_dispatch_is_deduped_per_job(self):
        from django.core.cache import cache

        cache.clear()
        Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        job = Job.objects.create(user=self.user, kind="pdfa", status=Job.STATUS_DONE)
        with patch("pdfeditor.tasks.deliver_webhook.delay") as delay:
            tasks._trigger_webhooks(job)
            tasks._trigger_webhooks(job)  # second terminal publish
        self.assertEqual(delay.call_count, 1)

    def test_publish_only_triggers_on_terminal(self):
        with patch("pdfeditor.tasks._trigger_webhooks") as trig:
            running = Job.objects.create(user=self.user, kind="pdfa", status=Job.STATUS_RUNNING)
            tasks._publish_job_event(running)
            trig.assert_not_called()
            done = Job.objects.create(user=self.user, kind="pdfa", status=Job.STATUS_DONE)
            tasks._publish_job_event(done)
            trig.assert_called_once_with(done)


class WebhookViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("wh_user", password="pw")
        self.client.force_login(self.user)

    def test_anonymous_is_redirected(self):
        self.client.logout()
        resp = self.client.get(reverse("webhooks"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/login", resp.url)

    def test_create_public_https_webhook(self):
        self.client.post(reverse("create_webhook"), {"url": "https://8.8.8.8/hook", "description": "prod"})
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), 1)

    def test_create_internal_url_is_rejected(self):
        self.client.post(reverse("create_webhook"), {"url": "https://127.0.0.1/hook"})
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), 0)

    def test_create_non_https_is_rejected(self):
        self.client.post(reverse("create_webhook"), {"url": "http://8.8.8.8/hook"})
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), 0)

    def test_create_is_capped(self):
        from .views.webhooks import MAX_WEBHOOKS_PER_USER

        for _ in range(MAX_WEBHOOKS_PER_USER):
            Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        self.client.post(reverse("create_webhook"), {"url": "https://8.8.8.8/hook"})
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), MAX_WEBHOOKS_PER_USER)

    def test_toggle_and_delete_own_webhook(self):
        hook = Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        self.client.post(reverse("toggle_webhook", args=[hook.id]))
        hook.refresh_from_db()
        self.assertFalse(hook.is_active)
        self.client.post(reverse("delete_webhook", args=[hook.id]))
        self.assertFalse(Webhook.objects.filter(id=hook.id).exists())

    def test_cannot_touch_another_users_webhook(self):
        other = User.objects.create_user("intruder", password="x")
        hook = Webhook.objects.create(user=other, url="https://8.8.8.8/hook")
        self.assertEqual(self.client.post(reverse("delete_webhook", args=[hook.id])).status_code, 404)
        self.assertEqual(self.client.post(reverse("toggle_webhook", args=[hook.id])).status_code, 404)
        self.assertTrue(Webhook.objects.filter(id=hook.id).exists())

    def test_secret_shown_once_then_cleared(self):
        self.client.post(reverse("create_webhook"), {"url": "https://8.8.8.8/hook"})
        first = self.client.get(reverse("webhooks"))
        hook = Webhook.objects.get(user=self.user)
        self.assertContains(first, hook.secret)  # flashed once
        second = self.client.get(reverse("webhooks"))
        self.assertNotContains(second, hook.secret)  # gone on reload


class WebhookApiTests(TestCase):
    """REST API surface: /api/v1/webhooks/ (X-API-Key authenticated)."""

    BASE = "/api/v1/webhooks/"

    def setUp(self):
        from .models import ApiKey

        self.user = User.objects.create_user("api_user", password="x")
        _, self.token = ApiKey.create_for_user(self.user, label="test")

    def _auth(self):
        return {"HTTP_X_API_KEY": self.token}

    def test_unauthenticated_is_rejected(self):
        self.assertIn(self.client.get(self.BASE).status_code, (401, 403))

    def test_create_returns_secret_and_stores_public_url(self):
        r = self.client.post(
            self.BASE,
            {"url": "https://8.8.8.8/hook", "description": "prod"},
            content_type="application/json",
            **self._auth(),
        )
        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertTrue(body["secret"])  # signing secret returned to the owner
        hook = Webhook.objects.get(user=self.user)
        self.assertEqual(str(hook.id), body["id"])
        self.assertEqual(hook.url, "https://8.8.8.8/hook")

    def test_create_internal_url_is_rejected(self):
        r = self.client.post(
            self.BASE,
            {"url": "https://127.0.0.1/hook"},
            content_type="application/json",
            **self._auth(),
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), 0)

    def test_create_is_capped(self):
        for _ in range(webhooks.MAX_WEBHOOKS_PER_USER):
            Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        r = self.client.post(
            self.BASE, {"url": "https://8.8.8.8/hook"}, content_type="application/json", **self._auth()
        )
        self.assertEqual(r.status_code, 400)
        self.assertEqual(Webhook.objects.filter(user=self.user).count(), webhooks.MAX_WEBHOOKS_PER_USER)

    def test_list_and_detail_are_owner_scoped(self):
        mine = Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        other = User.objects.create_user("other", password="x")
        theirs = Webhook.objects.create(user=other, url="https://8.8.8.8/hook")
        listing = self.client.get(self.BASE, **self._auth()).json()
        ids = {row["id"] for row in listing["results"]}
        self.assertIn(str(mine.id), ids)
        self.assertNotIn(str(theirs.id), ids)
        self.assertEqual(self.client.get(f"{self.BASE}{theirs.id}/", **self._auth()).status_code, 404)

    def test_patch_toggles_active_and_delete_removes(self):
        hook = Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        r = self.client.patch(
            f"{self.BASE}{hook.id}/", {"is_active": False}, content_type="application/json", **self._auth()
        )
        self.assertEqual(r.status_code, 200)
        hook.refresh_from_db()
        self.assertFalse(hook.is_active)
        self.assertEqual(self.client.delete(f"{self.BASE}{hook.id}/", **self._auth()).status_code, 204)
        self.assertFalse(Webhook.objects.filter(id=hook.id).exists())

    def test_test_action_reports_delivery_outcome(self):
        hook = Webhook.objects.create(user=self.user, url="https://8.8.8.8/hook")
        resp = MagicMock(status_code=200)
        with patch("pdfeditor.webhooks.requests.post", return_value=resp):
            r = self.client.post(f"{self.BASE}{hook.id}/test/", **self._auth())
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"ok": True, "status": "200"})
        hook.refresh_from_db()
        self.assertEqual(hook.last_status, "200")
