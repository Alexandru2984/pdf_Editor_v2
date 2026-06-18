"""Tests for the auth-aware rate limiter."""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.http import HttpResponse
from django.test import RequestFactory, TestCase, override_settings

from .ratelimiting import _compute_key_and_rate, auth_aware_ratelimit, check_rate_limit

User = get_user_model()


class _Anon:
    is_authenticated = False
    pk = None


class _BrokenLazyUser:
    @property
    def is_authenticated(self):
        raise IndexError("session user lookup failed")


class ComputeKeyAndRateTests(TestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def test_anon_uses_ip_key_and_anon_rate(self):
        request = self.factory.post("/x/")
        request.user = _Anon()
        request.META["REMOTE_ADDR"] = "203.0.113.7"
        key, rate = _compute_key_and_rate(request, anon_rate="20/h", user_rate="100/h")
        self.assertEqual(key, "ip:203.0.113.7")
        self.assertEqual(rate, "20/h")

    def test_authenticated_uses_user_key_and_user_rate(self):
        user = User.objects.create_user(username="alice", password="pw")
        request = self.factory.post("/x/")
        request.user = user
        key, rate = _compute_key_and_rate(request, anon_rate="20/h", user_rate="100/h")
        self.assertEqual(key, f"user:{user.pk}")
        self.assertEqual(rate, "100/h")

    def test_missing_ip_falls_back_to_noip_bucket(self):
        request = self.factory.post("/x/")
        request.user = _Anon()
        request.META.pop("REMOTE_ADDR", None)
        key, _rate = _compute_key_and_rate(request, anon_rate="1/h", user_rate="1/h")
        self.assertEqual(key, "ip:noip")

    def test_broken_lazy_user_falls_back_to_anon_bucket(self):
        request = self.factory.post("/x/")
        request.user = _BrokenLazyUser()
        request.META["REMOTE_ADDR"] = "203.0.113.8"
        key, rate = _compute_key_and_rate(request, anon_rate="20/h", user_rate="100/h")
        self.assertEqual(key, "ip:203.0.113.8")
        self.assertEqual(rate, "20/h")


class AuthAwareRatelimitDecoratorTests(TestCase):
    """Functional test of the decorator end-to-end through the cache."""

    def setUp(self):
        cache.clear()  # rate-limit state lives in Django's default cache
        self.factory = RequestFactory()

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_authenticated_uses_higher_user_rate(self):
        @auth_aware_ratelimit(anon_rate="1/h", user_rate="3/h", method="POST")
        def view(request):
            return HttpResponse("ok")

        user = User.objects.create_user(username="bob", password="pw")
        for _ in range(3):
            request = self.factory.post("/")
            request.user = user
            request.META["REMOTE_ADDR"] = "1.2.3.4"
            self.assertEqual(view(request).content, b"ok")

        # 4th request hits the user_rate ceiling.
        from django_ratelimit.exceptions import Ratelimited

        request = self.factory.post("/")
        request.user = user
        request.META["REMOTE_ADDR"] = "1.2.3.4"
        with self.assertRaises(Ratelimited):
            view(request)

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_anon_uses_lower_anon_rate(self):
        @auth_aware_ratelimit(anon_rate="2/h", user_rate="100/h", method="POST")
        def view(request):
            return HttpResponse("ok")

        from django_ratelimit.exceptions import Ratelimited

        for _ in range(2):
            request = self.factory.post("/")
            request.user = _Anon()
            request.META["REMOTE_ADDR"] = "9.9.9.9"
            self.assertEqual(view(request).content, b"ok")

        request = self.factory.post("/")
        request.user = _Anon()
        request.META["REMOTE_ADDR"] = "9.9.9.9"
        with self.assertRaises(Ratelimited):
            view(request)

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_user_quota_isolated_from_ip_quota(self):
        """Two anon users from same IP should share quota; one user should not affect another's quota."""

        @auth_aware_ratelimit(anon_rate="2/h", user_rate="2/h", method="POST")
        def view(request):
            return HttpResponse("ok")

        alice = User.objects.create_user(username="alice", password="pw")
        bob = User.objects.create_user(username="bob", password="pw")

        # Use up alice's quota.
        for _ in range(2):
            r = self.factory.post("/")
            r.user = alice
            r.META["REMOTE_ADDR"] = "1.2.3.4"
            view(r)

        # Bob, from the same IP, should still have his quota intact —
        # because the key includes user.pk, not just IP.
        r = self.factory.post("/")
        r.user = bob
        r.META["REMOTE_ADDR"] = "1.2.3.4"
        self.assertEqual(view(r).content, b"ok")

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_get_method_not_counted(self):
        @auth_aware_ratelimit(anon_rate="1/h", user_rate="1/h", method="POST")
        def view(request):
            return HttpResponse("ok")

        # Many GETs should be free.
        for _ in range(5):
            r = self.factory.get("/")
            r.user = _Anon()
            r.META["REMOTE_ADDR"] = "5.5.5.5"
            self.assertEqual(view(r).content, b"ok")


class CheckRateLimitAsyncHelperTests(TestCase):
    """The standalone helper used by async views."""

    def setUp(self):
        cache.clear()
        self.factory = RequestFactory()

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_helper_increments_and_blocks(self):
        request = self.factory.post("/")
        request.user = _Anon()
        request.META["REMOTE_ADDR"] = "10.0.0.1"

        kwargs = {"group": "async_test", "anon_rate": "2/h", "user_rate": "2/h", "method": "POST"}
        self.assertFalse(check_rate_limit(request, **kwargs))
        self.assertFalse(check_rate_limit(request, **kwargs))
        # Third call hits the limit.
        self.assertTrue(check_rate_limit(request, **kwargs))

    @override_settings(RATELIMIT_USE_CACHE="default")
    def test_increment_false_does_not_consume_quota(self):
        request = self.factory.post("/")
        request.user = _Anon()
        request.META["REMOTE_ADDR"] = "10.0.0.2"

        # Repeated peeks should never tip the limit.
        for _ in range(10):
            limited = check_rate_limit(
                request,
                group="peek",
                anon_rate="1/h",
                user_rate="1/h",
                method="POST",
                increment=False,
            )
            self.assertFalse(limited)
