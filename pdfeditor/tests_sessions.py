"""Tests for the "Sessions & security" feature.

Covers: UserSession bookkeeping via auth signals, the new-device alert
email, the sessions page, per-session revocation (the revoked browser is
really signed out), bulk "sign out everywhere else", and cross-user
isolation.
"""

from __future__ import annotations

from django.contrib.auth import get_user_model
from django.core import mail
from django.test import Client, TestCase
from django.urls import reverse

from .models import UserSession

_PW = "s3cret-pw-123"
_UA_DESKTOP = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/125.0"
_UA_PHONE = "Mozilla/5.0 (Linux; Android 14) Firefox/126.0"


class _SessionTestBase(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="alice", password=_PW, email="alice@example.com", is_active=True
        )

    def _login(self, client: Client, ua: str = _UA_DESKTOP) -> None:
        # POST through the real login view — Client.login() bypasses the
        # request cycle, so the signals would see an empty META.
        resp = client.post(reverse("login"), {"username": "alice", "password": _PW})
        self.assertEqual(resp.status_code, 302)


class SessionTrackingTests(_SessionTestBase):
    def test_login_records_a_session_row(self):
        client = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(client)
        row = UserSession.objects.get(user=self.user)
        self.assertEqual(row.session_key, client.session.session_key)
        self.assertIn("Chrome", row.user_agent)

    def test_logout_removes_the_row(self):
        client = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(client)
        client.post(reverse("logout"))
        self.assertFalse(UserSession.objects.filter(user=self.user).exists())

    def test_first_login_sends_no_alert(self):
        client = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(client)
        self.assertEqual(len(mail.outbox), 0)

    def test_new_device_login_sends_alert_email(self):
        self._login(Client(HTTP_USER_AGENT=_UA_DESKTOP))
        self._login(Client(HTTP_USER_AGENT=_UA_PHONE))
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("New sign-in", mail.outbox[0].subject)
        self.assertEqual(mail.outbox[0].to, ["alice@example.com"])

    def test_same_device_relogin_sends_no_alert(self):
        self._login(Client(HTTP_USER_AGENT=_UA_DESKTOP))
        self._login(Client(HTTP_USER_AGENT=_UA_DESKTOP))
        self.assertEqual(len(mail.outbox), 0)


class SessionsPageTests(_SessionTestBase):
    def test_page_lists_sessions_and_marks_current(self):
        client = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(client)
        self._login(Client(HTTP_USER_AGENT=_UA_PHONE))

        resp = client.get(reverse("security_sessions"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "This device")
        self.assertContains(resp, "Chrome")
        self.assertContains(resp, "Firefox")

    def test_requires_login(self):
        resp = Client().get(reverse("security_sessions"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])


class RevocationTests(_SessionTestBase):
    def test_revoking_another_session_signs_that_browser_out(self):
        desktop = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        phone = Client(HTTP_USER_AGENT=_UA_PHONE)
        self._login(desktop)
        self._login(phone)

        phone_row = UserSession.objects.get(session_key=phone.session.session_key)
        desktop.post(reverse("revoke_session", args=[phone_row.id]))

        self.assertFalse(UserSession.objects.filter(id=phone_row.id).exists())
        # The phone's very next authenticated request bounces to login.
        resp = phone.get(reverse("profile"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("login", resp["Location"])

    def test_cannot_revoke_current_session_via_button(self):
        client = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(client)
        row = UserSession.objects.get(user=self.user)
        client.post(reverse("revoke_session", args=[row.id]))
        self.assertTrue(UserSession.objects.filter(id=row.id).exists())
        self.assertEqual(client.get(reverse("profile")).status_code, 200)  # still signed in

    def test_cannot_revoke_other_users_session(self):
        get_user_model().objects.create_user(username="mallory", password=_PW, is_active=True)
        victim = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        self._login(victim)
        victim_row = UserSession.objects.get(user=self.user)

        attacker = Client(HTTP_USER_AGENT=_UA_PHONE)
        attacker.login(username="mallory", password=_PW)
        attacker.post(reverse("revoke_session", args=[victim_row.id]))

        self.assertTrue(UserSession.objects.filter(id=victim_row.id).exists())
        self.assertEqual(victim.get(reverse("profile")).status_code, 200)

    def test_sign_out_everywhere_else_keeps_current(self):
        desktop = Client(HTTP_USER_AGENT=_UA_DESKTOP)
        phone = Client(HTTP_USER_AGENT=_UA_PHONE)
        tablet = Client(HTTP_USER_AGENT=_UA_PHONE + " Tablet")
        for c in (desktop, phone, tablet):
            self._login(c)

        desktop.post(reverse("revoke_other_sessions"))

        rows = UserSession.objects.filter(user=self.user)
        self.assertEqual(rows.count(), 1)
        self.assertEqual(rows.first().session_key, desktop.session.session_key)
        self.assertEqual(desktop.get(reverse("profile")).status_code, 200)
        self.assertEqual(phone.get(reverse("profile")).status_code, 302)
