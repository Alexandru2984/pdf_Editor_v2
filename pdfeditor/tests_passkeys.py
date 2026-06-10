"""Tests for WebAuthn passkey registration, login and removal.

The authenticator-side cryptography can't run in a unit test, so the
py_webauthn verifier functions are mocked at the view-module boundary; what
these tests pin is everything the views own: challenge lifecycle (one-shot,
session-bound), credential storage, the no-enumeration error contract,
sign-count bookkeeping, MFA bypass semantics and owner-scoped deletion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from .models import WebAuthnCredential

_PW = "s3cret-pw-123"


def _post_json(client: Client, url: str, payload: dict):
    return client.post(url, json.dumps(payload), content_type="application/json")


class _PasskeyTestBase(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="alice", password=_PW, email="alice@example.com", is_active=True
        )
        self.client.login(username="alice", password=_PW)


class RegistrationOptionsTests(_PasskeyTestBase):
    def test_options_require_auth(self):
        resp = Client().post(reverse("passkey_register_options"))
        self.assertEqual(resp.status_code, 403)

    def test_options_store_challenge_and_have_rp(self):
        resp = self.client.post(reverse("passkey_register_options"))
        self.assertEqual(resp.status_code, 200)
        data = json.loads(resp.content)
        self.assertIn("challenge", data)
        self.assertEqual(data["authenticatorSelection"]["userVerification"], "required")
        self.assertIn("webauthn_reg_challenge", self.client.session)

    def test_passkey_limit_enforced(self):
        for i in range(10):
            WebAuthnCredential.objects.create(
                user=self.user, credential_id=f"cred-{i}", public_key="pk", sign_count=0
            )
        resp = self.client.post(reverse("passkey_register_options"))
        self.assertEqual(resp.status_code, 400)


class RegistrationVerifyTests(_PasskeyTestBase):
    def _prime_challenge(self):
        self.client.post(reverse("passkey_register_options"))

    def test_register_without_pending_challenge_fails(self):
        resp = _post_json(self.client, reverse("passkey_register"), {"credential": {"id": "x"}})
        self.assertEqual(resp.status_code, 400)

    def test_successful_registration_stores_credential(self):
        self._prime_challenge()
        verified = SimpleNamespace(
            credential_id=b"\x01\x02\x03", credential_public_key=b"\x04\x05", sign_count=7
        )
        with patch("pdfeditor.views.passkeys.verify_registration_response", return_value=verified):
            resp = _post_json(
                self.client,
                reverse("passkey_register"),
                {"credential": {"id": "AQID", "response": {"transports": ["internal"]}}, "label": "Laptop"},
            )
        self.assertEqual(resp.status_code, 200)
        row = WebAuthnCredential.objects.get(user=self.user)
        self.assertEqual(row.sign_count, 7)
        self.assertEqual(row.label, "Laptop")
        self.assertEqual(row.transports, "internal")
        # Challenge is one-shot: replaying the same POST has no challenge left.
        with patch("pdfeditor.views.passkeys.verify_registration_response", return_value=verified):
            resp2 = _post_json(self.client, reverse("passkey_register"), {"credential": {"id": "AQID"}})
        self.assertEqual(resp2.status_code, 400)

    def test_failed_verification_stores_nothing(self):
        from webauthn.helpers.exceptions import InvalidRegistrationResponse

        self._prime_challenge()
        with patch(
            "pdfeditor.views.passkeys.verify_registration_response",
            side_effect=InvalidRegistrationResponse("bad"),
        ):
            resp = _post_json(self.client, reverse("passkey_register"), {"credential": {"id": "x"}})
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(WebAuthnCredential.objects.exists())


class LoginTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username="alice", password=_PW, is_active=True)
        self.cred = WebAuthnCredential.objects.create(
            user=self.user, credential_id="AQID", public_key="BAUG", sign_count=5
        )

    def _prime(self, client: Client):
        resp = client.post(reverse("passkey_auth_options"))
        self.assertEqual(resp.status_code, 200)
        self.assertIn("challenge", json.loads(resp.content))

    def test_successful_passkey_login_creates_session(self):
        client = Client()
        self._prime(client)
        verified = SimpleNamespace(new_sign_count=6)
        with patch("pdfeditor.views.passkeys.verify_authentication_response", return_value=verified):
            resp = _post_json(client, reverse("passkey_login"), {"credential": {"id": "AQID"}})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(client.session.get("_auth_user_id"), str(self.user.pk))
        self.cred.refresh_from_db()
        self.assertEqual(self.cred.sign_count, 6)
        self.assertIsNotNone(self.cred.last_used_at)

    def test_passkey_login_bypasses_totp_step(self):
        from .models import MfaDevice

        MfaDevice.objects.create(user=self.user, confirmed=True)
        client = Client()
        self._prime(client)
        with patch(
            "pdfeditor.views.passkeys.verify_authentication_response",
            return_value=SimpleNamespace(new_sign_count=6),
        ):
            _post_json(client, reverse("passkey_login"), {"credential": {"id": "AQID"}})
        # Fully signed in — no pending-MFA half-state.
        self.assertEqual(client.session.get("_auth_user_id"), str(self.user.pk))
        self.assertEqual(client.get(reverse("profile")).status_code, 200)

    def test_unknown_credential_is_not_an_oracle(self):
        client = Client()
        self._prime(client)
        resp = _post_json(client, reverse("passkey_login"), {"credential": {"id": "nope"}})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(json.loads(resp.content)["error"], "sign-in could not be verified")
        self.assertNotIn("_auth_user_id", client.session)

    def test_failed_assertion_does_not_log_in(self):
        from webauthn.helpers.exceptions import InvalidAuthenticationResponse

        client = Client()
        self._prime(client)
        with patch(
            "pdfeditor.views.passkeys.verify_authentication_response",
            side_effect=InvalidAuthenticationResponse("bad"),
        ):
            resp = _post_json(client, reverse("passkey_login"), {"credential": {"id": "AQID"}})
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn("_auth_user_id", client.session)

    def test_login_without_challenge_fails(self):
        resp = _post_json(Client(), reverse("passkey_login"), {"credential": {"id": "AQID"}})
        self.assertEqual(resp.status_code, 400)

    def test_inactive_user_cannot_sign_in(self):
        self.user.is_active = False
        self.user.save(update_fields=["is_active"])
        client = Client()
        self._prime(client)
        resp = _post_json(client, reverse("passkey_login"), {"credential": {"id": "AQID"}})
        self.assertEqual(resp.status_code, 400)
        self.assertNotIn("_auth_user_id", client.session)


class DeletionTests(_PasskeyTestBase):
    def test_owner_can_delete(self):
        row = WebAuthnCredential.objects.create(
            user=self.user, credential_id="AQID", public_key="pk", sign_count=0
        )
        resp = self.client.post(reverse("passkey_delete", args=[row.id]))
        self.assertEqual(resp.status_code, 302)
        self.assertFalse(WebAuthnCredential.objects.exists())

    def test_other_user_cannot_delete(self):
        get_user_model().objects.create_user(username="mallory", password=_PW, is_active=True)
        row = WebAuthnCredential.objects.create(
            user=self.user, credential_id="AQID", public_key="pk", sign_count=0
        )
        attacker = Client()
        attacker.login(username="mallory", password=_PW)
        attacker.post(reverse("passkey_delete", args=[row.id]))
        self.assertTrue(WebAuthnCredential.objects.filter(id=row.id).exists())
