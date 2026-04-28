"""Tests for the auth flow + ownership rewriting (anonymous vs. logged-in)."""

import io
import os
import re
import shutil
import tempfile

import fitz
from django.contrib.auth import get_user_model
from django.contrib.auth.tokens import default_token_generator
from django.core import mail
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase, override_settings
from django.urls import reverse
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode

from .models import ProcessedPDF, UploadedPDF

User = get_user_model()


def _pdf_bytes(num_pages=1, prefix="hi"):
    doc = fitz.open()
    for i in range(num_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"{prefix} {i + 1}", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _upload(client, name="x.pdf"):
    return client.post(
        reverse("upload"),
        {"pdf_file": SimpleUploadedFile(name, _pdf_bytes(), content_type="application/pdf")},
    )


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="pdfedit_auth_"))
class _AuthTestBase(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from django.conf import settings as dj_settings

        os.makedirs(dj_settings.MEDIA_ROOT, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from django.conf import settings as dj_settings

        shutil.rmtree(dj_settings.MEDIA_ROOT, ignore_errors=True)
        super().tearDownClass()


class RegisterViewTests(_AuthTestBase):
    def setUp(self):
        self.client = Client()

    def test_get_renders_form(self):
        resp = self.client.get(reverse("register"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Create Account")

    def test_post_creates_inactive_user_and_redirects_to_login(self):
        from django.core import mail

        resp = self.client.post(
            reverse("register"),
            {
                "username": "alice",
                "email": "alice@example.com",
                "password1": "Sup3rSecret!Long",
                "password2": "Sup3rSecret!Long",
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(resp.url.endswith(reverse("login")))

        # User exists but cannot sign in yet.
        user = User.objects.get(username="alice")
        self.assertFalse(user.is_active)

        # A confirmation email was sent.
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("alice@example.com", mail.outbox[0].to)
        self.assertIn("/accounts/confirm/", mail.outbox[0].body)

    def test_post_rejects_duplicate_email(self):
        User.objects.create_user(username="bob", email="bob@example.com", password="x")
        resp = self.client.post(
            reverse("register"),
            {
                "username": "bob2",
                "email": "BOB@example.com",  # case-insensitive collision
                "password1": "Sup3rSecret!Long",
                "password2": "Sup3rSecret!Long",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "already exists")

    def test_post_rejects_password_mismatch(self):
        resp = self.client.post(
            reverse("register"),
            {
                "username": "carol",
                "email": "carol@example.com",
                "password1": "Sup3rSecret!Long",
                "password2": "different",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(User.objects.filter(username="carol").exists())

    def test_authenticated_user_redirects_away(self):
        User.objects.create_user(username="dave", password="pw12345!")
        self.client.login(username="dave", password="pw12345!")
        resp = self.client.get(reverse("register"))
        self.assertEqual(resp.status_code, 302)


class LoginLogoutTests(_AuthTestBase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="eve", password="pw12345!Long")

    def test_login_get_renders(self):
        resp = self.client.get(reverse("login"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Sign In")

    def test_login_with_correct_credentials(self):
        resp = self.client.post(reverse("login"), {"username": "eve", "password": "pw12345!Long"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(int(self.client.session["_auth_user_id"]), self.user.pk)

    def test_login_with_wrong_password_stays_on_form(self):
        resp = self.client.post(reverse("login"), {"username": "eve", "password": "wrong"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_logout_clears_session(self):
        self.client.login(username="eve", password="pw12345!Long")
        self.assertIn("_auth_user_id", self.client.session)
        resp = self.client.post(reverse("logout"))
        self.assertEqual(resp.status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)


class OwnershipScopingTests(_AuthTestBase):
    """The big one: PDFs uploaded while authenticated belong to the user; uploaded
    anonymously they belong to the session. The two never alias each other."""

    def setUp(self):
        self.client = Client()
        self.alice = User.objects.create_user(username="alice", password="pw12345!Long")
        self.bob = User.objects.create_user(username="bob", password="pw12345!Long")

    def test_anonymous_upload_has_no_user(self):
        _upload(self.client, name="anon.pdf")
        pdf = UploadedPDF.objects.get(name="anon.pdf")
        self.assertIsNone(pdf.user)
        self.assertTrue(pdf.session_key)

    def test_authenticated_upload_attaches_user(self):
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="alice.pdf")
        pdf = UploadedPDF.objects.get(name="alice.pdf")
        self.assertEqual(pdf.user, self.alice)
        self.assertEqual(pdf.session_key, "")

    def test_user_pdfs_are_not_visible_to_other_users(self):
        # Alice uploads.
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="alice_secret.pdf")
        self.client.logout()

        # Bob logs in — should NOT see alice's PDF on the dashboard.
        bob_client = Client()
        bob_client.login(username="bob", password="pw12345!Long")
        resp = bob_client.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "alice_secret.pdf")

    def test_user_pdfs_are_not_visible_to_anonymous_browser(self):
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="alice_only.pdf")
        self.client.logout()

        resp = self.client.get(reverse("dashboard"))
        self.assertEqual(resp.status_code, 200)
        self.assertNotContains(resp, "alice_only.pdf")

    def test_user_pdfs_persist_across_sessions(self):
        # Login + upload, then logout + new client + login → still see PDF.
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="persist.pdf")

        fresh = Client()
        fresh.login(username="alice", password="pw12345!Long")
        resp = fresh.get(reverse("dashboard"))
        self.assertContains(resp, "persist.pdf")

    def test_anon_pdfs_dont_leak_to_authenticated_user(self):
        # Browse anonymously, upload, then log in — anon PDFs stay anon.
        _upload(self.client, name="anon_first.pdf")
        # Consume the upload success message so it doesn't pollute the next page.
        self.client.get(reverse("dashboard"))
        # Confirm the row exists with no user.
        anon_pdf = UploadedPDF.objects.get(name="anon_first.pdf")
        self.assertIsNone(anon_pdf.user)

        self.client.login(username="alice", password="pw12345!Long")
        # Alice's dashboard query must not match anonymous rows.
        from .views._common import owner_filter

        request_factory_resp = self.client.get(reverse("dashboard"))
        # Direct DB check (faster + bypasses message rendering completely).
        self.assertFalse(
            UploadedPDF.objects.filter(owner_filter(request_factory_resp.wsgi_request))
            .filter(id=anon_pdf.id)
            .exists()
        )

    def test_processed_outputs_are_user_scoped(self):
        # Alice runs find/replace, then bob logs in — bob shouldn't see her output.
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client)
        self.client.post(
            reverse("edit"),
            {"search_text": "hi", "replace_text": "hello", "case_sensitive": True, "page_range": ""},
        )
        alice_outputs = ProcessedPDF.objects.filter(user=self.alice).count()
        self.assertGreaterEqual(alice_outputs, 1)

        self.client.logout()
        bob_client = Client()
        bob_client.login(username="bob", password="pw12345!Long")
        resp = bob_client.get(reverse("history"))
        self.assertContains(resp, "No operations yet")

    def test_history_persists_for_user_across_sessions(self):
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client)
        self.client.post(reverse("compress"), {"quality": "medium"})

        fresh = Client()
        fresh.login(username="alice", password="pw12345!Long")
        resp = fresh.get(reverse("history"))
        self.assertContains(resp, "Compress")

    def test_delete_pdf_only_works_for_owner(self):
        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="alice_only.pdf")
        pdf = UploadedPDF.objects.get(name="alice_only.pdf")

        bob_client = Client()
        bob_client.login(username="bob", password="pw12345!Long")
        bob_client.get(reverse("delete_pdf", args=[pdf.id]))

        self.assertTrue(UploadedPDF.objects.filter(id=pdf.id).exists())

    def test_serve_media_blocks_other_users(self):
        from django.conf import settings as dj_settings

        self.client.login(username="alice", password="pw12345!Long")
        _upload(self.client, name="alice_pdf.pdf")
        pdf = UploadedPDF.objects.get(name="alice_pdf.pdf")
        rel = os.path.relpath(pdf.path, dj_settings.MEDIA_ROOT)

        bob_client = Client()
        bob_client.login(username="bob", password="pw12345!Long")
        resp = bob_client.get(f"/media/{rel}")
        self.assertEqual(resp.status_code, 404)


class EmailConfirmationTests(_AuthTestBase):
    """Inactive users created by register can only sign in after confirming."""

    def setUp(self):
        self.client = Client()

    def _register_and_get_token(self, username="alice", email="alice@example.com"):
        self.client.post(
            reverse("register"),
            {
                "username": username,
                "email": email,
                "password1": "Sup3rSecret!Long",
                "password2": "Sup3rSecret!Long",
            },
        )
        user = User.objects.get(username=username)
        uidb64 = urlsafe_base64_encode(force_bytes(user.pk))
        token = default_token_generator.make_token(user)
        return user, uidb64, token

    def test_inactive_user_cannot_login(self):
        self._register_and_get_token()
        resp = self.client.post(reverse("login"), {"username": "alice", "password": "Sup3rSecret!Long"})
        # Stays on form; not logged in.
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_valid_confirmation_activates_and_logs_in(self):
        user, uidb64, token = self._register_and_get_token()
        self.assertFalse(user.is_active)

        resp = self.client.get(reverse("confirm_email", args=[uidb64, token]))
        self.assertEqual(resp.status_code, 302)

        user.refresh_from_db()
        self.assertTrue(user.is_active)
        self.assertIn("_auth_user_id", self.client.session)

    def test_invalid_token_returns_400_page(self):
        user, uidb64, _token = self._register_and_get_token()
        resp = self.client.get(reverse("confirm_email", args=[uidb64, "totally-bogus-token"]))
        self.assertEqual(resp.status_code, 400)
        user.refresh_from_db()
        self.assertFalse(user.is_active)

    def test_invalid_uid_returns_400_page(self):
        # Garbage uidb64 (not a valid base64-encoded PK).
        resp = self.client.get(reverse("confirm_email", args=["@@@notbase64@@@", "any-token"]))
        self.assertEqual(resp.status_code, 400)

    def test_token_invalidates_after_first_use(self):
        # default_token_generator hashes is_active into the seed, so once the
        # account is active the same token rejects.
        _user, uidb64, token = self._register_and_get_token()
        first = self.client.get(reverse("confirm_email", args=[uidb64, token]))
        self.assertEqual(first.status_code, 302)

        replay = self.client.get(reverse("confirm_email", args=[uidb64, token]))
        # User is already active, so the same token is now invalid.
        self.assertEqual(replay.status_code, 400)


class ResendConfirmationTests(_AuthTestBase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()  # rate-limit state lives in the cache between tests
        self.client = Client()

    def test_get_renders_form(self):
        resp = self.client.get(reverse("resend_confirmation"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Resend Confirmation")

    def test_post_with_inactive_user_email_sends_mail(self):
        User.objects.create_user(
            username="hank", email="hank@example.com", password="pw", is_active=False,
        )
        resp = self.client.post(
            reverse("resend_confirmation"), {"email": "hank@example.com"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Check Your Email")
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("hank@example.com", mail.outbox[0].to)
        self.assertIn("/accounts/confirm/", mail.outbox[0].body)

    def test_post_with_active_user_email_does_not_send(self):
        # Already-active accounts shouldn't get a new confirmation email.
        User.objects.create_user(
            username="ivy", email="ivy@example.com", password="pw", is_active=True,
        )
        resp = self.client.post(
            reverse("resend_confirmation"), {"email": "ivy@example.com"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(mail.outbox), 0)

    def test_post_with_unknown_email_silently_succeeds(self):
        # No existence-oracle: same response page, no email sent.
        resp = self.client.post(
            reverse("resend_confirmation"), {"email": "ghost@example.com"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Check Your Email")
        self.assertEqual(len(mail.outbox), 0)

    def test_post_with_invalid_email_re_renders_form(self):
        resp = self.client.post(
            reverse("resend_confirmation"), {"email": "not-an-email"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Resend Confirmation")  # form re-rendered

    def test_authenticated_user_redirected(self):
        User.objects.create_user(username="jules", password="pw")
        self.client.login(username="jules", password="pw")
        resp = self.client.get(reverse("resend_confirmation"))
        self.assertEqual(resp.status_code, 302)


class PasswordChangeTests(_AuthTestBase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username="frank", password="OldPw!12345")
        self.client.login(username="frank", password="OldPw!12345")

    def test_get_renders_form(self):
        resp = self.client.get(reverse("password_change"))
        self.assertEqual(resp.status_code, 200)

    def test_change_with_correct_old_password(self):
        resp = self.client.post(
            reverse("password_change"),
            {
                "old_password": "OldPw!12345",
                "new_password1": "BrandNewPw!9876",
                "new_password2": "BrandNewPw!9876",
            },
        )
        self.assertEqual(resp.status_code, 302)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("BrandNewPw!9876"))

    def test_wrong_old_password_rejected(self):
        resp = self.client.post(
            reverse("password_change"),
            {
                "old_password": "wrong",
                "new_password1": "BrandNewPw!9876",
                "new_password2": "BrandNewPw!9876",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("OldPw!12345"))

    def test_anonymous_redirected_away(self):
        Client().logout()  # ensure unrelated client is anon
        anon = Client()
        resp = anon.get(reverse("password_change"))
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/accounts/login/", resp.url)


class PasswordResetTests(_AuthTestBase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="grace",
            email="grace@example.com",
            password="OldPw!12345",
        )

    def test_request_sends_email_with_link(self):
        resp = self.client.post(reverse("password_reset"), {"email": "grace@example.com"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn("grace@example.com", mail.outbox[0].to)
        # The body has the reset URL with uidb64/token.
        self.assertRegex(mail.outbox[0].body, r"/accounts/password/reset/[^/]+/[^/]+/")

    def test_unknown_email_silently_succeeds(self):
        # Django's default is to NOT reveal whether the email exists.
        resp = self.client.post(reverse("password_reset"), {"email": "nobody@example.com"})
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(len(mail.outbox), 0)

    def test_full_reset_flow_sets_new_password(self):
        # Trigger the reset email.
        self.client.post(reverse("password_reset"), {"email": "grace@example.com"})
        body = mail.outbox[0].body

        # Extract the reset link from the email body.
        match = re.search(r"/accounts/password/reset/([^/]+)/([^/\s]+)/", body)
        self.assertIsNotNone(match)
        uidb64, token = match.group(1), match.group(2)

        # GET the confirm URL → Django redirects to a "set-password" URL with
        # token replaced by a session marker; follow it.
        confirm_url = reverse("password_reset_confirm", args=[uidb64, token])
        resp = self.client.get(confirm_url, follow=True)
        self.assertEqual(resp.status_code, 200)

        # Submit the new password to the redirected URL (final URL after follow).
        post_url = resp.redirect_chain[-1][0] if resp.redirect_chain else confirm_url
        resp2 = self.client.post(
            post_url,
            {"new_password1": "FreshPw!7777", "new_password2": "FreshPw!7777"},
        )
        self.assertEqual(resp2.status_code, 302)

        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password("FreshPw!7777"))
