"""Base class for browser-driven E2E tests.

Unit/integration tests in ``pdfeditor/tests_*.py`` exercise the Django
stack at the view/model layer. E2E tests in this package drive a real
browser through ``StaticLiveServerTestCase`` so we catch regressions
that only show up when JS + CSS + the server actually run together
(form submits that depend on a click handler, redirects that depend on
HX-Trigger headers, file-upload flows that need real multipart).

Conventions:

- One Playwright instance + one browser per test class. A fresh
  ``BrowserContext`` per test gives us a clean cookie jar and storage
  without the cost of a browser restart.
- ``StaticLiveServerTestCase`` runs each test in its own DB transaction
  that's rolled back, so we never need to clean up users by hand.
- Email confirmation is bypassed: we mark fixture users ``is_active``
  directly. The auth flow itself is covered by ``tests_auth.py``;
  re-driving it here would just slow the suite down.
"""

from __future__ import annotations

import os
from typing import ClassVar

# Playwright's sync API runs its own event loop. Django interprets that
# as "we're inside an async context" and refuses ORM calls. The escape
# hatch is documented for exactly this case — setting it before Django
# checks (i.e. before importing anything that uses the ORM) is required.
os.environ.setdefault("DJANGO_ALLOW_ASYNC_UNSAFE", "true")

from django.contrib.auth import get_user_model  # noqa: E402
from django.contrib.staticfiles.testing import StaticLiveServerTestCase  # noqa: E402
from django.test import override_settings, tag  # noqa: E402

User = get_user_model()

# Default password used by make_user() / login(). Deliberately structured so
# it doesn't pattern-match a real credential — GitHub Secret Scanning flagged
# an earlier plausible-looking value as a leak. This one carries the word
# "fixture" so a human (or a scanner heuristic) reading it can tell at a
# glance. Cannot authenticate against anything outside the transaction-scoped
# test DB; never use a real password here.
_FIXTURE_PASSWORD = "e2e-fixture-not-a-real-credential"  # noqa: S105 — test fixture


# Tagged "e2e" so CI can include or exclude this whole bucket with
# --tag=e2e / --exclude-tag=e2e. The unit-test job runs without playwright
# installed, so it MUST exclude these or every test class would error in
# setUpClass.
@tag("e2e")
@override_settings(
    # Ratelimit defaults trip after a few POSTs and would flake the suite.
    # The ratelimit logic itself is covered by tests_ratelimiting.py.
    RATELIMIT_ENABLE=False,
    # In-memory cache so ratelimit counters don't bleed across processes
    # if some leftover code still references them.
    CACHES={"default": {"BACKEND": "django.core.cache.backends.locmem.LocMemCache"}},
    # E2E uses runserver-style static files; no need for CDN-friendly hashes.
    STATICFILES_STORAGE="django.contrib.staticfiles.storage.StaticFilesStorage",
)
class PlaywrightTestCase(StaticLiveServerTestCase):
    """Spin up Chromium once per class and hand each test a fresh context."""

    # Set at class scope so subclasses can override (e.g. for headed debugging).
    headless: ClassVar[bool] = True

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        # Import inside setUpClass so non-E2E test runs don't need
        # playwright installed at all.
        from playwright.sync_api import sync_playwright

        cls._playwright = sync_playwright().start()
        cls.browser = cls._playwright.chromium.launch(headless=cls.headless)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls._playwright.stop()
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        # Each test gets an isolated context. Viewport matches what the
        # responsive layout was designed against; bumping it without
        # checking the dashboard layout will move things around.
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 800})
        self.page = self.context.new_page()

    def tearDown(self) -> None:
        self.context.close()
        super().tearDown()

    # ---- helpers ----------------------------------------------------------

    def make_user(self, username: str = "alice", password: str = _FIXTURE_PASSWORD) -> User:
        """Create an active user — bypasses the email-confirmation gate."""
        user = User.objects.create_user(
            username=username,
            email=f"{username}@example.com",
            password=password,
        )
        user.is_active = True
        user.save(update_fields=["is_active"])
        return user

    def login(self, username: str = "alice", password: str = _FIXTURE_PASSWORD) -> None:
        """Walk through the real login form so cookies + CSRF land in the
        browser context, not just the session backend."""
        self.page.goto(f"{self.live_server_url}/accounts/login/")
        # The header's language-switcher is also `<button type="submit">`,
        # so we have to scope the selector to the login form itself or
        # Playwright will submit the wrong form and we'll bounce back.
        login_form = self.page.locator("form.edit-form")
        login_form.locator('input[name="username"]').fill(username)
        login_form.locator('input[name="password"]').fill(password)
        login_form.locator('button[type="submit"]').click()
        self.page.wait_for_url(f"{self.live_server_url}/")
