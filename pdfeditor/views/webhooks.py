"""Logged-in-only management for outbound webhooks.

Webhooks push a signed POST to a user's endpoint when one of their async jobs
finishes (see :mod:`pdfeditor.webhooks` + the ``deliver_webhook`` task). This
module is just the CRUD surface; anonymous users have no account to attach an
endpoint to, so every view is ``@login_required``.
"""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import gettext as _
from django.views.decorators.http import require_http_methods

from .. import webhooks as webhook_utils
from ..models import Webhook
from ..ratelimiting import auth_aware_ratelimit

# Cap per user: each terminal job POSTs to every active endpoint, so this
# doubles as an amplification guard.
MAX_WEBHOOKS_PER_USER = 10


@login_required
def webhooks_view(request: HttpRequest) -> HttpResponse:
    hooks = Webhook.objects.filter(user=request.user)
    # Full signing secret is flashed exactly once, right after creation.
    new_secret = request.session.pop("new_webhook_secret", None)
    new_id = request.session.pop("new_webhook_id", None)
    return render(
        request,
        "pdfeditor/webhooks.html",
        {
            "webhooks": hooks,
            "new_webhook_secret": new_secret,
            "new_webhook_id": new_id,
            "max_webhooks": MAX_WEBHOOKS_PER_USER,
            "signature_header": webhook_utils.SIGNATURE_HEADER,
        },
    )


@login_required
@require_http_methods(["POST"])
# login_required rejects anon before this runs, so anon_rate is a formality;
# the real cap is per-user (each create does a DNS lookup for SSRF validation).
@auth_aware_ratelimit(anon_rate="30/h", user_rate="30/h", method="POST")
def create_webhook_view(request: HttpRequest) -> HttpResponse:
    url = (request.POST.get("url") or "").strip()
    description = (request.POST.get("description") or "").strip()[:100]

    if Webhook.objects.filter(user=request.user).count() >= MAX_WEBHOOKS_PER_USER:
        messages.error(
            request,
            _("You can have at most %(n)s webhooks.") % {"n": MAX_WEBHOOKS_PER_USER},
        )
        return redirect("webhooks")

    # Anti-SSRF: the target is user-supplied. Validation resolves DNS, so it
    # can take a moment, but create is a rare, deliberate action.
    try:
        webhook_utils.validate_webhook_url(url)
    except webhook_utils.InvalidWebhookURL as exc:
        messages.error(request, _("Invalid webhook URL: %(err)s") % {"err": exc})
        return redirect("webhooks")

    hook = Webhook.objects.create(user=request.user, url=url, description=description)
    request.session["new_webhook_secret"] = hook.secret
    request.session["new_webhook_id"] = str(hook.id)
    messages.success(request, _("Webhook created. Save the signing secret now — it's shown only once."))
    return redirect("webhooks")


@login_required
@require_http_methods(["POST"])
def toggle_webhook_view(request: HttpRequest, webhook_id: str) -> HttpResponse:
    hook = get_object_or_404(Webhook, user=request.user, id=webhook_id)
    hook.is_active = not hook.is_active
    # Re-enabling a webhook that auto-disabled clears its failure streak so it
    # gets a fresh set of retries.
    if hook.is_active:
        hook.failure_count = 0
    hook.save(update_fields=["is_active", "failure_count"])
    messages.success(
        request,
        _("Webhook enabled.") if hook.is_active else _("Webhook disabled."),
    )
    return redirect("webhooks")


@login_required
@require_http_methods(["POST"])
def delete_webhook_view(request: HttpRequest, webhook_id: str) -> HttpResponse:
    hook = get_object_or_404(Webhook, user=request.user, id=webhook_id)
    hook.delete()
    messages.success(request, _("Webhook deleted."))
    return redirect("webhooks")
