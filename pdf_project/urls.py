"""
URL configuration for pdf_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

import os

from django.conf import settings
from django.contrib import admin
from django.http import FileResponse, Http404
from django.urls import include, path, re_path
from django.views.decorators.http import require_safe
from django.views.generic import RedirectView
from django.views.i18n import JavaScriptCatalog

from pdfeditor.seo import robots_txt, sitemap_xml
from pdfeditor.views import admin_health_view, serve_media_view


def serve_service_worker(request):
    """Serve the PWA service worker from the site root so its scope covers
    the whole app. Browsers cap a SW's scope at its own URL's directory."""
    for candidate in (
        os.path.join(settings.BASE_DIR, "staticfiles", "sw.js"),  # collectstatic output
        os.path.join(settings.BASE_DIR, "static", "sw.js"),  # dev fallback
    ):
        if os.path.exists(candidate):
            resp = FileResponse(open(candidate, "rb"), content_type="application/javascript")  # noqa: SIM115 — FileResponse owns the handle
            resp["Service-Worker-Allowed"] = "/"
            resp["Cache-Control"] = "no-cache"  # bust stale SW between deploys
            return resp
    raise Http404("sw.js not found")


@require_safe
def serve_security_txt(request):
    """Publish the RFC 9116 security contact at its canonical location."""
    for candidate in (
        os.path.join(settings.BASE_DIR, "staticfiles", "security.txt"),
        os.path.join(settings.BASE_DIR, "static", "security.txt"),
    ):
        if os.path.exists(candidate):
            response = FileResponse(open(candidate, "rb"), content_type="text/plain; charset=utf-8")  # noqa: SIM115 — FileResponse owns the handle
            response["Cache-Control"] = "public, max-age=86400"
            return response
    raise Http404("security.txt not found")


urlpatterns = [
    path("robots.txt", robots_txt, name="robots_txt"),
    path("sitemap.xml", sitemap_xml, name="sitemap_xml"),
    path(".well-known/security.txt", serve_security_txt, name="security_txt"),
    path("admin/health/", admin_health_view, name="admin_health"),
    path("admin/", admin.site.urls),
    # Service worker must be served from the root path so its `scope` can
    # cover the whole site — browsers cap SW scope at the URL's directory.
    path("sw.js", serve_service_worker, name="service_worker"),
    path(
        "manifest.webmanifest",
        RedirectView.as_view(url="/static/manifest.webmanifest", permanent=True),
    ),
    re_path(r"^media/(?P<rel_path>.+)$", serve_media_view, name="serve_media"),
    # Django's set_language view writes the chosen language to a cookie
    # and redirects back; the LocaleMiddleware reads that cookie.
    path("i18n/", include("django.conf.urls.i18n")),
    # Serves a JS file that wires gettext()/ngettext() in the browser using
    # the active locale; static JS files import from this.
    path("jsi18n/", JavaScriptCatalog.as_view(packages=["pdfeditor"]), name="javascript-catalog"),
    path("api/v1/", include("pdfeditor.api.urls")),
    path("", include("pdfeditor.urls")),
]
