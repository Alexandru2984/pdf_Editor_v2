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

from django.contrib import admin
from django.urls import include, path, re_path
from django.views.i18n import JavaScriptCatalog

from pdfeditor.views import admin_health_view, serve_media_view

urlpatterns = [
    path("admin/health/", admin_health_view, name="admin_health"),
    path("admin/", admin.site.urls),
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
