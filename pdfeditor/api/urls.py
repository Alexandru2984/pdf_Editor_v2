"""URL conf for the REST API. Mounted at ``/api/v1/`` from the project urls."""

from django.urls import include, path
from django.views.decorators.cache import cache_page
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from rest_framework.routers import DefaultRouter

from . import ops_views, views

# Schema generation is expensive *and* triggers a race-condition assertion
# in drf_spectacular under concurrent load (observed in a 500-user locust
# run: thousands of 500s with "Schema generation REQUIRES a view instance").
# Caching the rendered response for 5 minutes sidesteps both — schemas
# don't change between requests anyway. Re-evaluated on each deploy.
_cached_schema = cache_page(60 * 5)(SpectacularAPIView.as_view())

app_name = "api"

router = DefaultRouter()
router.register(r"pdfs", views.UploadedPDFViewSet, basename="pdf")
router.register(r"outputs", views.ProcessedPDFViewSet, basename="output")
router.register(r"jobs", views.JobViewSet, basename="job")

op_patterns = [
    path("batch/", ops_views.BatchOpView.as_view(), name="op-batch"),
    path("summarize/", ops_views.SummarizeOpView.as_view(), name="op-summarize"),
    path("compress/", ops_views.CompressOpView.as_view(), name="op-compress"),
    path("split/", ops_views.SplitOpView.as_view(), name="op-split"),
    path("merge/", ops_views.MergeOpView.as_view(), name="op-merge"),
    path("redact/", ops_views.RedactOpView.as_view(), name="op-redact"),
    path("searchable/", ops_views.SearchableOpView.as_view(), name="op-searchable"),
    path("pdfa/", ops_views.PdfaOpView.as_view(), name="op-pdfa"),
    path("unprotect/", ops_views.UnprotectOpView.as_view(), name="op-unprotect"),
    path("protect/", ops_views.ProtectOpView.as_view(), name="op-protect"),
    path("rotate/", ops_views.RotateOpView.as_view(), name="op-rotate"),
    path("crop/", ops_views.CropOpView.as_view(), name="op-crop"),
    path("flatten/", ops_views.FlattenOpView.as_view(), name="op-flatten"),
    path("outline/", ops_views.OutlineOpView.as_view(), name="op-outline"),
    path("compare/", ops_views.CompareOpView.as_view(), name="op-compare"),
    path("metadata/", ops_views.MetadataOpView.as_view(), name="op-metadata"),
    path("to-images/", ops_views.ToImagesOpView.as_view(), name="op-to-images"),
    path("page-numbers/", ops_views.PageNumbersOpView.as_view(), name="op-page-numbers"),
    path("watermark/", ops_views.WatermarkOpView.as_view(), name="op-watermark"),
    path("convert-docx/", ops_views.ConvertDocxOpView.as_view(), name="op-convert-docx"),
    path("chat/", ops_views.ChatOpView.as_view(), name="op-chat"),
]

urlpatterns = [
    path("", ops_views.ApiRootView.as_view(), name="root"),
    path("", include(router.urls)),
    path("ops/", include(op_patterns)),
    path("schema/", _cached_schema, name="schema"),
    path("docs/", SpectacularSwaggerView.as_view(url_name="api:schema"), name="docs"),
    path("redoc/", SpectacularRedocView.as_view(url_name="api:schema"), name="redoc"),
]
