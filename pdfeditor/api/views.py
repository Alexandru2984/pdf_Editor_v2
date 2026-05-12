"""REST API views for PDF management.

These viewsets mirror the web flows but accept JSON / multipart input and
return JSON. They are owner-scoped — a key authenticates one user, and
that user only sees their own rows.
"""

from __future__ import annotations

import logging
import os

import fitz
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, Http404
from drf_spectacular.utils import OpenApiResponse, OpenApiTypes, extend_schema
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import ValidationError
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response

from ..models import Job, ProcessedPDF, UploadedPDF
from .serializers import JobSerializer, ProcessedPDFSerializer, UploadedPDFSerializer

logger = logging.getLogger(__name__)


def _count_pages_safely(path: str) -> int | None:
    try:
        with fitz.open(path) as doc:
            return len(doc)
    except Exception as exc:
        logger.warning("API upload — could not parse PDF %s: %s", path, exc)
        return None


@extend_schema(tags=["PDFs"])
class UploadedPDFViewSet(viewsets.ReadOnlyModelViewSet):
    """List, retrieve, and delete source PDFs uploaded by the current user."""

    # Class-level queryset enables drf-spectacular's schema introspection
    # (it can't call get_queryset() without a request). Actual ownership
    # scoping lives in get_queryset() and applies at runtime.
    queryset = UploadedPDF.objects.all()
    serializer_class = UploadedPDFSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]
    lookup_field = "id"

    def get_queryset(self):
        return UploadedPDF.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Upload a PDF",
        description=(
            "Upload a PDF file as `multipart/form-data` under the `pdf_file` "
            "key. Returns the stored PDF's metadata. Subject to the same "
            "size, page-count and storage-quota limits as the web UI."
        ),
        request={
            "multipart/form-data": {
                "type": "object",
                "properties": {"pdf_file": {"type": "string", "format": "binary"}},
            }
        },
        responses={201: UploadedPDFSerializer, 400: OpenApiTypes.OBJECT},
    )
    def create(self, request, *args, **kwargs):
        uploaded = request.FILES.get("pdf_file")
        if uploaded is None:
            raise ValidationError({"pdf_file": "This field is required."})

        max_bytes = getattr(settings, "PDF_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
        if uploaded.size > max_bytes:
            raise ValidationError({"pdf_file": f"File exceeds {max_bytes // (1024 * 1024)} MB limit."})

        # Storage quota.
        from ..views._common import storage_quota, storage_usage

        quota = storage_quota(request)
        if quota and storage_usage(request) + uploaded.size > quota:
            raise ValidationError({"pdf_file": "Storage quota exceeded."})

        header = uploaded.read(5)
        uploaded.seek(0)
        if header != b"%PDF-":
            raise ValidationError({"pdf_file": "Not a valid PDF file."})

        fs = FileSystemStorage(location=os.path.join(settings.MEDIA_ROOT, "uploads"))
        safe_name = os.path.basename(uploaded.name)
        filename = fs.save(safe_name, uploaded)
        file_path = fs.path(filename)

        max_pages = getattr(settings, "PDF_MAX_PAGES", 500)
        page_count = _count_pages_safely(file_path)
        if page_count is None:
            os.remove(file_path)
            raise ValidationError({"pdf_file": "PDF could not be parsed."})
        if page_count > max_pages:
            os.remove(file_path)
            raise ValidationError({"pdf_file": f"Document has {page_count} pages (max {max_pages})."})

        obj = UploadedPDF.objects.create(
            user=request.user,
            session_key="",
            name=uploaded.name,
            path=file_path,
            size=uploaded.size,
        )
        serializer = self.get_serializer(obj)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        obj = self.get_object()
        try:
            if obj.path and os.path.exists(obj.path):
                os.remove(obj.path)
        except OSError as exc:
            logger.warning("API delete — could not remove %s: %s", obj.path, exc)
        obj.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=["Outputs"])
class ProcessedPDFViewSet(viewsets.ReadOnlyModelViewSet):
    """List, retrieve, and download processed PDFs produced for the current user."""

    queryset = ProcessedPDF.objects.all()
    serializer_class = ProcessedPDFSerializer
    lookup_field = "id"

    def get_queryset(self):
        return ProcessedPDF.objects.filter(user=self.request.user)

    @extend_schema(
        summary="Download the processed PDF",
        responses={
            200: OpenApiResponse(description="Binary PDF stream"),
            404: OpenApiResponse(description="File missing"),
        },
    )
    @action(detail=True, methods=["get"], url_path="download")
    def download(self, request, id=None):
        obj = self.get_object()
        if not obj.path or not os.path.exists(obj.path):
            raise Http404("File missing")
        return FileResponse(
            open(obj.path, "rb"),
            as_attachment=True,
            filename=os.path.basename(obj.path),
        )


@extend_schema(tags=["Operations"])
class JobViewSet(viewsets.ReadOnlyModelViewSet):
    """Read async job state (queued/running/done/failed) + result link."""

    queryset = Job.objects.all()
    serializer_class = JobSerializer
    lookup_field = "id"

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user)
