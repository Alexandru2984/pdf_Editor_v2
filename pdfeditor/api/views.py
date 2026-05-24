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
from drf_spectacular.utils import OpenApiParameter, OpenApiResponse, OpenApiTypes, extend_schema
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
    # Default category for actions not listed in the map below.
    throttle_scope_category = "read"
    # Per-action override; create (upload) is the heaviest path, destroy
    # rewrites filesystem state so it costs more than a plain read.
    _action_throttle_categories = {"create": "upload", "destroy": "op"}

    def get_throttles(self):
        self.throttle_scope_category = self._action_throttle_categories.get(self.action, "read")
        return super().get_throttles()

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

        # Strip active content (JS, auto-actions) + rewrite structure. Best-effort.
        from ..pdf_processor import sanitize_pdf

        sanitize_pdf(file_path)

        obj = UploadedPDF.objects.create(
            user=request.user,
            session_key="",
            name=uploaded.name,
            path=file_path,
            size=os.path.getsize(file_path),
        )
        serializer = self.get_serializer(obj)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def destroy(self, request, *args, **kwargs):
        # post_delete signal in pdfeditor/signals.py cleans the file + thumbnail.
        self.get_object().delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


@extend_schema(tags=["Outputs"])
class ProcessedPDFViewSet(viewsets.ReadOnlyModelViewSet):
    """List, retrieve, and download processed PDFs produced for the current user."""

    queryset = ProcessedPDF.objects.all()
    serializer_class = ProcessedPDFSerializer
    lookup_field = "id"
    throttle_scope_category = "read"

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
    """Read async job state (queued/running/done/failed) + result link.

    The list endpoint supports ``?status=`` and ``?kind=`` query params
    for narrowing — e.g. ``?status=queued&status=running`` to see what's
    in flight, or ``?kind=ocr_layer`` to scope to one op type.
    """

    queryset = Job.objects.all()
    serializer_class = JobSerializer
    lookup_field = "id"
    throttle_scope_category = "read"
    # Posting to /jobs/<id>/cancel/ touches more state than a plain list,
    # so route it into the *_op bucket. Other actions stay "read".
    _action_throttle_categories = {"cancel": "op"}

    def get_throttles(self):
        self.throttle_scope_category = self._action_throttle_categories.get(self.action, "read")
        return super().get_throttles()

    def get_queryset(self):
        qs = Job.objects.filter(user=self.request.user)
        statuses = (
            self.request.query_params.getlist("status") if hasattr(self.request, "query_params") else []
        )
        if statuses:
            valid = {s for s, _ in Job.STATUS_CHOICES}
            statuses = [s for s in statuses if s in valid]
            if statuses:
                qs = qs.filter(status__in=statuses)
        kind = self.request.query_params.get("kind") if hasattr(self.request, "query_params") else None
        if kind:
            qs = qs.filter(kind=kind)
        return qs

    @extend_schema(
        summary="List jobs",
        parameters=[
            OpenApiParameter(
                "status",
                OpenApiTypes.STR,
                description="Filter by status. Repeat for OR (e.g. ?status=queued&status=running).",
                many=True,
                enum=[s for s, _ in Job.STATUS_CHOICES],
            ),
            OpenApiParameter(
                "kind",
                OpenApiTypes.STR,
                description="Filter by job kind (e.g. ocr_layer, pdfa, compare).",
            ),
        ],
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    @extend_schema(
        summary="Cancel a running job",
        description=(
            "Revoke the underlying Celery task (SIGTERM if already running) "
            "and mark the job failed with `error_message='Cancelled by user'`. "
            "Idempotent — calling on an already-terminal job returns 409."
        ),
        request=None,
        responses={
            200: JobSerializer,
            409: OpenApiResponse(description="Job already finished — nothing to cancel"),
        },
    )
    @action(detail=True, methods=["post"], url_path="cancel")
    def cancel(self, request, id=None):
        from ..tasks import cancel_job

        job = self.get_object()
        if not cancel_job(job):
            return Response(
                {"detail": f"Job is already in terminal state '{job.status}'."},
                status=status.HTTP_409_CONFLICT,
            )
        # Refresh from DB to pick up the updated finished_at/status.
        job.refresh_from_db()
        return Response(self.get_serializer(job).data)
