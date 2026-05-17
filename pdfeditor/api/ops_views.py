"""REST API endpoints for PDF processing operations.

Each operation takes a JSON body referencing a previously-uploaded
``pdf_id`` plus operation-specific parameters, runs the corresponding op
synchronously, and returns the resulting ``ProcessedPDF`` row. Async
execution will be wired in a future iteration via Celery — the contract
will not change.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

from drf_spectacular.utils import OpenApiResponse, extend_schema, inline_serializer
from rest_framework import serializers, status
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from ..models import Job, ProcessedPDF, UploadedPDF
from ..pdf_processor import (
    add_page_numbers,
    add_watermark,
    compress_pdf,
    convert_pdf_to_images,
    crop_pages,
    edit_pdf_metadata,
    flatten_pdf,
    merge_pdfs,
    pdf_page_count,
    protect_pdf,
    redact_text,
    remove_pdf_password,
    rotate_pages,
    set_pdf_outline,
    split_pdf,
    summarize_pdf,
)
from ..tasks import enqueue_job
from ..views._common import _client_ip
from .serializers import ProcessedPDFSerializer


def _user_pdf(user, pdf_id) -> UploadedPDF:
    pdf = UploadedPDF.objects.filter(user=user, id=pdf_id).first()
    if pdf is None:
        raise NotFound("PDF not found.")
    if not pdf.exists_on_disk():
        raise NotFound("PDF file is missing on disk.")
    return pdf


def _record(request, kind: str, path: str, source: UploadedPDF | None) -> ProcessedPDF:
    """Lightweight wrapper around the web-flow record_output. Logs audit too."""
    from ..models import AuditLog

    output = ProcessedPDF.objects.create(
        user=request.user,
        session_key="",
        kind=kind,
        source=source,
        name=os.path.basename(path),
        path=path,
        size=os.path.getsize(path) if os.path.exists(path) else 0,
    )
    with contextlib.suppress(Exception):
        AuditLog.objects.create(
            user=request.user,
            session_key="",
            kind=kind,
            source_name=getattr(source, "name", "") or "",
            output_name=output.name,
            output_size=output.size,
            ip_address=_client_ip(request),
            user_agent=request.META.get("HTTP_USER_AGENT", "")[:300],
        )
    return output


def _queue_async_api_job(
    user, kind: str, source: UploadedPDF, second_source=None, params: dict | None = None
) -> Job:
    job = Job.objects.create(
        user=user,
        session_key="",
        kind=kind,
        source=source,
        second_source=second_source,
        params=params or {},
    )
    enqueue_job(job)
    return job


def _job_response(job: Job, request) -> Response:
    """Standard 202 response for async ops — include URLs to poll status."""
    from django.urls import reverse

    payload = {
        "job_id": str(job.id),
        "kind": job.kind,
        "status": job.status,
        "status_url": request.build_absolute_uri(reverse("api:job-detail", args=[job.id])),
    }
    return Response(payload, status=status.HTTP_202_ACCEPTED)


class _BaseOpView(APIView):
    """Shared boilerplate for op endpoints."""

    permission_classes = [IsAuthenticated]
    # Picked up by ScopedAuthAwareThrottle to route into the *_op rate
    # buckets (api_key_op / user_op / anon_op). Top-level APIView classes
    # below that don't inherit this set the attribute themselves.
    throttle_scope_category = "op"
    kind: str
    op_callable: Callable
    request_serializer: type[serializers.Serializer]

    def post(self, request):
        in_ser = self.request_serializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        try:
            out_path = self.run(pdf.path, in_ser.validated_data)
        except ValueError as exc:
            raise ValidationError({"detail": str(exc)}) from exc
        if isinstance(out_path, tuple):
            out_path = out_path[0]
        output = _record(request, self.kind, out_path, source=pdf)
        return Response(
            ProcessedPDFSerializer(output, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

    def run(self, pdf_path: str, params: dict):
        raise NotImplementedError


# ---------- Request body schemas ----------


class _PdfIdSerializer(serializers.Serializer):
    pdf_id = serializers.UUIDField()


class _CompressSerializer(_PdfIdSerializer):
    quality = serializers.ChoiceField(choices=["low", "medium", "high"], default="medium")


class _SplitSerializer(_PdfIdSerializer):
    ranges = serializers.CharField(
        help_text='Range list "1-3,5-6". Each range is start-end, 1-indexed inclusive.'
    )


class _MergeSerializer(serializers.Serializer):
    pdf_ids = serializers.ListField(child=serializers.UUIDField(), min_length=2)


class _RedactSerializer(_PdfIdSerializer):
    search_terms = serializers.ListField(child=serializers.CharField(), min_length=1)
    page_range = serializers.CharField(required=False, allow_blank=True, default="")


class _OcrSerializer(_PdfIdSerializer):
    language = serializers.CharField(default="eng+ron")
    dpi = serializers.IntegerField(min_value=72, max_value=600, default=200)


class _PdfaSerializer(_PdfIdSerializer):
    version = serializers.ChoiceField(choices=["1b", "2b"], default="2b")


class _UnprotectSerializer(_PdfIdSerializer):
    password = serializers.CharField()


class _ProtectSerializer(_PdfIdSerializer):
    user_password = serializers.CharField()
    owner_password = serializers.CharField(required=False, allow_blank=True, default="")


class _RotateSerializer(_PdfIdSerializer):
    rotation_angle = serializers.ChoiceField(choices=[90, 180, 270])
    page_range = serializers.CharField(required=False, allow_blank=True, default="")


class _CropSerializer(_PdfIdSerializer):
    top = serializers.FloatField(min_value=0.0, max_value=49.0, default=0.0)
    right = serializers.FloatField(min_value=0.0, max_value=49.0, default=0.0)
    bottom = serializers.FloatField(min_value=0.0, max_value=49.0, default=0.0)
    left = serializers.FloatField(min_value=0.0, max_value=49.0, default=0.0)
    page_range = serializers.CharField(required=False, allow_blank=True, default="")


class _FlattenSerializer(_PdfIdSerializer):
    flatten_annotations = serializers.BooleanField(default=True)
    flatten_forms = serializers.BooleanField(default=True)


class _OutlineSerializer(_PdfIdSerializer):
    entries = serializers.ListField(child=serializers.DictField())


class _CompareSerializer(_PdfIdSerializer):
    second_pdf_id = serializers.UUIDField()


class _MetadataSerializer(_PdfIdSerializer):
    title = serializers.CharField(required=False, allow_blank=True, default="")
    author = serializers.CharField(required=False, allow_blank=True, default="")
    subject = serializers.CharField(required=False, allow_blank=True, default="")
    keywords = serializers.CharField(required=False, allow_blank=True, default="")
    creator = serializers.CharField(required=False, allow_blank=True, default="")
    producer = serializers.CharField(required=False, allow_blank=True, default="")
    clear_dates = serializers.BooleanField(default=False)


class _ToImagesSerializer(_PdfIdSerializer):
    fmt = serializers.ChoiceField(choices=["png", "jpg"], default="png")
    dpi = serializers.IntegerField(min_value=36, max_value=600, default=150)


class _PageNumbersSerializer(_PdfIdSerializer):
    position = serializers.CharField(default="bottom-right")
    start_at = serializers.IntegerField(min_value=1, default=1)
    page_range = serializers.CharField(required=False, allow_blank=True, default="")


class _WatermarkSerializer(_PdfIdSerializer):
    text = serializers.CharField()
    position = serializers.CharField(default="center")
    opacity = serializers.FloatField(min_value=0.05, max_value=1.0, default=0.3)


class _BatchSerializer(serializers.Serializer):
    """Apply one op to a list of PDFs. Params are passed through to the op
    handler unchanged — per-PDF validation lives in the handler."""

    op = serializers.ChoiceField(choices=[])  # populated in __init__
    pdf_ids = serializers.ListField(
        child=serializers.UUIDField(),
        min_length=1,
        help_text="UUIDs of UploadedPDFs to process.",
    )
    params = serializers.DictField(
        required=False,
        default=dict,
        help_text="Op-specific parameters, applied identically to each PDF.",
    )

    def __init__(self, *args, **kwargs):
        from .batch_ops import BATCH_OPS, MAX_BATCH_SIZE

        super().__init__(*args, **kwargs)
        self.fields["op"].choices = sorted(BATCH_OPS.keys())
        self._max_size = MAX_BATCH_SIZE

    def validate_pdf_ids(self, value):
        if len(value) > self._max_size:
            raise serializers.ValidationError(
                f"At most {self._max_size} PDFs per batch."
            )
        if len(set(value)) != len(value):
            raise serializers.ValidationError("Duplicate pdf_ids in batch.")
        return value


class _ConvertSerializer(_PdfIdSerializer):
    pass


# ---------- Endpoints ----------


@extend_schema(tags=["Operations"], request=_CompressSerializer, responses={201: ProcessedPDFSerializer})
class CompressOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_COMPRESS
    request_serializer = _CompressSerializer

    def run(self, pdf_path, params):
        out, *_ = compress_pdf(pdf_path, quality=params["quality"])
        return out


@extend_schema(
    tags=["Operations"], request=_SplitSerializer, responses={201: ProcessedPDFSerializer(many=True)}
)
class SplitOpView(_BaseOpView):
    """Split returns multiple outputs — overrides the base to return a list."""

    permission_classes = [IsAuthenticated]
    request_serializer = _SplitSerializer

    def post(self, request):
        in_ser = self.request_serializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])

        try:
            ranges_str = in_ser.validated_data["ranges"]
            ranges: list[tuple[int, int]] = []
            for piece in ranges_str.split(","):
                piece = piece.strip()
                if "-" in piece:
                    a, b = piece.split("-", 1)
                    ranges.append((int(a), int(b)))
                else:
                    n = int(piece)
                    ranges.append((n, n))
            out_paths = split_pdf(pdf.path, ranges)
        except (ValueError, IndexError) as exc:
            raise ValidationError({"detail": str(exc)}) from exc

        outputs = [_record(request, ProcessedPDF.KIND_SPLIT, p, source=pdf) for p in out_paths]
        return Response(
            ProcessedPDFSerializer(outputs, many=True, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


@extend_schema(tags=["Operations"], request=_MergeSerializer, responses={201: ProcessedPDFSerializer})
class MergeOpView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"
    request_serializer = _MergeSerializer

    def post(self, request):
        in_ser = self.request_serializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdfs = [_user_pdf(request.user, pid) for pid in in_ser.validated_data["pdf_ids"]]
        try:
            out = merge_pdfs([p.path for p in pdfs])
        except ValueError as exc:
            raise ValidationError({"detail": str(exc)}) from exc
        output = _record(request, ProcessedPDF.KIND_MERGE, out, source=pdfs[0])
        return Response(
            ProcessedPDFSerializer(output, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


@extend_schema(tags=["Operations"], request=_RedactSerializer, responses={201: ProcessedPDFSerializer})
class RedactOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_REDACT
    request_serializer = _RedactSerializer

    def run(self, pdf_path, params):
        out, _ = redact_text(pdf_path, params["search_terms"], page_range=params.get("page_range") or None)
        return out


@extend_schema(tags=["Operations"], request=_OcrSerializer, responses={202: None})
class SearchableOpView(APIView):
    """Queue an OCR job. Returns 202 + job_id; poll /api/v1/jobs/<id>/ for status."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _OcrSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        job = _queue_async_api_job(
            request.user,
            ProcessedPDF.KIND_OCR_LAYER,
            pdf,
            params={"language": in_ser.validated_data["language"], "dpi": in_ser.validated_data["dpi"]},
        )
        return _job_response(job, request)


@extend_schema(tags=["Operations"], request=_PdfaSerializer, responses={202: None})
class PdfaOpView(APIView):
    """Queue a PDF/A conversion. Returns 202 + job_id."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _PdfaSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        job = _queue_async_api_job(
            request.user,
            ProcessedPDF.KIND_PDFA,
            pdf,
            params={"version": in_ser.validated_data["version"]},
        )
        return _job_response(job, request)


@extend_schema(tags=["Operations"], request=_UnprotectSerializer, responses={201: ProcessedPDFSerializer})
class UnprotectOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_UNPROTECT
    request_serializer = _UnprotectSerializer

    def run(self, pdf_path, params):
        return remove_pdf_password(pdf_path, params["password"])


@extend_schema(tags=["Operations"], request=_ProtectSerializer, responses={201: ProcessedPDFSerializer})
class ProtectOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_PROTECT
    request_serializer = _ProtectSerializer

    def run(self, pdf_path, params):
        return protect_pdf(
            pdf_path,
            user_password=params["user_password"],
            owner_password=params.get("owner_password") or None,
        )


@extend_schema(tags=["Operations"], request=_RotateSerializer, responses={201: ProcessedPDFSerializer})
class RotateOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_ROTATE
    request_serializer = _RotateSerializer

    def run(self, pdf_path, params):
        return rotate_pages(
            pdf_path,
            rotation_angle=params["rotation_angle"],
            page_range=params.get("page_range") or None,
        )


@extend_schema(tags=["Operations"], request=_CropSerializer, responses={201: ProcessedPDFSerializer})
class CropOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_CROP
    request_serializer = _CropSerializer

    def run(self, pdf_path, params):
        return crop_pages(
            pdf_path,
            top=params["top"],
            right=params["right"],
            bottom=params["bottom"],
            left=params["left"],
            page_range=params.get("page_range") or None,
        )


@extend_schema(tags=["Operations"], request=_FlattenSerializer, responses={201: ProcessedPDFSerializer})
class FlattenOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_FLATTEN
    request_serializer = _FlattenSerializer

    def run(self, pdf_path, params):
        return flatten_pdf(
            pdf_path,
            flatten_annotations=params["flatten_annotations"],
            flatten_forms=params["flatten_forms"],
        )


@extend_schema(tags=["Operations"], request=_OutlineSerializer, responses={201: ProcessedPDFSerializer})
class OutlineOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_OUTLINE
    request_serializer = _OutlineSerializer

    def run(self, pdf_path, params):
        return set_pdf_outline(pdf_path, params["entries"])


@extend_schema(tags=["Operations"], request=_CompareSerializer, responses={202: None})
class CompareOpView(APIView):
    """Queue a comparison job. Returns 202 + job_id."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _CompareSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        a = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        b = _user_pdf(request.user, in_ser.validated_data["second_pdf_id"])
        if a.id == b.id:
            raise ValidationError({"detail": "Choose two different PDFs."})
        job = _queue_async_api_job(
            request.user, ProcessedPDF.KIND_COMPARE, a, second_source=b, params={"second_name": b.name}
        )
        return _job_response(job, request)


@extend_schema(tags=["Operations"], request=_MetadataSerializer, responses={201: ProcessedPDFSerializer})
class MetadataOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_METADATA
    request_serializer = _MetadataSerializer

    def run(self, pdf_path, params):
        metadata = {
            k: params.get(k, "") for k in ("title", "author", "subject", "keywords", "creator", "producer")
        }
        return edit_pdf_metadata(pdf_path, metadata, clear_dates=params["clear_dates"])


@extend_schema(
    tags=["Operations"],
    request=_ToImagesSerializer,
    responses={
        201: ProcessedPDFSerializer,
        202: OpenApiResponse(description="Large PDF — queued async, poll status_url"),
    },
)
class ToImagesOpView(APIView):
    """Rasterize PDF pages to PNG/JPG.

    Sync for small PDFs (< ASYNC_THRESHOLD_PAGES); above the threshold we
    queue the work to Celery and return 202 + job_id. Sync monopolizes a
    gunicorn worker for several seconds per page at typical DPI — the
    threshold keeps small uploads instant while heavy ones don't block
    the request pool.
    """

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        from django.conf import settings as dj_settings

        in_ser = _ToImagesSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])

        threshold = getattr(dj_settings, "ASYNC_THRESHOLD_PAGES", 5)
        page_count = pdf_page_count(pdf.path) or 0
        if page_count >= threshold:
            params = {"fmt": in_ser.validated_data["fmt"], "dpi": in_ser.validated_data["dpi"]}
            job = _queue_async_api_job(request.user, ProcessedPDF.KIND_TO_IMAGES, pdf, params=params)
            return _job_response(job, request)

        try:
            out, _ = convert_pdf_to_images(
                pdf.path, fmt=in_ser.validated_data["fmt"], dpi=in_ser.validated_data["dpi"]
            )
        except ValueError as exc:
            raise ValidationError({"detail": str(exc)}) from exc
        output = _record(request, ProcessedPDF.KIND_TO_IMAGES, out, source=pdf)
        return Response(
            ProcessedPDFSerializer(output, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )


@extend_schema(tags=["Operations"], request=_PageNumbersSerializer, responses={201: ProcessedPDFSerializer})
class PageNumbersOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_PAGE_NUMBERS
    request_serializer = _PageNumbersSerializer

    def run(self, pdf_path, params):
        return add_page_numbers(
            pdf_path,
            options={
                "position": params["position"],
                "start_at": params["start_at"],
                "page_range": params.get("page_range") or None,
            },
        )


@extend_schema(tags=["Operations"], request=_WatermarkSerializer, responses={201: ProcessedPDFSerializer})
class WatermarkOpView(_BaseOpView):
    kind = ProcessedPDF.KIND_WATERMARK
    request_serializer = _WatermarkSerializer

    def run(self, pdf_path, params):
        return add_watermark(
            pdf_path,
            watermark_type="text",
            watermark_content=params["text"],
            options={"position": params["position"], "opacity": params["opacity"]},
        )


class _ChatSerializer(serializers.Serializer):
    pdf_id = serializers.UUIDField()
    message = serializers.CharField(max_length=2000)


@extend_schema(
    tags=["Operations"],
    request=_ChatSerializer,
    responses={
        200: OpenApiResponse(description="Answer + per-excerpt citations"),
        409: OpenApiResponse(description="PDF not indexed yet"),
        502: OpenApiResponse(description="LLM upstream error"),
    },
)
class ChatOpView(APIView):
    """RAG chat over an indexed PDF. POST {'pdf_id', 'message'}.

    Same retrieval + Groq flow as the web view. If the PDF has no
    embeddings, returns 409 — clients should kick off indexing via the
    chat web flow or by enqueueing a chat_index Job manually."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _ChatSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])

        from ..models import Embedding
        from ..pdf_processor.rag import embed_query
        from ..views.chat import _build_rag_prompt, _call_groq, _retrieve

        if not Embedding.objects.filter(uploaded_pdf=pdf).exists():
            return Response(
                {"detail": "PDF not indexed for chat yet."},
                status=status.HTTP_409_CONFLICT,
            )

        qvec = embed_query(in_ser.validated_data["message"])
        chunks = _retrieve(pdf, qvec)
        answer, error = _call_groq(_build_rag_prompt(in_ser.validated_data["message"], chunks))
        if error:
            return Response({"detail": error}, status=status.HTTP_502_BAD_GATEWAY)

        return Response(
            {
                "answer": answer,
                "citations": [
                    {
                        "index": i + 1,
                        "page": c.page_number,
                        "excerpt": c.chunk_text[:300] + ("…" if len(c.chunk_text) > 300 else ""),
                    }
                    for i, c in enumerate(chunks)
                ],
            }
        )


@extend_schema(tags=["Operations"], request=_ConvertSerializer, responses={202: None})
class ConvertDocxOpView(APIView):
    """Queue PDF→DOCX conversion. Returns 202 + job_id."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _ConvertSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        job = _queue_async_api_job(request.user, ProcessedPDF.KIND_CONVERT, pdf)
        return _job_response(job, request)


class _SummarizeSerializer(_PdfIdSerializer):
    language = serializers.CharField(default="English", help_text="Output language (e.g. English, Romanian).")


@extend_schema(
    tags=["Operations"],
    request=_SummarizeSerializer,
    responses={
        200: inline_serializer(
            name="SummaryResponse",
            fields={
                "summary": serializers.CharField(),
                "truncated": serializers.BooleanField(),
                "chars_used": serializers.IntegerField(),
                "language": serializers.CharField(),
            },
        ),
        409: OpenApiResponse(description="No extractable text — run OCR first"),
        502: OpenApiResponse(description="Upstream LLM error"),
    },
)
class SummarizeOpView(APIView):
    """Single-shot document summary via Groq.

    Sync — the LLM call is bounded (~30s) and there's no per-PDF state
    to maintain, so we don't need a Job. Output is not persisted;
    re-call to get a fresh summary."""

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        in_ser = _SummarizeSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        pdf = _user_pdf(request.user, in_ser.validated_data["pdf_id"])
        language = in_ser.validated_data["language"]
        try:
            result = summarize_pdf(pdf.path, language=language)
        except ValueError as exc:
            # "No extractable text" → 409 so clients can react (e.g. offer
            # to run OCR first). Other ValueErrors (LLM upstream) → 502.
            msg = str(exc)
            if "OCR" in msg:
                return Response({"detail": msg}, status=status.HTTP_409_CONFLICT)
            return Response({"detail": msg}, status=status.HTTP_502_BAD_GATEWAY)
        result["language"] = language
        return Response(result)


@extend_schema(tags=["Operations"], request=_BatchSerializer, responses={202: None})
class BatchOpView(APIView):
    """Apply one op to many PDFs in a single async job.

    The same op + params are applied to each PDF; results live in the
    Job's ``params['results']`` list (one entry per input). Progress
    advances after each PDF completes, so SSE/poll clients see live
    movement instead of one big jump at the end.
    """

    permission_classes = [IsAuthenticated]
    throttle_scope_category = "op"

    def post(self, request):
        from .batch_ops import BATCH_OPS

        in_ser = _BatchSerializer(data=request.data)
        in_ser.is_valid(raise_exception=True)
        op_name = in_ser.validated_data["op"]
        pdf_ids = in_ser.validated_data["pdf_ids"]
        op_params = in_ser.validated_data["params"]

        # Validate ownership for every requested PDF before we accept
        # the job — partial-batch rejection is friendlier than queueing
        # work that's going to fail per-row on the worker.
        owned = UploadedPDF.objects.filter(user=request.user, id__in=pdf_ids)
        owned_ids = {str(pid) for pid in owned.values_list("id", flat=True)}
        missing = [str(p) for p in pdf_ids if str(p) not in owned_ids]
        if missing:
            raise ValidationError({"pdf_ids": f"Not found: {missing}"})

        # Sanity: confirm the op is in the registry. (DRF's ChoiceField
        # already guards this; the extra check is for clarity + future
        # registry edits.)
        if op_name not in BATCH_OPS:
            raise ValidationError({"op": f"Unknown op '{op_name}'."})

        job = Job.objects.create(
            user=request.user,
            session_key="",
            kind=f"batch:{op_name}",
            params={
                "op": op_name,
                "pdf_ids": [str(p) for p in pdf_ids],
                "op_params": op_params,
            },
        )
        from ..tasks import run_batch_task

        result = run_batch_task.delay(str(job.id))
        if getattr(result, "id", None):
            Job.objects.filter(pk=job.pk).update(celery_task_id=result.id)
            job.celery_task_id = result.id

        return _job_response(job, request)


@extend_schema(
    tags=["PDFs"],
    responses={
        200: inline_serializer(
            name="OpenAPIRoot",
            fields={
                "version": serializers.CharField(),
                "docs_url": serializers.CharField(),
                "endpoints": serializers.DictField(),
            },
        ),
    },
)
class ApiRootView(APIView):
    """Discover endpoints + version, no auth required."""

    authentication_classes: list = []
    permission_classes: list = []  # public
    throttle_scope_category = "read"

    def get(self, request):
        from django.urls import reverse

        def url(name):
            return request.build_absolute_uri(reverse(name))

        return Response(
            {
                "version": "1.0.0",
                "docs_url": url("api:docs"),
                "schema_url": url("api:schema"),
                "redoc_url": url("api:redoc"),
                "endpoints": {
                    "pdfs": url("api:pdf-list"),
                    "outputs": url("api:output-list"),
                    "operations_root": "/api/v1/ops/",
                },
            }
        )
