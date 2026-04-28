"""Profile view — account info + activity counters for the logged-in user."""

from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from ..models import ProcessedPDF, UploadedPDF


@login_required
def profile_view(request: HttpRequest) -> HttpResponse:
    user = request.user
    uploaded_count = UploadedPDF.objects.filter(user=user).count()
    processed_count = ProcessedPDF.objects.filter(user=user).count()

    return render(
        request,
        "pdfeditor/profile.html",
        {
            "uploaded_count": uploaded_count,
            "processed_count": processed_count,
        },
    )
