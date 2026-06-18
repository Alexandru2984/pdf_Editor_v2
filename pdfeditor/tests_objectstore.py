"""Tests for the Cloudflare R2 output mirror (pdfeditor/objectstore.py).

boto3 is never contacted: the client factory is patched. What's pinned here:
the enabled() gate, the key scheme, the Content-Disposition sanitisation,
the mirror task, the presigned-redirect download paths and their local-disk
fallback, and R2 object deletion on row delete.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

from django.test import TestCase, override_settings

from . import objectstore
from .models import ProcessedPDF, ShareLink
from .tasks import mirror_output_to_r2

_R2_SETTINGS = {
    "R2_ENABLED": True,
    "R2_ENDPOINT_URL": "https://acc.r2.cloudflarestorage.com",
    "R2_BUCKET": "pdf-editor",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
}


def _make_output(session_key: str, *, name: str = "out.pdf", r2_key: str = "") -> ProcessedPDF:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(b"%PDF-1.4 test")
    return ProcessedPDF.objects.create(
        user=None,
        session_key=session_key,
        kind=ProcessedPDF.KIND_COMPRESS,
        name=name,
        path=path,
        size=13,
        r2_key=r2_key,
    )


class EnabledGateTests(TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(objectstore.enabled())

    @override_settings(**_R2_SETTINGS)
    def test_enabled_when_fully_configured(self):
        self.assertTrue(objectstore.enabled())

    @override_settings(**{**_R2_SETTINGS, "R2_ACCESS_KEY_ID": ""})
    def test_partial_config_stays_disabled(self):
        self.assertFalse(objectstore.enabled())

    def test_disabled_mirror_and_presign_are_noops(self):
        output = _make_output("s1", r2_key="processed/x/out.pdf")
        self.assertIsNone(objectstore.mirror_processed(output))
        self.assertIsNone(objectstore.presigned_download_url(output))


@override_settings(**_R2_SETTINGS)
class MirrorTests(TestCase):
    def test_mirror_task_uploads_and_stores_key(self):
        output = _make_output("s1")
        client = MagicMock()
        with patch.object(objectstore, "_client", return_value=client):
            mirror_output_to_r2(str(output.id))
        client.upload_file.assert_called_once()
        args = client.upload_file.call_args[0]
        self.assertEqual(args[1], "pdf-editor")
        self.assertEqual(args[2], f"processed/{output.id}/out.pdf")
        output.refresh_from_db()
        self.assertEqual(output.r2_key, f"processed/{output.id}/out.pdf")

    def test_mirror_uses_output_content_type(self):
        output = _make_output("s1", name="pages.zip")
        client = MagicMock()
        with patch.object(objectstore, "_client", return_value=client):
            mirror_output_to_r2(str(output.id))
        extra_args = client.upload_file.call_args[1]["ExtraArgs"]
        self.assertEqual(extra_args["ContentType"], "application/zip")

    def test_upload_failure_leaves_key_empty(self):
        output = _make_output("s1")
        client = MagicMock()
        client.upload_file.side_effect = RuntimeError("boom")
        with patch.object(objectstore, "_client", return_value=client):
            mirror_output_to_r2(str(output.id))
        output.refresh_from_db()
        self.assertEqual(output.r2_key, "")

    def test_presign_sanitises_content_disposition(self):
        output = _make_output("s1", name='evil";drop.pdf', r2_key="processed/x/evil.pdf")
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://r2.example/signed"
        with patch.object(objectstore, "_client", return_value=client):
            url = objectstore.presigned_download_url(output)
        self.assertEqual(url, "https://r2.example/signed")
        disposition = client.generate_presigned_url.call_args[1]["Params"]["ResponseContentDisposition"]
        self.assertNotIn('";', disposition.replace('filename="', ""))
        self.assertNotIn("\\", disposition)

    def test_presign_uses_output_content_type(self):
        output = _make_output("s1", name="pages.zip", r2_key="processed/x/pages.zip")
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://r2.example/signed"
        with patch.object(objectstore, "_client", return_value=client):
            objectstore.presigned_download_url(output)
        params = client.generate_presigned_url.call_args[1]["Params"]
        self.assertEqual(params["ResponseContentType"], "application/zip")


@override_settings(**_R2_SETTINGS)
class DownloadPathTests(TestCase):
    def _share_link(self, output: ProcessedPDF) -> ShareLink:
        from datetime import timedelta

        from django.utils import timezone

        return ShareLink.objects.create(
            processed_pdf=output,
            creator=None,
            session_key="other-session",
            expires_at=timezone.now() + timedelta(hours=1),
            max_downloads=0,
        )

    def test_share_download_redirects_to_presigned_url(self):
        output = _make_output("s1", r2_key="processed/x/out.pdf")
        link = self._share_link(output)
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://r2.example/signed"
        with patch.object(objectstore, "_client", return_value=client):
            resp = self.client.get(f"/s/{link.token}/")
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp["Location"], "https://r2.example/signed")

    def test_share_download_falls_back_to_disk_without_mirror(self):
        output = _make_output("s1")  # no r2_key
        link = self._share_link(output)
        resp = self.client.get(f"/s/{link.token}/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("attachment", resp["Content-Disposition"])

    def test_mirrored_output_survives_local_cleanup(self):
        output = _make_output("s1", r2_key="processed/x/out.pdf")
        os.remove(output.path)  # simulate retention sweep of the media volume
        link = self._share_link(output)
        client = MagicMock()
        client.generate_presigned_url.return_value = "https://r2.example/signed"
        with patch.object(objectstore, "_client", return_value=client):
            resp = self.client.get(f"/s/{link.token}/")
        self.assertEqual(resp.status_code, 302)


@override_settings(**_R2_SETTINGS)
class DeleteMirrorTests(TestCase):
    def test_row_delete_removes_r2_object(self):
        output = _make_output("s1", r2_key="processed/x/out.pdf")
        client = MagicMock()
        with patch.object(objectstore, "_client", return_value=client):
            output.delete()
        client.delete_object.assert_called_once_with(Bucket="pdf-editor", Key="processed/x/out.pdf")

    def test_row_delete_without_mirror_skips_r2(self):
        output = _make_output("s1")
        client = MagicMock()
        with patch.object(objectstore, "_client", return_value=client):
            output.delete()
        client.delete_object.assert_not_called()
