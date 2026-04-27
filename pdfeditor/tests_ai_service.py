"""Tests for ai_service — OllamaProvider + GroqProvider with mocked HTTP."""

from unittest.mock import AsyncMock, MagicMock, patch

from django.test import TestCase, override_settings

from . import ai_service
from .ai_service import (
    REPHRASE_STYLES,
    GroqProvider,
    OllamaProvider,
    get_all_models,
    get_provider,
)


def _async_response(status_code=200, json_payload=None, text=""):
    """A MagicMock shaped like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload or {}
    resp.text = text
    return resp


class _FakeAsyncClient:
    """Minimal async-context-manager stub that mimics httpx.AsyncClient."""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.post = AsyncMock(side_effect=exc) if exc else AsyncMock(return_value=response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _mock_response(status_code=200, json_payload=None, text=""):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_payload or {}
    resp.text = text
    return resp


class GetProviderTests(TestCase):
    def test_groq_returns_groq_provider(self):
        self.assertIsInstance(get_provider("groq"), GroqProvider)

    def test_anything_else_returns_ollama(self):
        self.assertIsInstance(get_provider("ollama"), OllamaProvider)
        self.assertIsInstance(get_provider("unknown"), OllamaProvider)
        self.assertIsInstance(get_provider(""), OllamaProvider)


class OllamaGetModelsTests(TestCase):
    @patch("pdfeditor.ai_service.requests.get")
    def test_returns_model_names_on_success(self, mock_get):
        mock_get.return_value = _mock_response(
            json_payload={"models": [{"name": "llama3"}, {"name": "mistral"}]},
        )
        self.assertEqual(OllamaProvider().get_models(), ["llama3", "mistral"])

    @patch("pdfeditor.ai_service.requests.get")
    def test_returns_empty_on_non_200(self, mock_get):
        mock_get.return_value = _mock_response(status_code=500)
        self.assertEqual(OllamaProvider().get_models(), [])

    @patch("pdfeditor.ai_service.requests.get")
    def test_returns_empty_on_connection_error(self, mock_get):
        mock_get.side_effect = ConnectionError("nope")
        self.assertEqual(OllamaProvider().get_models(), [])

    @patch("pdfeditor.ai_service.requests.get")
    def test_returns_empty_when_models_key_missing(self, mock_get):
        mock_get.return_value = _mock_response(json_payload={})
        self.assertEqual(OllamaProvider().get_models(), [])


class OllamaRephraseTests(TestCase):
    @patch("pdfeditor.ai_service.requests.post")
    def test_happy_path_returns_text_and_success(self, mock_post):
        mock_post.return_value = _mock_response(
            json_payload={"response": "  rephrased version  "},
        )
        text, ok, err = OllamaProvider().rephrase(
            text="hello world",
            style="formal",
            model="llama3",
        )
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(text, "rephrased version")  # stripped

    @patch("pdfeditor.ai_service.requests.post")
    def test_uses_default_model_when_none_given(self, mock_post):
        mock_post.return_value = _mock_response(json_payload={"response": "out"})
        OllamaProvider().rephrase(text="hi", style="formal", model="")
        # The POST body should contain *some* model name (either the configured
        # default or the in-code "llama3" fallback).
        body = mock_post.call_args.kwargs["json"]
        self.assertTrue(body["model"])

    @patch("pdfeditor.ai_service.requests.post")
    def test_non_200_returns_failure(self, mock_post):
        mock_post.return_value = _mock_response(status_code=500, text="boom")
        text, ok, err = OllamaProvider().rephrase("x", "formal", "llama3")
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertIn("boom", err)

    @patch("pdfeditor.ai_service.requests.post")
    def test_connection_error_returns_failure(self, mock_post):
        mock_post.side_effect = TimeoutError("timed out")
        text, ok, err = OllamaProvider().rephrase("x", "formal", "llama3")
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertIn("Connection Error", err)

    @patch("pdfeditor.ai_service.requests.post")
    def test_custom_prompt_overrides_style(self, mock_post):
        mock_post.return_value = _mock_response(json_payload={"response": "out"})
        OllamaProvider().rephrase(
            text="hi",
            style="formal",
            model="llama3",
            custom_prompt="Translate to pig latin.",
        )
        body = mock_post.call_args.kwargs["json"]
        self.assertIn("pig latin", body["prompt"])
        # Shouldn't include the formal-style wording.
        self.assertNotIn(REPHRASE_STYLES["formal"]["prompt"], body["prompt"])


class GroqGetModelsTests(TestCase):
    @override_settings(GROQ_API_KEY="")
    def test_no_api_key_returns_empty(self):
        # Ensure neither settings nor env provides a key.
        with patch.dict("os.environ", {}, clear=False):
            import os as _os

            _os.environ.pop("GROQ_API_KEY", None)
            self.assertEqual(GroqProvider().get_models(), [])

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.get")
    def test_returns_model_ids_on_success(self, mock_get):
        mock_get.return_value = _mock_response(
            json_payload={"data": [{"id": "llama-3.1-70b"}, {"id": "mixtral-8x7b"}]},
        )
        self.assertEqual(
            GroqProvider().get_models(),
            ["llama-3.1-70b", "mixtral-8x7b"],
        )

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.get")
    def test_uses_authorization_header(self, mock_get):
        mock_get.return_value = _mock_response(json_payload={"data": []})
        GroqProvider().get_models()
        headers = mock_get.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.get")
    def test_non_200_returns_empty(self, mock_get):
        mock_get.return_value = _mock_response(status_code=401)
        self.assertEqual(GroqProvider().get_models(), [])

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.get")
    def test_exception_returns_empty(self, mock_get):
        mock_get.side_effect = ConnectionError("offline")
        self.assertEqual(GroqProvider().get_models(), [])


class GroqRephraseTests(TestCase):
    @override_settings(GROQ_API_KEY="")
    def test_no_api_key_returns_failure_with_message(self):
        import os as _os

        _os.environ.pop("GROQ_API_KEY", None)
        text, ok, err = GroqProvider().rephrase("hi", "formal", "llama-3.1-70b")
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertIn("API Key", err)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.post")
    def test_happy_path_extracts_choice_content(self, mock_post):
        mock_post.return_value = _mock_response(
            json_payload={"choices": [{"message": {"content": "  groq said hi  "}}]},
        )
        text, ok, err = GroqProvider().rephrase("hello", "casual", "llama-3.1-70b")
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertEqual(text, "groq said hi")

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.post")
    def test_request_payload_includes_model_and_messages(self, mock_post):
        mock_post.return_value = _mock_response(
            json_payload={"choices": [{"message": {"content": "x"}}]},
        )
        GroqProvider().rephrase("hello", "formal", "mixtral-8x7b")
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["model"], "mixtral-8x7b")
        self.assertEqual(len(body["messages"]), 2)
        roles = [m["role"] for m in body["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertIn("hello", body["messages"][1]["content"])

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.post")
    def test_non_200_returns_failure_with_text(self, mock_post):
        mock_post.return_value = _mock_response(status_code=429, text="rate limited")
        text, ok, err = GroqProvider().rephrase("hi", "formal", "llama-3.1-70b")
        self.assertFalse(ok)
        self.assertIn("rate limited", err)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.requests.post")
    def test_exception_returns_connection_error(self, mock_post):
        mock_post.side_effect = TimeoutError("timeout")
        text, ok, err = GroqProvider().rephrase("hi", "formal", "llama-3.1-70b")
        self.assertFalse(ok)
        self.assertIn("Connection Error", err)


class GetAllModelsTests(TestCase):
    @patch.object(ai_service.GroqProvider, "get_models", return_value=["g1", "g2"])
    @patch.object(ai_service.OllamaProvider, "get_models", return_value=["o1"])
    def test_groups_by_provider(self, _ollama_mock, _groq_mock):
        result = get_all_models()
        self.assertEqual(result, {"ollama": ["o1"], "groq": ["g1", "g2"]})


class OllamaArephraseTests(TestCase):
    """Async variant of OllamaProvider.rephrase, used by ASGI views."""

    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_happy_path(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(
            response=_async_response(json_payload={"response": "  async out  "}),
        )
        text, ok, err = await OllamaProvider().arephrase("hi", "formal", "llama3")
        self.assertTrue(ok)
        self.assertEqual(text, "async out")
        self.assertEqual(err, "")

    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_non_200_returns_failure(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(
            response=_async_response(status_code=500, text="boom"),
        )
        text, ok, err = await OllamaProvider().arephrase("hi", "formal", "llama3")
        self.assertFalse(ok)
        self.assertEqual(text, "")
        self.assertIn("boom", err)

    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_connection_exception_returns_failure(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(exc=TimeoutError("timed out"))
        text, ok, err = await OllamaProvider().arephrase("hi", "formal", "llama3")
        self.assertFalse(ok)
        self.assertIn("Connection Error", err)

    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_default_model_when_empty(self, mock_client_cls):
        client = _FakeAsyncClient(response=_async_response(json_payload={"response": "x"}))
        mock_client_cls.return_value = client
        await OllamaProvider().arephrase("hi", "formal", "")
        body = client.post.call_args.kwargs["json"]
        self.assertTrue(body["model"])


class GroqArephraseTests(TestCase):
    @override_settings(GROQ_API_KEY="")
    async def test_no_api_key_returns_failure(self):
        import os as _os

        _os.environ.pop("GROQ_API_KEY", None)
        text, ok, err = await GroqProvider().arephrase("hi", "formal", "llama-3.1-70b")
        self.assertFalse(ok)
        self.assertIn("API Key", err)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_happy_path(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(
            response=_async_response(
                json_payload={"choices": [{"message": {"content": "  groq async  "}}]},
            ),
        )
        text, ok, err = await GroqProvider().arephrase("hi", "casual", "mixtral-8x7b")
        self.assertTrue(ok)
        self.assertEqual(text, "groq async")

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_authorization_header_set(self, mock_client_cls):
        client = _FakeAsyncClient(
            response=_async_response(
                json_payload={"choices": [{"message": {"content": "x"}}]},
            ),
        )
        mock_client_cls.return_value = client
        await GroqProvider().arephrase("hi", "formal", "m")
        headers = client.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_non_200_returns_text(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(
            response=_async_response(status_code=429, text="rate limited"),
        )
        text, ok, err = await GroqProvider().arephrase("hi", "formal", "m")
        self.assertFalse(ok)
        self.assertIn("rate limited", err)

    @override_settings(GROQ_API_KEY="test-key")
    @patch("pdfeditor.ai_service.httpx.AsyncClient")
    async def test_connection_exception(self, mock_client_cls):
        mock_client_cls.return_value = _FakeAsyncClient(exc=TimeoutError("timeout"))
        text, ok, err = await GroqProvider().arephrase("hi", "formal", "m")
        self.assertFalse(ok)
        self.assertIn("Connection Error", err)
