import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_BASE_URL = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Rephrase styles (shared)
REPHRASE_STYLES = {
    'formal': {
        'name': 'Formal/Professional',
        'prompt': 'Rephrase the following text in a formal, professional tone. Keep the same meaning but use more sophisticated vocabulary and sentence structure.'
    },
    'casual': {
        'name': 'Casual/Conversational', 
        'prompt': 'Rephrase the following text in a casual, friendly, conversational tone. Make it sound natural and easy to read.'
    },
    'simplified': {
        'name': 'Simplified',
        'prompt': 'Rephrase the following text using simpler words and shorter sentences. Make it easy to understand for anyone.'
    },
    'concise': {
        'name': 'Concise',
        'prompt': 'Rephrase the following text to be more concise. Remove unnecessary words while preserving the core meaning.'
    },
    'expanded': {
        'name': 'Expanded/Detailed',
        'prompt': 'Rephrase the following text with more detail and explanation. Expand on the ideas while keeping the original meaning.'
    }
}

class AIProvider:
    def get_models(self) -> List[str]:
        raise NotImplementedError

    def rephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        raise NotImplementedError

    async def arephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        """Async variant of `rephrase`. Override in subclasses to avoid blocking ASGI workers."""
        raise NotImplementedError


def _build_prompt(text: str, style: str, custom_prompt: Optional[str]) -> str:
    prompt_text = custom_prompt or REPHRASE_STYLES.get(style, REPHRASE_STYLES['formal'])['prompt']
    return f"{prompt_text}\n\nText to rephrase:\n{text}\n\nRephrased text:"


def _groq_payload(text: str, style: str, model: str, custom_prompt: Optional[str]) -> Dict[str, Any]:
    prompt_text = custom_prompt or REPHRASE_STYLES.get(style, REPHRASE_STYLES['formal'])['prompt']
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful writing assistant. Output ONLY the rephrased text, without any intro or outro."},
            {"role": "user", "content": f"{prompt_text}\n\nText:\n{text}"},
        ],
        "model": model,
        "temperature": 0.7,
    }

class OllamaProvider(AIProvider):
    def get_models(self) -> List[str]:
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception:
            return []

    def rephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        if not model:
            model = getattr(settings, 'OLLAMA_DEFAULT_MODEL', 'llama3')

        payload = {
            "model": model,
            "prompt": _build_prompt(text, style, custom_prompt),
            "stream": False,
            "options": {"temperature": 0.7},
        }

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60,
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip(), True, ""
            return "", False, f"Ollama Error: {response.text}"
        except Exception as e:
            return "", False, f"Connection Error: {str(e)}"

    async def arephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        if not model:
            model = getattr(settings, 'OLLAMA_DEFAULT_MODEL', 'llama3')

        payload = {
            "model": model,
            "prompt": _build_prompt(text, style, custom_prompt),
            "stream": False,
            "options": {"temperature": 0.7},
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get('response', '').strip(), True, ""
            return "", False, f"Ollama Error: {response.text}"
        except Exception as e:
            return "", False, f"Connection Error: {str(e)}"

class GroqProvider(AIProvider):
    def __init__(self) -> None:
        self.api_key = getattr(settings, 'GROQ_API_KEY', '') or os.environ.get('GROQ_API_KEY', '')
        
    def get_models(self) -> List[str]:
        if not self.api_key:
            return []
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(GROQ_MODELS_URL, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Filter for chat models if needed, or just return all IDs
                return [m['id'] for m in data.get('data', [])]
            return []
        except Exception as e:
            logger.warning("Groq API error: %s", e)
            return []

    def rephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        if not self.api_key:
            return "", False, "Groq API Key not found. Please add GROQ_API_KEY to .env"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = _groq_payload(text, style, model, custom_prompt)

        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                return content, True, ""
            return "", False, f"Groq Error: {response.text}"
        except Exception as e:
            return "", False, f"Connection Error: {str(e)}"

    async def arephrase(self, text: str, style: str, model: str, custom_prompt: Optional[str] = None) -> Tuple[str, bool, str]:
        if not self.api_key:
            return "", False, "Groq API Key not found. Please add GROQ_API_KEY to .env"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = _groq_payload(text, style, model, custom_prompt)

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content'].strip()
                return content, True, ""
            return "", False, f"Groq Error: {response.text}"
        except Exception as e:
            return "", False, f"Connection Error: {str(e)}"

def get_provider(provider_name: str) -> AIProvider:
    if provider_name == 'groq':
        return GroqProvider()
    return OllamaProvider()

def get_all_models() -> Dict[str, List[str]]:
    """Returns models grouped by provider"""
    ollama = OllamaProvider().get_models()
    groq = GroqProvider().get_models()
    return {
        'ollama': ollama,
        'groq': groq
    }
