"""
Ollama Service Module - AI client for text rephrasing.

Provides functions to communicate with Ollama API for text generation
and rephrasing operations.
"""
import requests
from typing import List, Optional, Tuple
from django.conf import settings


# Default configuration (can be overridden in settings.py)
OLLAMA_BASE_URL = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_DEFAULT_MODEL = getattr(settings, 'OLLAMA_DEFAULT_MODEL', 'gemma3:latest')
OLLAMA_TIMEOUT = getattr(settings, 'OLLAMA_TIMEOUT', 60)

# API endpoints
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"


# Rephrase style prompts
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


def check_ollama_connection() -> Tuple[bool, str]:
    """
    Check if Ollama is available and responding.
    
    Returns:
        Tuple (is_connected: bool, message: str)
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        if response.status_code == 200:
            return True, "Ollama is connected and ready"
        else:
            return False, f"Ollama returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        return False, "Connection to Ollama timed out"
    except Exception as e:
        return False, f"Error connecting to Ollama: {str(e)}"


def get_available_models() -> List[str]:
    """
    Get list of available Ollama models.
    
    Returns:
        List of model names, empty list if connection fails
    """
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        return []
    except Exception:
        return []


def rephrase_text(
    text: str,
    style: str = 'formal',
    model: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> Tuple[str, bool, str]:
    """
    Rephrase text using Ollama AI.
    
    Args:
        text: The text to rephrase
        style: One of 'formal', 'casual', 'simplified', 'concise', 'expanded'
        model: Ollama model to use (defaults to OLLAMA_DEFAULT_MODEL)
        custom_prompt: Optional custom prompt (overrides style)
        
    Returns:
        Tuple (rephrased_text: str, success: bool, error_message: str)
    """
    if not model:
        model = OLLAMA_DEFAULT_MODEL
    
    # Build the prompt
    if custom_prompt:
        system_prompt = custom_prompt
    elif style in REPHRASE_STYLES:
        system_prompt = REPHRASE_STYLES[style]['prompt']
    else:
        system_prompt = REPHRASE_STYLES['formal']['prompt']
    
    # Create the full prompt
    full_prompt = f"""{system_prompt}

Text to rephrase:
"{text}"

Provide ONLY the rephrased text, nothing else. Do not include quotes, explanations, or any additional text."""

    try:
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        response = requests.post(
            OLLAMA_GENERATE_URL, 
            json=payload, 
            timeout=OLLAMA_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            rephrased = data.get('response', '').strip()
            
            # Clean up the response - remove quotes if present
            if rephrased.startswith('"') and rephrased.endswith('"'):
                rephrased = rephrased[1:-1]
            if rephrased.startswith("'") and rephrased.endswith("'"):
                rephrased = rephrased[1:-1]
            
            if rephrased:
                return rephrased, True, ""
            else:
                return "", False, "Ollama returned empty response"
        else:
            # Try to get detailed error message from Ollama
            try:
                error_data = response.json()
                error_msg = error_data.get('error', f"HTTP {response.status_code}")
            except:
                error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            return "", False, f"Ollama error: {error_msg}"
            
    except requests.exceptions.ConnectionError:
        return "", False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}"
    except requests.exceptions.Timeout:
        return "", False, f"Ollama request timed out after {OLLAMA_TIMEOUT}s. Try a smaller model."
    except Exception as e:
        return "", False, f"Error: {str(e)}"



def get_style_choices():
    """
    Get list of style choices for form dropdown.
    
    Returns:
        List of tuples (value, display_name) for Django form ChoiceField
    """
    return [(key, info['name']) for key, info in REPHRASE_STYLES.items()]
