#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # Load .env file
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        loaded = load_dotenv(env_path)
        print(f"DEBUG: Attempting to load .env from {env_path}")
        print(f"DEBUG: load_dotenv returned: {loaded}")
        print(f"DEBUG: GROQ_API_KEY in env: {'GROQ_API_KEY' in os.environ}")
    except ImportError:
        print("DEBUG: python-dotenv not installed or failed to import.")
        
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pdf_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
