"""
Environment utilities

Handles loading and managing environment variables from .env files.
"""

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


def load_env(env_file: str | None = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_file: Path to .env file. If None, looks for .env in project root.
    """
    if not HAS_DOTENV:
        return

    if env_file is None:
        # Look for .env in current directory and parent directories
        current = Path.cwd()
        while current != current.parent:
            env_path = current / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                break
            current = current.parent
    else:
        load_dotenv(env_file)

    # Ensure llm plugin key environment variables are populated
    if not os.getenv("LLM_GEMINI_KEY") and os.getenv("GEMINI_API_KEY"):
        os.environ["LLM_GEMINI_KEY"] = os.getenv("GEMINI_API_KEY", "")
    if not os.getenv("LLM_OPENAI_KEY") and os.getenv("OPENAI_API_KEY"):
        os.environ["LLM_OPENAI_KEY"] = os.getenv("OPENAI_API_KEY", "")
    if not os.getenv("LLM_ANTHROPIC_KEY") and os.getenv("ANTHROPIC_API_KEY"):
        os.environ["LLM_ANTHROPIC_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")


def get_env(key: str, default: Any = None) -> Any:
    """
    Get environment variable with optional default.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_required_env(key: str) -> str:
    """
    Get required environment variable, raising error if not found.

    Args:
        key: Environment variable name

    Returns:
        Environment variable value

    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def get_llm_config() -> dict[str, Any]:
    """
    Get LLM configuration from environment variables.

    Returns:
        Dictionary with LLM configuration
    """
    return {
        "provider": get_env("DEFAULT_LLM_PROVIDER", "gemini"),
        "gemini_model": get_env("DEFAULT_GEMINI_MODEL", "gemini-2.0-flash-exp"),
        "openai_model": get_env("DEFAULT_OPENAI_MODEL", "gpt-4"),
        "anthropic_model": get_env("DEFAULT_ANTHROPIC_MODEL", "claude-3-sonnet-20241022"),
        "gemini_key": get_env("GEMINI_API_KEY"),
        "openai_key": get_env("OPENAI_API_KEY"),
        "anthropic_key": get_env("ANTHROPIC_API_KEY"),
    }
