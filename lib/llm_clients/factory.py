"""
LLM Client Factory

Creates LLM clients based on configuration and environment variables.
"""

import os
from typing import Any

from .base import BaseLLMClient
from .gemini_client import GeminiClient
from .openai_client import OpenAIClient


class LLMClientFactory:
    """Factory for creating LLM clients."""

    # Registry of available clients
    _clients = {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
    }

    @classmethod
    def create_client(
        cls, provider: str | None = None, model: str | None = None, **kwargs: Any
    ) -> BaseLLMClient:
        """
        Create an LLM client based on provider.

        Args:
            provider: LLM provider name (gemini, openai)
            model: Specific model to use
            **kwargs: Additional arguments for client initialization

        Returns:
            Initialized LLM client
        """
        # Get provider from environment if not specified
        if not provider:
            provider = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")

        provider = provider.lower()
        if provider not in cls._clients:
            raise ValueError(
                f"Unknown provider: {provider}. Available: {list(cls._clients.keys())}"
            )

        client_class = cls._clients[provider]

        # Set default model if not provided
        if not model:
            if provider == "gemini":
                model = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.0-flash-exp")
            elif provider == "openai":
                model = os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4")

        # Create client
        if provider == "gemini":
            gemini_client: BaseLLMClient = client_class(
                model_name=model or "gemini-2.0-flash-exp", **kwargs
            )
            return gemini_client
        elif provider == "openai":
            openai_client: BaseLLMClient = client_class(model=model or "gpt-4", **kwargs)
            return openai_client
        else:
            default_client: BaseLLMClient = client_class(**kwargs)
            return default_client

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._clients.keys())

    @classmethod
    def register_client(cls, name: str, client_class: type[BaseLLMClient]) -> None:
        """Register a new client type."""
        cls._clients[name.lower()] = client_class


def create_llm_client(
    provider: str | None = None, model: str | None = None, **kwargs: Any
) -> BaseLLMClient:
    """Convenience function to create an LLM client."""
    return LLMClientFactory.create_client(provider=provider, model=model, **kwargs)
