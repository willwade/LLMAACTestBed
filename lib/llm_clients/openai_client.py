"""
OpenAI LLM Client Implementation

Implements the BaseLLMClient interface for OpenAI models.
"""

import os
from typing import Any

from openai import OpenAI

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """OpenAI implementation of LLM client."""

    def __init__(self, model: str = "gpt-4", api_key: str | None = None):
        """Initialize OpenAI client."""
        self.model_name = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)

    def _supports_temperature(self) -> bool:
        """
        Some lightweight OpenAI models (e.g., gpt-5-nano) only support the default
        temperature. Skip sending temperature for those to avoid 400s.
        """
        model = self.model_name.lower()
        # Conservatively disable for nano-tier and any model that starts with "o1"
        return not ("nano" in model or model.startswith("o1"))

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generate text using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Handle different parameter names for different models
        create_params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            **kwargs,
        }
        if temperature is not None and self._supports_temperature():
            create_params["temperature"] = temperature

        # Some models use max_completion_tokens instead of max_tokens
        if "gpt-4" in self.model_name or "gpt-3.5" in self.model_name:
            create_params["max_tokens"] = max_tokens
        else:
            create_params["max_completion_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**create_params)

            return response.choices[0].message.content or ""

        except Exception as e:
            # Retry once without temperature if the model rejects it
            if "Unsupported value: 'temperature'" in str(e):
                create_params.pop("temperature", None)
                try:
                    response = self.client.chat.completions.create(**create_params)
                    return response.choices[0].message.content or ""
                except Exception as e2:
                    raise RuntimeError(f"OpenAI generation failed after retry: {e2}") from e2
            raise RuntimeError(f"OpenAI generation failed: {e}") from e

    def judge_similarity(self, target: str, prediction: str) -> int:
        """Judge similarity between target and prediction using OpenAI (1-10 scale)."""
        judge_prompt = f"""You are scoring how well an AAC assistant's prediction matches the user's intended utterance.

TARGET: "{target}"
PREDICTION: "{prediction}"

Score 1-10, where:
- 10 = Perfectly conveys the same action/intent and specificity
- 8-9 = Minor wording differences but correct action and detail
- 5-7 = Partially correct; misses key action OR is too vague
- 3-4 = Vague acknowledgement without the requested action/detail
- 1-2 = Unrelated or wrong intent

Missing the requested action (e.g., predicting generic pain when target is \"apply gel to back\") should score 4 or below.
Respond with only the number 1-10."""

        try:
            response = self.generate(judge_prompt, temperature=0.0, max_tokens=10)

            # Extract number from response
            import re

            raw_response = response.strip() if isinstance(response, str) else str(response)
            if os.getenv("LOG_JUDGE_RESPONSES") == "1":
                print(f"[judge-debug] raw: {raw_response}")

            match = re.search(r"\b(10|[1-9])\b", raw_response)
            if match:
                score = int(match.group(1))
                return min(10, max(1, score))

            # Default to conservative mid-low score if no number found
            return 4

        except Exception:
            return 4

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "openai"

    def get_model_info(self) -> dict[str, Any]:
        """
        Get OpenAI model information.

        Returns:
            Dictionary with model details
        """
        return {
            "provider": "OpenAI",
            "model_name": self.model_name,
            "api_key_set": bool(self.api_key),
        }
