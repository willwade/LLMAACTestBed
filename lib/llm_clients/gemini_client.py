"""
Gemini LLM Client Implementation

Client for Google's Gemini models, implementing the BaseLLMClient interface.
"""

import os
from typing import Any

import llm
from llm import Model
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """
    Gemini-specific LLM client implementation.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp", temperature: float = 0.2):
        """
        Initialize Gemini client.

        Args:
            model_name: Gemini model name
            temperature: Temperature for generation
        """
        # Initialize attributes without calling parent __init__ to avoid type conflict
        self.model_name = model_name
        self.temperature = temperature
        self._gemini_client: Model = llm.get_model(model_name)
        self._gemini_judge_client: Model = llm.get_model(model_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate text using Gemini.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        try:
            # Prepare options, avoiding temperature conflicts
            options = kwargs.copy()
            options["temperature"] = self.temperature

            response = self._gemini_client.prompt(prompt, system=system_prompt, **options)
            text = response.text().strip()

            # Clean thinking tags if present
            if "</thinking>" in text:
                text = text.split("</thinking>")[-1].strip()

            return text.replace("\n", " ")
        except Exception as e:
            return f"Error: {e}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def judge_similarity(self, target: str, prediction: str) -> int:
        """
        Judge semantic similarity using Gemini.

        Args:
            target: Ground truth response
            prediction: Model prediction

        Returns:
            Integer score 1-10
        """
        judge_system = "You are a semantic evaluator for an AAC system. Rate the similarity between the target intent and the AI prediction on a scale of 1-10, where 10 means perfectly matched in intent and 1 means completely unrelated."

        judge_prompt = f"""
TARGET INTENT: "{target}"
AI PREDICTION: "{prediction}"

Rate the semantic similarity (1-10):
Consider:
1. Does the prediction fulfill the user's intent?
2. Is it appropriate for AAC communication?
3. Does it match the urgency and specificity needed?

Score:"""

        try:
            response = self._gemini_judge_client.prompt(
                judge_prompt, system=judge_system, temperature=0.1
            )
            score_text = response.text().strip()

            # Extract numeric score
            import re

            score_match = re.search(r"\b([1-9]|10)\b", score_text)
            if score_match:
                return int(score_match.group(1))
            else:
                # Fallback: try to convert entire response
                try:
                    score = int(float(score_text))
                    return max(1, min(10, score))  # Clamp to 1-10
                except Exception:
                    return 5  # Default middle score
        except Exception as e:
            print(f"Judge error: {e}")
            return 5

    def get_model_info(self) -> dict[str, Any]:
        """
        Get Gemini model information.

        Returns:
            Dictionary with model details
        """
        return {
            "provider": "Google",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "api_key_set": bool(os.getenv("LLM_GEMINI_KEY")),
        }
