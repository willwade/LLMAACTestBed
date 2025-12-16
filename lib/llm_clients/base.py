"""
Base LLM Client Interface

Abstract base class defining the interface for all LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    This class defines the common interface that all LLM clients must implement,
    ensuring consistency across different providers (Gemini, OpenAI, etc.).
    """

    def __init__(self, model_name: str, temperature: float = 0.2):
        """
        Initialize the LLM client.

        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation (default: 0.2 for consistency)
        """
        self.model_name = model_name
        self.temperature = temperature
        self._client = None

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate text from the LLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    def judge_similarity(self, target: str, prediction: str) -> int:
        """
        Judge the semantic similarity between target and prediction.

        Args:
            target: The ground truth/expected response
            prediction: The model's prediction

        Returns:
            Integer score (typically 1-10)
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary with model details
        """
        pass

    def batch_generate(
        self, prompts: list[str], system_prompt: str | None = None, **kwargs
    ) -> list[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt, **kwargs)
            responses.append(response)
        return responses
