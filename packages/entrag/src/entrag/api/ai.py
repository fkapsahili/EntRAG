from abc import ABC, abstractmethod


class BaseAIEngine(ABC):
    """
    This class defines the interface for interacting with an AI model.
    """

    @abstractmethod
    def chat_completion(self, model: str, user: str, system: str | None = None, temperature: float = 1.0) -> str:
        """
        Override this method to return a chat completion response.

        Args:
            model: The model name.
            system: The optional system message
            user: The user message
            temperature: The temperature value for the completion.

        Returns:
            str: The response from the AI model.
        """
        pass
