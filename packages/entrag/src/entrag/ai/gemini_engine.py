import os
from typing import override

from google.genai import Client
from google.genai.types import GenerateContentConfigDict, Part, UserContent
from loguru import logger

from entrag.api.ai import BaseAIEngine


class GeminiEngine(BaseAIEngine):
    @override
    def chat_completion(
        self,
        model: str,
        user: str,
        system: str | None = None,
        temperature: float = 1.0,
    ) -> str:
        try:
            contents = [UserContent(parts=[Part.from_text(text=user)])]
            client = Client(api_key=os.getenv("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=GenerateContentConfigDict(temperature=temperature, system_instruction=system),
            )
            logger.debug(
                "AI response: {extra}",
                extra={
                    "response": response.text,
                    "messages": contents,
                },
            )
            return response.text
        except Exception as e:
            logger.error("Error calling Gemini chat completion", e)
            raise
