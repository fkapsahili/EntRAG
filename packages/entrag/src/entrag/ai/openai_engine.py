import os
from typing import override

from loguru import logger
from openai import APIError, OpenAI, RateLimitError

from entrag.api.ai import BaseAIEngine


class OpenAIEngine(BaseAIEngine):
    @override
    def chat_completion(
        self,
        model: str,
        user: str,
        system: str | None = None,
        temperature: float = 1.0,
    ) -> str:
        try:
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user}
            ]
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            chat_completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            response = chat_completion.choices[0].message.content
            logger.debug(
                "AI response: {extra}",
                extra={
                    "response": response,
                    "messages": messages,
                },
            )
            return response
        except APIError as e:
            logger.error("Error calling OpenAI chat completion", e)
            raise
        except (RateLimitError, TimeoutError) as e:
            logger.error("Rate limit error calling OpenAI chat completion", e)
            raise
        except Exception as e:
            logger.error("Error calling OpenAI chat completion", e)
            raise
