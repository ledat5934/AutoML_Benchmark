from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import openai
from openai import OpenAI

from .env_config import Config
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class OpenAIClient:
    """Simple wrapper around OpenAI ChatCompletion API using the new SDK."""

    config: Config
    client: OpenAI = None  # type: ignore

    def __post_init__(self) -> None:  # type: ignore[override]
        # Initialize OpenAI sdk client (v1)
        # Sanitize custom base URL (if provided) to avoid accidental duplication of path segments
        base_url = self.config.OPENAI_API_URL
        if base_url:
            # remove surrounding quotes and whitespace
            base_url = base_url.strip().strip('"').strip("'").rstrip("/")
            # If user accidentally included the endpoint suffix, strip it
            for suffix in ("/chat/completions", "chat/completions"):
                if base_url.endswith(suffix):
                    base_url = base_url[: -len(suffix)]
        self.client = OpenAI(
            api_key=self.config.OPENAI_API_KEY,
            base_url=base_url or None,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the ChatCompletion endpoint and return content string."""
        logger.debug("Sending prompt with %d messages to OpenAI", len(messages))
        response = self.client.chat.completions.create(
            model=self.config.OPENAI_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        # take first choice
        content = response.choices[0].message.content
        logger.debug("Received response with %d tokens", len(content or ""))
        return content or "" 