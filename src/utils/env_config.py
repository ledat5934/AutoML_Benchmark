from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Holds environment configuration loaded from `.env`."""

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_API_URL: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Config":
        """Load variables from environment / .env file and return Config instance."""
        # Load variables from .env into environment if present
        load_dotenv()
        key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        url = os.getenv("OPENAI_API_URL")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set in environment or .env")
        return cls(OPENAI_API_KEY=key, OPENAI_MODEL=model, OPENAI_API_URL=url)


def load_config() -> Config:
    """Helper shortcut to get a Config object."""
    return Config.from_env() 