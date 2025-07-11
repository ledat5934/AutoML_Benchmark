from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class Config:
    """Holds environment configuration loaded from `.env`."""

    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls) -> "Config":
        """Load variables from environment / .env file and return Config instance."""
        # Load variables from .env into environment if present
        load_dotenv()
        key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set in environment or .env")
        return cls(GEMINI_API_KEY=key, GEMINI_MODEL=model)


def load_config() -> Config:
    """Helper shortcut to get a Config object."""
    return Config.from_env() 