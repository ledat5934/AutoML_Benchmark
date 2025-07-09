from __future__ import annotations

import ast
import textwrap
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CodeGenerator:
    """Validate generated code snippets and save them to disk."""

    def validate(self, code: str) -> None:
        """Raise SyntaxError if code is invalid."""
        try:
            ast.parse(code)
        except SyntaxError as exc:
            logger.error("Generated code failed syntax check: %s", exc)
            raise

    def save(self, code: str, path: Path) -> None:
        """Validate then write code to *path*. Creates parent dirs."""
        self.validate(code)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Dedent code so that triple-quoted code blocks align
        dedented = textwrap.dedent(code)
        path.write_text(dedented, encoding="utf-8")
        logger.info("Saved generated code to %s", path)

    # convenience
    def validate_and_extract(self, raw_response: str) -> str:
        """Extract code between markers and validate."""
        start_tok = "# ==="
        if "# ===" in raw_response:
            # assume markers present
            lines = raw_response.splitlines()
            inside = False
            collected: list[str] = []
            for ln in lines:
                if ln.strip().startswith("# ===") and "START" in ln:
                    inside = True
                    continue
                if ln.strip().startswith("# ===") and "END" in ln:
                    inside = False
                    break
                if inside:
                    collected.append(ln)
            code_block = "\n".join(collected)
        else:
            code_block = raw_response
        self.validate(code_block)
        return code_block 