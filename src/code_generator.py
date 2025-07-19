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
            # Attempt to strip triple backtick fenced blocks
            if "```" in raw_response:
                segments = raw_response.split("```")

                # Strategy:
                # 1) Prefer segments explicitly tagged as python
                # 2) Otherwise try each segment until one parses without SyntaxError
                chosen: Optional[str] = None

                for seg in segments[1:]:
                    if not seg.strip():
                        continue

                    # Separate optional language tag from content
                    first_nl = seg.find("\n")
                    if first_nl != -1:
                        lang_tag = seg[:first_nl].strip().lower()
                        tentative_code = seg[first_nl + 1 :]
                    else:
                        lang_tag = ""
                        tentative_code = seg

                    # Prefer explicit python tag
                    if lang_tag.startswith("python"):
                        chosen = tentative_code
                        break

                # If no explicit python tag chosen, fall back to first segment that parses
                if chosen is None:
                    for seg in segments[1:]:
                        if not seg.strip():
                            continue
                        first_nl = seg.find("\n")
                        tentative_code = seg[first_nl + 1 :] if first_nl != -1 else seg
                        try:
                            ast.parse(tentative_code)
                            chosen = tentative_code
                            break
                        except SyntaxError:
                            continue

                # Final fallback: first non-empty segment
                if chosen is None:
                    for seg in segments[1:]:
                        if seg.strip():
                            first_nl = seg.find("\n")
                            tentative_code = seg[first_nl + 1 :] if first_nl != -1 else seg
                            chosen = tentative_code
                            break

                code_block = chosen or ""
            else:
                code_block = raw_response
        # Dedent code to avoid indentation errors when code is pasted with leading spaces
        dedented_code = textwrap.dedent(code_block)
        self.validate(dedented_code)
        return dedented_code 