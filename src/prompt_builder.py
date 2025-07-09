from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape, Template

from src.data_analyzer import ProjectInfo
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PromptBuilder:
    """Assemble system/user messages for each stage based on templates."""

    def __init__(self, templates_dir: Optional[Path] = None) -> None:
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(disabled=True),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.templates_dir = templates_dir
        logger.debug("Prompt templates loaded from %s", templates_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_stage_prompt(
        self,
        stage: int,
        project: ProjectInfo,
        analysis_report: str,
        prev_code: str = "",
    ) -> List[Dict[str, str]]:
        """Return list of messages to feed ChatCompletion for given stage."""
        template_name = f"stage{stage}.j2"
        template = self._load_template(template_name)
        prompt_text = template.render(
            project=project,
            analysis_report=analysis_report,
            prev_code=prev_code,
        )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a senior ML engineer."},
            {"role": "user", "content": prompt_text},
        ]
        return messages

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_template(self, name: str) -> Template:
        try:
            return self.env.get_template(name)
        except Exception as exc:
            raise FileNotFoundError(f"Prompt template {name} not found in {self.templates_dir}") from exc 