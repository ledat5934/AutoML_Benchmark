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
        # Disable autoescaping entirely because templates generate plain text prompts, not HTML.
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
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
        accelerator: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Return list of messages to feed ChatCompletion for given stage."""
        template_name = f"stage{stage}.j2"
        template = self._load_template(template_name)
        # Read contest/task description from description.txt, if available
        description_text = ""
        try:
            with open(project.desc_path, "r", encoding="utf-8") as f:
                description_text = f.read()
        except Exception as exc:
            logger.warning("Could not read description.txt at %s: %s", project.desc_path, exc)

        prompt_text = template.render(
            project=project,
            analysis_report=analysis_report,
            prev_code=prev_code,
            description_text=description_text,
            accelerator=accelerator,
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