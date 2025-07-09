from __future__ import annotations

from pathlib import Path
from typing import List

import nbformat as nbf

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FileAssembler:
    """Combine stage scripts into one .py and .ipynb output."""

    def assemble(
        self,
        project_name: str,
        stage_files: List[Path],
        output_root: Path,
    ) -> None:
        if not stage_files:
            raise ValueError("No stage files provided to assemble")
        output_root.mkdir(parents=True, exist_ok=True)
        py_path = output_root / f"{project_name}.py"
        nb_path = output_root / f"{project_name}.ipynb"

        logger.info("Assembling %d stage files into %s and %s", len(stage_files), py_path, nb_path)

        all_code: List[str] = []
        for p in stage_files:
            code = p.read_text(encoding="utf-8")
            all_code.append(code)

        # Combine python file
        combined = "\n\n".join(all_code) + "\n\nif __name__ == \"__main__\":\n    main()\n"
        py_path.write_text(combined, encoding="utf-8")

        # Build notebook
        nb = nbf.v4.new_notebook()
        for code in all_code:
            nb.cells.append(nbf.v4.new_code_cell(code))
        nbf.write(nb, str(nb_path))

        logger.info("Wrote assembled outputs to %s", output_root) 