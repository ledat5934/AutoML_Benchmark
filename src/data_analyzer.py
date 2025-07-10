from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectInfo:
    """Metadata describing one ML project discovered in the input folder."""

    name: str
    desc_path: Path  # points to *description.txt*
    project_dir: Path
    data_files: Dict[str, List[Path]] = field(default_factory=dict)


class DataAnalyzer:
    """Walk a root directory, locate *description.txt* files and associated data."""

    SUPPORTED_DATA_EXTS: set[str] = {".csv", ".parquet"}

    def analyze(self, root: Path) -> List[ProjectInfo]:
        """Return list of ProjectInfo for every project description found under *root*."""
        if not root.exists():
            raise FileNotFoundError(f"Path {root} does not exist")

        projects: List[ProjectInfo] = []

        for desc_file in root.rglob("description.txt"):
            project_dir = desc_file.parent
            name = self._infer_project_name(desc_file)
            data_files = self._collect_data_files(project_dir, desc_file)
            projects.append(
                ProjectInfo(
                    name=name,
                    desc_path=desc_file,
                    project_dir=project_dir,
                    data_files=data_files,
                )
            )
            logger.debug(
                "Detected project '%s' with %d data files (description at %s)",
                name,
                sum(len(v) for v in data_files.values()),
                desc_file,
            )

        if not projects:
            logger.warning("No project description.txt files found under %s", root)
        return projects

    def build_eda_report(self, project: ProjectInfo) -> str:
        """Generate a simple text EDA report for primary train/test CSVs."""
        import pandas as pd  # local import to avoid heavy cost if not used

        lines: list[str] = [f"# EDA Report for {project.name}"]
        # Heuristic: find file containing 'train' in name
        train_files = [p for lst in project.data_files.values() for p in lst if "train" in str(p).lower()]
        if train_files:
            train_path = train_files[0]
            df = pd.read_csv(train_path, nrows=5000)  # sample first 5k rows
            lines.append(f"Train file: {train_path}")

            from io import StringIO

            info_buf = StringIO()
            df.info(buf=info_buf, verbose=True, show_counts=True, memory_usage="deep")
            lines.append(info_buf.getvalue())
            desc = df.describe(include="all").T
            lines.append("\nDescriptive stats:\n" + desc.to_string())
        else:
            lines.append("No train csv found for detailed report.")
        return "\n\n".join(lines)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    # Previously we inspected JSON structure; for plain text description we just accept any file named
    # *description.txt*, so no extra validation helper is needed.

    def _infer_project_name(self, desc_path: Path) -> str:
        """Use parent folder name as project name if we cannot parse more from description."""
        return desc_path.parent.name

    def _collect_data_files(self, project_dir: Path, desc_path: Path) -> Dict[str, List[Path]]:
        files: Dict[str, List[Path]] = {}
        for ext in self.SUPPORTED_DATA_EXTS:
            for p in project_dir.rglob(f"*{ext}"):
                # skip the description file
                if p == desc_path:
                    continue
                key = p.stem
                files.setdefault(key, []).append(p)
        return files 