from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectInfo:
    """Metadata describing one ML project discovered in the input folder."""

    name: str
    json_path: Path
    project_dir: Path
    data_files: Dict[str, List[Path]] = field(default_factory=dict)


class DataAnalyzer:
    """Walk a root directory, locate project description JSON files and associated data."""

    SUPPORTED_DATA_EXTS: set[str] = {".csv", ".parquet"}

    def analyze(self, root: Path) -> List[ProjectInfo]:
        """Return list of ProjectInfo for every project json found under *root*."""
        if not root.exists():
            raise FileNotFoundError(f"Path {root} does not exist")

        projects: List[ProjectInfo] = []

        for json_file in root.rglob("*.json"):
            if not self._looks_like_project_json(json_file):
                continue

            project_dir = json_file.parent
            name = self._infer_project_name(json_file)
            data_files = self._collect_data_files(project_dir, json_file)
            projects.append(
                ProjectInfo(
                    name=name,
                    json_path=json_file,
                    project_dir=project_dir,
                    data_files=data_files,
                )
            )
            logger.debug("Detected project '%s' with %d data files", name, sum(len(v) for v in data_files.values()))

        if not projects:
            logger.warning("No project JSON files found under %s", root)
        return projects

    def build_eda_report(self, project: ProjectInfo) -> str:
        """Generate a simple text EDA report for primary train/test CSVs."""
        import pandas as pd  # local import to avoid heavy cost if not used

        lines: list[str] = [f"# EDA Report for {project.name}"]
        # Heuristic: find file containing 'train' in name
        train_files = [p for key, lst in project.data_files.items() for p in lst if "train" in key.lower()]
        if train_files:
            train_path = train_files[0]
            df = pd.read_csv(train_path, nrows=5000)  # sample first 5k rows
            lines.append(f"Train file: {train_path}")
            lines.append(df.info(verbose=True, show_counts=True, memory_usage="deep").to_string())
            desc = df.describe(include="all").T
            lines.append("\nDescriptive stats:\n" + desc.to_string())
        else:
            lines.append("No train csv found for detailed report.")
        return "\n\n".join(lines)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _looks_like_project_json(self, path: Path) -> bool:
        """Quick heuristic: must contain the key \"competition_brief\" in first 4 KB."""
        try:
            with open(path, "r", encoding="utf-8") as fp:
                snippet = fp.read(4096)
            return "\"competition_brief\"" in snippet
        except Exception:
            return False

    def _infer_project_name(self, json_path: Path) -> str:
        try:
            with open(json_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data.get("kaggle_dataset_name", json_path.stem)
        except Exception:
            return json_path.stem

    def _collect_data_files(self, project_dir: Path, json_path: Path) -> Dict[str, List[Path]]:
        files: Dict[str, List[Path]] = {}
        for ext in self.SUPPORTED_DATA_EXTS:
            for p in project_dir.rglob(f"*{ext}"):
                # skip the project json file and any potential helper label jsons
                if p == json_path:
                    continue
                key = p.stem
                files.setdefault(key, []).append(p)
        return files 