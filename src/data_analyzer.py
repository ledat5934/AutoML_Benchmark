from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .utils.logger import get_logger

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

    SUPPORTED_DATA_EXTS: set[str] = {".csv", ".parquet", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls"}
    SUPPORTED_IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    # Additional modalities
    SUPPORTED_TEXT_EXTS: set[str] = {".txt", ".json", ".xml", ".html", ".md"}
    SUPPORTED_AUDIO_EXTS: set[str] = {".wav", ".mp3", ".flac", ".ogg", ".aac"}

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
        """Generate a simple text EDA report for primary train/test data files."""
        import pandas as pd  # local import to avoid heavy cost if not used

        lines: list[str] = [f"# EDA Report for {project.name}"]
        # Heuristic: find file containing 'train' in name
        train_files = [p for lst in project.data_files.values() for p in lst if "train" in str(p).lower()]
        if train_files:
            train_path = train_files[0]
            df = self._load_tabular_file(train_path)  # load full dataset for comprehensive analysis
            if df is not None:
                lines.append(f"Train file: {train_path}")

                from io import StringIO

                info_buf = StringIO()
                df.info(buf=info_buf, verbose=True, show_counts=True, memory_usage="deep")
                lines.append(info_buf.getvalue())
                desc = df.describe(include="all").T
                lines.append("\nDescriptive stats:\n" + desc.to_string())
            else:
                lines.append(f"Could not load train file: {train_path}")
        else:
            lines.append("No train data file found for detailed report.")
        return "\n\n".join(lines)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _load_tabular_file(self, file_path: Path) -> Optional["pd.DataFrame"]:
        """Load a tabular file (CSV, TSV, JSON, etc.) into a pandas DataFrame."""
        import pandas as pd
        
        try:
            ext = file_path.suffix.lower()
            
            if ext == ".csv":
                return pd.read_csv(file_path)
            elif ext == ".tsv":
                return pd.read_csv(file_path, sep='\t')
            elif ext == ".txt":
                # Try to detect separator for text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        return pd.read_csv(file_path, sep='\t')
                    elif ',' in first_line:
                        return pd.read_csv(file_path, sep=',')
                    else:
                        # Try default CSV first, then TSV
                        try:
                            return pd.read_csv(file_path)
                        except:
                            return pd.read_csv(file_path, sep='\t')
            elif ext == ".parquet":
                return pd.read_parquet(file_path)
            elif ext in [".json", ".jsonl"]:
                if ext == ".jsonl":
                    return pd.read_json(file_path, lines=True)
                else:
                    return pd.read_json(file_path)
            elif ext in [".xlsx", ".xls"]:
                return pd.read_excel(file_path)
            else:
                logger.warning("Unsupported file format: %s", ext)
                return None
                
        except Exception as e:
            logger.error("Failed to load file %s: %s", file_path, e)
            return None

    def _infer_project_name(self, desc_path: Path) -> str:
        """Use parent folder name as project name if we cannot parse more from description."""
        return desc_path.parent.name

    def _collect_data_files(self, project_dir: Path, desc_path: Path) -> Dict[str, List[Path]]:
        files: Dict[str, List[Path]] = {}
        # ------------------------------------------------------------------
        # 1) Tabular data (.csv / .parquet)
        # ------------------------------------------------------------------
        for ext in self.SUPPORTED_DATA_EXTS:
            for p in project_dir.rglob(f"*{ext}"):
                if p == desc_path:
                    continue
                key = p.stem
                files.setdefault(key, []).append(p)

        # ------------------------------------------------------------------
        # 2) Generic data folders (images, text, audio, etc.)
        # ------------------------------------------------------------------
        def _is_supported_file(path: Path) -> bool:
            ext = path.suffix.lower()
            return (
                ext in self.SUPPORTED_IMG_EXTS
                or ext in self.SUPPORTED_TEXT_EXTS
                or ext in self.SUPPORTED_AUDIO_EXTS
            )

        for d in project_dir.rglob("*"):
            if not d.is_dir():
                continue
            if d == project_dir:
                continue  # skip root dir itself
            if d.name.startswith("."):
                continue  # ignore hidden/system dirs

            # Check if directory contains at least one supported file
            if any(_is_supported_file(f) for f in d.rglob("*")):
                files.setdefault(d.name, []).append(d)

        return files

    # ------------------------------------------------------------------
    # Dataset JSON generation using ydata_profiling
    # ------------------------------------------------------------------
    def generate_dataset_json(
        self,
        project: ProjectInfo,
        openai_client: Optional["GeminiClient"] = None,
        message: str | None = None,
    ) -> Path:
        """Create a `<dataset>.json` file under *project.project_dir* containing
        1. dataset_info – file paths and inferred roles
        2. profiling_summary – overview produced by *ydata_profiling*
        3. task_definition – extracted from *description.txt* using an LLM if provided.

        The JSON structure mirrors the example `src/example.json` file.
        
        IMPORTANT: This JSON file is GENERATED by the AutoML pipeline and does NOT exist
        in the original dataset. It is created to provide structured metadata about the
        dataset for the code generation process. The generated code should NOT expect
        this JSON file to exist in the actual dataset directory.
        
        Returns the path to the newly written JSON file.
        """
        import pandas as pd  # local import to avoid heavy cost for users not needing this feature
        from pathlib import Path as _Path

        logger.info("Generating dataset JSON for project '%s'", project.name)

        # --------------------------------------------------------------
        # 1) DATASET INFO
        # --------------------------------------------------------------
        # Convert to path string that works both locally and on Kaggle
        kaggle_input_prefix = "/kaggle/input"
        project_dir_posix = project.project_dir.as_posix()

        if project_dir_posix.startswith(kaggle_input_prefix):
            # Running inside Kaggle or pointing to Kaggle dataset – keep absolute path so that
            # subsequent code can read files directly without additional joins.
            base_path_str = project_dir_posix
        else:
            # Try to build a relative path to make the JSON portable across OSes.
            try:
                base_path_str = project.project_dir.relative_to(_Path.cwd()).as_posix()
            except ValueError:
                # Fallback to absolute posix path (Windows -> C:/...)
                base_path_str = project_dir_posix

        dataset_info: Dict[str, object] = {
            "name": project.name,
            "base_path": base_path_str,
            "description_file": project.desc_path.relative_to(project.project_dir).as_posix(),
            "files": []
        }

        def _infer_role(path: _Path) -> str:
            """Infer role from filename or directory patterns."""
            name_lower = path.name.lower()
            # Special-case image folders so we preserve the richer role names
            if path.is_dir():
                if "train" in name_lower:
                    return "train_images"
                if "test" in name_lower:
                    return "test_images"
                if "valid" in name_lower or "val" in name_lower:
                    return "validation_images"
                return "image_folder"

            # File-based heuristics (fallbacks)
            stem_lower = path.stem.lower()
            if "train" in stem_lower:
                return "train"
            elif "test" in stem_lower:
                return "test"
            elif "valid" in stem_lower or "val" in stem_lower:
                return "validation"
            elif "sample" in stem_lower:
                return "sample"
            else:
                return "data"

        def _infer_type(path: _Path) -> str:
            """Infer data type from path (file or directory)."""
            if path.is_dir():
                # Peek at one file inside to guess modality
                try:
                    sample_file = next(f for f in path.rglob("*") if f.is_file())
                    ext = sample_file.suffix.lower()
                    if ext in self.SUPPORTED_IMG_EXTS:
                        return "image_folder"
                    if ext in self.SUPPORTED_AUDIO_EXTS:
                        return "audio_folder"
                    if ext in self.SUPPORTED_TEXT_EXTS:
                        return "text_folder"
                except StopIteration:
                    pass  # empty directory – unknown
                return "folder"

            ext = path.suffix.lower()
            if ext in self.SUPPORTED_DATA_EXTS:
                return "tabular"
            if ext in self.SUPPORTED_IMG_EXTS:
                return "image"
            if ext in self.SUPPORTED_AUDIO_EXTS:
                return "audio"
            if ext in self.SUPPORTED_TEXT_EXTS:
                return "text"
            return "other"

        # Collect all candidate files
        candidate_paths = [p for lst in project.data_files.values() for p in lst]
        seen = set()
        files_meta = []

        for p in candidate_paths:
            if p in seen or not p.exists():
                continue
            seen.add(p)
            files_meta.append(
                {
                    "path": p.relative_to(project.project_dir).as_posix(),
                    "role": _infer_role(p),
                    "type": _infer_type(p),
                }
            )
        dataset_info["files"] = files_meta

        # --------------------------------------------------------------
        # 2) PROFILING SUMMARY using ydata_profiling
        # --------------------------------------------------------------
        profiling_summary: Dict[str, object] = {}
        # Attempt to locate a primary *train* data file for profiling
        train_files = [p for p in candidate_paths if p.is_file() and "train" in p.name.lower() and p.suffix.lower() in self.SUPPORTED_DATA_EXTS]
        profile_df: Optional["pd.DataFrame"] = None
        if train_files:
            train_path = train_files[0]
            logger.info("Building ydata_profiling report from %s (full dataset)", train_path)
            profile_df = self._load_tabular_file(train_path)
            
            if profile_df is not None:
                logger.info("Loaded dataset with shape: %s", profile_df.shape)
            else:
                logger.error("Failed to load train file: %s", train_path)
        else:
            logger.warning("No train data file found for profiling in project '%s'", project.name)

        if profile_df is not None:
            try:
                from ydata_profiling import ProfileReport  # type: ignore
                # Import the Settings object for configuration
                from ydata_profiling.config import Settings
                logger.info("Creating ydata_profiling report (optimized for size)...")
                
                # Create highly optimized config to minimize JSON size
                config_dict = {
                    # Disable expensive computations
                    "samples": {"head": 0, "tail": 0},  # No sample data
                    "duplicates": {"head": 0},  # No duplicate examples
                    "missing_diagrams": {"heatmap": False, "dendrogram": False},
                    "correlations": {
                        "pearson": {"calculate": True, "warn_high_cardinality": False},
                        "spearman": {"calculate": False},  # Disable Spearman
                        "kendall": {"calculate": False},   # Disable Kendall
                        "phi_k": {"calculate": False},     # Disable Phi-k
                        "cramers": {"calculate": False}    # Disable Cramer's V
                    },
                    "interactions": {"continuous": False, "targets": []},
                    "html": {"minify_html": True},
                    "plot": {
                        "histogram": {"bins": 10},  # Reduce histogram bins
                        "correlation": {"cmap": "RdYlBu_r", "bad": "#000000"}
                    },
                    # Most important: limit value counts to reduce JSON size
                    "vars": {
                        "cat": {
                            "length": False,  # Disable length analysis for strings
                            "characters": False,  # Disable character analysis
                            "words": False,  # Disable word analysis
                            "n_obs": 10  # Only show top 10 most frequent values
                        },
                        "bool": {"n_obs": 3},  # Only show 3 obs for boolean
                        "num": {
                            "low_categorical_threshold": 0,  # Don't treat numeric as categorical
                        }
                    }
                }
                
                # Convert the dictionary to a Settings object
                profiling_settings = Settings(config_dict)
                
                profile = ProfileReport(
                    profile_df, 
                    minimal=True,  # Use minimal mode
                    config=profiling_settings,  # Pass the Settings object
                    title=f"Dataset Profile: {project.name}"
                )
                
                # Try to convert to JSON format first
                try:
                    profile_json = json.loads(profile.to_json())
                    # Post-process to reduce size if needed
                    profiling_summary = self._optimize_ydata_profiling_output(profile_json)
                    logger.info("Successfully generated and optimized ydata_profiling JSON report")
                except Exception as json_exc:
                    logger.warning("ydata_profiling to_json() failed: %s", json_exc)
                    # Fallback: use description dict which contains structured data
                    profiling_summary = profile.get_description()
                    logger.info("Used ydata_profiling description fallback")
                    
            except ImportError as import_exc:
                logger.error("ydata_profiling is not available: %s", import_exc)
                raise ImportError(f"ydata_profiling is required but not installed: {import_exc}")
            except Exception as exc:
                logger.error("ydata_profiling failed: %s", exc)
                logger.info("Attempting to create basic profiling summary as fallback...")
                profiling_summary = self._create_basic_profile_fallback(profile_df)
                logger.info("Created basic profile fallback successfully")
        else:
            profiling_summary = {"warning": "No profiling available – training file not found."}

        # --------------------------------------------------------------
        # 3) TASK DEFINITION from description.txt, powered by LLM
        # --------------------------------------------------------------
        task_definition: Dict[str, object] = {}
        description_text = project.desc_path.read_text(encoding="utf-8")

        if openai_client:
            logger.info("Extracting task definition from description.txt using LLM...")
            task_definition = self._extract_task_with_llm(
                project_name=project.name,
                description=description_text,
                client=openai_client,
                profiling_summary=profiling_summary,
                message=message,
            )
        else:
            logger.warning("No OpenAI client provided. Creating a fallback task definition.")
            task_definition = self._create_fallback_task_definition(project, profile_df, description_text)

        # --------------------------------------------------------------
        # 4) ASSEMBLE and WRITE FINAL JSON
        # --------------------------------------------------------------
        dataset_dict = {
            "dataset_info": dataset_info,
            "profiling_summary": profiling_summary,
            "task_definition": task_definition,
        }

        out_path = project.project_dir / f"{project.name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        logger.info("Dataset metadata written to %s", out_path)
        return out_path

    def _extract_task_with_llm(
        self,
        project_name: str,
        description: str,
        client: "GeminiClient",
        profiling_summary: Dict[str, object],
        message: str | None = None,
    ) -> Dict[str, object]:
        """Use an LLM to parse the description.txt and profiling summary into a structured task definition."""

        # Select a few key fields from profiling to give the LLM context without overloading it
        context_fields = {
            "table": profiling_summary.get("table"),
            "variables": {
                k: {
                    "type": v.get("type"),
                    "description": v.get("description"),
                    "n_missing": v.get("n_missing"),
                    "p_missing": v.get("p_missing"),
                    "n_unique": v.get("n_unique"),
                    "is_highly_correlated": v.get("is_highly_correlated"),
                }
                for k, v in profiling_summary.get("variables", {}).items()
            },
            "analysis": profiling_summary.get("analysis"),
        }
        profiling_context = json.dumps(context_fields, indent=2)

        # --------------------------------------------------------------
        # Incorporate optional user message into the prompt
        # --------------------------------------------------------------
        user_message_section = f"\n**User Additional Instructions:**\n{message}\n" if message else ""

        prompt = f"""You are an expert machine learning engineer analyzing a new dataset.
Your goal is to create a structured JSON object that defines the machine learning task.
You will be given the dataset description and a JSON summary from a profiling tool.{user_message_section}

Analyze the provided text and JSON to perform the following:
1.  **Summarize Description**: Read the 'Dataset Description' and create a concise summary.
2.  **Identify Special Notes**: Carefully review the description for any critical details that would affect modeling, such as:
    -   It is a multi-output or multi-label classification task.
    -   The dataset is imbalanced.
    -   Specific evaluation metrics are required (e.g., "must use AUC ROC").
    -   The data has unique properties (e.g., time-series, hierarchical).
    If you find any such details, add them to a 'note' field. If not, omit the note. The note should be a brief, impactful statement for the developer. For example, for the steel defect problem which is multi-output, a good note would be: "This is a multi-output classification problem; all outputs must be predicted correctly for a successful prediction."
3.  **Determine Task Type**: Based on all available information, classify the task. Common types include:
    -   `binary_classification`
    -   `multi_class_classification`
    -   `multi_label_classification`
    -   `regression`
    -   `clustering`
    -   `time_series_forecasting`
4.  **Identify Target Column(s)**: Find the name of the target column(s) or dependent variable(s). It might be explicitly named or implied. List them.
5.  **Identify Evaluation Metric**: Determine the primary metric for evaluating model performance. If not specified, suggest a standard metric for the task type (e.g., 'accuracy' for classification, 'rmse' for regression).

**Input:**
**Project Name:** {project_name}
**Dataset Description:**
---
{description}
---
**Profiling Summary:**
---
{profiling_context}
---

**Output Format:**
Return ONLY a valid JSON object with the following structure. Do NOT include any explanations or markdown formatting.

{{
  "description_summary": "<Your concise summary of the dataset description>",
  "note": "<Your critical note about the dataset, if any. Omit this field if there's nothing special.>",
  "task_type": "<The determined machine learning task type>",
  "target_columns": ["<list>", "<of>", "<target>", "<column>", "<names>"],
  "evaluation_metric": "<The primary evaluation metric>"
}}
"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response_str = client.chat_completion(messages=messages, temperature=0.0, clean_response=True)
            return json.loads(response_str)
        except json.JSONDecodeError as e:
            logger.error("Failed to decode JSON from LLM response: %s\nResponse: %s", e, response_str)
            return {"error": "Failed to parse LLM response as JSON", "raw_response": response_str}
        except Exception as e:
            logger.error("An unexpected error occurred during LLM task extraction: %s", e)
            return {"error": f"An unexpected error occurred: {e}"}

    def _create_basic_profile_fallback(self, df: "pd.DataFrame") -> Dict[str, object]:
        """Create a basic profiling summary using pandas when ydata_profiling fails."""
        import pandas as pd
        import numpy as np
        
        logger.info("Creating basic profile fallback for dataset shape: %s", df.shape)
        
        # Basic dataset overview
        total_cells = len(df) * len(df.columns)
        missing_cells = int(df.isnull().sum().sum())
        
        overview = {
            "number_of_variables": int(len(df.columns)),
            "number_of_observations": int(len(df)),
            "missing_cells": missing_cells,
            "missing_cells_percentage": f"{(missing_cells / total_cells * 100):.1f}%",
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": f"{(df.duplicated().sum() / len(df) * 100):.1f}%",
            "memory_size_mb": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}",
        }
        
        # Analyze each variable
        variables = {}
        for col in df.columns:
            col_data = df[col]
            
            # Basic stats
            count = int(col_data.count())
            missing = int(col_data.isnull().sum())
            distinct = int(col_data.nunique())
            
            col_info = {
                "type": str(col_data.dtype),
                "count": count,
                "missing": missing,
                "missing_percentage": f"{(missing / len(col_data) * 100):.1f}%",
                "distinct": distinct,
                "unique_percentage": f"{(distinct / count * 100):.1f}%" if count > 0 else "0.0%",
            }
            
            # Type-specific stats
            if pd.api.types.is_numeric_dtype(col_data):
                if count > 0:
                    col_info.update({
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median()),
                        "q25": float(col_data.quantile(0.25)),
                        "q75": float(col_data.quantile(0.75)),
                    })
            else:
                # Categorical/text - only basic info, no detailed value_counts
                if count > 0:
                    value_counts = col_data.value_counts()
                    # Only get essential info, avoid listing all values
                    col_info.update({
                        "most_frequent": str(value_counts.index[0]) if not value_counts.empty else None,
                        "most_frequent_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                        "most_frequent_percentage": f"{(value_counts.iloc[0] / count * 100):.1f}%" if not value_counts.empty else "0.0%",
                        # Only show a few sample values instead of all
                        "sample_values": [str(val) for val in value_counts.head(5).index.tolist()] if not value_counts.empty else []
                    })
            
            variables[col] = col_info
        
        # Basic correlation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlations = {}
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            # Get top correlations
            correlations_list = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if not pd.isna(corr_val):
                        correlations_list.append({
                            "feature_1": col1,
                            "feature_2": col2, 
                            "correlation": f"{corr_val:.3f}",
                            "abs_correlation": abs(corr_val)
                        })
            
            # Sort by absolute correlation and take top 10
            correlations_list.sort(key=lambda x: x["abs_correlation"], reverse=True)
            top_correlations = correlations_list[:10]
            
            # Remove abs_correlation from output
            for corr in top_correlations:
                del corr["abs_correlation"]
            
            correlations = {
                "top_correlations": top_correlations,
                "total_numeric_features": len(numeric_cols)
            }
        
        return {
            "overview": overview,
            "variables": variables,
            "correlations": correlations,
            "methodology": "Basic pandas profiling (ydata_profiling fallback)",
            "note": "This is a simplified profile due to ydata_profiling compatibility issues"
        }

    def _optimize_ydata_profiling_output(self, profile_json: Dict) -> Dict:
        """Optimize ydata_profiling output to minimize JSON size - only keep counts, not value lists."""
        logger.info("Aggressively optimizing ydata_profiling output to minimize size...")
        
        # Make a copy to avoid modifying the original
        optimized = profile_json.copy()
        size_reductions = []
        
        # Aggressively reduce variable details
        if 'variables' in optimized:
            for var_name, var_data in optimized['variables'].items():
                if isinstance(var_data, dict):
                    original_keys = list(var_data.keys())
                    
                    # REMOVE all value lists - only keep counts and basic stats
                    items_to_remove = []
                    for key in var_data.keys():
                        # Remove any field that contains actual values
                        if any(pattern in key.lower() for pattern in [
                            'value_counts', 'values', 'categories', 'histogram', 
                            'frequent_patterns', 'word_counts', 'character_counts',
                            'sample', 'examples', 'mode', 'first_rows', 'last_rows'
                        ]):
                            items_to_remove.append(key)
                    
                    # Actually remove the items
                    for key in items_to_remove:
                        if key in var_data:
                            del var_data[key]
                            size_reductions.append(f"{var_name}.{key}")
                    
                    # Keep only essential stats for each variable type
                    essential_stats = {
                        'count', 'distinct', 'missing', 'memory_size', 'type',
                        'mean', 'std', 'min', 'max', 'variance', 'kurtosis', 'skewness',
                        'sum', 'mad', 'range', 'iqr', 'cv', 'quantile_25', 'quantile_50', 'quantile_75',
                        'n_distinct', 'p_distinct', 'is_unique', 'n_unique', 'p_unique',
                        'n_missing', 'p_missing', 'n_infinite', 'p_infinite'
                    }
                    
                    # Remove non-essential fields
                    keys_to_check = list(var_data.keys())
                    for key in keys_to_check:
                        if key not in essential_stats and not key.startswith('p_') and not key.startswith('n_'):
                            # Keep basic numeric stats but remove complex objects
                            if isinstance(var_data[key], (dict, list)) and key not in ['type']:
                                del var_data[key]
                                size_reductions.append(f"{var_name}.{key}")
        
        # Heavily reduce correlations section
        if 'correlations' in optimized:
            corr_data = optimized['correlations']
            if isinstance(corr_data, dict):
                # Remove all correlation matrices, keep only high-level stats
                for corr_type in ['pearson', 'spearman', 'kendall', 'phi_k', 'cramers']:
                    if corr_type in corr_data and isinstance(corr_data[corr_type], dict):
                        corr_section = corr_data[corr_type]
                        # Remove matrix data but keep summary stats
                        items_to_remove = ['matrix', 'values', 'index']
                        for item in items_to_remove:
                            if item in corr_section:
                                del corr_section[item]
                                size_reductions.append(f"correlations.{corr_type}.{item}")
        
        # Remove large data sections completely
        sections_to_remove = [
            'sample', 'duplicates', 'missing', 'interactions', 
            'alerts', 'package', 'analysis'
        ]
        for section in sections_to_remove:
            if section in optimized:
                del optimized[section]
                size_reductions.append(f"root.{section}")
        
        # Keep only essential table info
        if 'table' in optimized:
            table_data = optimized['table']
            if isinstance(table_data, dict):
                # Keep only basic table stats
                essential_table_keys = {
                    'n', 'n_var', 'memory_size', 'record_size', 
                    'n_cells', 'n_cells_missing', 'p_cells_missing',
                    'n_duplicates', 'p_duplicates'
                }
                keys_to_remove = [k for k in table_data.keys() if k not in essential_table_keys]
                for key in keys_to_remove:
                    del table_data[key]
                    size_reductions.append(f"table.{key}")
        
        # Add comprehensive optimization note
        if 'table' not in optimized:
            optimized['table'] = {}
        optimized['table'].update({
            'size_optimized': True,
            'optimization_level': 'aggressive',
            'optimization_note': 'All value lists removed - only counts and basic statistics retained',
            'removed_sections': len(size_reductions),
            'optimization_strategy': 'Minimal JSON for maximum compatibility with LLM token limits'
        })
        
        logger.info("Aggressive optimization completed - removed %d sections/fields", len(size_reductions))
        logger.debug("Removed items: %s", ', '.join(size_reductions[:10]) + ('...' if len(size_reductions) > 10 else ''))
        
        return optimized

    def _create_fallback_task_definition(self, project: ProjectInfo, profile_df: Optional["pd.DataFrame"], desc_text: str) -> Dict[str, object]:
        """Create a fallback task definition when Gemini API fails or is unavailable."""
        import pandas as pd
        
        logger.info("Creating fallback task definition for project: %s", project.name)
        
        # Basic task definition structure
        task_def = {
            "problem_type": "Unknown",
            "objective": "Automated machine learning task",
            "method": "fallback_analysis",
            "note": "Generated without LLM due to API issues"
        }
        
        # Try to infer basic info from dataset if available
        if profile_df is not None:
            # Try to detect target column
            potential_targets = []
            for col in profile_df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['target', 'label', 'class', 'y', 'outcome']):
                    potential_targets.append(col)
                elif profile_df[col].nunique() <= 20 and profile_df[col].nunique() > 1:  # Low cardinality, likely categorical target
                    potential_targets.append(col)
            
            # Infer problem type
            if potential_targets:
                target_col = potential_targets[0]
                unique_values = profile_df[target_col].nunique()
                
                if pd.api.types.is_numeric_dtype(profile_df[target_col]):
                    if unique_values <= 10:
                        task_def["problem_type"] = "Classification"
                    else:
                        task_def["problem_type"] = "Regression"
                else:
                    task_def["problem_type"] = "Classification"
                
                task_def["target_column"] = {
                    "name": target_col,
                    "description": f"Detected target column with {unique_values} unique values"
                }
                
                task_def["objective"] = f"Predict {target_col} using available features"
                
            else:
                # No clear target found
                task_def["target_column"] = {
                    "name": "unknown",
                    "description": "No clear target column detected"
                }
                task_def["objective"] = "General data analysis and modeling"
            
            # Basic feature info
            numeric_cols = profile_df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = profile_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            task_def["features"] = {
                "total_features": len(profile_df.columns),
                "numeric_features": len(numeric_cols),
                "categorical_features": len(categorical_cols),
                "sample_numeric": numeric_cols[:5],
                "sample_categorical": categorical_cols[:5]
            }
            
            task_def["dataset_info"] = {
                "rows": len(profile_df),
                "columns": len(profile_df.columns),
                "memory_usage": f"{profile_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
            }
        
        # Use description text for additional context
        if desc_text:
            # Simple keyword detection
            desc_lower = desc_text.lower()
            if any(keyword in desc_lower for keyword in ['classification', 'classify', 'predict class']):
                task_def["problem_type"] = "Classification"
            elif any(keyword in desc_lower for keyword in ['regression', 'predict value', 'continuous']):
                task_def["problem_type"] = "Regression"
            
            # Extract a brief objective from description
            if len(desc_text) > 0:
                # Use first sentence or first 100 chars as objective
                first_sentence = desc_text.split('.')[0]
                if len(first_sentence) <= 100:
                    task_def["objective"] = first_sentence.strip()
                else:
                    task_def["objective"] = desc_text[:100].strip() + "..."
        
        logger.info("Fallback task definition created successfully")
        return task_def 