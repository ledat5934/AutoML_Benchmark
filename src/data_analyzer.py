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
        runtime_path: str | None = None,
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
        
        Parameters:
        -----------
        runtime_path : str | None
            Optional path where the dataset will exist during code execution (e.g., Kaggle path).
            If not provided, uses the generation path (where dataset exists during analysis).
        
        Returns the path to the newly written JSON file.
        """
        import pandas as pd  # local import to avoid heavy cost for users not needing this feature
        from pathlib import Path as _Path

        logger.info("Generating dataset JSON for project '%s'", project.name)

        # --------------------------------------------------------------
        # 1) DATASET INFO
        # --------------------------------------------------------------
        # Handle dual path concept: generation path vs runtime path
        kaggle_input_prefix = "/kaggle/input"
        project_dir_posix = project.project_dir.as_posix()

        # Generation path (for analysis) - current logic
        if project_dir_posix.startswith(kaggle_input_prefix):
            generation_base_path = project_dir_posix
        else:
            try:
                generation_base_path = project.project_dir.relative_to(_Path.cwd()).as_posix()
            except ValueError:
                generation_base_path = project_dir_posix

        # Runtime path (for execution) - use provided runtime_path or fallback to generation path
        if runtime_path is not None:
            runtime_base_path = runtime_path
        else:
            runtime_base_path = generation_base_path

        dataset_info: Dict[str, object] = {
            "name": project.name,
            "base_path": runtime_base_path,  # Use runtime path for code generation
            "generation_path": generation_base_path,  # Keep generation path for reference
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

        # --------------------------------------------------------------
        # Handle potentially HUGE collections of individual files (e.g. thousands
        # of .json items in datasets like *pet_finder*).  Listing every single
        # file path inflates the JSON which in turn bloats the prompt and token
        # count for Gemini/LLM.  Therefore we cap the number of file metadata
        # entries and append a concise summary for the remainder.
        # --------------------------------------------------------------

        MAX_FILE_ENTRIES = 20  # Keep at most this many detailed file records

        for p in candidate_paths:
            if p in seen or not p.exists():
                continue
            seen.add(p)

            # Stop adding detailed entries once we reach the cap – we'll summarise later
            if len(files_meta) >= MAX_FILE_ENTRIES:
                continue

            files_meta.append(
                {
                    "path": p.relative_to(project.project_dir).as_posix(),
                    "role": _infer_role(p),
                    "type": _infer_type(p),
                }
            )

        omitted_count = len(candidate_paths) - len(files_meta)

        if omitted_count > 0:
            # Append a lightweight summary object so the LLM still knows additional
            # files exist without enumerating every path.
            files_meta.append(
                {
                    "path": "<omitted>",
                    "role": "bulk_files_summary",
                    "type": "summary",
                    "omitted_count": omitted_count,
                }
            )

        dataset_info["files"] = files_meta

        # --------------------------------------------------------------
        # Detect sample submission file and record its column schema
        # --------------------------------------------------------------
        sample_submission_info = None
        for p in candidate_paths:
            if p.is_file() and "sample" in p.stem.lower() and p.suffix.lower() in {".csv", ".tsv"}:
                try:
                    # Read only the header to avoid loading large files
                    import pandas as pd  # local import within function scope
                    df_header = pd.read_csv(p, nrows=0)
                    sample_submission_info = {
                        "path": p.relative_to(project.project_dir).as_posix(),
                        "columns": df_header.columns.tolist(),
                    }
                    logger.info(
                        "Detected sample submission file at %s with columns: %s",
                        p,
                        sample_submission_info["columns"],
                    )
                    break  # Use the first detected sample file
                except Exception as exc:
                    logger.warning("Failed to read sample submission file %s: %s", p, exc)

        if sample_submission_info:
            dataset_info["sample_submission"] = sample_submission_info

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
                logger.info("Creating ydata_profiling report (optimized for size)...")
                
                # Create profile with simplified configuration (from alt/profile_data.py)
                profile = ProfileReport(
                    profile_df,
                    title=f"Dataset Profile: {project.name}",
                    minimal=True,
                    samples={
                        "random": 5
                    },
                    correlations={
                        "auto": {"calculate": False},
                        "pearson": {"calculate": False},
                        "spearman": {"calculate": False},
                        "kendall": {"calculate": False},
                        "phi_k": {"calculate": False}
                    },
                    missing_diagrams={
                        "bar": False,
                        "matrix": False,
                        "heatmap": False,
                        "dendrogram": False
                    },
                    interactions={"targets": []},
                    explorative=False,
                    progress_bar=False,
                    infer_dtypes=True
                )
                
                # Get JSON and apply filtering (inspired by alt/profile_data.py)
                try:
                    profile_json_str = profile.to_json()
                    filtered_json_str = self._filter_value_counts(profile_json_str)
                    profiling_summary = json.loads(filtered_json_str)
                    logger.info("Successfully generated and filtered ydata_profiling JSON report")
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
            logger.info("Extracting comprehensive guidelines from description.txt using LLM...")
            task_definition = self._extract_task_with_llm(
                project_name=project.name,
                description=description_text,
                client=openai_client,
                profiling_summary=profiling_summary,
                message=message,
            )
        else:
            logger.warning("No LLM client provided. Skipping task definition extraction.")
            task_definition = {"error": "No LLM client available for task definition extraction"}

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

    def _create_variables_summary(self, variables: Dict) -> Dict:
        """Create intelligent summary of variables for LLM (inspired by alt/guideline_create.py)."""
        if not variables:
            return {}
        
        var_types = {"numerical": [], "categorical": [], "text": [], "datetime": [], "other": []}
        
        for var_name, var_info in variables.items():
            var_type = var_info.get("type", "")
            var_summary = {
                "name": var_name, 
                "type": var_type,
                "missing_pct": round(var_info.get("p_missing", 0), 3),
                "n_distinct": var_info.get("n_distinct", 0)
            }
            
            if var_type == "Categorical":
                var_summary.update({
                    "imbalance": round(var_info.get("imbalance", 0), 3),
                    "is_binary": var_info.get("n_distinct", 0) == 2
                })
                if var_info.get("n_distinct", 0) <= 10:
                    value_counts = var_info.get("value_counts_without_nan", {})
                    if value_counts:
                        var_summary["top_values"] = dict(list(value_counts.items())[:5])
            elif var_type == "Numeric":
                var_summary.update({
                    "min": var_info.get("min"), 
                    "max": var_info.get("max"),
                    "mean": round(var_info.get("mean", 0), 3) if var_info.get("mean") else None,
                    "std": round(var_info.get("std", 0), 3) if var_info.get("std") else None
                })
            
            # Categorize by type
            if var_type == "Numeric": 
                var_types["numerical"].append(var_summary)
            elif var_type == "Categorical": 
                var_types["categorical"].append(var_summary)
            elif var_type == "Text": 
                var_types["text"].append(var_summary)
            elif var_type in ["DateTime", "Date", "Time"]: 
                var_types["datetime"].append(var_summary)
            else: 
                var_types["other"].append(var_summary)

        # Sort by missing percentage and distinct count
        for v_type in var_types:
            var_types[v_type] = sorted(var_types[v_type], key=lambda x: (x["missing_pct"], -x.get("n_distinct", 0)))
            
        # Summary statistics
        total_rows = max((v.get("n", 1) for v in variables.values()), default=1)
        summary_stats = {
            "total_variables": len(variables),
            "by_type": {k: len(v) for k, v in var_types.items() if v},
            "missing_data": {
                "variables_with_missing": sum(1 for v in variables.values() if v.get("p_missing", 0) > 0),
                "avg_missing_pct": round(sum(v.get("p_missing", 0) for v in variables.values()) / (len(variables) or 1), 3)
            },
            "data_issues": {
                "id_like_features": [],
                "high_cardinality_features": [],
                "highly_imbalanced_features": []
            }
        }
        
        # Detect data issues
        for var_name, var_info in variables.items():
            n_distinct = var_info.get("n_distinct", 0)
            if n_distinct > total_rows * 0.95 and total_rows > 1:
                summary_stats["data_issues"]["id_like_features"].append(var_name)
            if n_distinct > 100:
                summary_stats["data_issues"]["high_cardinality_features"].append(var_name)
            if var_info.get("type") == "Categorical" and var_info.get("imbalance", 0) > 0.95:
                summary_stats["data_issues"]["highly_imbalanced_features"].append({
                    "name": var_name, 
                    "imbalance": round(var_info.get("imbalance", 0), 3)
                })
                
        return {"summary_stats": summary_stats, "variables_by_type": var_types}



    def _extract_task_with_llm(
        self,
        project_name: str,
        description: str,
        client: "GeminiClient",
        profiling_summary: Dict[str, object],
        message: str | None = None,
    ) -> Dict[str, object]:
        """Use an LLM to parse the description.txt and profiling summary into a comprehensive task definition (inspired by alt/guideline_create.py)."""

        # Extract table info
        table_info = profiling_summary.get("table", {})
        n_rows = table_info.get("n", 0)
        n_cols = table_info.get("n_var", 0)
        
        # Create intelligent variables summary
        variables = profiling_summary.get("variables", {})
        variables_summary = self._create_variables_summary(variables)
        variables_summary_str = json.dumps(variables_summary, indent=2, ensure_ascii=False)
        
        # Extract alerts if available
        alerts = profiling_summary.get("alerts", [])
        alerts_summary = alerts[:3] if alerts else ['None']

        # --------------------------------------------------------------
        # Incorporate optional user message into the prompt
        # --------------------------------------------------------------
        user_message_section = f"\n**User Additional Instructions:**\n{message}\n" if message else ""

        prompt = f"""You are an expert Machine Learning architect. Your task is to analyze the provided dataset information and create a specific, actionable, and justified guideline for an AutoML pipeline.{user_message_section}

## Dataset Information:
- **Dataset**: {project_name}
- **Task Description**: {description[:500]}{'...' if len(description) > 500 else ''}
- **Size**: {n_rows:,} rows, {n_cols} columns
- **Key Quality Alerts**: {alerts_summary}

## Variables Analysis Summary:
```json
{variables_summary_str}
```

## Guideline Generation Principles & Examples
Your response must be guided by the following principles. Refer to these examples to understand the expected level of detail.

**BE SPECIFIC AND ACTIONABLE**: Your recommendations must be concrete actions.
- Bad (Generic): "Handle missing values"
- Good (Specific): "Impute 'Age' with the median"

**JUSTIFY YOUR CHOICES INTERNALLY**: Even though the final JSON doesn't have a reason for every single step, your internal reasoning process must be sound. Base your choices on the data's properties (type, statistics, alerts).

**IT'S OKAY TO OMIT**: If a step is not necessary (e.g., feature selection for a dataset with very few features), provide an empty list [] or null for that key in the JSON output.

**CONSIDER FEATURE SCALING FOR LARGE NUMERIC VALUES**: If any numerical feature (including the target variable) has a very large mean or standard deviation (e.g., >10,000), consider applying scaling such as StandardScaler or MinMaxScaler.

## High-Quality Examples

**Example 1: Feature Engineering for a DateTime column**
If you see a DateTime column like 'transaction_date', a good feature_engineering list would be ["Extract 'month' from 'transaction_date'", "Extract 'day_of_week' from 'transaction_date'"].

**Example 2: Handling High Cardinality Categorical Data**
If a categorical column 'product_id' has over 100 unique values, a good feature_engineering recommendation would be ["Apply frequency encoding to 'product_id'"] instead of one-hot encoding to avoid a memory explosion.

**Example 3: Handling Missing Numerical Data**
If you see a numeric column 'income' with 25% missing values and a skewed distribution, a good missing_values recommendation would be ["Impute 'income' with its median"].

## Required Thinking Process (Do not output this part)
Before generating the final JSON, think step-by-step:
1. First, carefully identify the target variable and the task type (classification/regression).
2. Second, review each variable. What are its type, statistics, and potential issues?
3. Third, based on the data properties and the examples above, decide on the most appropriate, specific ML or DL algorithm for this task.
4. Fourth, think the suitable preprocessing for the algorithm (Example: If use pretrained model for NLP tasks, feature engineering should not have 'generate embedding' step).
5. Consider using pretrained model for NLP or CV tasks if necessary.
6. With text data, consider between pretrained model or BOW, TF-IDF, ... based on task.
7. Finally, compile these specific actions into the required JSON format below.

## Output Format: Your response must be the JSON format below:
Please provide your response in JSON format. It is acceptable to provide an empty list or null for recommendations if none are suitable.

**IMPORTANT**: Ensure the generated JSON is perfectly valid.
- All strings must be enclosed in double quotes.
- All backslashes inside strings must be properly escaped (e.g., "C:\\\\\\\\path" not "C:\\\\path").
- There should be no unescaped newline characters within a string value.
- Do not add trailing commas.
- Do not include comments (// or #) within the JSON output.

{{
    "target_identification": {{
        "target_variable": "identified_target_column_name",
        "reasoning": "explanation for target selection",
        "task_type": "classification/regression/etc"
    }},
    "modeling": {{
        "recommended_algorithms": ["algorithm"],
        "explanation": "explanation for the recommended algorithms",
        "model_selection": ["model_name1", "model_name2"],
        "model_selection_reasoning": "explanation for the model selection",
        "output_file_structure": {{"submission.csv": "submission file for the test dataset, contain n Columns:[...], have the same columns but not the same rows with sample_submission.csv"}}
    }},
    "preprocessing": {{
        "data_cleaning": ["specific step 1", "specific step 2"],
        "feature_engineering": ["specific technique 1", "specific technique 2"],
        "explanation": "explanation for the feature engineering",
        "missing_values": ["strategy 1", "strategy 2"],
        "feature_selection": ["method 1", "method 2"],
        "data_splitting": {{"train": 0.8, "val": 0.2, "strategy": "stratified"}}
    }},
    "evaluation": {{
        "metrics": ["metric 1", "metric 2"],
        "validation_strategy": ["approach 1", "approach 2"],
        "performance_benchmarking": ["baseline 1", "baseline 2"],
        "result_interpretation": ["interpretation 1", "interpretation 2"]
    }}
}}"""

        try:
            messages = [{"role": "user", "content": prompt}]
            # Use enhanced configuration inspired by alt
            response_str = client.chat_completion(
                messages=messages, 
                temperature=0.0, 
                clean_response=True
            )
            
            # Enhanced JSON parsing with fallback (inspired by alt)
            try:
                guidelines = json.loads(response_str)
                logger.info("Guidelines parsed successfully for project: %s", project_name)
                return guidelines
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error, attempting to fix: %s", e)
                # Advanced fixing logic from alt
                response_fixed = response_str.strip()
                if response_fixed.startswith("```json"):
                    response_fixed = response_fixed[7:]
                if response_fixed.endswith("```"):
                    response_fixed = response_fixed[:-3]
                # Remove trailing commas
                import re
                response_fixed = re.sub(r",\s*([}\]])", r"\1", response_fixed)
                
                try:
                    guidelines = json.loads(response_fixed)
                    logger.info("Successfully parsed after manual fixing")
                    return guidelines
                except json.JSONDecodeError as e2:
                    logger.error("Failed to parse even after fixing: %s", e2)
                    return {
                        "error": "Failed to parse LLM response as JSON",
                        "raw_response": response_str,
                        "parse_error": str(e2)
                    }
                    
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

    def _filter_value_counts(self, profile_json_str: str) -> str:
        """Filter and optimize ydata_profiling JSON output (inspired by alt/profile_data.py)."""
        try:
            profile_dict = json.loads(profile_json_str)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s", e)
            return profile_json_str

        if "variables" not in profile_dict:
            logger.warning("No 'variables' section found in profile")
            return profile_json_str

        for var_name, var_info in profile_dict["variables"].items():
            var_type = var_info.get("type", "")
            n_unique = var_info.get("n_unique", 0)

            # Nếu là kiểu văn bản hoặc có quá nhiều giá trị rời rạc → xoá các phần nặng
            should_remove = (
                var_type in ["Text", "Numeric", "Date", "DateTime", "Time", "URL", "Path"]
                or (var_type == "Categorical" and n_unique > 50)
            )

            if should_remove:
                keys_to_remove = [
                    "value_counts_without_nan",
                    "value_counts_index_sorted",
                    "histogram",
                    "length_histogram",
                    "histogram_length",
                    "block_alias_char_counts",
                    "word_counts",
                    "category_alias_char_counts",
                    "script_char_counts",
                    "block_alias_values",
                    "category_alias_values",
                    "character_counts",
                    "block_alias_counts",
                    "script_counts",
                    "category_alias_counts",
                    "n_block_alias",
                    "n_scripts",
                    "n_category",
                ]

                for key in keys_to_remove:
                    var_info.pop(key, None)

        return json.dumps(profile_dict, ensure_ascii=False, indent=2)

 