#!/usr/bin/env python
"""AutoML Benchmark Orchestrator CLI"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import json
from typing import List

from src.utils.env_config import load_config
from src.utils.logger import setup_logging, get_logger
from src.data_analyzer import DataAnalyzer, ProjectInfo
from src.prompt_builder import PromptBuilder
from src.code_generator import CodeGenerator
from src.file_assembler import FileAssembler
from src.utils.gemini_client import GeminiClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoML Benchmark Orchestrator")
    parser.add_argument("path", type=str, help="Root path containing project data")
    parser.add_argument("--all", action="store_true", help="Process every project found instead of just the first one")

    # Kaggle runtime accelerators (mutually exclusive)
    accel_group = parser.add_mutually_exclusive_group()
    accel_group.add_argument("-tpu", action="store_true", help="Use TPU pods on Kaggle")
    accel_group.add_argument("-t4", action="store_true", help="Use 2× Nvidia T4 GPUs on Kaggle")
    accel_group.add_argument("-p100", action="store_true", help="Use Nvidia P100 GPU on Kaggle")

    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # Custom user prompt for the LLM
    parser.add_argument(
        "-m", "--message", type=str, default="", help="Additional user prompt passed to the AI data analyzer"
    )
    # Gemini thinking mode
    parser.add_argument(
        "-thinking", "--thinking", type=str, default="off",
        help="Gemini thinking mode: off | on | <int>. Example: -thinking=on or --thinking 256"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(level=args.log_level)

    # ------------------------------------------------------------------
    # Parse thinking argument (off | on | integer budget)
    # ------------------------------------------------------------------
    thinking_arg = args.thinking.strip() if isinstance(args.thinking, str) else "off"
    if thinking_arg.lower() in {"off", "false", "no"}:
        use_thinking: bool | int = False
    elif thinking_arg.lower() in {"on", "true", "yes"}:
        use_thinking = True
    else:
        try:
            use_thinking = int(thinking_arg)
            if use_thinking < 0:
                raise ValueError
        except ValueError:
            raise SystemExit(f"Invalid --thinking value: {thinking_arg}. Use off, on, or non-negative integer.")

    get_logger(__name__).info("Gemini thinking mode set to: %s", use_thinking)

    # Determine requested accelerator
    accelerator = None
    if args.tpu:
        accelerator = "TPU"
    elif args.t4:
        accelerator = "T4x2"
    elif args.p100:
        accelerator = "P100"

    logger = get_logger(__name__)

    # Load environment variables
    try:
        config = load_config()
        logger.debug("Environment config loaded: %s", config)
    except Exception as exc:
        logger.error("Failed to load environment configuration: %s", exc)
        raise SystemExit(1)

    root_path = Path(args.path).expanduser().resolve()
    logger.info("Scanning input path: %s", root_path)
    if accelerator:
        logger.info("Requested Kaggle accelerator: %s", accelerator)

    analyzer = DataAnalyzer()
    projects: List[ProjectInfo] = analyzer.analyze(root_path)

    if not projects:
        logger.error("No project definitions found. Exiting.")
        raise SystemExit(1)

    if not args.all:
        projects = projects[:1]

    for project in projects:
        logger.info("\n[PROJECT] %s", project.name)
        logger.info("  DESC: %s", project.desc_path)
        # ------------------------------------------------------------------
        # Friendly logging of project data files:
        #   • Avoid printing thousands of individual JSON files (pet_finder, etc.)
        #   • Cap long lists and provide a concise summary instead.
        # ------------------------------------------------------------------
        MAX_FILES_TO_DISPLAY = 10  # How many file paths per key we show at most

        # First, detect single-file JSON entries (common when a folder contains many
        # small JSON docs). If there are a lot, summarise rather than enumerate.
        json_single_entries = [
            key for key, lst in project.data_files.items()
            if len(lst) == 1 and lst[0].suffix.lower() in {".json", ".jsonl"}
        ]

        JSON_SUMMARY_THRESHOLD = 25  # If more than this many JSON entries, summarise

        if len(json_single_entries) > JSON_SUMMARY_THRESHOLD:
            logger.info("  %-20s %s", "json_files", f"{len(json_single_entries)} individual .json/.jsonl files (omitted)")

        for key, lst in project.data_files.items():
            # Skip per-file logging for the bulk JSON entries we summarised above
            if key in json_single_entries and len(json_single_entries) > JSON_SUMMARY_THRESHOLD:
                continue

            # Convert Path objects to strings only for a limited slice
            display_paths = [str(p) for p in lst[:MAX_FILES_TO_DISPLAY]]

            # If there are more files than the cap, append a concise summary
            if len(lst) > MAX_FILES_TO_DISPLAY:
                display_paths.append(f"... (+{len(lst) - MAX_FILES_TO_DISPLAY} more)")

            logger.info("  %-20s %s", key, display_paths)

        # Skip EDA report generation – dataset JSON already contains sufficient profiling summary.
        analysis_report = ""

        # Initialize helpers
        prompt_builder = PromptBuilder()
        gemini_client = GeminiClient(config)
        # NEW: generate <dataset>.json metadata file
        # Build dataset metadata JSON once – reuse for error regeneration prompts
        dataset_json_content = ""
        try:
            dataset_json_content = analyzer.generate_dataset_json(project, openai_client=gemini_client, message=args.message).read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to create dataset JSON for project %s: %s", project.name, exc)

        codegen = CodeGenerator()
        stage_paths: List[Path] = []
        stage_prompts: List[str] = []  # store original prompts for potential regeneration
        stage_code_blocks: List[str] = []  # store generated code blocks per stage
        prev_code = ""

        # Generate stages 1..3
        for stage in (1, 2, 3):
            logger.info("Generating Stage %d code via Gemini...", stage)
            messages = prompt_builder.build_stage_prompt(
                stage,
                project,
                analysis_report,
                prev_code,
                accelerator,
                dataset_json_content,
            )

            # Keep a copy of the user-facing prompt so we can resend it if we need to regenerate
            stage_prompts.append(messages[1]["content"])
            try:
                response_content = gemini_client.chat_completion(messages, temperature=0, use_thinking=use_thinking)
            except Exception as exc:
                logger.error("Gemini request failed: %s", exc)
                raise SystemExit(1)

            code_block = codegen.validate_and_extract(response_content)

            # Save the code block for possible later use when regenerating
            stage_code_blocks.append(code_block)

            stage_file = project.project_dir / f"stage{stage}.py"
            codegen.save(code_block, stage_file)
            stage_paths.append(stage_file)

            # Update prev_code context
            prev_code += "\n\n" + code_block

        # Assemble final outputs (flatten: outputs/<dataset>.py, .ipynb)
        assembler = FileAssembler()
        output_root = Path("outputs")
        assembler.assemble(project.name, stage_paths, output_root)
        logger.info("Project %s processing completed. Outputs in %s", project.name, output_root / f"{project.name}.py")

        # ------------------------------------------------------------------
        # Interactive validation – give the user a chance to report errors.
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Helper: get user feedback (supports multi-line traceback). The user
        # can simply type "-c" and press Enter to confirm success. Otherwise,
        # they may paste the traceback/error and end with an empty line.
        # ------------------------------------------------------------------

        def _collect_feedback() -> str:
            """Read either a single-line '-c' or a multi-line error message.

            The user can:
            1. Type '-c' <Enter>  ➜ generation succeeded.
            2. Paste an error/traceback (may contain blank lines) and, when done,
               type '-e' <Enter> on a new line to signal end of input.
            """
            prompt_msg = (
                "\nGeneration completed.\n"
                "  • Enter '-c' and press <Enter> if everything worked fine.\n"
                "  • Otherwise, paste the error/traceback. When finished, type '-e' on a new line and press <Enter>.\n> "
            )
            try:
                first_line = input(prompt_msg)
            except EOFError:
                # Non-interactive environment: assume success
                return "-c"

            # Quick success path
            if first_line.strip() == "-c":
                return "-c"

            # Collect error lines until sentinel '-e' is entered
            lines: list[str] = []
            if first_line.strip() != "-e":  # '-e' alone means no error text provided
                lines.append(first_line)

            while True:
                try:
                    line = input()
                except EOFError:
                    break  # End of input stream
                if line.strip() == "-e":
                    break
                lines.append(line)
            return "\n".join(lines)

        # Enter regeneration loop – continue until user confirms success (-c)
        while True:
            user_feedback = _collect_feedback()

            if user_feedback == "-c":
                logger.info("User confirmed success. Exiting.")
                # --------------------------------------------------
                # Save Gemini token usage statistics to JSON
                # --------------------------------------------------
                try:
                    usage = gemini_client.get_token_usage()
                    # Simple cost estimate assuming $0.0000003 per prompt token and $0.0000025 per completion token
                    usage["estimated_cost_usd"] = round(
                        (usage["total_prompt_tokens"] * 0.3 + usage["total_completion_tokens"] * 2.5) / 1_000_000,
                        6,
                    )

                    token_usage_path = output_root / "token_usage.json"
                    with open(token_usage_path, "w", encoding="utf-8") as f:
                        json.dump(usage, f, indent=2)
                    logger.info("Token usage summary written to %s", token_usage_path)
                except Exception as exc:
                    logger.warning("Failed to write token usage JSON: %s", exc)
                break

            logger.info("User reported issue – attempting regeneration with error context…")

            # Prepare for regeneration – overwrite existing stage files
            stage_paths_regen: List[Path] = []

            for stage in (1, 2, 3):
                # Build regeneration prompt with FULL code for all stages (no snippets)
                all_previous_code = "\n\n".join([
                    f"### Stage {i+1} Full Code ###\n{code}" for i, code in enumerate(stage_code_blocks)
                ])

                # Build simplified regeneration prompt
                regen_prompt = (
                    f"### DATASET METADATA JSON ###\n{dataset_json_content}\n\n"
                    f"### ERROR TRACEBACK ###\n{user_feedback}\n\n"
                    f"### EXISTING CODE ###\n{all_previous_code}\n\n"
                    f"You are tasked with fixing **Stage {stage}** based on the traceback.\n"
                    f"Follow these common repair patterns:\n"
                    f"- *ImportError*: add or correct `import` statements.\n"
                    f"- *NameError*: ensure all variables and functions are defined/renamed consistently.\n"
                    f"- *FileNotFoundError*: adjust file paths using `Path(base_path)/...`.\n"
                    f"- *TypeError / ValueError*: validate and convert data types or reshape arrays as needed.\n"
                    f"- *AttributeError*: confirm objects support the attribute/method, or convert to correct type.\n\n"
                    f"Output only the fixed Python code for stage {stage} inside triple backticks."
                )

                messages = [
                    {"role": "system", "content": "You are a senior ML engineer with access to complete project context. Use all available information to provide the best solution."},
                    {"role": "user", "content": regen_prompt},
                ]

                try:
                    response_content = gemini_client.chat_completion(messages, temperature=0, use_thinking=use_thinking)
                except Exception as exc:
                    logger.error("Gemini request failed during regeneration: %s", exc)
                    raise SystemExit(1)

                new_code_block = codegen.validate_and_extract(response_content)

                stage_file = project.project_dir / f"stage{stage}.py"
                codegen.save(new_code_block, stage_file)  # overwrite existing stage file
                stage_paths_regen.append(stage_file)

                # Update stored code for potential further regenerations
                stage_code_blocks[stage - 1] = new_code_block

            # Re-assemble final outputs with regenerated code
            assembler.assemble(project.name, stage_paths_regen, output_root)
            logger.info(
                "Regeneration completed. Updated outputs in %s",
                output_root / f"{project.name}.py",
            )

    # Print token usage summary at the end
    if 'gemini_client' in locals():
        gemini_client.print_token_summary()


if __name__ == "__main__":
    main() 