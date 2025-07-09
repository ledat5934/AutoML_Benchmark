#!/usr/bin/env python
"""AutoML Benchmark Orchestrator CLI"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from src.utils.env_config import load_config
from src.utils.logger import setup_logging, get_logger
from src.data_analyzer import DataAnalyzer, ProjectInfo
from src.prompt_builder import PromptBuilder
from src.code_generator import CodeGenerator
from src.file_assembler import FileAssembler
from src.utils.openai_client import OpenAIClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoML Benchmark Orchestrator")
    parser.add_argument("path", type=str, help="Root path containing project data")
    parser.add_argument("--all", action="store_true", help="Process every project found instead of just the first one")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    setup_logging(level=args.log_level)
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

    analyzer = DataAnalyzer()
    projects: List[ProjectInfo] = analyzer.analyze(root_path)

    if not projects:
        logger.error("No project definitions found. Exiting.")
        raise SystemExit(1)

    if not args.all:
        projects = projects[:1]

    for project in projects:
        logger.info("\n[PROJECT] %s", project.name)
        logger.info("  JSON: %s", project.json_path)
        for key, lst in project.data_files.items():
            logger.info("  %-20s %s", key, [str(p) for p in lst])

        # Build EDA report
        analysis_report = analyzer.build_eda_report(project)

        # Initialize helpers
        prompt_builder = PromptBuilder()
        openai_client = OpenAIClient(config)
        codegen = CodeGenerator()
        stage_paths: List[Path] = []
        prev_code = ""

        # Generate stages 1..3
        for stage in (1, 2, 3):
            logger.info("Generating Stage %d code via OpenAI...", stage)
            messages = prompt_builder.build_stage_prompt(stage, project, analysis_report, prev_code)
            try:
                response_content = openai_client.chat_completion(messages, temperature=0)
            except Exception as exc:
                logger.error("OpenAI request failed: %s", exc)
                raise SystemExit(1)

            code_block = codegen.validate_and_extract(response_content)

            stage_file = project.project_dir / f"stage{stage}.py"
            codegen.save(code_block, stage_file)
            stage_paths.append(stage_file)

            # Update prev_code context
            prev_code += "\n\n" + code_block

        # Assemble final outputs
        assembler = FileAssembler()
        output_root = Path("outputs") / project.name
        assembler.assemble(project.name, stage_paths, output_root)
        logger.info("Project %s processing completed. Outputs in %s", project.name, output_root)


if __name__ == "__main__":
    main() 