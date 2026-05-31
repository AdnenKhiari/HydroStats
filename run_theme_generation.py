"""
run_theme_generation.py — Gemini Batch theme generation across versions.

Consumes `codification_analysis.json` for each version, aggregates all Step 2
codebook entries within that version, and writes `theme_generation_analysis.json`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from theme_generation_analysis.batch_processor import GeminiThemeGenerationBatchProcessor
from theme_generation_analysis.data_loader import combine_versions, load_version
from theme_generation_analysis.models import GeneratedTheme, ThemeGenerationResult, ThemeSourceCode
from theme_generation_analysis.pipeline import save_results

DATA_ROOT = Path(__file__).parent / "data"
ALL_VERSIONS = ["Baseline V0", "V1", "V2", "V1+2"]
STEP3_PROMPT_FILE = Path(__file__).parent / "theme_generation_analysis" / "prompt.txt"
STEP3_COMBINED_PROMPT_FILE = Path(__file__).parent / "theme_generation_analysis" / "prompt_combined.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_theme_generation")

_COMBINED_CORPUS_LABEL = "ALL_VERSIONS"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini Batch theme generation over Step 2 codebooks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--versions", nargs="+", default=None, metavar="VERSION")
    parser.add_argument("--questions", nargs="+", default=None, metavar="QUERY_KEY")
    parser.add_argument("--limit", type=int, default=None, metavar="N")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default=None, metavar="DIR")
    parser.add_argument(
        "--per-version-input",
        action="store_true",
        help="Legacy mode: send one Step 3 prompt per version instead of one combined prompt across all selected versions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _output_path(version: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir / version / "theme_generation_analysis.json"
    return DATA_ROOT / version / "theme_generation_analysis.json"


def _input_log_path(version: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir / version / "theme_generation_input.txt"
    return DATA_ROOT / version / "theme_generation_input.txt"


def _raw_log_root(output_dir: Optional[Path]) -> Path:
    return output_dir if output_dir is not None else DATA_ROOT


def _prompt_file_for_mode(*, use_combined_input: bool) -> Path:
    return STEP3_COMBINED_PROMPT_FILE if use_combined_input else STEP3_PROMPT_FILE


def _copy_text_file(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def _project_combined_result(
    combined_result: ThemeGenerationResult,
    version_corpus,
) -> ThemeGenerationResult:
    prefix = f"{version_corpus.version}::"

    def to_local_code_id(scoped_code_id: str) -> str:
        return scoped_code_id[len(prefix):] if scoped_code_id.startswith(prefix) else scoped_code_id

    local_source_codes: List[ThemeSourceCode] = [
        ThemeSourceCode(
            version=code.version,
            code_id=to_local_code_id(code.code_id),
            prompt_id=code.prompt_id,
            question_id=code.question_id,
            tag=code.tag,
            code_name=code.code_name,
            description=code.description,
            representative_excerpt=code.representative_excerpt,
        )
        for code in version_corpus.source_codes
    ]

    projected_themes: List[GeneratedTheme] = []
    for theme in combined_result.themes:
        local_code_ids: List[str] = []
        for scoped_code_id in theme.code_ids:
            if scoped_code_id.startswith(prefix):
                local_code_id = to_local_code_id(scoped_code_id)
                if local_code_id not in local_code_ids:
                    local_code_ids.append(local_code_id)
        if not local_code_ids:
            continue
        projected_themes.append(GeneratedTheme(
            theme_name=theme.theme_name,
            description=theme.description,
            code_ids=local_code_ids,
        ))

    return ThemeGenerationResult(
        version=version_corpus.version,
        question_count=version_corpus.question_count,
        code_count=len(local_source_codes),
        themes=projected_themes,
        source_codes=local_source_codes,
    )


def main() -> None:
    args = build_parser().parse_args()
    verbose = args.verbose and not args.quiet
    use_combined_input = not args.per_version_input

    requested_versions: List[str] = args.versions or ALL_VERSIONS
    versions = [version for version in requested_versions if (DATA_ROOT / version).is_dir()]
    missing = set(requested_versions) - set(versions)
    if missing:
        logger.warning("Version folders not found, skipping: %s", ", ".join(sorted(missing)))
    if not versions:
        logger.error("No valid version folders found under %s", DATA_ROOT)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.dry_run:
        print("\nDRY RUN — no API calls will be made\n")
        version_corpora = [load_version(DATA_ROOT, version, question_ids=args.questions) for version in versions]
        if args.limit:
            version_corpora = version_corpora[:args.limit]
            versions = [corpus.version for corpus in version_corpora]

        if use_combined_input:
            combined_corpus = combine_versions(DATA_ROOT, versions, question_ids=args.questions)
            print(
                f"  combined input: {len(combined_corpus.source_codes)} Step 2 codes "
                f"across {combined_corpus.question_count} version-question pairs"
            )
            for corpus in version_corpora:
                print(
                    f"  projected output → {corpus.version}: {len(corpus.source_codes)} Step 2 codes "
                    f"across {corpus.question_count} questions → {_output_path(corpus.version, output_dir)}"
                )
        else:
            for corpus in version_corpora:
                print(
                    f"  {corpus.version}: {len(corpus.source_codes)} Step 2 codes "
                    f"across {corpus.question_count} questions → {_output_path(corpus.version, output_dir)}"
                )
        print()
        return

    processor = GeminiThemeGenerationBatchProcessor(
        api_key=args.api_key or os.environ.get("GOOGLE_API_KEY"),
        model=args.model,
        prompt_file=str(_prompt_file_for_mode(use_combined_input=use_combined_input)),
    )

    version_corpora = [load_version(DATA_ROOT, version, question_ids=args.questions) for version in versions]
    if args.limit:
        version_corpora = version_corpora[:args.limit]
        versions = [corpus.version for corpus in version_corpora]

    total_saved = 0

    if use_combined_input:
        combined_corpus = combine_versions(DATA_ROOT, versions, question_ids=args.questions)
        combined_input_text = processor._generator.build_user_message(combined_corpus)

        for version in versions:
            input_log_path = _input_log_path(version, output_dir)
            input_log_path.parent.mkdir(parents=True, exist_ok=True)
            input_log_path.write_text(combined_input_text, encoding="utf-8")
            logger.info("Saved combined Step 3 input log to %s", input_log_path)

        combined_results = processor.process([combined_corpus], verbose=verbose, raw_log_root=_raw_log_root(output_dir))
        combined_result = combined_results[0] if combined_results else ThemeGenerationResult(
            version=_COMBINED_CORPUS_LABEL,
            question_count=0,
            code_count=0,
            themes=[],
            source_codes=[],
        )

        combined_raw_log_path = _raw_log_root(output_dir) / _COMBINED_CORPUS_LABEL / "theme_generation_raw_response.json"
        for corpus in version_corpora:
            projected_result = _project_combined_result(combined_result, corpus)
            out_path = _output_path(projected_result.version, output_dir)
            save_results([projected_result], out_path)
            _copy_text_file(combined_raw_log_path, out_path.parent / "theme_generation_raw_response.json")
            print(f"\n  ✓ 1 projected result saved → {out_path}")
            total_saved += 1
    else:
        for corpus in version_corpora:
            input_log_path = _input_log_path(corpus.version, output_dir)
            input_log_path.parent.mkdir(parents=True, exist_ok=True)
            input_log_path.write_text(processor._generator.build_user_message(corpus), encoding="utf-8")
            logger.info("Saved Step 3 input log to %s", input_log_path)

        results = processor.process(version_corpora, verbose=verbose, raw_log_root=_raw_log_root(output_dir))
        for result in results:
            out_path = _output_path(result.version, output_dir)
            save_results([result], out_path)
            print(f"\n  ✓ 1 result saved → {out_path}")
            total_saved += 1

    print(f"\n{'─' * 60}")
    print(f"  Done. Total results saved: {total_saved}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()