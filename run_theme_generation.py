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
from theme_generation_analysis.data_loader import load_version
from theme_generation_analysis.pipeline import save_results

DATA_ROOT = Path(__file__).parent / "data"
ALL_VERSIONS = ["Baseline V0", "V1", "V2", "V1+2"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_theme_generation")


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


def main() -> None:
    args = build_parser().parse_args()
    verbose = args.verbose and not args.quiet

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
        corpora = []
        for version in versions:
            corpora.append(load_version(DATA_ROOT, version, question_ids=args.questions))
        if args.limit:
            corpora = corpora[:args.limit]
        for corpus in corpora:
            print(
                f"  {corpus.version}: {len(corpus.source_codes)} Step 2 codes "
                f"across {corpus.question_count} questions → {_output_path(corpus.version, output_dir)}"
            )
        print()
        return

    processor = GeminiThemeGenerationBatchProcessor(
        api_key=args.api_key or os.environ.get("GOOGLE_API_KEY"),
        model=args.model,
    )

    corpora = [load_version(DATA_ROOT, version, question_ids=args.questions) for version in versions]
    if args.limit:
        corpora = corpora[:args.limit]

    for corpus in corpora:
        input_log_path = _input_log_path(corpus.version, output_dir)
        input_log_path.parent.mkdir(parents=True, exist_ok=True)
        input_log_path.write_text(processor._generator._build_user_message(corpus), encoding="utf-8")
        logger.info("Saved Step 3 input log to %s", input_log_path)

    total_saved = 0
    results = processor.process(corpora, verbose=verbose, raw_log_root=_raw_log_root(output_dir))
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