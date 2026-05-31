"""
run_codification.py — Gemini Batch open codification across versions and questions.

Consumes `thematic_analysis.json` plus the original grouped chatbot answers for
matching question/version units and writes `codification_analysis.json`.
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

from codification_analysis.batch_processor import GeminiCodificationBatchProcessor
from codification_analysis.data_loader import load_version
from codification_analysis.pipeline import save_results

DATA_ROOT = Path(__file__).parent / "data"
ALL_VERSIONS = ["Baseline V0", "V1", "V2", "V1+2"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_codification")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gemini Batch open codification over thematic analyses.",
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
        return output_dir / version / "codification_analysis.json"
    return DATA_ROOT / version / "codification_analysis.json"


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
        for version in versions:
            items = load_version(DATA_ROOT, version, question_ids=args.questions)
            if args.limit:
                items = items[:args.limit]
            print(f"  {version}: {len(items)} codification inputs → {_output_path(version, output_dir)}")
        print()
        return

    processor = GeminiCodificationBatchProcessor(
        api_key=args.api_key or os.environ.get("GOOGLE_API_KEY"),
        model=args.model,
    )

    total_saved = 0
    for version in versions:
        print(f"\n{'═' * 60}")
        print(f"  Version: {version}")
        print(f"{'═' * 60}")

        items = load_version(DATA_ROOT, version, question_ids=args.questions)
        if not items:
            logger.warning("No codification inputs found for version '%s' — skipping", version)
            continue
        if args.limit:
            items = items[:args.limit]

        results = processor.process(items, verbose=verbose)
        out_path = _output_path(version, output_dir)
        save_results(results, out_path)
        print(f"\n  ✓ {len(results)} results saved → {out_path}")
        total_saved += len(results)

    print(f"\n{'─' * 60}")
    print(f"  Done. Total results saved: {total_saved}")
    print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
