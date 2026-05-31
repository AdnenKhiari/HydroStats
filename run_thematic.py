"""
run_thematic.py — Gemini Batch thematic analysis across versions and questions.

For each (question, version) pair, concatenates the 3 chatbot answers and
submits them through the Gemini Batch API for inductive thematic analysis.
Results are saved as
``thematic_analysis.json`` inside each version folder.

Usage
─────
    # Full run — all versions, all questions
    GOOGLE_API_KEY=... python run_thematic.py

    # Specific versions only
    python run_thematic.py --versions V1 "Baseline V0"

    # Specific questions (by queryKey)
    python run_thematic.py --questions query_01khxksa6nenarkcx5nrapv411

    # Limit number of questions per version (useful for testing)
    python run_thematic.py --limit 3

    # Dry-run: show what would be processed without calling the API
    python run_thematic.py --dry-run

    # Choose model and output location
    python run_thematic.py --model gemini-2.5-flash --output-dir results/thematic
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

from them_desc_analysis.batch_processor import GeminiThematicBatchProcessor
from them_desc_analysis.pipeline import save_results

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT    = Path(__file__).parent / "data"
ALL_VERSIONS = ["Baseline V0", "V1", "V2", "V1+2"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_thematic")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gemini Batch thematic analysis of grouped AI chatbot answers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--versions", nargs="+", default=None,
        metavar="VERSION",
        help="Version folders to process (default: all discovered).",
    )
    p.add_argument(
        "--questions", nargs="+", default=None,
        metavar="QUERY_KEY",
        help="Restrict processing to these question IDs (queryKey values).",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Process only the first N questions per version (useful for testing).",
    )
    p.add_argument(
        "--model", default="gemini-2.5-flash",
        help="LLM model identifier (default: gemini-2.5-flash).",
    )
    p.add_argument(
        "--api-key", default=None,
        help="API key (defaults to GOOGLE_API_KEY env var).",
    )
    p.add_argument(
        "--output-dir", default=None,
        metavar="DIR",
        help=(
            "Directory to write output JSON files. "
            "Default: same folder as the version data (data/{version}/thematic_analysis.json)."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without making any API calls.",
    )
    p.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print per-question progress (default: on).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-question progress output.",
    )
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Output path helper
# ──────────────────────────────────────────────────────────────────────────────

def _output_path(version: str, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir / version / "thematic_analysis.json"
    return DATA_ROOT / version / "thematic_analysis.json"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    verbose = args.verbose and not args.quiet

    # Resolve versions
    requested_versions: List[str] = args.versions or ALL_VERSIONS
    versions: List[str] = [
        v for v in requested_versions
        if (DATA_ROOT / v).is_dir()
    ]
    missing = set(requested_versions) - set(versions)
    if missing:
        logger.warning("Version folders not found, skipping: %s", ", ".join(sorted(missing)))
    if not versions:
        logger.error("No valid version folders found under %s", DATA_ROOT)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else None

    # ── dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        from them_desc_analysis.data_loader import load_version
        print("\nDRY RUN — no API calls will be made\n")
        for version in versions:
            groups = load_version(DATA_ROOT, version)
            if args.questions:
                groups = [g for g in groups if g.question_id in args.questions]
            if args.limit:
                groups = groups[:args.limit]
            out = _output_path(version, output_dir)
            print(f"  {version}: {len(groups)} questions → {out}")
        print()
        return

    # ── real run ──────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")

    processor = GeminiThematicBatchProcessor(
        api_key=api_key,
        model=args.model,
    )

    # Process version by version so we can save incrementally
    total_saved = 0
    for version in versions:
        print(f"\n{'═'*60}")
        print(f"  Version: {version}")
        print(f"{'═'*60}")

        from them_desc_analysis.data_loader import load_version
        groups = load_version(DATA_ROOT, version)

        if args.questions:
            groups = [g for g in groups if g.question_id in args.questions]

        if not groups:
            logger.warning("No groups found for version '%s' — skipping", version)
            continue

        if args.limit:
            groups = groups[:args.limit]

        results = processor.process(groups, verbose=verbose)

        out_path = _output_path(version, output_dir)
        save_results(results, out_path)
        print(f"\n  ✓ {len(results)} results saved → {out_path}")
        total_saved += len(results)

    print(f"\n{'─'*60}")
    print(f"  Done. Total results saved: {total_saved}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
