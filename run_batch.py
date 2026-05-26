"""
run_batch.py — Official batch tagging run across all chatbots and versions.

Discovers every source file under data/, tags all answers through Claude,
and writes the result to tagged.<filename>.json in the same folder.

Usage
─────
    # Full run (all versions, all bots, all answers)
    ANTHROPIC_API_KEY=sk-ant-... python run_batch.py

    # Validation run — only first 5 answers per file
    python run_batch.py --limit 5

    # Specific versions and/or bots
    python run_batch.py --versions "Baseline V0" V1 --bots chatgpt openai

    # Dry-run: show what would be processed without calling the API
    python run_batch.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from answer_analysis.data_loader import load_file
from answer_analysis.pipeline import tagged_answer_to_dict
from answer_analysis.tagger import AnswerTagger

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).parent / "data"

# Folders to scan (in processing order)
ALL_VERSIONS = ["Baseline V0", "V1", "V2", "V1+2"]

# File stem fragments that identify each bot
ALL_BOTS = ["chatgpt", "openai", "gemini", "perplexity"]


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_files(
    versions: list[str],
    bots: list[str],
) -> list[tuple[Path, Path]]:
    """
    Return a list of (source_file, output_file) pairs.
    Skips files whose stem starts with 'tagged.' (already processed outputs).
    """
    pairs: list[tuple[Path, Path]] = []

    for version in versions:
        folder = DATA_ROOT / version
        if not folder.is_dir():
            print(f"  [warn] version folder not found, skipping: {folder}")
            continue

        for src in sorted(folder.iterdir()):
            if src.suffix.lower() not in (".txt", ".json"):
                continue
            if src.stem.startswith("tagged."):
                continue  # skip previous outputs

            # Only include if any bot name appears in the file stem
            stem_lower = src.stem.lower()
            if not any(bot in stem_lower for bot in bots):
                continue

            out = src.parent / f"tagged.{src.stem}.json"
            pairs.append((src, out))

    return pairs


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch answer tagging runner")
    p.add_argument(
        "--versions", "-V",
        nargs="+",
        default=ALL_VERSIONS,
        metavar="VERSION",
        help=f"Version folders to process (default: all). Choices: {ALL_VERSIONS}",
    )
    p.add_argument(
        "--bots", "-b",
        nargs="+",
        default=ALL_BOTS,
        metavar="BOT",
        help=f"Bot name substrings to include (default: all). Choices: {ALL_BOTS}",
    )
    p.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        metavar="N",
        help="Max answers to process per file (default: all). Use 5–10 for validation.",
    )
    p.add_argument(
        "--model", "-m",
        default="gemini-3.5-flash",
        help="Model to use (default: gemini-3.5-flash)",
    )
    p.add_argument(
        "--provider", "-p",
        default="gemini",
        choices=["gemini", "anthropic"],
        help="LLM provider to use (default: gemini)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without calling the API.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-answer progress.",
    )
    p.add_argument(
        "--batch-api",
        action="store_true",
        help=(
            "Submit all answers as one batch job (no per-request 503s). "
            "Uses Gemini Batch API by default; pair with --provider anthropic "
            "for Anthropic Message Batches (50%% cheaper input tokens)."
        ),
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── file discovery ────────────────────────────────────────────────────────
    pairs = discover_files(args.versions, args.bots)

    if not pairs:
        print("No files matched the given versions/bots filter.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'═'*64}")
    print(f"  Batch tagging run")
    print(f"  Versions : {args.versions}")
    print(f"  Bots     : {args.bots}")
    print(f"  Limit    : {args.limit if args.limit else 'all answers'} per file")
    _provider_label = args.provider + (" [batch]" if args.batch_api else "")
    print(f"  Provider : {_provider_label}")
    print(f"  Model    : {args.model}")
    print(f"  Batch API: {'yes' if args.batch_api else 'no'}")
    print(f"  Files    : {len(pairs)}")
    print(f"{'═'*64}\n")

    for src, out in pairs:
        version = src.parent.name
        print(f"  {version} / {src.name}  →  {out.name}")

    if args.dry_run:
        print("\n[dry-run] No API calls made.")
        return

    # ── provider / API key setup ──────────────────────────────────────────────
    from answer_analysis.providers import AnthropicProvider, GeminiProvider

    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "\nERROR: ANTHROPIC_API_KEY is not set.\n"
                "Add it to the .env file:\n  ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)
    else:  # gemini (default)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(
                "\nERROR: GOOGLE_API_KEY is not set.\n"
                "Add it to the .env file:\n  GOOGLE_API_KEY=AIza...",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.batch_api:
        if args.provider == "anthropic":
            from answer_analysis.batch_processor import AnthropicBatchProcessor
            batch_processor = AnthropicBatchProcessor(api_key=api_key, model=args.model)
        else:
            from answer_analysis.batch_processor import GeminiBatchProcessor
            batch_processor = GeminiBatchProcessor(api_key=api_key, model=args.model)
        tagger = None
    elif args.provider == "anthropic":
        provider = AnthropicProvider(api_key=api_key, model=args.model)
        tagger = AnswerTagger(provider=provider)
        batch_processor = None
    else:
        provider = GeminiProvider(api_key=api_key, model=args.model)
        tagger = AnswerTagger(provider=provider)
        batch_processor = None

    # ── per-file processing ───────────────────────────────────────────────────
    total_files   = len(pairs)
    total_answers = 0
    total_spans   = 0
    run_start     = time.monotonic()

    for file_idx, (src, out) in enumerate(pairs, 1):
        version = src.parent.name
        print(f"\n[{file_idx}/{total_files}] {version} / {src.name}")

        raw_answers = load_file(src)
        if args.limit:
            raw_answers = raw_answers[: args.limit]

        n = len(raw_answers)
        print(f"  Processing {n} answer(s) …")

        file_result: dict = {}
        file_spans = 0

        if batch_processor is not None:
            # ── Batch API: one submission for the whole file ───────────────────
            tagged_list = batch_processor.process(raw_answers, verbose=True)
            for tagged in tagged_list:
                file_result[tagged.answer_id] = tagged_answer_to_dict(tagged)
                file_spans += len(tagged.all_spans)
        else:
            # ── Per-answer mode (Gemini / Anthropic sync) ───────────────────
            for ans_idx, raw in enumerate(raw_answers, 1):
                if args.verbose:
                    print(f"  [{ans_idx}/{n}] {raw['answer_id'][-20:]}")
                tagged = tagger.tag_answer(raw, verbose=False)
                file_result[tagged.answer_id] = tagged_answer_to_dict(tagged)
                file_spans += len(tagged.all_spans)

        # Write output JSON (always overwrite)
        out.write_text(
            json.dumps(file_result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        total_answers += n
        total_spans   += file_spans
        print(f"  ✓ {n} answers  |  {file_spans} spans  →  {out}")

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - run_start
    print(f"\n{'═'*64}")
    print(f"  Run complete in {elapsed:.0f}s")
    print(f"  Files processed : {total_files}")
    print(f"  Answers tagged  : {total_answers}")
    print(f"  Total spans     : {total_spans}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
