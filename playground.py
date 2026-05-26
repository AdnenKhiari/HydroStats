"""
playground.py — Quick test harness for the answer tagging pipeline.

Loads one AI response, runs it through Claude, and saves the result to a JSON file.

Usage
─────
    # Tag the first answer from any experiment file (auto-discovers)
    ANTHROPIC_API_KEY=sk-ant-... python playground.py

    # Tag a specific file (first answer in it)
    ANTHROPIC_API_KEY=sk-ant-... python playground.py --file data/V1/openai.responses_c1.txt

    # Tag the Nth answer (0-indexed)
    ANTHROPIC_API_KEY=sk-ant-... python playground.py --index 3

    # Choose experiment folder
    ANTHROPIC_API_KEY=sk-ant-... python playground.py --experiment "Baseline V0"

    # Change output file name
    ANTHROPIC_API_KEY=sk-ant-... python playground.py --out my_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── make the package importable when running from the project root ────────────
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")  # loads ANTHROPIC_API_KEY etc.

from answer_analysis.data_loader import load_directory, load_file
from answer_analysis.pipeline import tagged_answer_to_dict
from answer_analysis.tagger import AnswerTagger

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Answer-tagging playground")
    p.add_argument(
        "--file", "-f",
        default=None,
        help="Path to a single answer file (JSON or txt). "
             "If omitted, auto-discovers the first file under --experiment.",
    )
    p.add_argument(
        "--experiment", "-e",
        default="Baseline V0",
        help="Experiment folder name under data/ (default: 'Baseline V0')",
    )
    p.add_argument(
        "--index", "-i",
        type=int,
        default=0,
        help="0-based index of the answer to process (default: 0)",
    )
    p.add_argument(
        "--out", "-o",
        default=None,
        help="Output JSON file path (default: tagged_<answer_id>.json)",
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
        "--verbose", "-v",
        action="store_true",
        help="Print per-line progress",
    )
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    from answer_analysis.providers import AnthropicProvider, GeminiProvider
    if args.provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(
                "ERROR: ANTHROPIC_API_KEY is not set.\n"
                "Add it to the .env file:\n  ANTHROPIC_API_KEY=sk-ant-...",
                file=sys.stderr,
            )
            sys.exit(1)
        provider = AnthropicProvider(api_key=api_key, model=args.model)
    else:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(
                "ERROR: GOOGLE_API_KEY is not set.\n"
                "Add it to the .env file:\n  GOOGLE_API_KEY=AIza...",
                file=sys.stderr,
            )
            sys.exit(1)
        provider = GeminiProvider(api_key=api_key, model=args.model)

    # ── load answers ──────────────────────────────────────────────────────────
    if args.file:
        source_path = Path(args.file)
        if not source_path.exists():
            print(f"ERROR: file not found: {source_path}", file=sys.stderr)
            sys.exit(1)
        answers = load_file(source_path)
        source_label = str(source_path)
    else:
        experiment_dir = Path("data") / args.experiment
        if not experiment_dir.exists():
            print(f"ERROR: experiment folder not found: {experiment_dir}", file=sys.stderr)
            print("Available experiments:")
            for d in sorted(Path("data").iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
            sys.exit(1)
        answers = load_directory(experiment_dir)
        source_label = str(experiment_dir)

    if not answers:
        print(f"ERROR: no answers found in {source_label}", file=sys.stderr)
        sys.exit(1)

    if args.index >= len(answers):
        print(
            f"ERROR: index {args.index} out of range "
            f"(found {len(answers)} answers in {source_label})",
            file=sys.stderr,
        )
        sys.exit(1)

    raw = answers[args.index]

    print(f"\n{'─'*60}")
    print(f"Source     : {source_label}")
    print(f"Answer #{args.index}")
    print(f"  ID       : {raw['answer_id']}")
    print(f"  Provider : {raw.get('provider', 'unknown')}")
    print(f"  Query ID : {raw.get('query_id', 'n/a')}")
    preview = raw["text"][:200].replace("\n", " ")
    print(f"  Preview  : {preview}…")
    print(f"{'─'*60}\n")

    # ── tag ───────────────────────────────────────────────────────────────────
    tagger = AnswerTagger(provider=provider)

    print(f"Tagging with {args.provider} / {args.model} …")
    tagged = tagger.tag_answer(raw, verbose=args.verbose)

    # ── build output JSON ─────────────────────────────────────────────────────
    # Schema: { answer_id: { full tagged_answer dict } }
    output = {
        tagged.answer_id: tagged_answer_to_dict(tagged)
    }

    out_path = Path(args.out) if args.out else Path(f"tagged_{tagged.answer_id[-12:]}.json")
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Result saved to: {out_path}")
    print(f"  Lines processed : {len(tagged.lines)}")
    print(f"  Signal spans    : {len(tagged.all_spans)}")

    # Quick per-category breakdown
    from answer_analysis.models import Category
    for cat in Category:
        count = len(tagged.spans_for(cat))
        if count:
            print(f"  {cat.value:<22}: {count} span(s)")

    # Print a preview of the tagged output
    print(f"\n{'─'*60}")
    print("Tagged output preview (first 5 non-blank lines):")
    shown = 0
    for line in tagged.lines:
        if not line.raw_text.strip():
            continue
        print(f"\n  [{line.line_index:03d}] {line.raw_text}")
        print(f"       {line.tagged_text}")
        shown += 1
        if shown >= 5:
            break
    print(f"{'─'*60}")
    print(f"\nFull result → {out_path}")


if __name__ == "__main__":
    main()
