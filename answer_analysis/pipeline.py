"""
answer_analysis/pipeline.py

Orchestrates the full tagging pipeline:
  load answers → tag each line with Claude → collect TaggedAnswer objects

Also provides a pretty-print helper and a JSON serialiser for inspection.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .data_loader import RawAnswer, load_directory, load_experiment, load_file
from .models import Category, TaggedAnswer, TaggedLine, TaggedSpan
from .tagger import AnswerTagger


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helpers  (dataclasses → plain dicts for JSON / display)
# ──────────────────────────────────────────────────────────────────────────────

def span_to_dict(span: TaggedSpan) -> Dict[str, Any]:
    return {
        "text":       span.text,
        "category":   span.category.value if span.category else None,
        "char_start": span.char_start,
        "char_end":   span.char_end,
    }


def line_to_dict(line: TaggedLine) -> Dict[str, Any]:
    return {
        "line_index":  line.line_index,
        "raw_text":    line.raw_text,
        "tagged_text": line.tagged_text,
        "spans":       [span_to_dict(s) for s in line.spans],
    }


def tagged_answer_to_dict(ta: TaggedAnswer) -> Dict[str, Any]:
    return {
        "answer_id": ta.answer_id,
        "query_id":  ta.query_id,
        "model":     ta.model,
        "provider":  ta.provider,
        "lines":     [line_to_dict(ln) for ln in ta.lines],
        "summary":   ta.summary(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────────────

_CATEGORY_COLORS = {
    Category.STATISTICAL_USE:      "\033[94m",   # blue
    Category.CREDIBILITY_SIGNAL:   "\033[92m",   # green
    Category.STRONG_RECOMMENDATION: "\033[93m",  # yellow
    Category.BRAND_POSITIONING:    "\033[95m",   # magenta
}
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def pretty_print(tagged_answer: TaggedAnswer, *, use_color: bool = True) -> None:
    """Print a TaggedAnswer to stdout in a readable format."""

    def c(cat: Category, text: str) -> str:
        if not use_color:
            return text
        return f"{_CATEGORY_COLORS.get(cat, '')}{text}{_RESET}"

    print(f"\n{'═'*70}")
    print(f"{_BOLD}Answer {tagged_answer.answer_id}{_RESET}")
    print(f"  Provider : {tagged_answer.provider}   Model: {tagged_answer.model}")
    print(f"  Query ID : {tagged_answer.query_id}")
    print(f"{'─'*70}")

    for line in tagged_answer.lines:
        if not line.raw_text.strip():
            continue
        print(f"\n  Line {line.line_index:03d}: {line.raw_text}")
        # Render the tagged_text with colour substitutions
        coloured = line.tagged_text
        for cat in Category:
            open_tag  = f"<{cat.tag}>"
            close_tag = f"</{cat.tag}>"
            colour = _CATEGORY_COLORS.get(cat, "")
            coloured = coloured.replace(open_tag,  f"{colour}[{cat.value}]↦")
            coloured = coloured.replace(close_tag, _RESET)
        coloured = re.sub(r"</?null>", "", coloured)
        print(f"         {coloured}")

    print(f"\n{'─'*70}")
    s = tagged_answer.summary()
    print(f"  Summary: {s['total_lines']} lines | spans: {s['span_counts']}")
    print(f"{'═'*70}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ──────────────────────────────────────────────────────────────────────────────

class TaggingPipeline:
    """
    End-to-end pipeline: load → tag → collect results.

    Parameters
    ----------
    api_key      : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    model        : Claude model identifier
    max_tokens   : Claude response token budget per line
    rpm_limit    : minimum seconds between API calls (crude rate limiter)
    skip_short   : skip lines shorter than this many chars
    verbose      : print progress while tagging
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        max_tokens: int = 1024,
        rpm_limit: float = 1.0,
        skip_short: int = 10,
        verbose: bool = True,
    ) -> None:
        self.tagger = AnswerTagger(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            rpm_limit=rpm_limit,
            skip_short=skip_short,
        )
        self.verbose = verbose

    def run_on_answers(self, raw_answers: List[RawAnswer]) -> List[TaggedAnswer]:
        """Tag a pre-loaded list of RawAnswer dicts."""
        results: List[TaggedAnswer] = []
        total = len(raw_answers)
        for i, raw in enumerate(raw_answers, 1):
            if self.verbose:
                print(f"\n[{i}/{total}] Tagging answer {raw['answer_id']} "
                      f"({raw.get('provider', '?')}) …")
            tagged = self.tagger.tag_answer(raw, verbose=self.verbose)
            results.append(tagged)
        return results

    def run_on_file(self, file_path: str | Path) -> List[TaggedAnswer]:
        """Load a single file and tag all answers in it."""
        raw_answers = load_file(file_path)
        return self.run_on_answers(raw_answers)

    def run_on_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
    ) -> List[TaggedAnswer]:
        """Load all files under *directory* and tag every answer."""
        raw_answers = load_directory(directory, recursive=recursive)
        return self.run_on_answers(raw_answers)

    def run_on_experiment(
        self,
        data_root: str | Path,
        experiment: str,
    ) -> List[TaggedAnswer]:
        """Convenience: tag all answers in a named experiment folder."""
        raw_answers = load_experiment(data_root, experiment)
        return self.run_on_answers(raw_answers)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test entry-point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Usage:
        ANTHROPIC_API_KEY=sk-ant-... python -m answer_analysis.pipeline

    Tags the first answer from each experiment and prints a summary.
    Set N_ANSWERS_PER_EXPERIMENT to control how many are processed.
    """
    import sys

    N_ANSWERS_PER_EXPERIMENT = 1  # keep costs low during testing

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: set ANTHROPIC_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    data_root = Path(__file__).parent.parent / "data"
    pipeline = TaggingPipeline(api_key=api_key, verbose=True)

    for experiment in ["Baseline V0", "V1", "V2"]:
        folder = data_root / experiment
        if not folder.exists():
            print(f"Skipping {experiment} (folder not found)")
            continue

        from .data_loader import load_directory as _ld
        raw = _ld(folder)[:N_ANSWERS_PER_EXPERIMENT]
        if not raw:
            continue

        print(f"\n{'#'*70}")
        print(f"# Experiment: {experiment}  ({len(raw)} answer(s))")
        print(f"{'#'*70}")

        tagged_list = pipeline.run_on_answers(raw)
        for ta in tagged_list:
            pretty_print(ta)
            # Also dump JSON for inspection
            out_path = Path(f"tagged_{experiment.replace(' ', '_')}_{ta.answer_id[-8:]}.json")
            out_path.write_text(
                json.dumps(tagged_answer_to_dict(ta), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  → saved to {out_path}")
