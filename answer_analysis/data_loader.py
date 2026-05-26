"""
answer_analysis/data_loader.py

Loads raw AI answers from the project's data directory into plain dicts
ready for the tagging pipeline.

Supported file formats
──────────────────────
• JSON  – files that contain a top-level "items" list (Baseline V0, V1, V1+2, V2)
          Each item is expected to have at least: provider, text
          Optional fields: queryKey, key (used as answer_id), sources

Returns a list of RawAnswer dicts with normalised fields so the rest of the
pipeline doesn't have to worry about file-format differences.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Internal type alias
# ──────────────────────────────────────────────────────────────────────────────
RawAnswer = Dict[str, Any]
"""
Keys guaranteed to be present after loading:
    answer_id   str
    query_id    str | None
    model       str | None   (e.g. "gpt-4o")
    provider    str | None   (e.g. "OPENAI", "GEMINI", "PERPLEXITY")
    text        str          full answer text, with the leading "Query: …" prefix stripped
    sources     list[dict]   may be empty
    source_file str          path of the originating file
"""

_PROVIDER_HINT: Dict[str, str] = {
    "chatgpt": "OPENAI",
    "openai":  "OPENAI",
    "gemini":  "GEMINI",
    "perplexity": "PERPLEXITY",
}

# Matches the "Query: User query: …\n\n" prefix that some answers carry
_QUERY_PREFIX_RE = re.compile(
    r"^\s*(?:\*\*)?Query:\s*(?:User query:\s*)?(.*?)\n\n",
    re.DOTALL,
)


def _infer_provider_from_filename(path: Path) -> Optional[str]:
    stem = path.stem.lower()
    for key, val in _PROVIDER_HINT.items():
        if key in stem:
            return val
    return None


def _strip_query_prefix(text: str) -> str:
    """Remove the embedded 'Query: …' header that some answer files include."""
    m = _QUERY_PREFIX_RE.match(text)
    if m:
        return text[m.end():].strip()
    return text.strip()


def _parse_json_file(path: Path) -> Iterator[RawAnswer]:
    """Parse a JSON file that has a top-level 'items' list."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    items = data if isinstance(data, list) else data.get("items", [])
    provider_fallback = _infer_provider_from_filename(path)

    for item in items:
        raw_text = item.get("text", "")
        yield {
            "answer_id":   item.get("key", str(uuid.uuid4())),
            "query_id":    item.get("queryKey"),
            "model":       item.get("model"),
            "provider":    item.get("provider") or provider_fallback,
            "text":        _strip_query_prefix(raw_text),
            "sources":     item.get("sources", []),
            "source_file": str(path),
        }


def _parse_txt_file(path: Path) -> Iterator[RawAnswer]:
    """
    Fallback: treat a plain-text .txt file as a single answer block,
    or split on blank-line-separated blocks if multiple answers are present.
    Some .txt files in this project are actually JSON – handle that first.
    """
    provider_fallback = _infer_provider_from_filename(path)
    raw = path.read_text(encoding="utf-8").strip()

    # Try JSON first (some .txt files are actually JSON)
    try:
        data = json.loads(raw)
        items = data if isinstance(data, list) else data.get("items", [])
        for item in items:
            raw_text = item.get("text", "")
            yield {
                "answer_id":   item.get("key", str(uuid.uuid4())),
                "query_id":    item.get("queryKey"),
                "model":       item.get("model"),
                "provider":    item.get("provider") or provider_fallback,
                "text":        _strip_query_prefix(raw_text),
                "sources":     item.get("sources", []),
                "source_file": str(path),
            }
        return
    except json.JSONDecodeError:
        pass

    # Plain text: split on double-newline blocks
    blocks = [b.strip() for b in re.split(r"\n{2,}", raw) if b.strip()]
    for i, block in enumerate(blocks):
        yield {
            "answer_id":   f"{path.stem}_{i}",
            "query_id":    None,
            "model":       None,
            "provider":    provider_fallback,
            "text":        _strip_query_prefix(block),
            "sources":     [],
            "source_file": str(path),
        }


def load_file(path: str | Path) -> List[RawAnswer]:
    """Load all answers from a single file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".json":
        return list(_parse_json_file(path))
    # .txt may still be JSON inside
    return list(_parse_txt_file(path))


def load_directory(
    directory: str | Path,
    *,
    recursive: bool = True,
    extensions: tuple[str, ...] = (".json", ".txt"),
) -> List[RawAnswer]:
    """
    Load every supported file under *directory*.

    Parameters
    ----------
    directory : path to a folder (e.g. "data/V1")
    recursive : whether to descend into sub-folders
    extensions: file extensions to include
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pattern = "**/*" if recursive else "*"
    answers: List[RawAnswer] = []

    for ext in extensions:
        for fp in sorted(directory.glob(pattern + ext)):
            answers.extend(load_file(fp))

    return answers


def load_experiment(
    data_root: str | Path,
    experiment: str,
) -> List[RawAnswer]:
    """
    Convenience: load a named experiment folder.

    Example
    -------
    >>> answers = load_experiment("data", "V1")
    >>> answers = load_experiment("data", "Baseline V0")
    """
    return load_directory(Path(data_root) / experiment)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent / "data"
    for experiment in ["Baseline V0", "V1", "V2", "V1+2"]:
        folder = root / experiment
        if not folder.exists():
            continue
        answers = load_directory(folder)
        print(f"{experiment}: {len(answers)} answers loaded")
        if answers:
            a = answers[0]
            preview = a["text"][:120].replace("\n", " ")
            print(f"  [{a['provider']}] {preview}…")
    sys.exit(0)
