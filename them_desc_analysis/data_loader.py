"""
them_desc_analysis/data_loader.py

Loads and groups raw chatbot answers by (question, version) so they can be
sent together to the thematic analysis LLM.

Strategy
────────
• Re-uses ``answer_analysis.data_loader.load_file`` to read individual chatbot
  answer files — no duplication of parsing logic.
• Answers within a version are grouped by ``query_id`` (the ``queryKey`` field).
  When ``query_id`` is missing, positional order (file index) is used as fallback.
• The three chatbot answers are concatenated with labelled headers so the LLM
  knows which AI produced each response.
• If a version has fewer than three chatbots, the available ones are used and
  a warning is logged.

Typical directory layout expected:
    data/
      V1/
        gemini.responses_c1.txt   (or .json)
        openai.responses_c1.txt
        perplexity.responses_c1.txt
      Baseline V0/
        chatgpt.answers.txt
        gemini.answers.txt
        perplexity.answers.txt
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from answer_analysis.data_loader import load_file as _load_file
from .models import AnswerGroup

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_BOT_HINTS: Dict[str, str] = {
    "chatgpt":    "ChatGPT",
    "openai":     "ChatGPT",
    "gemini":     "Gemini",
    "perplexity": "Perplexity",
}

_SEPARATOR = "\n\n" + "─" * 60 + "\n\n"


def _infer_label(path: Path) -> str:
    """Return a human-readable chatbot label from the file name."""
    stem = path.stem.lower()
    for hint, label in _BOT_HINTS.items():
        if hint in stem:
            return label
    return path.stem   # fallback: use the file stem as-is


# ──────────────────────────────────────────────────────────────────────────────
# Core grouping logic
# ──────────────────────────────────────────────────────────────────────────────

def _group_by_question(
    version_dir: Path,
) -> List[AnswerGroup]:
    """
    Load all chatbot files in *version_dir*, group answers by question,
    and return one AnswerGroup per question.

    Parameters
    ----------
    version_dir : path to a version folder (e.g. ``data/V1``).
    """
    if not version_dir.is_dir():
        raise NotADirectoryError(version_dir)

    # Discover chatbot source files (skip tagged outputs)
    source_files: List[Path] = sorted(
        fp for fp in version_dir.iterdir()
        if fp.suffix.lower() in (".txt", ".json")
        and not fp.stem.startswith("tagged.")
        and not fp.stem.startswith("thematic")
        and not fp.stem.startswith("codification")
    )

    if not source_files:
        logger.warning("No source files found in %s", version_dir)
        return []

    # Load each file and track label → {question_id → answer_text}
    # Structure: per_file[label] = list of (question_id, text)
    per_file: List[Tuple[str, List[Tuple[str, str]]]] = []

    for fp in source_files:
        label = _infer_label(fp)
        try:
            raw_answers = _load_file(fp)
        except Exception as exc:
            logger.warning("Could not load %s: %s", fp, exc)
            continue

        entries: List[Tuple[str, str]] = []
        for idx, raw in enumerate(raw_answers):
            q_id = raw.get("query_id") or f"q_{idx:04d}"
            text = raw.get("text", "").strip()
            entries.append((q_id, text))

        per_file.append((label, entries))

    if not per_file:
        return []

    # Use the first file's question order as the canonical list
    canonical_label, canonical_entries = per_file[0]
    canonical_ids = [qid for qid, _ in canonical_entries]

    # Build a mapping: label → {question_id → text}
    text_map: Dict[str, Dict[str, str]] = {}
    for label, entries in per_file:
        if len(entries) != len(canonical_ids):
            logger.warning(
                "%s has %d entries but canonical (%s) has %d — using positional alignment",
                label, len(entries), canonical_label, len(canonical_ids),
            )
        mapping: Dict[str, str] = {}
        for i, (qid, text) in enumerate(entries):
            # Prefer queryKey match; fall back to position-aligned canonical id
            resolved_id = qid if qid in canonical_ids else (
                canonical_ids[i] if i < len(canonical_ids) else qid
            )
            mapping[resolved_id] = text
        text_map[label] = mapping

    # Build AnswerGroups
    version = version_dir.name
    groups: List[AnswerGroup] = []
    labels_present = [label for label, _ in per_file]

    for q_id in canonical_ids:
        parts: List[str] = []
        providers_present: List[str] = []

        for label in labels_present:
            answer_text = text_map.get(label, {}).get(q_id, "")
            if answer_text:
                parts.append(f"[{label}]:\n{answer_text}")
                providers_present.append(label)
            else:
                logger.debug("No answer for question %s from %s in %s", q_id, label, version)

        if not parts:
            logger.warning("All chatbots missing for question %s in %s — skipping", q_id, version)
            continue

        concatenated = _SEPARATOR.join(parts)

        groups.append(AnswerGroup(
            question_id=q_id,
            version=version,
            providers=providers_present,
            concatenated_text=concatenated,
        ))

    return groups


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def load_version(
    data_root: str | Path,
    version: str,
) -> List[AnswerGroup]:
    """
    Load and group all chatbot answers for one version.

    Parameters
    ----------
    data_root : root data directory (e.g. ``Path("data")``).
    version   : folder name, e.g. ``"V1"`` or ``"Baseline V0"``.
    """
    return _group_by_question(Path(data_root) / version)


def load_versions(
    data_root: str | Path,
    versions: Optional[List[str]] = None,
    *,
    question_ids: Optional[List[str]] = None,
) -> List[AnswerGroup]:
    """
    Load and group chatbot answers across multiple versions.

    Parameters
    ----------
    data_root    : root data directory.
    versions     : list of version names to include.  ``None`` → all discovered.
    question_ids : if given, only include groups matching these question IDs.
    """
    data_root = Path(data_root)

    if versions is None:
        versions = sorted(
            d.name for d in data_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    all_groups: List[AnswerGroup] = []
    for version in versions:
        groups = load_version(data_root, version)
        if question_ids is not None:
            groups = [g for g in groups if g.question_id in question_ids]
        all_groups.extend(groups)
        logger.info("Loaded %d groups from version '%s'", len(groups), version)

    return all_groups
