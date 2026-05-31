"""
codification_analysis/data_loader.py

Loads Step 2 codification inputs by joining:
1. the original grouped chatbot answers for a given (question, version), and
2. the thematic analysis previously generated for the same (question, version).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from them_desc_analysis.data_loader import load_version as _load_answer_groups

from .models import CodificationInput

logger = logging.getLogger(__name__)


def _load_thematic_file(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return data


def load_version(
    data_root: str | Path,
    version: str,
    *,
    question_ids: Optional[List[str]] = None,
) -> List[CodificationInput]:
    data_root = Path(data_root)
    version_dir = data_root / version
    thematic_path = version_dir / "thematic_analysis.json"

    answer_groups = _load_answer_groups(data_root, version)
    answer_map = {group.question_id: group for group in answer_groups}

    thematic_items = _load_thematic_file(thematic_path)
    joined: List[CodificationInput] = []

    for item in thematic_items:
        question_id = str(item.get("question_id", ""))
        if not question_id:
            logger.warning("Skipping thematic item without question_id in %s", thematic_path)
            continue
        if question_ids is not None and question_id not in question_ids:
            continue

        group = answer_map.get(question_id)
        if group is None:
            logger.warning(
                "No grouped source answers found for %s in %s; skipping",
                question_id,
                version,
            )
            continue

        raw_analysis = str(item.get("analysis", "")).strip()
        providers = item.get("providers") or group.providers
        joined.append(CodificationInput(
            question_id=question_id,
            version=version,
            providers=[str(provider) for provider in providers],
            raw_analysis=raw_analysis,
            source_answers=group.concatenated_text,
        ))

    logger.info("Loaded %d codification inputs from version '%s'", len(joined), version)
    return joined


def load_versions(
    data_root: str | Path,
    versions: Optional[List[str]] = None,
    *,
    question_ids: Optional[List[str]] = None,
) -> List[CodificationInput]:
    data_root = Path(data_root)
    if versions is None:
        versions = sorted(
            folder.name for folder in data_root.iterdir()
            if folder.is_dir() and not folder.name.startswith(".")
        )

    inputs: List[CodificationInput] = []
    for version in versions:
        inputs.extend(load_version(data_root, version, question_ids=question_ids))
    return inputs
