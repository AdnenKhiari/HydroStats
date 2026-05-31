"""
theme_generation_analysis/data_loader.py

Loads Step 3 inputs by collecting all Step 2 codebook entries for one version.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .models import ThemeGenerationCorpus, ThemeSourceCode

logger = logging.getLogger(__name__)


def _local_code_id(question_id: str, tag: str) -> str:
    return f"{question_id}::{tag}"


def _scoped_code_id(version: str, question_id: str, tag: str) -> str:
    return f"{version}::{question_id}::{tag}"


def _with_prompt_ids(source_codes: List[ThemeSourceCode]) -> List[ThemeSourceCode]:
    assigned: List[ThemeSourceCode] = []
    for index, code in enumerate(source_codes, start=1):
        assigned.append(ThemeSourceCode(
            version=code.version,
            code_id=code.code_id,
            prompt_id=f"C{index}",
            question_id=code.question_id,
            tag=code.tag,
            code_name=code.code_name,
            description=code.description,
            representative_excerpt=code.representative_excerpt,
        ))
    return assigned


def _load_codification_file(path: Path) -> List[Dict[str, object]]:
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
    scoped_code_ids: bool = False,
) -> ThemeGenerationCorpus:
    data_root = Path(data_root)
    codification_path = data_root / version / "codification_analysis.json"
    items = _load_codification_file(codification_path)

    source_codes: List[ThemeSourceCode] = []

    for item in items:
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            logger.warning("Skipping codification item without question_id in %s", codification_path)
            continue
        if question_ids is not None and question_id not in question_ids:
            continue

        raw_codebook = item.get("codebook")
        if not isinstance(raw_codebook, list):
            continue

        for entry in raw_codebook:
            if not isinstance(entry, dict):
                continue
            tag = str(entry.get("tag", "")).strip()
            code_name = str(entry.get("code_name", "")).strip()
            description = str(entry.get("description", "")).strip()
            representative_excerpt = str(entry.get("representative_excerpt", "")).strip()
            if not tag or not code_name:
                continue
            code_id = _scoped_code_id(version, question_id, tag) if scoped_code_ids else _local_code_id(question_id, tag)
            source_codes.append(ThemeSourceCode(
                version=version,
                code_id=code_id,
                question_id=question_id,
                tag=tag,
                code_name=code_name,
                description=description,
                representative_excerpt=representative_excerpt,
            ))

    logger.info("Loaded %d Step 2 codebook entries from version '%s'", len(source_codes), version)
    return ThemeGenerationCorpus(version=version, source_codes=_with_prompt_ids(source_codes))


def load_versions(
    data_root: str | Path,
    versions: Optional[List[str]] = None,
    *,
    question_ids: Optional[List[str]] = None,
    scoped_code_ids: bool = False,
) -> List[ThemeGenerationCorpus]:
    data_root = Path(data_root)
    if versions is None:
        versions = sorted(
            folder.name for folder in data_root.iterdir()
            if folder.is_dir() and not folder.name.startswith(".")
        )

    corpora: List[ThemeGenerationCorpus] = []
    for version in versions:
        corpora.append(load_version(
            data_root,
            version,
            question_ids=question_ids,
            scoped_code_ids=scoped_code_ids,
        ))
    return corpora


def combine_versions(
    data_root: str | Path,
    versions: List[str],
    *,
    question_ids: Optional[List[str]] = None,
) -> ThemeGenerationCorpus:
    combined_codes: List[ThemeSourceCode] = []
    for version in versions:
        corpus = load_version(
            data_root,
            version,
            question_ids=question_ids,
            scoped_code_ids=True,
        )
        combined_codes.extend(corpus.source_codes)

    combined_label = "ALL_VERSIONS"
    logger.info(
        "Loaded %d Step 2 codebook entries across %d versions into combined Step 3 corpus",
        len(combined_codes),
        len(versions),
    )
    return ThemeGenerationCorpus(version=combined_label, source_codes=_with_prompt_ids(combined_codes))