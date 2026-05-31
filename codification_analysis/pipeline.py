"""
codification_analysis/pipeline.py

Serialization and optional non-batch pipeline for Step 2 codification.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .codifier import OpenCodifier
from .data_loader import load_version, load_versions
from .models import CodedAnalysis, CodificationInput

logger = logging.getLogger(__name__)


def results_to_json(results: List[CodedAnalysis], *, indent: int = 2) -> str:
    return json.dumps([result.to_dict() for result in results], indent=indent, ensure_ascii=False)


def save_results(results: List[CodedAnalysis], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(results_to_json(results), encoding="utf-8")
    logger.info("Saved %d results to %s", len(results), output_path)


class CodificationPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> None:
        self.codifier = OpenCodifier(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            rpm_limit=rpm_limit,
            prompt_file=prompt_file,
        )
        self.verbose = verbose

    def run_on_inputs(
        self,
        items: List[CodificationInput],
        *,
        limit: Optional[int] = None,
    ) -> List[CodedAnalysis]:
        if limit is not None:
            items = items[:limit]
        results: List[CodedAnalysis] = []
        total = len(items)
        for index, item in enumerate(items, 1):
            if self.verbose:
                print(f"\n[{index}/{total}] {item.version} / {item.question_id}")
            results.append(self.codifier.codify(item, verbose=self.verbose))
        return results

    def run_on_version(
        self,
        data_root: str | Path,
        version: str,
        *,
        question_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[CodedAnalysis]:
        return self.run_on_inputs(load_version(data_root, version, question_ids=question_ids), limit=limit)

    def run_on_versions(
        self,
        data_root: str | Path,
        versions: Optional[List[str]] = None,
        *,
        question_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[CodedAnalysis]:
        return self.run_on_inputs(load_versions(data_root, versions, question_ids=question_ids), limit=limit)
