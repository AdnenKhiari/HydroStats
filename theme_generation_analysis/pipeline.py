"""
theme_generation_analysis/pipeline.py

Serialization and optional non-batch pipeline for Step 3 theme generation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from .data_loader import load_version, load_versions
from .generator import ThemeGenerator
from .models import ThemeGenerationCorpus, ThemeGenerationResult

logger = logging.getLogger(__name__)


def results_to_json(results: List[ThemeGenerationResult], *, indent: int = 2) -> str:
    return json.dumps([result.to_dict() for result in results], indent=indent, ensure_ascii=False)


def save_results(results: List[ThemeGenerationResult], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(results_to_json(results), encoding="utf-8")
    logger.info("Saved %d results to %s", len(results), output_path)


class ThemeGenerationPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        verbose: bool = True,
    ) -> None:
        self.generator = ThemeGenerator(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            rpm_limit=rpm_limit,
            prompt_file=prompt_file,
        )
        self.verbose = verbose

    def run_on_corpora(
        self,
        corpora: List[ThemeGenerationCorpus],
        *,
        limit: Optional[int] = None,
    ) -> List[ThemeGenerationResult]:
        if limit is not None:
            corpora = corpora[:limit]
        results: List[ThemeGenerationResult] = []
        total = len(corpora)
        for index, corpus in enumerate(corpora, 1):
            if self.verbose:
                print(f"\n[{index}/{total}] {corpus.version}")
            results.append(self.generator.generate(corpus, verbose=self.verbose))
        return results

    def run_on_version(
        self,
        data_root: str | Path,
        version: str,
        *,
        question_ids: Optional[List[str]] = None,
    ) -> List[ThemeGenerationResult]:
        return self.run_on_corpora([load_version(data_root, version, question_ids=question_ids)])

    def run_on_versions(
        self,
        data_root: str | Path,
        versions: Optional[List[str]] = None,
        *,
        question_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[ThemeGenerationResult]:
        return self.run_on_corpora(load_versions(data_root, versions, question_ids=question_ids), limit=limit)