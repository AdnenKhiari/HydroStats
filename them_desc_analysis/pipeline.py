"""
them_desc_analysis/pipeline.py

Orchestrates the full thematic analysis pipeline:
  load grouped answers → analyse each group → collect ThematicResult objects.

Also provides a JSON serialiser for saving results to disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analyzer import ThematicAnalyzer
from .data_loader import load_version, load_versions
from .models import AnswerGroup, ThematicResult

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

def results_to_json(results: List[ThematicResult], *, indent: int = 2) -> str:
    """Serialise a list of ThematicResult objects to a JSON string."""
    return json.dumps([r.to_dict() for r in results], indent=indent, ensure_ascii=False)


def save_results(
    results: List[ThematicResult],
    output_path: str | Path,
) -> None:
    """Write results as a JSON array to *output_path* (creates parent dirs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(results_to_json(results), encoding="utf-8")
    logger.info("Saved %d results to %s", len(results), output_path)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class ThematicPipeline:
    """
    End-to-end pipeline: load → analyse → collect results.

    Parameters
    ----------
    api_key    : API key for the default provider (Gemini).
    model      : Model identifier.
    max_tokens : Max tokens the model may generate per call.
    rpm_limit  : Minimum seconds between consecutive API calls.
    prompt_file: Override path to the system-prompt file.
    verbose    : Print progress while analysing.
    provider   : Custom LLMProvider (overrides api_key / model).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        verbose: bool = True,
        provider: Optional[Any] = None,
    ) -> None:
        self.analyzer = ThematicAnalyzer(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            rpm_limit=rpm_limit,
            prompt_file=prompt_file,
            provider=provider,
        )
        self.verbose = verbose

    # ── core runner ───────────────────────────────────────────────────────────

    def run_on_groups(
        self,
        groups: List[AnswerGroup],
        *,
        limit: Optional[int] = None,
    ) -> List[ThematicResult]:
        """Analyse a pre-loaded list of AnswerGroup objects."""
        if limit is not None:
            groups = groups[:limit]

        results: List[ThematicResult] = []
        total = len(groups)

        for i, group in enumerate(groups, 1):
            if self.verbose:
                print(f"\n[{i}/{total}] {group.version} / {group.question_id}")
            result = self.analyzer.analyze(group, verbose=self.verbose)
            results.append(result)

        return results

    # ── convenience runners ───────────────────────────────────────────────────

    def run_on_version(
        self,
        data_root: str | Path,
        version: str,
        *,
        question_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[ThematicResult]:
        """Load a single version and analyse all questions."""
        groups = load_version(data_root, version)
        if question_ids:
            groups = [g for g in groups if g.question_id in question_ids]
        return self.run_on_groups(groups, limit=limit)

    def run_on_versions(
        self,
        data_root: str | Path,
        versions: Optional[List[str]] = None,
        *,
        question_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[ThematicResult]:
        """Load multiple (or all) versions and analyse all questions."""
        groups = load_versions(data_root, versions, question_ids=question_ids)
        return self.run_on_groups(groups, limit=limit)
