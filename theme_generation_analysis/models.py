"""
theme_generation_analysis/models.py

Data models for Step 3 theme generation over Step 2 codebooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ThemeSourceCode:
    """One Step 2 codebook entry, identified within a version-level corpus."""
    code_id: str
    question_id: str
    tag: str
    code_name: str
    description: str
    representative_excerpt: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "code_id": self.code_id,
            "question_id": self.question_id,
            "tag": self.tag,
            "code_name": self.code_name,
            "description": self.description,
            "representative_excerpt": self.representative_excerpt,
        }


@dataclass
class ThemeGenerationCorpus:
    """All Step 2 codebook entries for one version."""
    version: str
    source_codes: List[ThemeSourceCode] = field(default_factory=list)

    @property
    def question_count(self) -> int:
        return len({code.question_id for code in self.source_codes})


@dataclass
class GeneratedTheme:
    """One higher-level theme grouping similar Step 2 codes."""
    theme_name: str
    description: str
    code_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "theme_name": self.theme_name,
            "description": self.description,
            "code_ids": self.code_ids,
        }


@dataclass
class ThemeGenerationResult:
    """Step 3 result for one version-level codebook corpus."""
    version: str
    question_count: int
    code_count: int
    themes: List[GeneratedTheme] = field(default_factory=list)
    source_codes: List[ThemeSourceCode] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        return {
            theme.theme_name: len(theme.code_ids)
            for theme in self.themes
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "version": self.version,
            "question_count": self.question_count,
            "code_count": self.code_count,
            "themes": [theme.to_dict() for theme in self.themes],
            "source_codes": [code.to_dict() for code in self.source_codes],
            "summary": self.summary(),
        }