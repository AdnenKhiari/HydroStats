"""
answer_analysis/models.py
Data models for the answer tagging pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Category(str, Enum):
    """Rhetorical signal categories used as inline XML tags in Claude output."""
    STATISTICAL_USE      = "Statistical Use"
    CREDIBILITY_SIGNAL   = "Credibility Signal"
    STRONG_RECOMMENDATION = "Strong Recommendation"
    BRAND_POSITIONING    = "Brand Positioning"

    # XML tag name used in the LLM output (no spaces, PascalCase)
    @property
    def tag(self) -> str:
        return {
            "Statistical Use":     "StatisticalUse",
            "Credibility Signal":  "CredibilitySignal",
            "Strong Recommendation": "StrongRecommendation",
            "Brand Positioning":   "BrandPositioning",
        }[self.value]

    @classmethod
    def from_tag(cls, tag: str) -> "Category":
        mapping = {
            "StatisticalUse":       cls.STATISTICAL_USE,
            "CredibilitySignal":    cls.CREDIBILITY_SIGNAL,
            "StrongRecommendation": cls.STRONG_RECOMMENDATION,
            "BrandPositioning":     cls.BRAND_POSITIONING,
        }
        return mapping[tag]


# ──────────────────────────────────────────────────────────────────────────────
# Span-level annotation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaggedSpan:
    """
    A contiguous phrase in a line.
    category=None means the phrase was tagged <null> (no signal).
    """
    text: str
    category: Optional[Category]   # None → null span
    char_start: Optional[int] = None   # character offset in the parent line
    char_end:   Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────────
# Line-level annotation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaggedLine:
    line_index: int          # 0-based position inside the answer
    raw_text:   str          # original line
    tagged_text: str         # raw LLM output (inline XML tags)
    spans: List[TaggedSpan] = field(default_factory=list)

    def spans_for(self, category: Category) -> List[TaggedSpan]:
        return [s for s in self.spans if s.category == category]

    @property
    def signal_spans(self) -> List[TaggedSpan]:
        """Only spans that carry a real category (non-null)."""
        return [s for s in self.spans if s.category is not None]


# ──────────────────────────────────────────────────────────────────────────────
# Answer-level result
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaggedAnswer:
    """Full tagging result for one AI answer."""
    answer_id: str
    query_id:  Optional[str]
    model:     Optional[str]
    provider:  Optional[str]
    raw_text:  str
    lines: List[TaggedLine] = field(default_factory=list)

    @property
    def all_spans(self) -> List[TaggedSpan]:
        return [span for line in self.lines for span in line.spans]

    def spans_for(self, category: Category) -> List[TaggedSpan]:
        return [s for s in self.all_spans if s.category == category]

    def summary(self) -> dict:
        cat_counts = {cat.value: 0 for cat in Category}
        for span in self.all_spans:
            if span.category is not None:
                cat_counts[span.category.value] += 1
        return {
            "answer_id":  self.answer_id,
            "model":      self.model,
            "total_lines": len(self.lines),
            "span_counts": cat_counts,
        }
