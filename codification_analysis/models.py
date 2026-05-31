"""
codification_analysis/models.py

Data models for Step 2 open codification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CodeDefinition:
    """Human-readable definition for one inductively discovered code."""
    tag: str
    code_name: str
    description: str
    representative_excerpt: str


@dataclass
class CodedSpan:
    """A contiguous tagged span inside one grouped-answer line."""
    text: str
    tag: Optional[str] = None
    code_name: Optional[str] = None
    description: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


@dataclass
class CodedLine:
    """One line of the grouped answers with inline XML tagging."""
    line_index: int
    raw_text: str
    tagged_text: str
    spans: List[CodedSpan] = field(default_factory=list)


@dataclass
class CodificationInput:
    """Joined Step 2 input for one (question, version) unit."""
    question_id: str
    version: str
    providers: List[str]
    raw_analysis: str
    source_answers: str


@dataclass
class CodedAnalysis:
    """Full Step 2 result for one grouped-answer codification."""
    question_id: str
    version: str
    providers: List[str]
    raw_analysis: str
    source_answers: str
    tagged_text: str
    codebook: List[CodeDefinition] = field(default_factory=list)
    lines: List[CodedLine] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for line in self.lines:
            for span in line.spans:
                if span.code_name:
                    counts[span.code_name] = counts.get(span.code_name, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, object]:
        return {
            "question_id": self.question_id,
            "version": self.version,
            "providers": self.providers,
            "raw_analysis": self.raw_analysis,
            "source_answers": self.source_answers,
            "tagged_text": self.tagged_text,
            "codebook": [
                {
                    "tag": code.tag,
                    "code_name": code.code_name,
                    "description": code.description,
                    "representative_excerpt": code.representative_excerpt,
                }
                for code in self.codebook
            ],
            "lines": [
                {
                    "line_index": line.line_index,
                    "raw_text": line.raw_text,
                    "tagged_text": line.tagged_text,
                    "spans": [
                        {
                            "text": span.text,
                            "tag": span.tag,
                            "code_name": span.code_name,
                            "description": span.description,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                        }
                        for span in line.spans
                    ],
                }
                for line in self.lines
            ],
            "summary": self.summary(),
        }
