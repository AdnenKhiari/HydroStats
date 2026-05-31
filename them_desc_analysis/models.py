"""
them_desc_analysis/models.py

Data models for the thematic analysis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AnswerGroup:
    """
    The three chatbot answers for a single (question, version) pair,
    ready to be sent to the LLM as a concatenated block.
    """
    question_id: str                     # queryKey from the data files
    version: str                         # e.g. "V1", "Baseline V0"
    providers: List[str]                 # providers present, e.g. ["GEMINI", "OPENAI", "PERPLEXITY"]
    concatenated_text: str               # labeled concatenation of all answers


@dataclass
class ThematicResult:
    """
    Output of one thematic analysis call.
    """
    question_id: str
    version: str
    providers: List[str]                 # which providers contributed
    analysis: str                        # LLM-generated qualitative analysis

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "version":     self.version,
            "providers":   self.providers,
            "analysis":    self.analysis,
        }
