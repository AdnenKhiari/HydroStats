"""
them_desc_analysis — Inductive thematic analysis of grouped AI chatbot responses.

For each (question, version) pair, the three chatbot answers are concatenated
and sent to an LLM that performs a qualitative thematic analysis.  The result
is a single descriptive string saved as structured JSON.

Public API
──────────
    from them_desc_analysis import (
        GeminiThematicBatchProcessor,
        ThematicAnalyzer,
        ThematicPipeline,
    )
"""
from .analyzer import ThematicAnalyzer
from .batch_processor import GeminiThematicBatchProcessor
from .pipeline import ThematicPipeline

__all__ = ["GeminiThematicBatchProcessor", "ThematicAnalyzer", "ThematicPipeline"]
