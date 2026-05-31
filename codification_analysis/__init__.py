"""
codification_analysis — Step 2 open codification over thematic analyses.

Consumes each (question, version) thematic analysis together with the original
concatenated chatbot answers, then produces inline XML-tagged analysis text and
an inductive codebook suitable for later exploitation.
"""
from .codifier import OpenCodifier
from .batch_processor import GeminiCodificationBatchProcessor
from .pipeline import CodificationPipeline

__all__ = ["CodificationPipeline", "GeminiCodificationBatchProcessor", "OpenCodifier"]
