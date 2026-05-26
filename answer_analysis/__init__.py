"""
answer_analysis
~~~~~~~~~~~~~~~
Modular pipeline for rhetorical tagging of AI-generated answers.

Public API
──────────
    from answer_analysis.data_loader import load_file, load_directory, load_experiment
    from answer_analysis.tagger import AnswerTagger
    from answer_analysis.pipeline import TaggingPipeline, pretty_print, tagged_answer_to_dict
    from answer_analysis.models import Category, TaggedAnswer, TaggedLine, TaggedSpan
    from answer_analysis.providers import LLMProvider, AnthropicProvider
"""
from .models import (
    Category,
    TaggedAnswer,
    TaggedLine,
    TaggedSpan,
)
from .data_loader import load_file, load_directory, load_experiment
from .providers import LLMProvider, AnthropicProvider, GeminiProvider
from .tagger import AnswerTagger
from .batch_processor import AnthropicBatchProcessor, GeminiBatchProcessor
from .pipeline import TaggingPipeline, pretty_print, tagged_answer_to_dict

__all__ = [
    # models
    "Category",
    "TaggedAnswer",
    "TaggedLine",
    "TaggedSpan",
    # data loading
    "load_file",
    "load_directory",
    "load_experiment",
    # providers
    "LLMProvider",
    "AnthropicProvider",
    "GeminiProvider",
    # tagging
    "AnswerTagger",
    "AnthropicBatchProcessor",
    "GeminiBatchProcessor",
    # pipeline
    "TaggingPipeline",
    "pretty_print",
    "tagged_answer_to_dict",
]
