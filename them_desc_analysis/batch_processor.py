"""
them_desc_analysis/batch_processor.py

Batch-mode thematic analysis using the Gemini Developer API Batch endpoint.

Each request represents one (question, version) answer group: the three chatbot
answers are concatenated, submitted in one Gemini batch job, then reconstructed
into ``ThematicResult`` objects in input order.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .analyzer import ThematicAnalyzer, _TOOL_DEFINITION
from .models import AnswerGroup, ThematicResult

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 15
_GEMINI_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


class GeminiThematicBatchProcessor:
    """
    Process a list of AnswerGroup objects using the Gemini Batch API.

    Parameters
    ----------
    api_key     : Google AI API key (defaults to GOOGLE_API_KEY env var).
    model       : Gemini model to use.
    max_tokens  : Max output tokens per analysis.
    prompt_file : Override path to the system-prompt text file.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        prompt_file: Optional[str] = None,
    ) -> None:
        from google import genai as _genai
        from google.genai import types as _types

        from answer_analysis.providers import GeminiProvider

        self._genai = _genai
        self._types = _types
        self.client = _genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

        self._gemini_provider = GeminiProvider(api_key=api_key, model=model)
        self._analyzer = ThematicAnalyzer(
            provider=self._gemini_provider,
            max_tokens=max_tokens,
            prompt_file=prompt_file,
        )

    def _build_inlined_request(self, group: AnswerGroup) -> Any:
        messages = [{"role": "user", "content": group.concatenated_text}]
        contents = self._gemini_provider._to_gemini_contents(messages)
        system_text = self._gemini_provider._extract_text(self._analyzer._system_prompt)
        gemini_tool = self._gemini_provider._to_gemini_tool(_TOOL_DEFINITION)
        config = self._gemini_provider.build_generate_config(
            system_text=system_text,
            gemini_tools=[gemini_tool],
            tool_name=_TOOL_DEFINITION["name"],
            max_tokens=self.max_tokens,
        )

        return self._types.InlinedRequest(
            model=self.model,
            contents=contents,
            config=config,
            metadata={
                "question_id": group.question_id,
                "version": group.version,
            },
        )

    def process(
        self,
        groups: List[AnswerGroup],
        *,
        poll_interval: int = _POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[ThematicResult]:
        if not groups:
            return []

        requests = [self._build_inlined_request(group) for group in groups]

        if verbose:
            print(f"  Submitting {len(requests)} thematic analyses to Gemini Batch API ...")

        batch = self.client.batches.create(model=self.model, src=requests)

        if verbose:
            print(f"  Batch name : {batch.name}")
            print(f"  Polling every {poll_interval}s until complete ...")

        while True:
            batch = self.client.batches.get(name=batch.name)
            state_name = batch.state.name if batch.state else "UNKNOWN"
            if verbose:
                print(f"  [{state_name}]  {batch.completion_stats}")
            if state_name in _GEMINI_TERMINAL_STATES:
                break
            time.sleep(poll_interval)

        inlined_responses = (
            batch.dest.inlined_responses
            if batch.dest and batch.dest.inlined_responses
            else []
        )

        results: List[ThematicResult] = []
        succeeded = 0

        for i, group in enumerate(groups):
            analysis = ""
            if i < len(inlined_responses):
                response = inlined_responses[i]
                if response.error:
                    logger.warning(
                        "Gemini batch error for %s / %s: %s",
                        group.version,
                        group.question_id,
                        response.error,
                    )
                elif response.response and response.response.candidates:
                    candidate = response.response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = content.parts if content and content.parts else []
                    for part in parts:
                        if part.function_call is not None:
                            analysis = dict(part.function_call.args).get("analysis", "").strip()
                            if analysis:
                                succeeded += 1
                            break

            results.append(
                ThematicResult(
                    question_id=group.question_id,
                    version=group.version,
                    providers=group.providers,
                    analysis=analysis,
                )
            )

        if verbose:
            print(f"  Done: {succeeded}/{len(groups)} analyses succeeded")

        return results
