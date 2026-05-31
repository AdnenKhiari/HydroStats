"""
codification_analysis/batch_processor.py

Gemini Batch API processor for Step 2 open codification.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from answer_analysis.providers import GeminiProvider

from .codifier import OpenCodifier, _TOOL_DEFINITION
from .models import CodedAnalysis, CodificationInput

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 15
_GEMINI_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


class GeminiCodificationBatchProcessor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        prompt_file: Optional[str] = None,
    ) -> None:
        from google import genai as _genai
        from google.genai import types as _types

        self._genai = _genai
        self._types = _types
        self.client = _genai.Client(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

        self._gemini_provider = GeminiProvider(api_key=api_key, model=model)
        self._codifier = OpenCodifier(
            provider=self._gemini_provider,
            max_tokens=max_tokens,
            prompt_file=prompt_file,
            use_cache=False,
        )

    def _build_inlined_request(self, item: CodificationInput) -> Any:
        messages = [{"role": "user", "content": self._codifier._build_user_message(item)}]
        contents = self._gemini_provider._to_gemini_contents(messages)
        system_text = self._gemini_provider._extract_text(self._codifier._system_prompt)
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
                "question_id": item.question_id,
                "version": item.version,
            },
        )

    def process(
        self,
        items: List[CodificationInput],
        *,
        poll_interval: int = _POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[CodedAnalysis]:
        if not items:
            return []

        requests = [self._build_inlined_request(item) for item in items]

        if verbose:
            print(f"  Submitting {len(requests)} codification jobs to Gemini Batch API ...")

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

        results: List[CodedAnalysis] = []
        succeeded = 0

        for index, item in enumerate(items):
            tagged_text = ""
            codebook: List[Dict[str, Any]] = []
            if index < len(inlined_responses):
                response = inlined_responses[index]
                if response.error:
                    logger.warning(
                        "Gemini batch error for %s / %s: %s",
                        item.version,
                        item.question_id,
                        response.error,
                    )
                elif response.response and response.response.candidates:
                    candidate = response.response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = content.parts if content and content.parts else []
                    for part in parts:
                        if part.function_call is not None:
                            payload = dict(part.function_call.args)
                            tagged_text = str(payload.get("tagged_text", ""))
                            raw_codebook = payload.get("codebook", [])
                            codebook = raw_codebook if isinstance(raw_codebook, list) else []
                            if tagged_text:
                                succeeded += 1
                            break
            results.append(self._codifier.build_coded_analysis(item, tagged_text, codebook))

        if verbose:
            print(f"  Done: {succeeded}/{len(items)} codifications succeeded")

        return results
