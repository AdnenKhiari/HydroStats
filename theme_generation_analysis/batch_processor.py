"""
theme_generation_analysis/batch_processor.py

Gemini Batch API processor for Step 3 theme generation.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from answer_analysis.providers import GeminiProvider

from .generator import ThemeGenerator
from .models import ThemeGenerationCorpus, ThemeGenerationResult

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 15
_GEMINI_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


class GeminiThemeGenerationBatchProcessor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 65535,
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
        self._generator = ThemeGenerator(
            provider=self._gemini_provider,
            max_tokens=max_tokens,
            prompt_file=prompt_file,
        )

    def _call_with_retry(self, func: Callable[[], Any], *, action: str) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(self._gemini_provider.max_retries + 1):
            try:
                return func()
            except Exception as exc:
                if not self._gemini_provider._is_retriable(exc):
                    raise
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Gemini batch %s transient error (attempt %d/%d) — retrying in %ds … (%s)",
                    action,
                    attempt + 1,
                    self._gemini_provider.max_retries + 1,
                    wait,
                    exc,
                )
                time.sleep(wait)
        raise RuntimeError(f"Gemini batch {action} failed after retries") from last_exc

    def _build_inlined_request(self, corpus: ThemeGenerationCorpus) -> Any:
        messages = [{"role": "user", "content": self._generator._build_user_message(corpus)}]
        contents = self._gemini_provider._to_gemini_contents(messages)
        system_text = self._gemini_provider._extract_text(self._generator._system_prompt)
        config = self._gemini_provider.build_generate_config(
            system_text=system_text,
            max_tokens=self.max_tokens,
        )

        return self._types.InlinedRequest(
            model=self.model,
            contents=contents,
            config=config,
            metadata={"version": corpus.version},
        )

    @staticmethod
    def _serialize_part(part: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        text = getattr(part, "text", None)
        if text:
            payload["text"] = text

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            payload["function_call"] = {
                "name": getattr(function_call, "name", None),
                "args": dict(getattr(function_call, "args", {}) or {}),
            }

        executable_code = getattr(part, "executable_code", None)
        if executable_code is not None:
            payload["executable_code"] = str(executable_code)

        code_execution_result = getattr(part, "code_execution_result", None)
        if code_execution_result is not None:
            payload["code_execution_result"] = str(code_execution_result)

        if not payload:
            payload["repr"] = str(part)
        return payload

    def _write_raw_response_log(
        self,
        *,
        version: str,
        response: Any,
        raw_log_dir: Path,
    ) -> None:
        raw_log_dir.mkdir(parents=True, exist_ok=True)
        raw_log_path = raw_log_dir / "theme_generation_raw_response.json"

        payload: Dict[str, Any] = {
            "version": version,
            "error": str(response.error) if getattr(response, "error", None) else None,
            "candidates": [],
        }

        if response.response and response.response.candidates:
            for candidate in response.response.candidates:
                content = getattr(candidate, "content", None)
                parts = content.parts if content and content.parts else []
                payload["candidates"].append({
                    "finish_reason": getattr(getattr(candidate, "finish_reason", None), "name", None),
                    "parts": [self._serialize_part(part) for part in parts],
                })

        raw_log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved Step 3 raw response log to %s", raw_log_path)

    def process(
        self,
        corpora: List[ThemeGenerationCorpus],
        *,
        poll_interval: int = _POLL_INTERVAL,
        verbose: bool = True,
        raw_log_root: Optional[str | Path] = None,
    ) -> List[ThemeGenerationResult]:
        if not corpora:
            return []

        requests = [self._build_inlined_request(corpus) for corpus in corpora]

        if verbose:
            print(f"  Submitting {len(requests)} theme-generation jobs to Gemini Batch API ...")

        batch = self._call_with_retry(
            lambda: self.client.batches.create(model=self.model, src=requests),
            action="create",
        )

        if verbose:
            print(f"  Batch name : {batch.name}")
            print(f"  Polling every {poll_interval}s until complete ...")

        while True:
            batch = self._call_with_retry(
                lambda: self.client.batches.get(name=batch.name),
                action="poll",
            )
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

        results: List[ThemeGenerationResult] = []
        succeeded = 0

        for index, corpus in enumerate(corpora):
            raw_themes: List[Dict[str, Any]] = []
            if index < len(inlined_responses):
                response = inlined_responses[index]
                if raw_log_root is not None:
                    self._write_raw_response_log(
                        version=corpus.version,
                        response=response,
                        raw_log_dir=Path(raw_log_root) / corpus.version,
                    )
                if response.error:
                    logger.warning(
                        "Gemini batch error for %s: %s",
                        corpus.version,
                        response.error,
                    )
                elif response.response and response.response.candidates:
                    candidate = response.response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = content.parts if content and content.parts else []
                    text = "\n".join(
                        getattr(part, "text", "")
                        for part in parts
                        if getattr(part, "text", None)
                    )
                    raw_themes = self._generator._extract_raw_themes_from_text(text)
                    if raw_themes:
                        succeeded += 1

            results.append(ThemeGenerationResult(
                version=corpus.version,
                question_count=corpus.question_count,
                code_count=len(corpus.source_codes),
                themes=self._generator._normalize_themes(raw_themes, corpus.source_codes),
                source_codes=corpus.source_codes,
            ))

        if verbose:
            print(f"  Done: {succeeded}/{len(corpora)} theme generations succeeded")

        return results