"""
answer_analysis/batch_processor.py

Batch-mode processors: submit all answers at once, poll, retrieve results.

Two implementations are available:

GeminiBatchProcessor  (default — uses the Gemini Developer API)
──────────────────────────────────────────────────────────────────
• Uses ``client.batches.create(src=[InlinedRequest(...)])`` — no GCS required.
• Responses come back in ``batch.dest.inlined_responses`` in the same order
  as the input requests.
• No per-request 503 spikes — Gemini processes the queue on their end.

AnthropicBatchProcessor  (alternative — uses the Anthropic Message Batches API)
────────────────────────────────────────────────────────────────────────────────
• 50 % cheaper input tokens on the Anthropic Batch API.
• Results identified by ``custom_id = answer_id`` (order-independent).

Usage
─────
    from answer_analysis.batch_processor import GeminiBatchProcessor
    from answer_analysis.data_loader import load_file

    processor = GeminiBatchProcessor(api_key="AIza...")
    raw_answers = load_file("data/Baseline V0/chatgpt.answers.txt")
    tagged = processor.process(raw_answers, verbose=True)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import anthropic

from .tagger import AnswerTagger, _TOOL_DEFINITION
from .models import TaggedAnswer

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 15   # seconds between status checks while waiting for batch


class AnthropicBatchProcessor:
    """
    Process a list of raw answers using Anthropic's Message Batches API.

    All answers are submitted in a single batch request; the processor polls
    until processing is complete and returns ``TaggedAnswer`` objects in the
    same order as the input.

    Parameters
    ----------
    api_key      : Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
    model        : Claude model to use.
    max_tokens   : Max output tokens per answer.
    prompt_file  : Override path to the system-prompt text file.
    fewshot_file : Override path to the few-shot JSON file.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        max_tokens: int = 16384,
        prompt_file: Optional[str] = None,
        fewshot_file: Optional[str] = None,
    ) -> None:
        self.client    = anthropic.Anthropic(api_key=api_key)
        self.model     = model
        self.max_tokens = max_tokens

        # Reuse tagger's prompt-loading and message-building helpers.
        # use_cache=False: caching is irrelevant when all requests are batched.
        from .providers import AnthropicProvider
        self._tagger = AnswerTagger(
            provider=AnthropicProvider(api_key=api_key, model=model, use_cache=False),
            max_tokens=max_tokens,
            use_cache=False,
            prompt_file=prompt_file,
            fewshot_file=fewshot_file,
        )

    # ── request building ──────────────────────────────────────────────────────

    def _prepare_request(self, raw_answer: Dict[str, Any]) -> Dict[str, Any]:
        """Build one Anthropic batch request entry for a single answer."""
        text = raw_answer["text"]
        messages = self._tagger._cached_few_shot() + [
            {"role": "user", "content": text}
        ]
        return {
            "custom_id": raw_answer["answer_id"],
            "params": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self._tagger._cached_system(),
                "messages": messages,
                "tools": [_TOOL_DEFINITION],
                "tool_choice": {"type": "tool", "name": "submit_tagged_answer"},
            },
        }

    # ── public API ────────────────────────────────────────────────────────────

    def process(
        self,
        raw_answers: List[Dict[str, Any]],
        *,
        poll_interval: int = _POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[TaggedAnswer]:
        """
        Submit, poll, and return tagged answers for all input raw answers.

        Parameters
        ----------
        raw_answers   : List of raw answer dicts (from data_loader).
        poll_interval : Seconds between status-check polls.
        verbose       : Print progress to stdout.

        Returns
        -------
        List of TaggedAnswer in the same order as ``raw_answers``.
        Answers that errored or expired are returned with empty spans.
        """
        if not raw_answers:
            return []

        requests = [self._prepare_request(a) for a in raw_answers]

        if verbose:
            print(f"  Submitting {len(requests)} answers to Anthropic Batch API …")

        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id

        if verbose:
            print(f"  Batch ID : {batch_id}")
            print(f"  Polling every {poll_interval}s until complete …")

        # ── poll ──────────────────────────────────────────────────────────────
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            if verbose:
                print(
                    f"  [{batch.processing_status}]  "
                    f"processing={counts.processing}  "
                    f"succeeded={counts.succeeded}  "
                    f"errored={counts.errored}"
                )
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval)

        # ── collect results ───────────────────────────────────────────────────
        result_map: Dict[str, str] = {}   # answer_id → tagged_full text
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type == "succeeded":
                msg = result.result.message
                tool_block = next(
                    (b for b in msg.content if b.type == "tool_use"), None
                )
                if tool_block and "tagged_text" in tool_block.input:
                    result_map[result.custom_id] = tool_block.input["tagged_text"]
                else:
                    logger.warning(
                        "No tagged_text in batch result for %s", result.custom_id
                    )
            else:
                logger.warning(
                    "Batch result for %s: %s — %s",
                    result.custom_id,
                    result.result.type,
                    getattr(result.result, "error", ""),
                )

        if verbose:
            print(
                f"  Done: {len(result_map)}/{len(raw_answers)} answers succeeded"
            )

        # ── build TaggedAnswer objects in original order ───────────────────────
        tagged_answers: List[TaggedAnswer] = []
        for raw in raw_answers:
            aid = raw["answer_id"]
            tagged_full = result_map.get(aid, "")
            if not tagged_full:
                # Fallback: wrap every line in <null>
                tagged_full = "\n".join(
                    f"<null>{l}</null>" if l.strip() else ""
                    for l in raw["text"].splitlines()
                )
            tagged_answers.append(
                self._tagger.build_tagged_answer(raw, tagged_full)
            )

        return tagged_answers


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Batch API  (Developer API — no GCS required)
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_TERMINAL_STATES = frozenset({
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
})


class GeminiBatchProcessor:
    """
    Process a list of raw answers using the Gemini Developer API Batch endpoint.

    All answers are submitted as a list of ``InlinedRequest`` objects — no GCS
    bucket is needed.  The processor polls until the job finishes, then reads
    results from ``batch.dest.inlined_responses`` in the same order as input.

    Parameters
    ----------
    api_key      : Google AI API key (defaults to GOOGLE_API_KEY env var).
    model        : Gemini model to use.
    max_tokens   : Max output tokens per answer.
    prompt_file  : Override path to the system-prompt text file.
    fewshot_file : Override path to the few-shot JSON file.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        prompt_file: Optional[str] = None,
        fewshot_file: Optional[str] = None,
    ) -> None:
        from google import genai as _genai
        from google.genai import types as _types

        self._genai  = _genai
        self._types  = _types
        self.client  = _genai.Client(api_key=api_key)
        self.model   = model
        self.max_tokens = max_tokens

        # Reuse GeminiProvider for message/tool conversion helpers
        from .providers import GeminiProvider
        self._gemini_provider = GeminiProvider(api_key=api_key, model=model)

        # Reuse AnswerTagger for prompt loading and result parsing
        from .tagger import AnswerTagger
        self._tagger = AnswerTagger(
            provider=self._gemini_provider,
            max_tokens=max_tokens,
            use_cache=False,
            prompt_file=prompt_file,
            fewshot_file=fewshot_file,
        )

    # ── request building ──────────────────────────────────────────────────────

    def _build_inlined_request(self, raw_answer: Dict[str, Any]) -> Any:
        """Build one Gemini InlinedRequest for a single answer."""
        from .tagger import _TOOL_DEFINITION

        messages = self._tagger._cached_few_shot() + [
            {"role": "user", "content": raw_answer["text"]}
        ]
        contents    = self._gemini_provider._to_gemini_contents(messages)
        system_text = self._gemini_provider._extract_text(
            self._tagger._cached_system()
        )
        gemini_tool = self._gemini_provider._to_gemini_tool(_TOOL_DEFINITION)

        config = self._types.GenerateContentConfig(
            system_instruction=system_text,
            tools=[gemini_tool],
            tool_config=self._types.ToolConfig(
                function_calling_config=self._types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[_TOOL_DEFINITION["name"]],
                )
            ),
            max_output_tokens=self.max_tokens,
        )

        return self._types.InlinedRequest(
            model=self.model,
            contents=contents,
            config=config,
            metadata={"answer_id": raw_answer["answer_id"]},
        )

    # ── public API ────────────────────────────────────────────────────────────

    def process(
        self,
        raw_answers: List[Dict[str, Any]],
        *,
        poll_interval: int = _POLL_INTERVAL,
        verbose: bool = True,
    ) -> List[TaggedAnswer]:
        """
        Submit, poll, and return tagged answers for all input raw answers.

        Parameters
        ----------
        raw_answers   : List of raw answer dicts (from data_loader).
        poll_interval : Seconds between status-check polls.
        verbose       : Print progress to stdout.

        Returns
        -------
        List of TaggedAnswer in the same order as ``raw_answers``.
        Failed answers are returned with all lines wrapped in ``<null>``.
        """
        if not raw_answers:
            return []

        requests = [self._build_inlined_request(a) for a in raw_answers]

        if verbose:
            print(f"  Submitting {len(requests)} answers to Gemini Batch API …")

        batch = self.client.batches.create(model=self.model, src=requests)

        if verbose:
            print(f"  Batch name : {batch.name}")
            print(f"  Polling every {poll_interval}s until complete …")

        # ── poll ──────────────────────────────────────────────────────────────
        while True:
            batch = self.client.batches.get(name=batch.name)
            state_name = batch.state.name if batch.state else "UNKNOWN"
            if verbose:
                stats = batch.completion_stats
                print(f"  [{state_name}]  {stats}")
            if state_name in _GEMINI_TERMINAL_STATES:
                break
            time.sleep(poll_interval)

        # ── collect results ───────────────────────────────────────────────────
        tagged_answers: List[TaggedAnswer] = []

        inlined_responses = (
            batch.dest.inlined_responses
            if batch.dest and batch.dest.inlined_responses
            else []
        )

        succeeded = 0
        for i, raw in enumerate(raw_answers):
            tagged_full = ""
            if i < len(inlined_responses):
                ir = inlined_responses[i]
                if ir.error:
                    logger.warning(
                        "Gemini batch error for answer %d (%s): %s",
                        i, raw["answer_id"], ir.error,
                    )
                elif ir.response and ir.response.candidates:
                    candidate = ir.response.candidates[0]
                    content = getattr(candidate, "content", None)
                    parts = content.parts if content and content.parts else []
                    for part in parts:
                        if part.function_call is not None:
                            tagged_full = (
                                dict(part.function_call.args).get("tagged_text", "")
                            )
                            succeeded += 1
                            break
            tagged_answers.append(
                self._tagger.build_tagged_answer(raw, tagged_full)
            )

        if verbose:
            print(f"  Done: {succeeded}/{len(raw_answers)} answers succeeded")

        return tagged_answers
