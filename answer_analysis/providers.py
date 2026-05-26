"""
answer_analysis/providers.py

LLM provider abstraction layer.

Implement LLMProvider to swap the backend at any time (Anthropic, Google Gemini,
OpenAI, Ollama, …) without touching the tagger or pipeline code.

Usage
─────
# Default (Google Gemini 3.5 Flash):
from answer_analysis import AnswerTagger
tagger = AnswerTagger()                          # reads GOOGLE_API_KEY from env

# Explicit Gemini provider:
from answer_analysis.providers import GeminiProvider
tagger = AnswerTagger(provider=GeminiProvider(api_key="..."))

# Switch to Anthropic:
from answer_analysis.providers import AnthropicProvider
tagger = AnswerTagger(provider=AnthropicProvider(api_key="sk-ant-..."))
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Retry configuration
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_MAX_RETRIES = 4
_BASE_DELAY          = 1.0    # seconds before first retry
_MAX_DELAY           = 60.0   # upper bound after exponential growth

# HTTP status codes that indicate a transient server-side problem
_RETRIABLE_STATUS_CODES = {429, 500, 502, 503, 529}


def _jittered_backoff(attempt: int) -> float:
    """Full-jitter exponential backoff (capped at _MAX_DELAY)."""
    ceiling = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
    return random.uniform(0.0, ceiling)


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract interface for LLM backends.

    Implement `complete` to support any provider.  The tagger passes fully-formed
    messages (with caching headers already applied) and expects back the parsed
    tool-input dict from the model's forced tool call.

    The provider is responsible for:
      • Making the HTTP request (streaming or not)
      • Retrying transient failures
      • Extracting the tool-use input from the raw response

    The tagger remains provider-agnostic: it handles prompt construction,
    rate limiting, and result parsing.
    """

    @abstractmethod
    def complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: List[Dict[str, Any]],
        tool_name: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Make one LLM call and return the tool-input dict.

        Parameters
        ----------
        messages   : conversation turns (user/assistant) already built by the tagger.
                     May contain Anthropic-style ``cache_control`` blocks — a
                     non-Anthropic implementation should strip those.
        system     : system prompt — either a plain string or a list of content
                     blocks (with ``cache_control`` if caching is enabled).
        tools      : tool-definition list in Anthropic format.
        tool_name  : the tool to force-invoke.
        max_tokens : maximum tokens the model may generate.

        Returns
        -------
        dict
            The parsed tool-input dict (e.g. ``{"tagged_text": "…"}``).

        Raises
        ------
        Exception
            Re-raises after all retry attempts are exhausted, or immediately
            for non-retriable errors (bad credentials, malformed request, …).
        """
        ...


# ──────────────────────────────────────────────────────────────────────────────
# Anthropic / Claude implementation
# ──────────────────────────────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """
    Claude via the Anthropic API.

    Features
    --------
    • Streaming      — always on; avoids the 10-minute non-streaming limit.
    • Auto-retry     — exponential backoff with full jitter on rate-limit (429)
                       and transient server errors (500 / 529).
    • Prompt caching — enabled by default; Anthropic charges ~10 % of the
                       normal input-token price for cache hits.

    Parameters
    ----------
    api_key     : Anthropic API key (defaults to ``ANTHROPIC_API_KEY`` env var).
    model       : Claude model identifier, e.g. ``"claude-opus-4-5"``.
    use_cache   : add ``betas=["prompt-caching-2024-07-31"]`` to every request.
    max_retries : number of retry attempts after the first failure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5",
        use_cache: bool = True,
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        import anthropic as _anthropic  # lazy import — keeps the module importable without anthropic installed

        self._anthropic = _anthropic
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.use_cache = use_cache
        self.max_retries = max_retries

    def complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: List[Dict[str, Any]],
        tool_name: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        create_kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            tool_choice={"type": "tool", "name": tool_name},
        )
        if self.use_cache:
            create_kwargs["betas"] = ["prompt-caching-2024-07-31"]

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                with self.client.beta.messages.stream(**create_kwargs) as stream:
                    final = stream.get_final_message()

                tool_block = next(
                    b for b in final.content if b.type == "tool_use"
                )
                return tool_block.input

            except self._anthropic.RateLimitError as exc:
                last_exc = exc
                wait = _jittered_backoff(attempt)
                logger.warning(
                    "Rate limit (attempt %d/%d) — retrying in %.1f s …",
                    attempt + 1, self.max_retries + 1, wait,
                )
                time.sleep(wait)

            except self._anthropic.APIStatusError as exc:
                if exc.status_code in _RETRIABLE_STATUS_CODES:
                    last_exc = exc
                    wait = _jittered_backoff(attempt)
                    logger.warning(
                        "API %d error (attempt %d/%d) — retrying in %.1f s …",
                        exc.status_code, attempt + 1, self.max_retries + 1, wait,
                    )
                    time.sleep(wait)
                else:
                    raise  # 400 bad request, 401 auth, 403 forbidden — fail fast

            except self._anthropic.APIConnectionError as exc:
                last_exc = exc
                wait = _jittered_backoff(attempt)
                logger.warning(
                    "Connection error (attempt %d/%d) — retrying in %.1f s …",
                    attempt + 1, self.max_retries + 1, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"AnthropicProvider: all {self.max_retries + 1} attempts failed."
        ) from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Google Gemini implementation
# ──────────────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """
    Google Gemini via the ``google-genai`` SDK.

    Features
    --------
    • Function calling  — forced via ``tool_config`` (equivalent to Anthropic's
                          ``tool_choice``).
    • Auto-retry        — exponential backoff with jitter on quota (429) and
                          server errors (500 / 503).
    • No prompt caching — Gemini handles caching implicitly; ``cache_control``
                          blocks from the tagger are stripped automatically.

    Parameters
    ----------
    api_key     : Google AI API key (defaults to ``GOOGLE_API_KEY`` env var).
    model       : Gemini model identifier, e.g. ``"gemini-3.5-flash"``.
    max_retries : Number of retry attempts after the first failure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        max_retries: int = _DEFAULT_MAX_RETRIES,
    ) -> None:
        from google import genai as _genai
        from google.genai import types as _types
        from google.genai import errors as _errors

        self._genai   = _genai
        self._types   = _types
        self._errors  = _errors
        self.client   = _genai.Client(api_key=api_key)
        self.model    = model
        self.max_retries = max_retries

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Flatten Anthropic-style content (string or list-of-blocks) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return str(content)

    def _to_gemini_contents(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert Anthropic-style messages to ``google.genai.types.Content`` objects."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            text = self._extract_text(msg["content"])
            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part(text=text)],
                )
            )
        return contents

    def _to_gemini_tool(self, tool_def: Dict[str, Any]) -> Any:
        """Convert an Anthropic tool definition to a Gemini ``Tool`` object."""
        schema = dict(tool_def["input_schema"])
        schema.pop("additionalProperties", None)  # not supported by Gemini

        # Normalise type strings to uppercase (Gemini requires e.g. "STRING")
        def _upper_types(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {
                    k: obj[k].upper() if k == "type" and isinstance(obj[k], str) else _upper_types(obj[k])
                    for k in obj
                }
            if isinstance(obj, list):
                return [_upper_types(i) for i in obj]
            return obj

        schema = _upper_types(schema)

        return self._types.Tool(
            function_declarations=[
                self._types.FunctionDeclaration(
                    name=tool_def["name"],
                    description=tool_def.get("description", ""),
                    parameters=schema,
                )
            ]
        )

    @staticmethod
    def _is_retriable(exc: Exception) -> bool:
        """Return True if the error is a transient one worth retrying."""
        code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if code in _RETRIABLE_STATUS_CODES:
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in ("quota", "rate limit", "overloaded", "unavailable", "timeout"))

    # ── LLMProvider interface ─────────────────────────────────────────────────

    def complete(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: Any,
        tools: List[Dict[str, Any]],
        tool_name: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        contents      = self._to_gemini_contents(messages)
        system_text   = self._extract_text(system)
        gemini_tools  = [self._to_gemini_tool(t) for t in tools]

        config = self._types.GenerateContentConfig(
            system_instruction=system_text,
            tools=gemini_tools,
            tool_config=self._types.ToolConfig(
                function_calling_config=self._types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[tool_name],
                )
            ),
            max_output_tokens=max_tokens,
        )

        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                # Extract the forced function call from the first candidate
                for part in response.candidates[0].content.parts:
                    if part.function_call is not None:
                        return dict(part.function_call.args)
                raise ValueError("GeminiProvider: no function_call in response")

            except Exception as exc:
                if self._is_retriable(exc):
                    last_exc = exc
                    wait = _jittered_backoff(attempt)
                    logger.warning(
                        "Gemini transient error (attempt %d/%d) — retrying in %.1f s … (%s)",
                        attempt + 1, self.max_retries + 1, wait, exc,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"GeminiProvider: all {self.max_retries + 1} attempts failed."
        ) from last_exc
