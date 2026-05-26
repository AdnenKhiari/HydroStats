"""
answer_analysis/tagger.py

LLM-powered rhetorical tagger using the Anthropic Claude API.

Design
──────
• System prompt and few-shot examples are loaded from external files so you
  can edit them without touching code:
    answer_analysis/tagger_prompt.txt   ← system prompt (categories, rules)
    answer_analysis/tagger_fewshot.json ← few-shot conversation examples

• The FULL answer text is sent to Claude in ONE call (plain string, not a
  line-by-line JSON array).  Claude returns the fully-tagged text as one
  string; we split on newlines to reconstruct per-line TaggedLine objects.

• The four XML signal tags are:
    <StatisticalUse>      → concrete numbers, measurements, specifications
    <CredibilitySignal>   → named authorities, institutions, experts
    <StrongRecommendation>→ strong AI recommendations, superlatives, conviction
    <BrandPositioning>    → explicit/implicit brand or product focus
  Everything else goes into <null>.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Category, TaggedAnswer, TaggedLine, TaggedSpan
from .providers import AnthropicProvider, GeminiProvider, LLMProvider

# ──────────────────────────────────────────────────────────────────────────────
# Paths to external prompt files (relative to this module's directory)
# ──────────────────────────────────────────────────────────────────────────────

_MODULE_DIR   = Path(__file__).parent
_PROMPT_FILE  = _MODULE_DIR / "tagger_prompt.txt"
_FEWSHOT_FILE = _MODULE_DIR / "tagger_fewshot.json"

# ──────────────────────────────────────────────────────────────────────────────
# Tag registry
# ──────────────────────────────────────────────────────────────────────────────

_KNOWN_TAGS = {cat.tag for cat in Category} | {"null"}

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Tool schema (forces Claude to return valid JSON)
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_DEFINITION = {
    "name": "submit_tagged_answer",
    "description": (
        "Submit the fully XML-tagged version of the input answer text. "
        "Every character of the original must be reproduced inside exactly one XML tag. "
        "The output must have the same number of lines as the input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tagged_text": {
                "type": "string",
                "description": (
                    "The complete answer text with every character wrapped in "
                    "exactly one tag: <StatisticalUse>, <CredibilitySignal>, "
                    "<StrongRecommendation>, <BrandPositioning>, or <null>. "
                    "Line count must match the input exactly."
                ),
            }
        },
        "required": ["tagged_text"],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _parse_tagged_line(tagged_text: str, raw_line: str) -> List[TaggedSpan]:
    """Convert inline-XML tagged text into TaggedSpan objects with char offsets."""
    spans: List[TaggedSpan] = []
    cursor = 0

    for m in _TAG_RE.finditer(tagged_text):
        tag_name = m.group(1)
        content  = m.group(2)

        if tag_name not in _KNOWN_TAGS:
            tag_name = "null"

        try:
            category: Optional[Category] = (
                Category.from_tag(tag_name) if tag_name != "null" else None
            )
        except KeyError:
            category = None

        idx = raw_line.find(content, cursor)
        if idx != -1:
            char_start, char_end = idx, idx + len(content)
            cursor = char_end
        else:
            char_start = char_end = None

        spans.append(TaggedSpan(
            text=content,
            category=category,
            char_start=char_start,
            char_end=char_end,
        ))

    return spans


# ──────────────────────────────────────────────────────────────────────────────
# Tagger
# ──────────────────────────────────────────────────────────────────────────────

class AnswerTagger:
    """
    Batch-tags an answer by sending its full text to Claude in a single call.

    The provider abstraction (``LLMProvider``) lets you swap the LLM backend
    at any time — pass a custom ``provider`` instance instead of ``api_key``
    and ``model`` to use a different model or service.

    External files (edit freely to tune behaviour):
        tagger_prompt.txt   — system prompt with category definitions and rules
        tagger_fewshot.json — few-shot conversation examples

    Parameters
    ----------
    api_key      : Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
                   Ignored when a custom ``provider`` is supplied.
    model        : Claude model identifier.
                   Ignored when a custom ``provider`` is supplied.
    max_tokens   : Maximum tokens the model may generate per call.
                   The full capacity is used — no automatic cap.
    rpm_limit    : Minimum seconds between consecutive API calls.
    skip_short   : Kept for API compatibility; no longer filters lines from the
                   API call (the full text is always sent).
    prompt_file  : Override path to the system-prompt text file.
    fewshot_file : Override path to the few-shot JSON file.
    use_cache    : Add Anthropic prompt-caching headers so repeated calls for
                   the same system prompt + few-shots are cheaper.
    provider     : Custom ``LLMProvider`` implementation.  When supplied,
                   ``api_key``, ``model``, and ``use_cache`` are ignored for
                   the API call (the provider owns those settings).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        skip_short: int = 10,
        prompt_file: Optional[str | Path] = None,
        fewshot_file: Optional[str | Path] = None,
        use_cache: bool = True,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        # Provider — use the supplied one, or create the default Anthropic one
        if provider is not None:
            self.provider = provider
        else:
            self.provider = GeminiProvider(
                api_key=api_key,
                model=model,
            )
        self.model     = model       # kept for display / logging
        self.use_cache = use_cache   # controls cache_control headers in messages
        self.max_tokens = max_tokens
        self._min_interval = rpm_limit
        self._last_call_time: float = 0.0
        self.skip_short = skip_short

        # Load external prompt files
        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)
        self._few_shot      = self._load_fewshot(fewshot_file or _FEWSHOT_FILE)

    # ── prompt loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_prompt(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {path}\n"
                "Expected at answer_analysis/tagger_prompt.txt"
            )
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_fewshot(path: str | Path) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []   # few-shot is optional
        return json.loads(path.read_text(encoding="utf-8"))

    def reload_prompts(
        self,
        prompt_file: Optional[str | Path] = None,
        fewshot_file: Optional[str | Path] = None,
    ) -> None:
        """Hot-reload prompt files without recreating the tagger instance."""
        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)
        self._few_shot      = self._load_fewshot(fewshot_file or _FEWSHOT_FILE)

    # ── prompt-cache helpers ──────────────────────────────────────────────────

    def _cached_system(self) -> Any:
        """
        Return the system parameter for the API call.
        With caching: a list with cache_control on the text block.
        Without caching: a plain string.
        """
        if not self.use_cache:
            return self._system_prompt
        return [
            {
                "type": "text",
                "text": self._system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _cached_few_shot(self) -> List[Dict[str, Any]]:
        """
        Return few-shot messages, adding cache_control to the last assistant
        turn so the entire prefix (system + few-shots) is a single cache entry.
        Without caching: returns the messages unchanged.
        """
        if not self.use_cache or not self._few_shot:
            return self._few_shot

        messages = [m.copy() for m in self._few_shot]
        # Mark the last message as the cache breakpoint
        last = messages[-1]
        # Normalise content to a list of blocks so we can attach cache_control
        content = last["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            content = [block.copy() for block in content]
        content[-1]["cache_control"] = {"type": "ephemeral"}
        last["content"] = content
        return messages

    # ── API call ──────────────────────────────────────────────────────────────

    def _rate_limit_wait(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _call_claude(self, text: str) -> str:
        """
        Send the full answer text to the provider and return the full tagged text.
        """
        self._rate_limit_wait()

        messages = self._cached_few_shot() + [{"role": "user", "content": text}]

        result = self.provider.complete(
            messages=messages,
            system=self._cached_system(),
            tools=[_TOOL_DEFINITION],
            tool_name="submit_tagged_answer",
            max_tokens=self.max_tokens,
        )

        self._last_call_time = time.monotonic()
        return result["tagged_text"]

    # ── public ────────────────────────────────────────────────────────────────

    def tag_answer(
        self,
        raw_answer: Dict[str, Any],
        *,
        verbose: bool = False,
    ) -> TaggedAnswer:
        """
        Tag a full RawAnswer dict (as returned by data_loader) in one Claude call.

        The complete answer text is sent as a plain string; Claude returns the
        fully-tagged text as a single string.  We split on newlines to recover
        per-line TaggedLine objects, matching positions 1-to-1 with the original.

        Parameters
        ----------
        raw_answer : dict with keys: answer_id, query_id, model, provider, text
        verbose    : print progress to stdout
        """
        text: str = raw_answer["text"]
        raw_lines = text.splitlines()

        if verbose:
            non_empty = sum(1 for l in raw_lines if l.strip())
            print(f"  Sending full answer ({non_empty} non-empty lines) to Claude …")

        tagged_full = ""
        if text.strip():
            tagged_full = self._call_claude(text)

        return self.build_tagged_answer(raw_answer, tagged_full)

    def build_tagged_answer(
        self,
        raw_answer: Dict[str, Any],
        tagged_full: str,
    ) -> TaggedAnswer:
        """
        Reconstruct a TaggedAnswer from a raw answer dict and the tagged text
        string returned by the LLM.  Used by both tag_answer() and the batch
        processor so the parsing logic lives in one place.
        """
        text: str = raw_answer["text"]
        raw_lines = text.splitlines()
        tagged_line_texts = tagged_full.splitlines()

        if len(tagged_line_texts) != len(raw_lines):
            # Level-1 rescue: align on non-empty lines only.
            # Gemini often drops/adds blank lines but tags every content line.
            raw_nonempty_idx = [i for i, l in enumerate(raw_lines) if l.strip()]
            tagged_nonempty  = [l for l in tagged_line_texts if l.strip()]

            if len(tagged_nonempty) == len(raw_nonempty_idx):
                # Perfect non-empty match — reconstruct the full-length list
                aligned = [""] * len(raw_lines)
                for j, i in enumerate(raw_nonempty_idx):
                    aligned[i] = tagged_nonempty[j]
                tagged_line_texts = aligned
                logger.debug(
                    "Line-count mismatch rescued by non-empty alignment "
                    "(raw=%d tagged=%d non-empty=%d) for %s",
                    len(raw_lines), len(tagged_full.splitlines()),
                    len(tagged_nonempty), raw_answer.get("answer_id", "?"),
                )
            else:
                # True fallback: LLM response is structurally unusable
                logger.warning(
                    "Line-count mismatch, falling back to <null> wrapping "
                    "(raw=%d tagged=%d non-empty raw=%d non-empty tagged=%d) for %s",
                    len(raw_lines), len(tagged_full.splitlines()),
                    len(raw_nonempty_idx), len(tagged_nonempty),
                    raw_answer.get("answer_id", "?"),
                )
                tagged_line_texts = [
                    f"<null>{l}</null>" if l.strip() else ""
                    for l in raw_lines
                ]

        tagged_lines: List[TaggedLine] = []
        for i, raw_line in enumerate(raw_lines):
            ttext = tagged_line_texts[i]
            if not raw_line.strip():
                tagged_lines.append(TaggedLine(
                    line_index=i, raw_text=raw_line, tagged_text="", spans=[],
                ))
            else:
                tagged_lines.append(TaggedLine(
                    line_index=i,
                    raw_text=raw_line,
                    tagged_text=ttext,
                    spans=_parse_tagged_line(ttext, raw_line),
                ))

        return TaggedAnswer(
            answer_id=raw_answer["answer_id"],
            query_id=raw_answer.get("query_id"),
            model=raw_answer.get("model"),
            provider=raw_answer.get("provider"),
            raw_text=text,
            lines=tagged_lines,
        )

