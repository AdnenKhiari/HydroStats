"""
them_desc_analysis/analyzer.py

LLM-powered thematic analyzer using the provider abstraction from answer_analysis.

Design
──────
• System prompt is loaded from ``them_desc_analysis/prompt.txt`` (edit freely).
• The full concatenated text (3 chatbot answers) is sent to the LLM in ONE call.
• A forced tool call (``submit_thematic_analysis``) makes the model return a
  structured JSON with a single non-null ``analysis`` string — no post-parsing.
• Reuses ``answer_analysis.providers`` (GeminiProvider / AnthropicProvider) so
  no LLM plumbing is duplicated.

Tool schema
───────────
    {
      "name": "submit_thematic_analysis",
      "input_schema": {
        "type": "object",
        "properties": {
          "analysis": { "type": "string" }
        },
        "required": ["analysis"]
      }
    }
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from answer_analysis.providers import GeminiProvider, LLMProvider

from .models import AnswerGroup, ThematicResult

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_MODULE_DIR  = Path(__file__).parent
_PROMPT_FILE = _MODULE_DIR / "prompt.txt"

# ──────────────────────────────────────────────────────────────────────────────
# Tool schema — forces the LLM to return a single analysis string
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_DEFINITION: Dict[str, Any] = {
    "name": "submit_thematic_analysis",
    "description": (
        "Submit the qualitative thematic analysis of the provided AI-generated responses. "
        "The analysis must be a non-empty descriptive string covering observed patterns, "
        "visibility mechanisms, and semantic tendencies."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": (
                    "The full thematic analysis text. Must be a non-empty string. "
                    "Do not truncate or summarise — provide the complete analysis."
                ),
            }
        },
        "required": ["analysis"],
    },
}

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────

class ThematicAnalyzer:
    """
    Analyses a group of concatenated chatbot answers for one question/version.

    Parameters
    ----------
    api_key      : API key for the default provider (Gemini).
                   Ignored when a custom ``provider`` is passed.
    model        : Model identifier.
                   Ignored when a custom ``provider`` is passed.
    max_tokens   : Maximum tokens the model may generate.
    rpm_limit    : Minimum seconds between consecutive API calls.
    prompt_file  : Override path to the system-prompt text file.
    provider     : Custom ``LLMProvider``.  When supplied, ``api_key`` and
                   ``model`` are ignored.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        if provider is not None:
            self.provider = provider
        else:
            self.provider = GeminiProvider(api_key=api_key, model=model)

        self.model      = model
        self.max_tokens = max_tokens
        self._min_interval   = rpm_limit
        self._last_call_time: float = 0.0

        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)

    # ── prompt loading ────────────────────────────────────────────────────────

    @staticmethod
    def _load_prompt(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {path}\n"
                "Expected at them_desc_analysis/prompt.txt"
            )
        return path.read_text(encoding="utf-8").strip()

    def reload_prompt(self, prompt_file: Optional[str | Path] = None) -> None:
        """Hot-reload the prompt file without recreating the analyzer."""
        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)

    # ── rate limiting ─────────────────────────────────────────────────────────

    def _rate_limit_wait(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    # ── API call ──────────────────────────────────────────────────────────────

    def _call_llm(self, concatenated_text: str) -> str:
        """Send concatenated answers to the provider and return the analysis string."""
        self._rate_limit_wait()

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": concatenated_text}
        ]

        result = self.provider.complete(
            messages=messages,
            system=self._system_prompt,
            tools=[_TOOL_DEFINITION],
            tool_name="submit_thematic_analysis",
            max_tokens=self.max_tokens,
        )

        self._last_call_time = time.monotonic()

        analysis = result.get("analysis", "").strip()
        if not analysis:
            logger.warning("LLM returned an empty analysis — returning empty string")
        return analysis

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        group: AnswerGroup,
        *,
        verbose: bool = False,
    ) -> ThematicResult:
        """
        Perform thematic analysis on one AnswerGroup.

        Parameters
        ----------
        group   : the (question, version) answer group to analyse.
        verbose : print progress to stdout.
        """
        if verbose:
            print(
                f"  Analysing question {group.question_id} "
                f"[{group.version}] "
                f"({', '.join(group.providers)}) …"
            )

        if not group.concatenated_text.strip():
            logger.warning(
                "Empty concatenated text for %s / %s — skipping LLM call",
                group.question_id, group.version,
            )
            return ThematicResult(
                question_id=group.question_id,
                version=group.version,
                providers=group.providers,
                analysis="",
            )

        analysis = self._call_llm(group.concatenated_text)

        return ThematicResult(
            question_id=group.question_id,
            version=group.version,
            providers=group.providers,
            analysis=analysis,
        )
