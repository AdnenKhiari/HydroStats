"""
theme_generation_analysis/generator.py

LLM-powered Step 3 theme generation over Step 2 codebooks.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from answer_analysis.providers import GeminiProvider, LLMProvider

from .models import GeneratedTheme, ThemeGenerationCorpus, ThemeGenerationResult, ThemeSourceCode

_MODULE_DIR = Path(__file__).parent
_PROMPT_FILE = _MODULE_DIR / "prompt.txt"

logger = logging.getLogger(__name__)

class ThemeGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 65536,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        if provider is not None:
            self.provider = provider
        else:
            self.provider = GeminiProvider(api_key=api_key, model=model)

        self.model = model
        self.max_tokens = max_tokens
        self._min_interval = rpm_limit
        self._last_call_time: float = 0.0
        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)

    @staticmethod
    def _load_prompt(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {path}\nExpected at theme_generation_analysis/prompt.txt"
            )
        return path.read_text(encoding="utf-8").strip()

    def _rate_limit_wait(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    @staticmethod
    def _build_user_message(corpus: ThemeGenerationCorpus) -> str:
        lines = [
            f"VERSION: {corpus.version}",
            f"QUESTION COUNT: {corpus.question_count}",
            f"CODE COUNT: {len(corpus.source_codes)}",
            "",
            "STEP 2 CODEBOOK ENTRIES",
            "======================",
        ]
        for code in corpus.source_codes:
            lines.extend([
                f"CODE_ID: {code.code_id}",
                f"CODE_NAME: {code.code_name}",
                f"DESCRIPTION: {code.description}",
                "",
            ])
        lines.extend([
            "OUTPUT FORMAT",
            "============",
            "Return only a valid JSON array.",
            "Do not use markdown fences.",
            "Each item must be an object with exactly these keys:",
            '- "theme_name": string',
            '- "description": string',
            '- "code_ids": array of strings',
            "Keep descriptions concise.",
        ])
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return stripped

    def _call_llm(self, corpus: ThemeGenerationCorpus) -> Dict[str, Any]:
        self._rate_limit_wait()
        if not isinstance(self.provider, GeminiProvider):
            raise TypeError("ThemeGenerator plain-text mode requires GeminiProvider")

        messages = [{"role": "user", "content": self._build_user_message(corpus)}]
        contents = self.provider._to_gemini_contents(messages)
        system_text = self.provider._extract_text(self._system_prompt)
        config = self.provider.build_generate_config(
            system_text=system_text,
            max_tokens=self.max_tokens,
        )
        response = self.provider.client.models.generate_content(
            model=self.provider.model,
            contents=contents,
            config=config,
        )
        self._last_call_time = time.monotonic()
        text_parts: List[str] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            parts = content.parts if content and content.parts else []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(text)
        return {"themes": self._extract_raw_themes_from_text("\n".join(text_parts))}

    @staticmethod
    def _extract_raw_themes_from_text(text: str) -> List[Dict[str, Any]]:
        text = ThemeGenerator._strip_code_fences(text)
        if not text:
            return []

        candidates = [text]
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            candidates.insert(0, text[start:end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                return [entry for entry in parsed if isinstance(entry, dict)]

        logger.warning("Step 3 returned text but it could not be parsed as a JSON array")
        return []

    @staticmethod
    def _normalize_themes(
        raw_themes: List[Dict[str, Any]],
        source_codes: List[ThemeSourceCode],
    ) -> List[GeneratedTheme]:
        valid_ids = {code.code_id for code in source_codes}
        themes: List[GeneratedTheme] = []
        for entry in raw_themes:
            if not isinstance(entry, dict):
                continue
            theme_name = str(entry.get("theme_name", "")).strip()
            description = str(entry.get("description", "")).strip()
            raw_ids = entry.get("code_ids", [])
            if not theme_name or not description or not isinstance(raw_ids, list):
                continue

            code_ids: List[str] = []
            for code_id in raw_ids:
                normalized_id = str(code_id).strip()
                if normalized_id in valid_ids and normalized_id not in code_ids:
                    code_ids.append(normalized_id)

            if not code_ids:
                continue

            themes.append(GeneratedTheme(
                theme_name=theme_name,
                description=description,
                code_ids=code_ids,
            ))

        return themes

    def generate(
        self,
        corpus: ThemeGenerationCorpus,
        *,
        verbose: bool = False,
    ) -> ThemeGenerationResult:
        if verbose:
            print(
                f"  Generating themes for version {corpus.version} "
                f"({len(corpus.source_codes)} Step 2 codes) ..."
            )

        if not corpus.source_codes:
            return ThemeGenerationResult(
                version=corpus.version,
                question_count=0,
                code_count=0,
                themes=[],
                source_codes=[],
            )

        result = self._call_llm(corpus)
        raw_themes = result.get("themes", [])
        if not isinstance(raw_themes, list):
            raw_themes = []

        themes = self._normalize_themes(raw_themes, corpus.source_codes)

        return ThemeGenerationResult(
            version=corpus.version,
            question_count=corpus.question_count,
            code_count=len(corpus.source_codes),
            themes=themes,
            source_codes=corpus.source_codes,
        )