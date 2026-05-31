"""
codification_analysis/codifier.py

LLM-powered Step 2 open codifier.

The codifier receives the original grouped chatbot answers plus the thematic
analysis text for one (question, version) unit. It returns:
- inline XML-tagged grouped-answer text using neutral tags like Code_1
- a structured codebook mapping those tags to inductive code definitions
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from answer_analysis.providers import GeminiProvider, LLMProvider

from .models import CodeDefinition, CodedAnalysis, CodedLine, CodedSpan, CodificationInput

_MODULE_DIR = Path(__file__).parent
_PROMPT_FILE = _MODULE_DIR / "prompt.txt"
_FEWSHOT_FILE = _MODULE_DIR / "fewshot.json"

logger = logging.getLogger(__name__)

_TOOL_DEFINITION = {
    "name": "submit_open_codification",
    "description": (
        "Submit the fully XML-tagged grouped answers together with a codebook. "
        "Every character of the grouped answers must appear inside exactly one XML tag. "
        "Use neutral tags like Code_1, Code_2, plus null."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tagged_text": {
                "type": "string",
                "description": (
                    "The complete grouped-answer text with every character wrapped in exactly one XML tag. "
                    "Use only tags like <Code_1>...</Code_1>, <Code_2>...</Code_2>, and <null>...</null>. "
                    "Line count must match the grouped-answer input exactly."
                ),
            },
            "codebook": {
                "type": "array",
                "description": "Mapping from each neutral XML tag to its inductive code definition based on the grouped answers.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "code_name": {"type": "string"},
                        "description": {"type": "string"},
                        "representative_excerpt": {"type": "string"},
                    },
                    "required": ["tag", "code_name", "description", "representative_excerpt"],
                },
            },
        },
        "required": ["tagged_text", "codebook"],
    },
}

_TAG_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?(\w+)>")
_VALID_TAG_RE = re.compile(r"^Code_\d+$")


def _is_valid_tag(tag: str) -> bool:
    return tag == "null" or bool(_VALID_TAG_RE.match(tag))


def _parse_tagged_segments(tagged_text: str) -> List[tuple[str, str]]:
    """
    Tolerantly parse tagged text into sequential (tag, content) segments.

    Gemini occasionally returns malformed tag sequences such as:
    </Code_2>content</Code_2>

    In practice, that first closing tag is often acting like an opener. This
    parser treats stray closing tags encountered while in `null` mode as a tag
    switch so the content can still be aligned back onto the raw source text.
    """
    segments: List[tuple[str, str]] = []
    current_tag = "null"
    cursor = 0

    def append_segment(tag: str, text: str) -> None:
        if not text:
            return
        segments.append((tag, text))

    for match in _ANY_TAG_RE.finditer(tagged_text):
        token = match.group(0)
        tag = match.group(1)

        if cursor < match.start():
            append_segment(current_tag, tagged_text[cursor:match.start()])

        if _is_valid_tag(tag):
            is_closing = token.startswith("</")
            if is_closing:
                if current_tag == tag:
                    current_tag = "null"
                elif current_tag == "null":
                    current_tag = tag
            else:
                current_tag = tag

        cursor = match.end()

    if cursor < len(tagged_text):
        append_segment(current_tag, tagged_text[cursor:])

    return segments


class OpenCodifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 16384,
        rpm_limit: float = 1.0,
        prompt_file: Optional[str | Path] = None,
        fewshot_file: Optional[str | Path] = None,
        use_cache: bool = True,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        if provider is not None:
            self.provider = provider
        else:
            self.provider = GeminiProvider(api_key=api_key, model=model)

        self.model = model
        self.max_tokens = max_tokens
        self.use_cache = use_cache
        self._min_interval = rpm_limit
        self._last_call_time: float = 0.0
        self._system_prompt = self._load_prompt(prompt_file or _PROMPT_FILE)
        self._few_shot = self._load_fewshot(fewshot_file or _FEWSHOT_FILE)

    @staticmethod
    def _load_prompt(path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"System prompt file not found: {path}\nExpected at codification_analysis/prompt.txt"
            )
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_fewshot(path: str | Path) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _cached_system(self) -> Any:
        if not self.use_cache:
            return self._system_prompt
        return [{
            "type": "text",
            "text": self._system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]

    def _cached_few_shot(self) -> List[Dict[str, Any]]:
        if not self.use_cache or not self._few_shot:
            return self._few_shot
        messages = [message.copy() for message in self._few_shot]
        last = messages[-1]
        content = last["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        else:
            content = [block.copy() for block in content]
        content[-1]["cache_control"] = {"type": "ephemeral"}
        last["content"] = content
        return messages

    def _rate_limit_wait(self) -> None:
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _build_user_message(self, item: CodificationInput) -> str:
        return (
            "TEXT TO CODE: ORIGINAL GROUPED ANSWERS\n"
            "=====================================\n"
            f"{item.source_answers}\n\n"
            "THEMATIC ANALYSIS CONTEXT ONLY\n"
            "==============================\n"
            f"{item.raw_analysis}"
        )

    def _call_llm(self, item: CodificationInput) -> Dict[str, Any]:
        self._rate_limit_wait()
        messages = self._cached_few_shot() + [{
            "role": "user",
            "content": self._build_user_message(item),
        }]
        result = self.provider.complete(
            messages=messages,
            system=self._cached_system(),
            tools=[_TOOL_DEFINITION],
            tool_name="submit_open_codification",
            max_tokens=self.max_tokens,
        )
        self._last_call_time = time.monotonic()
        return result

    @staticmethod
    def _normalize_codebook(codebook: List[Dict[str, Any]]) -> List[CodeDefinition]:
        normalized: List[CodeDefinition] = []
        seen_tags = set()
        for entry in codebook:
            tag = str(entry.get("tag", "")).strip()
            if not _VALID_TAG_RE.match(tag) or tag in seen_tags:
                continue
            seen_tags.add(tag)
            normalized.append(CodeDefinition(
                tag=tag,
                code_name=str(entry.get("code_name", "")).strip() or tag,
                description=str(entry.get("description", "")).strip(),
                representative_excerpt=str(entry.get("representative_excerpt", "")).strip(),
            ))
        return normalized

    @staticmethod
    def _build_lines_from_tagged_text(
        tagged_text: str,
        code_map: Dict[str, CodeDefinition],
    ) -> List[CodedLine]:
        """Rebuild lines directly from the returned XML-tagged text."""
        if not tagged_text:
            return []

        lines: List[CodedLine] = []
        line_index = 0
        current_raw_parts: List[str] = []
        current_tagged_parts: List[str] = []
        current_spans: List[CodedSpan] = []

        def flush_line() -> None:
            nonlocal line_index, current_raw_parts, current_tagged_parts, current_spans
            lines.append(CodedLine(
                line_index=line_index,
                raw_text="".join(current_raw_parts),
                tagged_text="".join(current_tagged_parts),
                spans=current_spans,
            ))
            line_index += 1
            current_raw_parts = []
            current_tagged_parts = []
            current_spans = []

        def append_text(tag: Optional[str], text: str) -> None:
            if not text:
                return
            line_text = "".join(current_raw_parts)
            code = code_map.get(tag) if tag else None
            current_raw_parts.append(text)
            tag_name = tag or "null"
            current_tagged_parts.append(f"<{tag_name}>{text}</{tag_name}>")
            current_spans.append(CodedSpan(
                text=text,
                tag=tag,
                code_name=code.code_name if code else None,
                description=code.description if code else None,
                char_start=len(line_text),
                char_end=len(line_text) + len(text),
            ))

        for raw_tag, content in _parse_tagged_segments(tagged_text):
            tag = raw_tag if raw_tag in code_map else None
            remaining = content

            while True:
                newline_index = remaining.find("\n")
                if newline_index == -1:
                    append_text(tag, remaining)
                    break

                append_text(tag, remaining[:newline_index])
                flush_line()
                remaining = remaining[newline_index + 1:]

        if current_raw_parts or current_tagged_parts or current_spans or not lines:
            flush_line()

        return lines

    def build_coded_analysis(
        self,
        item: CodificationInput,
        tagged_text: str,
        raw_codebook: List[Dict[str, Any]],
    ) -> CodedAnalysis:
        raw_text = item.source_answers
        raw_lines = raw_text.splitlines()
        codebook = self._normalize_codebook(raw_codebook)
        code_map = {code.tag: code for code in codebook}

        if tagged_text.strip():
            lines = self._build_lines_from_tagged_text(tagged_text, code_map)
        else:
            logger.warning(
                "Missing tagged_text for %s / %s; falling back to <null> wrapping",
                item.version,
                item.question_id,
            )
            lines = []
            for index, raw_line in enumerate(raw_lines):
                if not raw_line.strip():
                    lines.append(CodedLine(line_index=index, raw_text=raw_line, tagged_text="", spans=[]))
                    continue
                lines.append(CodedLine(
                    line_index=index,
                    raw_text=raw_line,
                    tagged_text=f"<null>{raw_line}</null>",
                    spans=[CodedSpan(
                        text=raw_line,
                        tag=None,
                        code_name=None,
                        description=None,
                        char_start=0,
                        char_end=len(raw_line),
                    )],
                ))

        return CodedAnalysis(
            question_id=item.question_id,
            version=item.version,
            providers=item.providers,
            raw_analysis=item.raw_analysis,
            source_answers=item.source_answers,
            tagged_text=tagged_text,
            codebook=codebook,
            lines=lines,
        )

    def codify(self, item: CodificationInput, *, verbose: bool = False) -> CodedAnalysis:
        if verbose:
            print(f"  Codifying question {item.question_id} [{item.version}] ...")
        if not item.source_answers.strip():
            return self.build_coded_analysis(item, "", [])
        result = self._call_llm(item)
        tagged_text = str(result.get("tagged_text", ""))
        codebook = result.get("codebook", [])
        if not isinstance(codebook, list):
            codebook = []
        return self.build_coded_analysis(item, tagged_text, codebook)
