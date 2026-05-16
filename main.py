# file: parsing_layer.py
# Standard-library only. Suitable for later integration with Streamlit.

from __future__ import annotations
import json
import os
import uuid
from typing import Dict, List, Tuple, Optional, Any, Iterable, Union
from dataclasses import dataclass, field


# -----------------------------
# Data Models (in-memory schema)
# -----------------------------

@dataclass
class Query:
    query_id: str
    text: str
    category: Optional[str] = None
    task_type: Optional[str] = None
    created_at: Optional[str] = None
    source_file: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    answer_id: str
    query_id: Optional[str]  # may be None prior to linking if provided only by text
    product: Optional[str]
    model: Optional[str]
    response: str
    prompt_variant: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    usage: Dict[str, Any] = field(default_factory=dict)
    run_context: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    # Optional helper to assist linking when query_id is missing:
    query_text_fallback: Optional[str] = None
    source_file: Optional[str] = None


@dataclass
class Annotation:
    annotation_id: str
    answer_id: str
    query_id: Optional[str]
    type: str
    score: Optional[float] = None
    label: Optional[str] = None
    rater: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[str] = None


@dataclass
class ParsingReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)


@dataclass
class LinkedCorpus:
    # Canonical containers
    queries: Dict[str, Query]                    # query_id -> Query
    answers: Dict[str, Answer]                   # answer_id -> Answer
    annotations: Dict[str, Annotation]           # annotation_id -> Annotation
    # Indexes for fast UI rendering
    by_query: Dict[str, List[str]]               # query_id -> [answer_id, ...]
    by_product: Dict[str, List[str]]             # product -> [answer_id, ...]
    by_model: Dict[str, List[str]]               # model -> [answer_id, ...]
    # Helpful join maps
    query_text_to_id: Dict[str, str]             # exact text -> query_id
    # Diagnostics
    report: ParsingReport


# -----------------------------
# Utilities
# -----------------------------

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _read_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def _load_json_any(path: str) -> Any:
    """Load JSON, JSONL, or a .txt file that contains JSON.
    If JSONL, returns a list of objects.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jsonl":
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at {path}:{i}: {e}") from e
        return items
    elif ext in (".json", ".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {path}")


# -----------------------------
# Query Loading
# -----------------------------

def load_queries_from_txt(path: str, report: Optional[ParsingReport] = None) -> Dict[str, Query]:
    """
    Reads plain text queries (one per line) and generates Query objects with new query_ids.
    Empty or comment lines (# ...) are ignored.
    """
    report = report or ParsingReport()
    queries: Dict[str, Query] = {}
    for i, line in enumerate(_read_lines(path), start=1):
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        qid = _gen_id("q")
        queries[qid] = Query(query_id=qid, text=raw, source_file=f"{path}:{i}")
    report.add_info(f"Loaded {len(queries)} queries from TXT: {path}")
    return queries


def load_queries_from_json(path: str, report: Optional[ParsingReport] = None) -> Tuple[Dict[str, Query], List[Dict[str, Any]]]:
    """
    Reads queries from JSON / JSONL. Supports:
      - Normalized: {"queries": [ {...}, ... ]}
      - Raw list: [ {...}, ... ]
      - Embedded answers inside queries (returned separately as raw for later ingestion)
    Returns (queries_dict, embedded_answers_raw)
    """
    report = report or ParsingReport()
    data = _load_json_any(path)
    items: List[Dict[str, Any]]
    # Support "items" wrapper (actual export format), "queries" wrapper, or bare list
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, dict) and "queries" in data and isinstance(data["queries"], list):
        items = data["queries"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unrecognized queries JSON structure in {path}")

    queries: Dict[str, Query] = {}
    embedded_answers_raw: List[Dict[str, Any]] = []

    for idx, obj in enumerate(items, start=1):
        if not isinstance(obj, dict):
            report.add_error(f"{path}[{idx}] is not an object; skipping.")
            continue

        # Accept both normalized field "text" and actual export field "query"
        text = obj.get("text") or obj.get("query")
        if not text or not isinstance(text, str):
            report.add_warning(f"{path}[{idx}] missing or invalid 'text'/'query' field; skipping.")
            continue

        # Accept both normalized "query_id" and actual export "key"
        qid = obj.get("query_id") or obj.get("key") or _gen_id("q")

        # "intent" maps to both category and task_type in the actual format
        intent = obj.get("intent")

        # Preserve extra fields (e.g. branded) in metadata
        metadata = dict(obj.get("metadata") or {})
        if "branded" in obj:
            metadata["branded"] = obj["branded"]

        query = Query(
            query_id=qid,
            text=text,
            category=obj.get("category") or intent,
            task_type=obj.get("task_type") or intent,
            created_at=obj.get("created_at") or obj.get("createdAt"),
            source_file=f"{path}[{idx}]",
            metadata=metadata,
        )
        queries[qid] = query

        # Collect embedded answers if any
        ans_list = obj.get("answers")
        if isinstance(ans_list, list):
            for ans_idx, ans in enumerate(ans_list, start=1):
                if isinstance(ans, dict):
                    if "query_id" not in ans:
                        ans["query_id"] = qid
                    embedded_answers_raw.append(ans)
                else:
                    report.add_warning(f"{path}[{idx}].answers[{ans_idx}] is not an object; skipping.")

    report.add_info(f"Loaded {len(queries)} queries from JSON: {path}")
    if embedded_answers_raw:
        report.add_info(f"Found {len(embedded_answers_raw)} embedded answers in: {path}")
    return queries, embedded_answers_raw


def load_queries(path: str, report: Optional[ParsingReport] = None) -> Tuple[Dict[str, Query], List[Dict[str, Any]]]:
    """
    Convenience loader: detects extension and delegates.
    Returns (queries_dict, embedded_answers_raw).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        # The file may be a JSON document saved with a .txt extension; try JSON first.
        try:
            return load_queries_from_json(path, report)
        except (json.JSONDecodeError, ValueError):
            return load_queries_from_txt(path, report), []
    elif ext in (".json", ".jsonl"):
        return load_queries_from_json(path, report)
    else:
        raise ValueError(f"Unsupported queries file format: {path}")


# -----------------------------
# Answer Loading
# -----------------------------

def _normalize_answer_obj(obj: Dict[str, Any], source: str, report: ParsingReport) -> Optional[Answer]:
    """
    Convert a raw dict into an Answer. Flexible field recognition, with fallbacks.
    Required: response (string). Strongly recommended: product, model, query_id or query_text.
    """
    if not isinstance(obj, dict):
        report.add_warning(f"{source}: answer is not an object; skipping.")
        return None

    response = obj.get("response") or obj.get("text") or obj.get("content")
    if not isinstance(response, str) or not response.strip():
        report.add_warning(f"{source}: missing 'response' (or compatible field); skipping.")
        return None

    # Accept both normalized "answer_id" and actual export "key"
    answer_id = obj.get("answer_id") or obj.get("key") or _gen_id("a")
    # Accept both normalized "product" and actual export "provider"
    product = obj.get("product") or obj.get("provider")
    model = obj.get("model")

    # Prefer explicit query_id; fall back to actual export "queryKey"; then text fallback
    query_id = obj.get("query_id") or obj.get("queryKey")
    query_text_fallback = obj.get("query_text") or obj.get("prompt")

    # Optional known structures
    params = obj.get("params") or obj.get("generation_params") or {}

    # Build timing: use explicit timing dict or fall back to createdAt timestamp
    timing: Dict[str, Any] = dict(obj.get("timing") or {})
    if not timing and obj.get("createdAt"):
        timing["created_at"] = obj["createdAt"]

    usage = obj.get("usage") or {}

    # Build run_context: merge base dict + visibilityReportKey + stats
    run_context: Dict[str, Any] = dict(obj.get("run_context") or {})
    if obj.get("visibilityReportKey"):
        run_context["visibility_report_key"] = obj["visibilityReportKey"]
    if obj.get("stats") and isinstance(obj["stats"], dict):
        run_context["stats"] = obj["stats"]

    annotations = obj.get("annotations") or []
    sources = obj.get("sources") or []

    return Answer(
        answer_id=answer_id,
        query_id=query_id,
        product=product,
        model=model,
        response=response,
        prompt_variant=obj.get("prompt_variant"),
        params=params if isinstance(params, dict) else {},
        timing=timing if isinstance(timing, dict) else {},
        usage=usage if isinstance(usage, dict) else {},
        run_context=run_context if isinstance(run_context, dict) else {},
        annotations=annotations if isinstance(annotations, list) else [],
        sources=sources if isinstance(sources, list) else [],
        query_text_fallback=query_text_fallback if isinstance(query_text_fallback, str) else None,
        source_file=source,
    )


def load_answers_from_files(paths: List[str], report: Optional[ParsingReport] = None) -> Dict[str, Answer]:
    """
    Reads one or more JSON / JSONL files of answers.
    Accepts:
      - JSONL with one answer per line
      - JSON array
      - JSON dict { "answers": [ ... ] }
    Flexible field names handled by _normalize_answer_obj.
    """
    report = report or ParsingReport()
    answers: Dict[str, Answer] = {}

    for path in paths:
        data = _load_json_any(path)
        raw_items: List[Dict[str, Any]]

        # Support "items" wrapper (actual export format), "answers" wrapper, or bare list
        if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
            raw_items = data["items"]
        elif isinstance(data, dict) and "answers" in data and isinstance(data["answers"], list):
            raw_items = data["answers"]
        elif isinstance(data, list):
            raw_items = data
        else:
            report.add_error(f"Unrecognized answers structure in {path}; expected list or 'answers'/'items' array.")
            continue

        # Derive a product name from the filename as a fallback
        # e.g. "chatgpt" from "chatgpt.answers.txt"
        filename_product = os.path.splitext(os.path.basename(path))[0].split(".")[0]

        for idx, obj in enumerate(raw_items, start=1):
            if not isinstance(obj, dict):
                report.add_warning(f"{path}[{idx}] answer is not an object; skipping.")
                continue
            # Inject filename-based product only when the record carries neither field
            if not obj.get("product") and not obj.get("provider"):
                obj = dict(obj)  # shallow copy — do not mutate the source data
                obj["product"] = filename_product
            ans = _normalize_answer_obj(obj, f"{path}[{idx}]", report)
            if ans is None:
                continue
            # De-dup by answer_id, last write wins (warn on collision)
            if ans.answer_id in answers:
                report.add_warning(f"Duplicate answer_id {ans.answer_id} at {path}[{idx}]; overwriting previous.")
            answers[ans.answer_id] = ans

        report.add_info(f"Loaded {len(raw_items)} raw answers from: {path}")

    return answers


# -----------------------------
# Linking & Indexing
# -----------------------------

def link_corpus(
    queries: Dict[str, Query],
    answers: Dict[str, Answer],
    embedded_answers_raw: Optional[List[Dict[str, Any]]] = None,
    report: Optional[ParsingReport] = None
) -> LinkedCorpus:
    """
    Link answers to queries:
      1) If answer.query_id exists and matches → link
      2) Else if answer.query_text_fallback matches a query.text exactly → link
      3) Otherwise, leave unlinked (warn)
    Also ingests embedded answers if provided.
    Builds helpful indexes for UI.
    """
    report = report or ParsingReport()

    # Ingest embedded answers (if any) into the same 'answers' dict
    if embedded_answers_raw:
        for idx, obj in enumerate(embedded_answers_raw, start=1):
            ans = _normalize_answer_obj(obj, f"embedded[{idx}]", report)
            if ans is None:
                continue
            if ans.answer_id in answers:
                report.add_warning(f"Duplicate embedded answer_id {ans.answer_id}; overwriting existing.")
            answers[ans.answer_id] = ans

    # Build quick lookup by exact text for fallback
    query_text_to_id: Dict[str, str] = {}
    for qid, q in queries.items():
        if q.text in query_text_to_id:
            # If duplicate texts exist, matching by text becomes ambiguous
            report.add_warning(f"Duplicate query text encountered; exact-text linking may be ambiguous: '{q.text}'")
        query_text_to_id[q.text] = qid

    # Attempt linking for answers missing query_id
    for aid, ans in answers.items():
        if ans.query_id and ans.query_id in queries:
            continue
        if ans.query_id and ans.query_id not in queries:
            report.add_warning(f"Answer {aid} refers to unknown query_id={ans.query_id}; trying text fallback.")
        if not ans.query_text_fallback:
            report.add_warning(f"Answer {aid} does not have query_id nor query_text; remains unlinked.")
            continue
        match_qid = query_text_to_id.get(ans.query_text_fallback)
        if match_qid:
            ans.query_id = match_qid
        else:
            report.add_warning(f"Answer {aid}: no exact text match for query_text='{ans.query_text_fallback}'; remains unlinked.")

    # Build indexes
    by_query: Dict[str, List[str]] = {}
    by_product: Dict[str, List[str]] = {}
    by_model: Dict[str, List[str]] = {}

    for aid, ans in answers.items():
        # by_query
        if ans.query_id:
            by_query.setdefault(ans.query_id, []).append(aid)

        # by_product
        if ans.product:
            by_product.setdefault(ans.product, []).append(aid)
        else:
            by_product.setdefault("_unknown_product", []).append(aid)

        # by_model
        if ans.model:
            by_model.setdefault(ans.model, []).append(aid)
        else:
            by_model.setdefault("_unknown_model", []).append(aid)

    # No separate annotations input yet; collect from answers (flatten)
    annotations: Dict[str, Annotation] = {}
    for aid, ans in answers.items():
        for ann_idx, raw in enumerate(ans.annotations, start=1):
            if not isinstance(raw, dict):
                report.add_warning(f"Answer {aid} contains non-object annotation; skipping.")
                continue
            ann_id = raw.get("annotation_id") or _gen_id("ann")
            ann = Annotation(
                annotation_id=ann_id,
                answer_id=aid,
                query_id=ans.query_id,
                type=raw.get("type", "unspecified"),
                score=raw.get("score"),
                label=raw.get("label"),
                rater=raw.get("rater"),
                notes=raw.get("notes"),
                created_at=raw.get("created_at"),
            )
            if ann.annotation_id in annotations:
                report.add_warning(f"Duplicate annotation_id {ann.annotation_id}; overwriting.")
            annotations[ann.annotation_id] = ann

    # Final informational stats
    total_linked = sum(1 for a in answers.values() if a.query_id in queries)
    total_unlinked = len(answers) - total_linked
    report.add_info(f"Linked answers: {total_linked}, Unlinked answers: {total_unlinked}, Total queries: {len(queries)}")

    return LinkedCorpus(
        queries=queries,
        answers=answers,
        annotations=annotations,
        by_query=by_query,
        by_product=by_product,
        by_model=by_model,
        query_text_to_id=query_text_to_id,
        report=report
    )


# -----------------------------
# Convenience: Unified Entry
# -----------------------------

# Absolute path to the directory that contains this file — used for default paths.
_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
_DATA      = os.path.join(_WORKSPACE, "data")

_EXPERIMENT_ORDER  = ["Baseline V0", "V1", "V2", "V1+2"]
_EXCLUDED_FOLDERS  = {"question_themes"}


def list_experiments() -> List[str]:
    """Return experiment folder names found under data/, in canonical order."""
    if not os.path.isdir(_DATA):
        return []
    found  = [d for d in _EXPERIMENT_ORDER if os.path.isdir(os.path.join(_DATA, d))]
    others = sorted(
        d for d in os.listdir(_DATA)
        if os.path.isdir(os.path.join(_DATA, d))
        and d not in _EXPERIMENT_ORDER
        and d not in _EXCLUDED_FOLDERS
    )
    return found + others


# Resolve the default queries file: prefer data/queries.txt.txt, fall back to workspace root.
def _default_queries_file() -> str:
    candidates = [
        os.path.join(_DATA, "queries.txt.txt"),
        os.path.join(_WORKSPACE, "queries.txt.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]  # will raise a clear FileNotFoundError at load time


def build_corpus(
    queries_file: Optional[str] = None,
    answer_files: Optional[List[str]] = None,
    experiment: Optional[str] = None,
) -> LinkedCorpus:
    """
    High-level convenience that:
      - Loads queries (txt/json/jsonl)
      - Loads answers from the given experiment folder (all .txt files inside it),
        or falls back to scanning all known experiment subdirs when experiment=None.
      - Links everything, returning a LinkedCorpus with indexes and a ParsingReport.
    """
    if queries_file is None:
        queries_file = _default_queries_file()

    report = ParsingReport()
    if answer_files is None:
        if experiment is not None:
            # Load every .txt file from the selected experiment folder only.
            exp_dir = os.path.join(_DATA, experiment)
            answer_files = sorted(
                os.path.join(exp_dir, f)
                for f in os.listdir(exp_dir)
                if f.endswith(".txt") and os.path.isfile(os.path.join(exp_dir, f))
            )
        else:
            # Legacy fallback: scan all known experiment subdirs in order.
            _subdirs = [
                d for d in (os.path.join(_DATA, x) for x in _EXPERIMENT_ORDER)
                if os.path.isdir(d)
            ] + [_DATA, _WORKSPACE]
            seen: set = set()
            candidates: List[str] = []
            for d in _subdirs:
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if not f.endswith(".txt"):
                        continue
                    p = os.path.join(d, f)
                    if os.path.isfile(p) and p not in seen:
                        seen.add(p)
                        candidates.append(p)
            answer_files = candidates
    queries, embedded_answers_raw = load_queries(queries_file, report)
    answers: Dict[str, Answer] = {}
    if answer_files:
        answers = load_answers_from_files(answer_files, report)
    corpus = link_corpus(queries, answers, embedded_answers_raw, report)
    return corpus


# -----------------------------
# Serialization Helpers (Optional)
# -----------------------------

def to_normalized_dict(corpus: LinkedCorpus) -> Dict[str, Any]:
    """
    Export a normalized dictionary suitable for JSON export or UI caches.
    """
    return {
        "queries": [
            {
                "query_id": q.query_id,
                "text": q.text,
                "category": q.category,
                "task_type": q.task_type,
                "created_at": q.created_at,
                "source_file": q.source_file,
                "metadata": q.metadata,
            }
            for q in corpus.queries.values()
        ],
        "answers": [
            {
                "answer_id": a.answer_id,
                "query_id": a.query_id,
                "product": a.product,
                "model": a.model,
                "response": a.response,
                "prompt_variant": a.prompt_variant,
                "params": a.params,
                "timing": a.timing,
                "usage": a.usage,
                "run_context": a.run_context,
                "annotations": a.annotations,
                "sources": a.sources,
                "source_file": a.source_file,
            }
            for a in corpus.answers.values()
        ],
        "annotations": [
            {
                "annotation_id": an.annotation_id,
                "answer_id": an.answer_id,
                "query_id": an.query_id,
                "type": an.type,
                "score": an.score,
                "label": an.label,
                "rater": an.rater,
                "notes": an.notes,
                "created_at": an.created_at,
            }
            for an in corpus.annotations.values()
        ],
        "indexes": {
            "by_query": corpus.by_query,
            "by_product": corpus.by_product,
            "by_model": corpus.by_model,
        },
        "diagnostics": {
            "info": corpus.report.info,
            "warnings": corpus.report.warnings,
            "errors": corpus.report.errors,
        }
    }


def to_embedded_queries_view(corpus: LinkedCorpus) -> List[Dict[str, Any]]:
    """
    Export a query-centric view with embedded answers (handy for human review/UIs).
    """
    out = []
    for qid, q in corpus.queries.items():
        answers_ids = corpus.by_query.get(qid, [])
        answers_embedded = []
        for aid in answers_ids:
            a = corpus.answers[aid]
            answers_embedded.append({
                "answer_id": a.answer_id,
                "product": a.product,
                "model": a.model,
                "response": a.response,
                "prompt_variant": a.prompt_variant,
                "params": a.params,
                "timing": a.timing,
                "usage": a.usage,
                "run_context": a.run_context,
                "annotations": a.annotations,
                "sources": a.sources,
                "source_file": a.source_file,
            })
        out.append({
            "query_id": q.query_id,
            "text": q.text,
            "category": q.category,
            "task_type": q.task_type,
            "metadata": q.metadata,
            "answers": answers_embedded
        })
    return out


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    corpus = build_corpus()

    print(f"Queries loaded : {len(corpus.queries)}")
    print(f"Answers loaded : {len(corpus.answers)}")
    print(f"Annotations    : {len(corpus.annotations)}")
    print(f"By product     : { {k: len(v) for k, v in corpus.by_product.items()} }")
    print(f"By model       : { {k: len(v) for k, v in corpus.by_model.items()} }")

    print("\n--- Diagnostics ---")
    for line in corpus.report.info:
        print(f"  INFO : {line}")
    for line in corpus.report.warnings:
        print(f"  WARN : {line}")
    for line in corpus.report.errors:
        print(f"  ERROR: {line}")