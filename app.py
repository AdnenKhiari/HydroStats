# app.py  –  AI Answer Comparison + Hydromea Visibility  (Streamlit)
from __future__ import annotations
import difflib
import html as _html
import json
import pathlib
import re
import urllib.parse
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram
import streamlit as st
import streamlit.components.v1 as _components
from main import Answer, LinkedCorpus, build_corpus, list_experiments


def _fuzzy_match(needle: str, haystack: str) -> bool:
    """Return True if every word in needle loosely matches the haystack.

    A word matches if it appears as a substring OR its similarity ratio
    against any word in the haystack exceeds 0.75 (handles minor typos).
    """
    needle = needle.strip().lower()
    haystack = haystack.lower()
    if not needle:
        return True
    hay_words = haystack.split()
    for token in needle.split():
        if token in haystack:          # fast exact / substring check
            continue
        if any(
            difflib.SequenceMatcher(None, token, w).ratio() >= 0.75
            for w in hay_words
        ):
            continue
        return False
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Visibility Suite",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.main .block-container { padding-top: 3rem; max-width: 100%; }
section[data-testid="stSidebar"] { min-width: 320px; }

/* ── Query card ── */
.query-card {
    background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
    border-left: 5px solid #4361ee;
    border-radius: 8px;
    padding: 1.1rem 1.4rem 1rem;
    margin-bottom: 1.2rem;
}
.query-text {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a1a2e;
    line-height: 1.45;
    margin-bottom: .55rem;
}
.query-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 24px;
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .03em;
    text-transform: uppercase;
}
.badge-researching { background: #e8f5e9; color: #2e7d32; }
.badge-purchasing  { background: #fce4ec; color: #880e4f; }
.badge-branded     { background: #fff8e1; color: #f57c00; }
.badge-id          { background: #f3f4f6; color: #6b7280; font-family: monospace; font-size: .68rem; }
.badge-theme-t1    { background: #e0f2fe; color: #0369a1; }
.badge-theme-t2    { background: #f0fdf4; color: #166534; }
.badge-theme-t3    { background: #fef3c7; color: #92400e; }
.badge-theme-t4    { background: #fce7f3; color: #9d174d; }

/* ── Provider column headers ── */
/* Allow sticky to propagate through Streamlit's column wrappers */
[data-testid="column"],
[data-testid="column"] > div,
[data-testid="stVerticalBlock"] {
    overflow: visible !important;
}
.provider-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 8px 8px 0 0;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: .02em;
    margin-bottom: 0;
    position: sticky;
    top: 3.5rem;   /* clear Streamlit's top toolbar */
    z-index: 999;
}
.provider-openai      { background: #d8f5ed; color: #0a6b4e; border-bottom: 3px solid #10a37f; }
.provider-perplexity  { background: #ede9fe; color: #4c1d95; border-bottom: 3px solid #7c3aed; }
.provider-gemini      { background: #e8f0fe; color: #1a73e8; border-bottom: 3px solid #4285f4; }

/* ── Answer card ── */
.answer-card {
    border: 1px solid #e5e7eb;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1.1rem 1.2rem 1rem;
    background: #fff;
    box-shadow: 0 2px 6px rgba(0,0,0,.05);
}

/* ── Stat pill ── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: .8rem; }
.stat-pill {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .78rem;
    color: #374151;
}
.stat-pill strong { color: #111827; }

/* ── Response text ── */
.response-body {
    font-size: .93rem;
    line-height: 1.65;
    color: #1f2937;
}

/* ── Tagging panel ── */
.tagging-panel {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px dashed #dbe3ee;
}
.tagging-title {
    font-size: .76rem;
    font-weight: 800;
    letter-spacing: .05em;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.tagging-hint {
    font-size: .72rem;
    color: #94a3b8;
    margin-bottom: 8px;
}
.tag-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-bottom: 8px;
}
.tag-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    border-radius: 999px;
    padding: 2px 9px;
    font-size: .68rem;
    font-weight: 700;
    border: 1px solid transparent;
    white-space: nowrap;
}
.tagged-response {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 11px;
    font-size: .86rem;
    line-height: 1.58;
    color: #1e293b;
    white-space: pre-wrap;
}
.tagged-span {
    border-radius: 4px;
    padding: 0 1px;
    border-bottom: 2px solid transparent;
}

.tag-brand { background: #e0f2fe; border-color: #0284c7; }
.tag-credibility { background: #dcfce7; border-color: #16a34a; }
.tag-stat { background: #fef3c7; border-color: #d97706; }
.tag-reco { background: #fee2e2; border-color: #dc2626; }
.tag-other { background: #ede9fe; border-color: #7c3aed; }

.tag-chip-brand { background: #e0f2fe; color: #075985; border-color: #7dd3fc; }
.tag-chip-credibility { background: #dcfce7; color: #166534; border-color: #86efac; }
.tag-chip-stat { background: #fef3c7; color: #92400e; border-color: #fcd34d; }
.tag-chip-reco { background: #fee2e2; color: #991b1b; border-color: #fca5a5; }
.tag-chip-other { background: #ede9fe; color: #5b21b6; border-color: #c4b5fd; }

/* ── Sources ── */
.source-item {
    padding: 5px 0;
    font-size: .8rem;
    border-bottom: 1px solid #f3f4f6;
    line-height: 1.4;
}
.source-item a { color: #4361ee; text-decoration: none; font-weight: 500; }
.source-item a:hover { text-decoration: underline; }
.source-host { color: #9ca3af; margin-left: 6px; font-size: .72rem; }

/* ── Sidebar query buttons ── */
div[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background: #fff;
    color: #374151;
    font-size: .82rem;
    padding: 7px 10px;
    margin-bottom: 3px;
    transition: background .15s, border-color .15s;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #eef2ff !important;
    border-color: #4361ee !important;
    color: #1e3a8a !important;
}

/* ── No-answer placeholder ── */
.no-answer {
    text-align: center;
    color: #9ca3af;
    padding: 2rem 1rem;
    font-size: .9rem;
}

/* ── Hydromea visibility badges ── */
.vis-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 700;
    white-space: nowrap;
    margin: 2px 2px 2px 0;
}
.vis-sourced   { background: #d1fae5; color: #065f46; border: 1px solid #34d39966; }
.vis-unsourced { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a566; }
.vis-cited     { background: #dbeafe; color: #1e40af; border: 1px solid #93c5fd66; }
.vis-uncited   { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db66; }
.vis-ranked    { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d66; }

/* ── Stat metric cards ── */
.stat-card {
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Brand-visibility constants
# ─────────────────────────────────────────────────────────────────────────────
BRAND_DOMAINS = {"hydromea.com", "hydromea.ch"}

_TAG_STYLE = {
    "Brand Positioning": {"span": "tag-brand", "chip": "tag-chip-brand"},
    "Credibility Signal": {"span": "tag-credibility", "chip": "tag-chip-credibility"},
    "Statistical Use": {"span": "tag-stat", "chip": "tag-chip-stat"},
    "Strong Recommendation": {"span": "tag-reco", "chip": "tag-chip-reco"},
}
_TAG_STYLE_DEFAULT = {"span": "tag-other", "chip": "tag-chip-other"}

TAG_PARTITION_CATEGORIES: List[str] = [
    "Brand Positioning",
    "Credibility Signal",
    "Statistical Use",
    "Strong Recommendation",
]

TAG_PARTITION_COLORS: Dict[str, str] = {
    "Brand Positioning": "#0284c7",
    "Credibility Signal": "#16a34a",
    "Statistical Use": "#d97706",
    "Strong Recommendation": "#dc2626",
}

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT PATTERN REGISTRY
# To track a new product mention: add ONE entry here.
# label, icon, color are used in badges, charts, sidebar — no other file changes needed.
# ─────────────────────────────────────────────────────────────────────────────
# Each entry may include an optional `source_url` (str).
# When present, a `product_{key}_sourced` metric is auto-generated that checks
# whether any answer source URL contains that path — no other file changes needed.
_PRODUCT_PATTERNS: List[dict] = [
    {"key": "diskdrive", "label": "DiskDrive mentioned", "icon": "💿", "color": "#7c3aed",
     "pattern": re.compile(r"disk[\s\-_]?drive", re.IGNORECASE),
     "source_url": "hydromea.com/diskdrive-thrusters"},
    {"key": "luma",      "label": "Luma mentioned",      "icon": "💡", "color": "#db2777",
     "pattern": re.compile(r"\bluma\b",                  re.IGNORECASE),
     "source_url": "hydromea.com/luma-underwater-communication"},
    {"key": "exray",     "label": "Exray mentioned",     "icon": "🔬", "color": "#0891b2",
     "pattern": re.compile(r"\bex[\s\-_]?ray\b",         re.IGNORECASE),
     "source_url": "hydromea.com/exray-underwater-robot"},
]

PRODUCT_META: Dict[str, dict] = {
    "OPENAI":     {"label": "ChatGPT",    "color": "#10a37f", "bg": "#e8faf3", "icon": "🟢"},
    "PERPLEXITY": {"label": "Perplexity", "color": "#7c3aed", "bg": "#f5f3ff", "icon": "🟣"},
    "GEMINI":     {"label": "Gemini",     "color": "#4285f4", "bg": "#e8f0fe", "icon": "🔵"},
}
_DMETA = {"label": "Unknown", "color": "#888", "bg": "#f5f5f5", "icon": "❓"}

# ─────────────────────────────────────────────────────────────────────────────
# Theme taxonomy  —  loaded from groupe1.json
# ─────────────────────────────────────────────────────────────────────────────
_THEME_FILE = pathlib.Path(__file__).parent / "data" / "question_themes" / "groupe1.json"

@st.cache_data(show_spinner=False)
def _load_themes() -> tuple:
    """Returns (text_lower→theme_full_name, code→theme_full_name)."""
    try:
        raw = json.loads(_THEME_FILE.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, {}
    codes   = raw.get("themes", {})    # {"T1": "Comparative & Superlative Seeking", ...}
    mapping = raw.get("mapping", {})   # {"Q1": {"question": "...", "theme": "T1"}, ...}
    text_to_name: dict = {}
    for entry in mapping.values():
        q_text = entry.get("question", "").strip()
        code   = entry.get("theme", "")
        name   = codes.get(code, code)
        if q_text:
            text_to_name[q_text.lower()] = name
    return text_to_name, codes

_TEXT_TO_THEME, _THEME_CODES = _load_themes()

# Theme code → CSS badge class
_THEME_NAME_TO_CSS = {
    name: f"badge-theme-t{i+1}"
    for i, name in enumerate(_THEME_CODES.values())
}

def query_theme(text: str) -> str:
    """Resolve a query text to its full theme name (empty string if not found)."""
    return _TEXT_TO_THEME.get((text or "").strip().lower(), "")

# ─────────────────────────────────────────────────────────────────────────────
# Brand-visibility helpers  —  single source of truth
# ─────────────────────────────────────────────────────────────────────────────
def _nh(h: str) -> str:
    return h.lower().replace("www.", "").strip()

def brand_idxs(ans: Answer) -> List[int]:
    """1-based positions in sources that belong to BRAND_DOMAINS (checks hostname + url)."""
    result = []
    for i, s in enumerate(ans.sources, 1):
        hostname = _nh(s.get("hostname", ""))
        url      = s.get("url", "").lower()
        if hostname in BRAND_DOMAINS or any(d in url for d in BRAND_DOMAINS):
            result.append(i)
    return result


_HYDROMEA_MENTION_RE = re.compile(r"\bhydromea\b", re.IGNORECASE)
_ANY_URL_RE = re.compile(
    r"\b(?:https?://|www\.|(?:[\w-]+\.)+[a-z]{2,})[^\s<>)\]}]*",
    re.IGNORECASE,
)
_BRAND_URL_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?(?:[\w-]+\.)*hydromea\.(?:com|ch)(?:/[^\s<>)\]}]*)?",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BRACKET_CITATION_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")


def _normalize_url_candidate(raw_url: str) -> str:
    candidate = (raw_url or "").strip().rstrip(".,;:!?)]}")
    if not candidate:
        return ""
    if not re.match(r"^[a-z]+://", candidate, re.IGNORECASE):
        candidate = f"https://{candidate}"
    return candidate


def _is_brand_domain_url(raw_url: str) -> bool:
    candidate = _normalize_url_candidate(raw_url)
    if not candidate:
        return False
    parsed = urllib.parse.urlparse(candidate)
    hostname = _nh(parsed.netloc)
    normalized = candidate.lower()
    return hostname in BRAND_DOMAINS or any(domain in normalized for domain in BRAND_DOMAINS)


def _normalized_brand_url(raw_url: str) -> str:
    candidate = _normalize_url_candidate(raw_url)
    if not candidate:
        return ""
    parsed = urllib.parse.urlparse(candidate)
    hostname = _nh(parsed.netloc)
    if hostname not in BRAND_DOMAINS and not any(domain in candidate.lower() for domain in BRAND_DOMAINS):
        return ""
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    normalized = urllib.parse.urlunparse(
        (
            parsed.scheme.lower() or "https",
            hostname,
            path,
            parsed.params,
            parsed.query,
            "",
        )
    )
    return normalized


def _count_hydromea_mentions(text: str) -> int:
    if not text:
        return 0

    url_spans = [
        (match.start(), match.end(), _is_brand_domain_url(match.group(0)))
        for match in _ANY_URL_RE.finditer(text)
    ]
    mention_count = 0
    for match in _HYDROMEA_MENTION_RE.finditer(text):
        match_start = match.start()
        inside_non_brand_url = any(
            start <= match_start < end and not is_brand_url
            for start, end, is_brand_url in url_spans
        )
        if not inside_non_brand_url:
            mention_count += 1
    return mention_count


def _expand_bracket_citation_numbers(raw_text: str) -> List[int]:
    numbers: List[int] = []
    for piece in (raw_text or "").split(","):
        token = piece.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            if start_text.strip().isdigit() and end_text.strip().isdigit():
                start_num = int(start_text.strip())
                end_num = int(end_text.strip())
                if start_num <= end_num:
                    numbers.extend(range(start_num, end_num + 1))
                    continue
        if token.isdigit():
            numbers.append(int(token))
    return numbers


def _count_brand_text_references(text: str, brand_source_idxs: List[int]) -> int:
    brand_idx_set = set(brand_source_idxs or [])
    if not text:
        return len(brand_idx_set)

    url_hits = sum(1 for match in _BRAND_URL_RE.finditer(text) if _is_brand_domain_url(match.group(0)))
    markdown_duplicate_hits = 0
    for match in _MARKDOWN_LINK_RE.finditer(text):
        label_text = match.group(1).strip()
        target_text = match.group(2).strip()
        if _normalized_brand_url(label_text) and _normalized_brand_url(label_text) == _normalized_brand_url(target_text):
            markdown_duplicate_hits += 1
    url_hits = max(0, url_hits - markdown_duplicate_hits)

    if not brand_idx_set:
        return url_hits

    bracket_hits = 0
    for match in _BRACKET_CITATION_RE.finditer(text):
        citation_numbers = _expand_bracket_citation_numbers(match.group(1))
        bracket_hits += sum(1 for number in citation_numbers if number in brand_idx_set)

    return url_hits + bracket_hits + len(brand_idx_set)

def compute_answer_metrics(ans: Answer) -> dict:
    """
    Single source of truth for all per-answer metrics.
    Auto-populates `product_<key>` for every entry in _PRODUCT_PATTERNS.
    """
    _bi   = brand_idxs(ans)
    _text = ans.response or ""
    _hydromea_mention_count = _count_hydromea_mentions(_text)
    metrics: dict = {
        "sourced":         bool(_bi),
        "mentioned":       _hydromea_mention_count > 0,
        "hydromea_mention_count": _hydromea_mention_count,
        "source_position": _bi[0] if _bi else -1,
        "citation_count":  len(_bi),
        "n_sources":       ans.run_context.get("stats", {}).get("totalSources", len(ans.sources)),
        "brand_idxs":      _bi,
    }
    for p in _PRODUCT_PATTERNS:
        _matches = p["pattern"].findall(_text)
        metrics[f'product_{p["key"]}'] = bool(_matches)
        metrics[f'product_{p["key"]}_count'] = len(_matches)
        if "source_url" in p:
            needle = p["source_url"].lower()
            metrics[f'product_{p["key"]}_sourced'] = any(
                needle in s.get("url", "").lower()
                for s in ans.sources
            )
    # Total occurrences of any product mention across all patterns
    metrics["product_mention_count"] = sum(
        metrics[f'product_{p["key"]}_count'] for p in _PRODUCT_PATTERNS
    )
    metrics["hydromea_text_reference_count"] = _count_brand_text_references(_text, _bi)
    metrics["global_score"] = (
        metrics["hydromea_mention_count"]
        + metrics["product_mention_count"]
        + metrics["hydromea_text_reference_count"]
    )
    metrics["global_mentioned"] = metrics["global_score"] > 0
    return metrics


def _build_answer_metrics_index(corpus: LinkedCorpus) -> Dict[str, dict]:
    """Compute metrics once per answer and return an answer_id-indexed store."""
    return {aid: compute_answer_metrics(ans) for aid, ans in corpus.answers.items()}

# Thin convenience wrappers
def is_sourced(ans: Answer) -> bool:
    return compute_answer_metrics(ans)["sourced"]

def is_mentioned(ans: Answer) -> bool:
    return compute_answer_metrics(ans)["mentioned"]

# ─────────────────────────────────────────────────────────────────────────────
# METRIC REGISTRY  —  the single place to define what shows up everywhere.
#
# Hydromea brand metrics are declared directly here (they are not regex-driven).
# Product mention metrics are AUTO-GENERATED from _PRODUCT_PATTERNS below —
# do NOT add product entries manually here; edit _PRODUCT_PATTERNS instead.
#
# Every entry propagates automatically to:
#   • Sidebar filters (3-state selectbox, grouped)
#   • Explorer per-answer badges
#   • Stats provider cards (metric widget per spec)
#   • Stats comparison charts (one chart per group)
#   • Stats summary table (one column pair per spec)
#   • Stats per-query breakdown table
# ─────────────────────────────────────────────────────────────────────────────
FILTER_SPECS: List[dict] = [
    # ── Hydromea brand (manually declared) ───────────────────────────────────
    {"key": "sourced",   "label": "Sourced",          "icon": "✅", "group": "Hydromea", "color": "#059669",
     "fn": lambda m: m["sourced"]},
    {"key": "mentioned", "label": "Hydromea cited",   "icon": "💬", "group": "Hydromea", "color": "#2563eb",
     "fn": lambda m: m["mentioned"]},
    {"key": "global_mentioned", "label": "Global Mentioned", "icon": "🌐", "group": "Hydromea", "color": "#ea580c",
     "fn": lambda m: m.get("global_mentioned", False)},
]
# ── Auto-append product mention + product sourced specs from _PRODUCT_PATTERNS ──
for _pp in _PRODUCT_PATTERNS:
    # Text mention metric
    _key = f'product_{_pp["key"]}'
    FILTER_SPECS.append({
        "key":   _key,
        "label": _pp["label"],
        "icon":  _pp["icon"],
        "group": "Products — Mentioned",
        "color": _pp["color"],
        "fn":    (lambda k: lambda m: m.get(k, False))(_key),
    })
    # Source URL metric (only when source_url is defined)
    if "source_url" in _pp:
        _skey = f'product_{_pp["key"]}_sourced'
        FILTER_SPECS.append({
            "key":   _skey,
            "label": f'{_pp["label"].split()[0]} page sourced',  # e.g. "DiskDrive page sourced"
            "icon":  "🔗",
            "group": "Products — Page Sourced",
            "color": _pp["color"],
            "fn":    (lambda k: lambda m: m.get(k, False))(_skey),
        })

# Ordered unique groups for UI sectioning
_METRIC_GROUPS: List[str] = list(dict.fromkeys(s["group"] for s in FILTER_SPECS))

# ─────────────────────────────────────────────────────────────────────────────
# Experiments + Data
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS: List[str] = list_experiments()


@st.cache_data(show_spinner="Loading corpus…")
def _load(experiment: str) -> LinkedCorpus:
    return build_corpus(experiment=experiment)


@st.cache_data(show_spinner=False)
def _load_metrics_index(experiment: str) -> Dict[str, dict]:
    """Cached global metrics store for one experiment."""
    return _build_answer_metrics_index(_load(experiment))


def build_stats_df(corpus: LinkedCorpus, metrics_index: Dict[str, dict] | None = None) -> pd.DataFrame:
    """
    Wide dataframe: one row per query, one column per (product × FILTER_SPEC).
    Adding a new metric to FILTER_SPECS automatically adds its column here.
    """
    metrics_index = metrics_index or _build_answer_metrics_index(corpus)
    _products = sorted(p for p in corpus.by_product if not p.startswith("_"))
    rows = []
    for qid, q in corpus.queries.items():
        row: dict = {
            "query_id": qid, "Query": q.text,
            "Theme":    query_theme(q.text),
            "Branded":  bool(q.metadata.get("branded", False)),
            "Date":     (q.created_at or "")[:10],
        }
        for p in _products:
            ans_list = [corpus.answers[a] for a in corpus.by_query.get(qid, [])
                        if corpus.answers[a].product == p]
            if ans_list:
                m = metrics_index[ans_list[0].answer_id]
                row[f"{p}__total"] = 1        # always 1 when an answer exists
                for spec in FILTER_SPECS:
                    row[f"{p}__{spec['key']}"] = spec["fn"](m)
                row[f"{p}__source_position"] = m["source_position"]
                row[f"{p}__citation_count"] = m["citation_count"]
                row[f"{p}__nsrc"]     = m["n_sources"]
            else:
                row[f"{p}__total"] = 0
                for spec in FILTER_SPECS:
                    row[f"{p}__{spec['key']}"] = None
                row[f"{p}__source_position"] = None
                row[f"{p}__citation_count"] = None
                row[f"{p}__nsrc"]     = None
        rows.append(row)
    return pd.DataFrame(rows)


def _on_exp_change() -> None:
    """Called by the selectbox on_change."""
    pass

_PAGE_URL_MAP = {
    "explorer": "📋 Explorer",
    "stats":    "📊 Hydromea Stats",
    "tags":     "🗂️ Aggregated Tags",
    "themes":   "🧩 Theme Generation",
    "dendrogram": "🌳 Dendrogram Analysis",
    "wordcloud": "☁️ Word Cloud",
}
_PAGE_KEY_MAP = {v: k for k, v in _PAGE_URL_MAP.items()}

# ── Bootstrap session state from URL query params (first load only) ───────────
_qp = st.query_params
if "sel_exp" not in st.session_state:
    # Prefer URL param, fall back to first experiment
    _exp_from_url = _qp.get("exp", "")
    st.session_state["sel_exp"] = (
        _exp_from_url if _exp_from_url in EXPERIMENTS
        else (EXPERIMENTS[0] if EXPERIMENTS else "")
    )
if "qid" not in st.session_state and "qid" in _qp:
    st.session_state["qid"] = _qp["qid"]
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = _PAGE_URL_MAP.get(_qp.get("page", ""), "📋 Explorer")

# Resolve current experiment from session state before any rendering
_cur_exp = st.session_state.get("sel_exp", EXPERIMENTS[0] if EXPERIMENTS else "")
corpus   = _load(_cur_exp)
answer_metrics_index = _load_metrics_index(_cur_exp)
products = sorted(p for p in corpus.by_product if not p.startswith("_"))


def _clean_response(text: str) -> str:
    """Strip the repeated 'Query: User query:' header OpenAI sometimes prepends."""
    return re.sub(r"^\*{0,2}Query:\s*User query:\s*\*{0,2}", "", text.strip())


@st.cache_data(show_spinner=False)
def _load_tagged_answers(experiment: str) -> Dict[str, dict]:
    """Load tagged answer payloads keyed by answer_id for one experiment folder."""
    out: Dict[str, dict] = {}
    exp_dir = pathlib.Path(__file__).parent / "data" / experiment
    if not exp_dir.exists() or not exp_dir.is_dir():
        return out

    for file_path in sorted(exp_dir.glob("tagged*.json")):
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        for answer_id, payload in raw.items():
            if isinstance(payload, dict):
                out[str(answer_id)] = payload
    return out


@st.cache_data(show_spinner=False)
def _load_theme_generation_analysis(experiment: str) -> Optional[dict]:
    """Load one experiment's Step 3 theme-generation payload."""
    file_path = pathlib.Path(__file__).parent / "data" / experiment / "theme_generation_analysis.json"
    if not file_path.exists():
        return None
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        return raw[0]
    if isinstance(raw, dict):
        return raw
    return None


def _build_theme_generation_frames(experiment: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (themes_df, codes_df, payload) for one experiment."""
    payload = _load_theme_generation_analysis(experiment) or {}
    themes = payload.get("themes") or []
    source_codes = payload.get("source_codes") or []

    source_index: Dict[str, dict] = {}
    for code in source_codes:
        if not isinstance(code, dict):
            continue
        code_id = str(code.get("code_id", "")).strip()
        if code_id:
            source_index[code_id] = code

    theme_rows: List[dict] = []
    code_rows: List[dict] = []
    for idx, theme in enumerate(themes, start=1):
        if not isinstance(theme, dict):
            continue

        theme_name = str(theme.get("theme_name", "")).strip() or f"Theme {idx}"
        description = str(theme.get("description", "")).strip()
        raw_code_ids = theme.get("code_ids") or []
        code_ids = [str(code_id).strip() for code_id in raw_code_ids if str(code_id).strip()]
        question_ids = {
            str(source_index.get(code_id, {}).get("question_id", "")).strip()
            for code_id in code_ids
            if source_index.get(code_id)
        }
        question_ids.discard("")

        theme_rows.append({
            "theme_rank": idx,
            "theme_name": theme_name,
            "description": description,
            "code_count": len(code_ids),
            "question_count": len(question_ids),
        })

        for order, code_id in enumerate(code_ids, start=1):
            source = source_index.get(code_id, {})
            code_rows.append({
                "theme_rank": idx,
                "theme_name": theme_name,
                "description": description,
                "code_order": order,
                "question_id": str(source.get("question_id", "")).strip(),
                "code_id": code_id,
                "code_name": str(source.get("code_name", "")).strip(),
                "tag": str(source.get("tag", "")).strip(),
                "code_description": str(source.get("description", "")).strip(),
                "representative_excerpt": str(source.get("representative_excerpt", "")).strip(),
            })

    return pd.DataFrame(theme_rows), pd.DataFrame(code_rows), payload


@st.cache_data(show_spinner=False)
def _load_dendrogram_analysis() -> Optional[dict]:
    """Load the cross-version dendrogram analysis bundle exported from the notebook."""
    bundle_path = pathlib.Path(__file__).parent / "data" / "ALL_VERSIONS" / "dendrogram_analysis" / "bundle.json"
    if not bundle_path.exists():
        return None
    try:
        raw = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _build_dendrogram_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (points_df, cluster_summary_df, payload) for the cross-version bundle."""
    payload = _load_dendrogram_analysis() or {}
    points_df = pd.DataFrame(payload.get("points") or [])
    cluster_summary_df = pd.DataFrame(payload.get("cluster_summary") or [])

    if not points_df.empty:
        points_df["cluster_label"] = pd.to_numeric(points_df["cluster_label"], errors="coerce").fillna(-1).astype(int)
        points_df["cluster_probability"] = pd.to_numeric(points_df["cluster_probability"], errors="coerce").fillna(0.0)
        points_df["umap_x"] = pd.to_numeric(points_df["umap_x"], errors="coerce")
        points_df["umap_y"] = pd.to_numeric(points_df["umap_y"], errors="coerce")
        points_df["cluster_name"] = points_df["cluster_label"].apply(
            lambda value: "Noise" if value < 0 else f"Cluster {value}"
        )

    if not cluster_summary_df.empty:
        cluster_summary_df["cluster_label"] = pd.to_numeric(
            cluster_summary_df["cluster_label"], errors="coerce"
        ).fillna(-1).astype(int)
        cluster_summary_df["cluster_name"] = cluster_summary_df["cluster_label"].apply(
            lambda value: "Noise" if value < 0 else f"Cluster {value}"
        )

    return points_df, cluster_summary_df, payload


def _build_cluster_label_map(cluster_summary_df: pd.DataFrame) -> Dict[int, str]:
    """Build short human-readable labels for clusters from their top example codes."""
    label_map: Dict[int, str] = {}
    if cluster_summary_df.empty:
        return label_map

    for row in cluster_summary_df.itertuples(index=False):
        cluster_id = int(getattr(row, "cluster_label", -1))
        top_codes = getattr(row, "top_codes", []) or []
        names: List[str] = []
        for code in top_codes:
            if not isinstance(code, dict):
                continue
            code_name = str(code.get("code_name", "")).strip()
            if code_name and code_name not in names:
                names.append(code_name)
            if len(names) == 2:
                break
        label_map[cluster_id] = " / ".join(names) if names else f"Cluster {cluster_id}"
    return label_map


def _compute_span_counts(lines: List[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for line in lines:
        for span in line.get("spans", []):
            cat = span.get("category")
            if cat:
                counts[cat] = counts.get(cat, 0) + 1
    return counts


def _render_tagged_lines(lines: List[dict]) -> str:
    """Render tagged spans as escaped HTML with category color overlays."""
    rendered_lines: List[str] = []
    for line in lines:
        spans = line.get("spans") or []
        if not spans:
            rendered_lines.append(_html.escape(line.get("raw_text", "")))
            continue

        chunks: List[str] = []
        for span in spans:
            txt = _html.escape(str(span.get("text", "")))
            cat = span.get("category")
            if not cat:
                chunks.append(txt)
                continue

            style = _TAG_STYLE.get(cat, _TAG_STYLE_DEFAULT)
            char_start = span.get("char_start")
            char_end = span.get("char_end")
            if isinstance(char_start, int) and isinstance(char_end, int):
                title = f"{cat} ({char_start}:{char_end})"
            else:
                title = cat
            chunks.append(
                f'<span class="tagged-span {style["span"]}" title="{_html.escape(title)}">{txt}</span>'
            )
        rendered_lines.append("".join(chunks))

    return "<br>".join(rendered_lines)


def _extract_tag_snippets(lines: List[dict]) -> Dict[str, List[str]]:
    """Extract plain-text tagged snippets grouped by category."""
    out: Dict[str, List[str]] = {k: [] for k in TAG_PARTITION_CATEGORIES}
    for line in lines or []:
        for span in line.get("spans", []) or []:
            cat = span.get("category")
            txt = (span.get("text") or "").strip()
            if not cat or not txt or cat not in out:
                continue
            out[cat].append(txt)
    return out


def _normalize_tag_counts(raw_counts: dict) -> Dict[str, int]:
    """Normalize arbitrary category counts into known partition categories."""
    normalized = {k: 0 for k in TAG_PARTITION_CATEGORIES}
    for key, value in (raw_counts or {}).items():
        if not key:
            continue
        if key in normalized:
            normalized[key] += int(value or 0)
    return normalized


def _build_tag_partition_rows(corpus: LinkedCorpus, experiment: str) -> tuple[pd.DataFrame, dict]:
    """Build one row per answer with tag-category counts + slicing metadata."""
    tagged_map = _load_tagged_answers(experiment)
    metrics_index = _load_metrics_index(experiment)
    rows: List[dict] = []
    n_answer_total = 0
    n_missing_tagged = 0
    n_missing_query = 0

    for answer in corpus.answers.values():
        if not answer.product or answer.product.startswith("_"):
            continue
        n_answer_total += 1
        query = corpus.queries.get(answer.query_id or "")
        if query is None:
            n_missing_query += 1
            continue

        tag_payload = tagged_map.get(answer.answer_id)
        if tag_payload:
            raw_counts = _compute_span_counts(tag_payload.get("lines") or [])
            if not raw_counts:
                raw_counts = (tag_payload.get("summary") or {}).get("span_counts") or {}
        else:
            raw_counts = {}
            n_missing_tagged += 1

        tag_counts = _normalize_tag_counts(raw_counts)
        metric = metrics_index[answer.answer_id]
        provider_meta = PRODUCT_META.get(answer.product, _DMETA)

        row = {
            "answer_id": answer.answer_id,
            "query_id": query.query_id,
            "query_text": query.text,
            "query_text_key": (query.text or "").strip().lower(),
            "provider": answer.product,
            "provider_label": provider_meta["label"],
            "theme": query_theme(query.text) or "❓ Unmatched",
            "hydromea_mentioned": bool(metric["mentioned"]),
            "global_mentioned": bool(metric.get("global_mentioned", False)),
            "hydromea_sourced": bool(metric["sourced"]),
            "mention_bucket": "Mentioned" if metric["mentioned"] else "Not Mentioned",
            "global_mention_bucket": "Global Mentioned" if metric.get("global_mentioned", False) else "Global Not Mentioned",
            "source_bucket": "Sourced" if metric["sourced"] else "Unsourced",
        }
        for cat in TAG_PARTITION_CATEGORIES:
            row[cat] = int(tag_counts.get(cat, 0))
        rows.append(row)

    df = pd.DataFrame(rows)
    meta = {
        "n_answer_total": n_answer_total,
        "n_rows": len(df),
        "n_missing_tagged": n_missing_tagged,
        "n_missing_query": n_missing_query,
    }
    return df, meta


def _aggregate_tag_partition(df: pd.DataFrame, key_col: str, categories: List[str]) -> pd.DataFrame:
    """Aggregate tag category counts by a breakdown key and return long format."""
    if df.empty:
        return pd.DataFrame(columns=[key_col, "Category", "Count"])
    # Ensure categories are numeric before aggregation
    for cat in categories:
        if cat in df.columns:
            df[cat] = pd.to_numeric(df[cat], errors='coerce').fillna(0).astype(int)
    grouped = df.groupby(key_col, dropna=False)[categories].sum().reset_index()
    long_df = grouped.melt(id_vars=[key_col], var_name="Category", value_name="Count")
    return long_df


def _compute_tag_delta(current_df: pd.DataFrame, baseline_df: pd.DataFrame, key_col: str, categories: List[str]) -> pd.DataFrame:
    """Compute current-vs-baseline delta by breakdown key and category."""
    cur = _aggregate_tag_partition(current_df, key_col, categories)
    base = _aggregate_tag_partition(baseline_df, key_col, categories)
    if cur.empty and base.empty:
        return pd.DataFrame(columns=[key_col, "Category", "Delta"])
    merged = cur.merge(base, on=[key_col, "Category"], how="outer", suffixes=("_cur", "_base")).fillna(0)
    merged["Delta"] = merged["Count_cur"] - merged["Count_base"]
    return merged[[key_col, "Category", "Delta"]]


tagged_answers = _load_tagged_answers(_cur_exp)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:21px;font-weight:900;color:#0f172a;margin-bottom:2px;">'
        '🔍 AI Visibility Suite</div>'
        '<div style="color:#64748b;font-size:12px;margin-bottom:10px;">'
        'Answer Comparison · Brand Tracking</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Experiment selector ───────────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:#64748b;'
        'margin-bottom:4px;">EXPERIMENT</div>',
        unsafe_allow_html=True,
    )
    st.selectbox(
        "exp",
        EXPERIMENTS,
        key="sel_exp",
        on_change=_on_exp_change,
        label_visibility="collapsed",
    )

    # ── Dynamic corpus summary ────────────────────────────────────────────────
    parts = [f"**{len(corpus.queries)}** queries"]
    for p in products:
        m = PRODUCT_META.get(p, _DMETA)
        parts.append(
            f'<span style="color:{m["color"]};font-weight:700;">'
            f'{m["icon"]} {m["label"]}: {len(corpus.by_product.get(p, []))}</span>'
        )
    st.markdown(" &nbsp;·&nbsp; ".join(parts), unsafe_allow_html=True)
    st.divider()

    _PAGE = st.radio(
        "nav", ["📋 Explorer", "📊 Hydromea Stats", "🗂️ Aggregated Tags", "🧩 Theme Generation", "🌳 Dendrogram Analysis", "☁️ Word Cloud"],
        key="nav_page",
        label_visibility="collapsed",
    )
    st.divider()

    search_q = st.text_input("🔍 Search", placeholder="keyword in query…", label_visibility="collapsed")

    intents = sorted({query_theme(q.text) for q in corpus.queries.values() if query_theme(q.text)})
    intent_sel = st.selectbox("Filter by theme", ["All themes"] + intents, label_visibility="collapsed")
    show_tagging = st.toggle("🏷️ Show tagging highlights", value=True)
    show_tagging_hints = st.toggle("🧭 Show span offset hints", value=True)

    # ── Metric filters (auto-generated from FILTER_SPECS, grouped) ────────────
    _STATE_OPTS = {"—": None, "✅ Yes": True, "✗ No": False}
    filter_states: List[tuple] = []   # (spec, bool|None)
    _last_group = None
    for spec in FILTER_SPECS:
        if spec["group"] != _last_group:
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
                f'color:#64748b;margin:10px 0 4px;">{spec["group"].upper()}</div>',
                unsafe_allow_html=True,
            )
            _last_group = spec["group"]
        sel = st.selectbox(
            f'{spec["icon"]} {spec["label"]}',
            options=list(_STATE_OPTS.keys()),
            index=0,
            key=f'flt_{spec["key"]}',
        )
        filter_states.append((spec, _STATE_OPTS[sel]))

    st.divider()

    all_queries = list(corpus.queries.values())
    all_queries.sort(key=lambda q: q.created_at or "")

    def _query_passes_filters(q) -> bool:
        """AND across active filters, OR across providers per filter."""
        ans_for_q = [
            corpus.answers[a] for a in corpus.by_query.get(q.query_id, [])
            if a in corpus.answers
        ]
        for spec, want in filter_states:
            if want is None:
                continue   # disabled
            result = any(spec["fn"](answer_metrics_index[a.answer_id]) for a in ans_for_q)
            if result != want:
                return False
        return True

    filtered = [
        q for q in all_queries
        if _fuzzy_match(search_q, q.text)
        and (intent_sel == "All themes" or query_theme(q.text) == intent_sel)
        and _query_passes_filters(q)
    ]

    st.caption(f"{len(filtered)} / {len(all_queries)} queries")

    if "qid" not in st.session_state:
        st.session_state.qid = filtered[0].query_id if filtered else None

    for q in filtered:
        is_active = q.query_id == st.session_state.qid
        short = q.text if len(q.text) <= 68 else q.text[:65] + "…"
        prefix = "▶ " if is_active else "   "
        if st.button(prefix + short, key=f"btn_{q.query_id}", use_container_width=True):
            st.session_state.qid = q.query_id
            st.rerun()

# ── Sync current state → URL (single authoritative point) ────────────────────
_url_params: dict = {
    "exp":  _cur_exp,
    "page": _PAGE_KEY_MAP.get(_PAGE, "explorer"),
}
if st.session_state.get("qid"):
    _url_params["qid"] = st.session_state.qid
st.query_params.from_dict(_url_params)

# ─────────────────────────────────────────────────────────────────────────────
# Stats page (shown when _PAGE == "📊 Hydromea Stats")
# ─────────────────────────────────────────────────────────────────────────────
if _PAGE == "🌳 Dendrogram Analysis":
    st.markdown("### Dendrogram Analysis")
    st.caption(
        "Cross-version code clustering exported from the notebook. This page always uses "
        "the shared ALL_VERSIONS artifact bundle."
    )

    _points_df, _cluster_summary_df, _dendro_payload = _build_dendrogram_frames()
    if not _dendro_payload or _points_df.empty:
        st.info(
            "No dendrogram-analysis bundle was found. Run the export step at the end of the notebook "
            "to generate data/ALL_VERSIONS/dendrogram_analysis/."
        )
        st.stop()

    _manifest = _dendro_payload.get("manifest") or {}
    _all_versions = sorted(_points_df["version"].dropna().astype(str).unique().tolist())
    _all_clusters = sorted(label for label in _points_df["cluster_label"].unique().tolist() if int(label) >= 0)
    _cluster_label_map = _build_cluster_label_map(_cluster_summary_df)

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Codes", int(_manifest.get("n_records", len(_points_df))))
    _m2.metric("Clusters", int(_manifest.get("n_clusters_excluding_noise", len(_all_clusters))))
    _m3.metric("Noise", f"{100.0 * float(_manifest.get('noise_ratio', 0.0)):.1f}%")
    _m4.metric("Versions", len(_manifest.get("versions") or _all_versions))

    st.markdown(
        "<div style='font-size:12px;color:#64748b;margin:8px 0 16px;'>"
        f"Model: {_html.escape(str(_manifest.get('model_name', 'unknown')))}"
        f" &nbsp;·&nbsp; UMAP neighbors: {_html.escape(str((_manifest.get('umap') or {}).get('n_neighbors', 'n/a')))}"
        f" &nbsp;·&nbsp; HDBSCAN min cluster size: {_html.escape(str((_manifest.get('hdbscan') or {}).get('min_cluster_size', 'n/a')))}"
        "</div>",
        unsafe_allow_html=True,
    )

    _flt1, _flt2, _flt3, _flt4 = st.columns([1.5, 1.3, 0.8, 1.4], gap="large")
    with _flt1:
        _version_sel = st.multiselect(
            "Versions",
            options=_all_versions,
            default=_all_versions,
            key="dendrogram_versions",
        )
    with _flt2:
        _cluster_sel = st.multiselect(
            "Clusters",
            options=_all_clusters,
            default=_all_clusters,
            format_func=lambda value: f"Cluster {value}",
            key="dendrogram_clusters",
        )
    with _flt3:
        _show_noise = st.toggle("Show noise", value=False, key="dendrogram_show_noise")
    with _flt4:
        _search_term = st.text_input(
            "Search codes",
            placeholder="code name, tag, description...",
            key="dendrogram_search",
        )

    _filtered_points = _points_df.copy()
    if _version_sel:
        _filtered_points = _filtered_points[_filtered_points["version"].isin(_version_sel)].copy()
    if _cluster_sel:
        _allowed_clusters = set(int(value) for value in _cluster_sel)
        if _show_noise:
            _allowed_clusters.add(-1)
        _filtered_points = _filtered_points[_filtered_points["cluster_label"].isin(_allowed_clusters)].copy()
    elif not _show_noise:
        _filtered_points = _filtered_points[_filtered_points["cluster_label"] >= 0].copy()

    _needle = (_search_term or "").strip().lower()
    if _needle:
        _haystack = _filtered_points[["tag", "code_name", "description", "representative_excerpt"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        _filtered_points = _filtered_points[_haystack.str.contains(re.escape(_needle), regex=True)].copy()

    if _filtered_points.empty:
        st.warning("No points match the current dendrogram filters.")
        st.stop()

    _filtered_points["cluster_name"] = _filtered_points["cluster_label"].apply(
        lambda value: "Noise" if int(value) < 0 else f"Cluster {int(value)}"
    )
    _filtered_points["cluster_short_label"] = _filtered_points["cluster_label"].apply(
        lambda value: "Noise" if int(value) < 0 else _cluster_label_map.get(int(value), f"Cluster {int(value)}")
    )
    _filtered_points["cluster_display_name"] = _filtered_points.apply(
        lambda row: row["cluster_name"] if int(row["cluster_label"]) < 0 else f"{row['cluster_name']} · {row['cluster_short_label']}",
        axis=1,
    )
    _color_order = [f"Cluster {value}" for value in _all_clusters]
    if _show_noise or (_filtered_points["cluster_label"] < 0).any():
        _color_order.append("Noise")

    _centroid_labels = (
        _filtered_points[_filtered_points["cluster_label"] >= 0]
        .groupby(["cluster_label", "cluster_name", "cluster_short_label"], as_index=False)[["umap_x", "umap_y"]]
        .mean()
        .sort_values("cluster_label")
    )

    _chart_col, _heatmap_col = st.columns([1.8, 1.0], gap="large")
    with _chart_col:
        st.markdown("**UMAP projection**")
        _scatter = px.scatter(
            _filtered_points,
            x="umap_x",
            y="umap_y",
            color="cluster_name",
            symbol="version",
            category_orders={"cluster_name": _color_order},
            hover_data={
                "version": True,
                "question_id": True,
                "tag": True,
                "code_name": True,
                "description": True,
                "cluster_short_label": True,
                "cluster_probability": ":.3f",
                "umap_x": ":.3f",
                "umap_y": ":.3f",
                "cluster_name": False,
                "cluster_display_name": False,
            },
            opacity=0.82,
            height=640,
            color_discrete_sequence=px.colors.qualitative.Alphabet,
        )
        _scatter.update_traces(marker=dict(size=8, line=dict(width=0.4, color="white")))
        for _row in _centroid_labels.itertuples(index=False):
            _scatter.add_annotation(
                x=float(_row.umap_x),
                y=float(_row.umap_y),
                text=f"C{int(_row.cluster_label)}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="rgba(31,41,55,0.25)",
                borderwidth=1,
                font=dict(size=10, color="#1f2937"),
                xanchor="center",
                yanchor="middle",
            )
        _scatter.update_layout(
            legend_title_text="Cluster",
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(_scatter, use_container_width=True)

    with _heatmap_col:
        st.markdown("**Cluster × version**")
        _heatmap_source = _filtered_points[_filtered_points["cluster_label"] >= 0].copy()
        if _heatmap_source.empty:
            st.info("No clustered points remain under the current filters.")
        else:
            _heatmap_df = (
                _heatmap_source.groupby(["cluster_label", "version"]).size()
                .unstack(fill_value=0)
                .reindex(index=sorted(_heatmap_source["cluster_label"].unique()), fill_value=0)
            )
            _heatmap = px.imshow(
                _heatmap_df,
                labels={"x": "Version", "y": "Cluster", "color": "Codes"},
                aspect="auto",
                color_continuous_scale="Blues",
                height=640,
            )
            _heatmap.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(_heatmap, use_container_width=True)

    with st.expander("Hierarchy overview", expanded=False):
        _dendrogram_payload = _dendro_payload.get("dendrogram") or {}
        _linkage_values = _dendrogram_payload.get("linkage_matrix") or []
        _leaf_labels = _dendrogram_payload.get("leaf_labels") or []
        if not _linkage_values or not _leaf_labels:
            st.info("The exported bundle does not include the notebook dendrogram payload.")
        else:
            _linkage_matrix = np.asarray(_linkage_values, dtype=float)
            _fig, _ax = plt.subplots(figsize=(18, 8))
            dendrogram(
                _linkage_matrix,
                labels=_leaf_labels,
                leaf_rotation=90,
                leaf_font_size=8,
                above_threshold_color="#4C72B0",
                ax=_ax,
            )
            _ax.set_title("Hierarchical Dendrogram of Codebook Embeddings", fontsize=14, fontweight="bold")
            _ax.set_xlabel("Code entry")
            _ax.set_ylabel("Cosine distance")
            st.pyplot(_fig, use_container_width=True)
            plt.close(_fig)
            st.caption("This is the same dendrogram exported from the notebook, using the original embedding-space linkage tree.")

    st.markdown("**Cluster catalog**")
    _catalog_source = _filtered_points[_filtered_points["cluster_label"] >= 0].copy()
    if _catalog_source.empty:
        st.info("No non-noise clusters remain under the current filters.")
    else:
        _cluster_order = (
            _catalog_source.groupby("cluster_label").size().sort_values(ascending=False).index.tolist()
        )
        _left_col, _right_col = st.columns(2, gap="large")
        for _idx, _cluster_id in enumerate(_cluster_order):
            _target_col = _left_col if _idx % 2 == 0 else _right_col
            _subset = _catalog_source[_catalog_source["cluster_label"] == _cluster_id].copy()
            _subset = _subset.sort_values(["cluster_probability", "version", "code_name"], ascending=[False, True, True])
            _version_line = " · ".join(
                f"{version}: {count}" for version, count in _subset["version"].value_counts().sort_index().items()
            )
            _cluster_short_label = _cluster_label_map.get(int(_cluster_id), f"Cluster {_cluster_id}")
            with _target_col:
                with st.expander(f"Cluster {_cluster_id} · {_cluster_short_label} · {len(_subset)} codes", expanded=False):
                    st.caption(_version_line)
                    st.dataframe(
                        _subset[["question_id", "tag", "version", "code_name", "description", "cluster_short_label"]].rename(columns={
                            "question_id": "Query ID",
                            "tag": "Code ID",
                            "version": "Version",
                            "code_name": "Name",
                            "description": "Description",
                            "cluster_short_label": "Cluster Name",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

    st.markdown("**Filtered points**")
    _download_df = _filtered_points[[
        "question_id",
        "tag",
        "version",
        "code_name",
        "description",
        "cluster_short_label",
    ]].copy()
    _download_df = _download_df.rename(columns={
        "question_id": "Query ID",
        "tag": "Code ID",
        "version": "Version",
        "code_name": "Name",
        "description": "Description",
        "cluster_short_label": "Cluster Name",
    })
    st.download_button(
        "Download filtered points as CSV",
        data=_download_df.to_csv(index=False).encode("utf-8"),
        file_name="dendrogram_filtered_points.csv",
        mime="text/csv",
    )
    st.dataframe(_download_df, use_container_width=True, hide_index=True)
    st.stop()

if _PAGE == "🧩 Theme Generation":
    st.markdown("### Theme Generation Analysis")
    st.caption(
        "Inspect Step 3 grouped themes for the selected version, including coverage, "
        "source questions, and the underlying Step 2 codes."
    )

    _themes_df, _codes_df, _theme_payload = _build_theme_generation_frames(_cur_exp)
    if not _theme_payload:
        st.info("No `theme_generation_analysis.json` file was found for this version.")
        st.stop()

    _theme_count = int(len(_themes_df))
    _question_count = int(_theme_payload.get("question_count", 0))
    _code_count = int(_theme_payload.get("code_count", 0))
    _assigned_code_count = int(_codes_df["code_id"].nunique()) if not _codes_df.empty else 0

    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Themes", _theme_count)
    _m2.metric("Questions", _question_count)
    _m3.metric("Step 2 codes", _code_count)
    _m4.metric("Codes assigned", _assigned_code_count)

    if _themes_df.empty:
        st.warning("The theme-generation file exists, but no themes were found in it.")
        st.stop()

    _chart_col, _table_col = st.columns([1.15, 1.85], gap="large")
    with _chart_col:
        st.markdown("**Theme sizes**")
        _chart_df = _themes_df.sort_values(["code_count", "theme_name"], ascending=[False, True]).copy()
        _height = max(320, 70 + 42 * len(_chart_df))
        fig = px.bar(
            _chart_df,
            x="code_count",
            y="theme_name",
            orientation="h",
            text="code_count",
            color="question_count",
            color_continuous_scale="Blues",
            height=_height,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Assigned Step 2 codes",
            yaxis_title="",
            coloraxis_colorbar_title="Questions",
        )
        st.plotly_chart(fig, use_container_width=True)

    with _table_col:
        st.markdown("**Theme summary table**")
        _summary_df = _themes_df.rename(columns={
            "theme_rank": "#",
            "theme_name": "Theme",
            "description": "Description",
            "code_count": "Codes",
            "question_count": "Questions",
        })
        st.dataframe(_summary_df, use_container_width=True, hide_index=True)

    st.divider()

    _search_col, _sort_col = st.columns([1.7, 1.1], gap="large")
    with _search_col:
        _theme_search = st.text_input(
            "Search themes or codes",
            placeholder="theme name, code name, code id, question id...",
            key="theme_generation_search",
        )
    with _sort_col:
        _sort_mode = st.selectbox(
            "Sort themes",
            ["Largest first", "Smallest first", "A-Z"],
            key="theme_generation_sort",
        )

    _theme_view = _themes_df.copy()
    if _sort_mode == "Largest first":
        _theme_view = _theme_view.sort_values(["code_count", "theme_name"], ascending=[False, True])
    elif _sort_mode == "Smallest first":
        _theme_view = _theme_view.sort_values(["code_count", "theme_name"], ascending=[True, True])
    else:
        _theme_view = _theme_view.sort_values(["theme_name"], ascending=[True])

    _needle = (_theme_search or "").strip().lower()
    if _needle:
        _matching_themes = set()
        if not _codes_df.empty:
            _matching_code_rows = _codes_df[[
                "theme_name",
                "code_id",
                "question_id",
                "code_name",
                "code_description",
                "representative_excerpt",
            ]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            _matching_themes.update(_codes_df.loc[_matching_code_rows.str.contains(re.escape(_needle), regex=True), "theme_name"].tolist())

        _matching_theme_rows = _theme_view[["theme_name", "description"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        _matching_themes.update(_theme_view.loc[_matching_theme_rows.str.contains(re.escape(_needle), regex=True), "theme_name"].tolist())
        _theme_view = _theme_view[_theme_view["theme_name"].isin(_matching_themes)].copy()

    st.caption(f"Showing {_theme_view.shape[0]} of {_themes_df.shape[0]} themes for {_cur_exp}.")

    for _theme_row in _theme_view.itertuples(index=False):
        _theme_codes = _codes_df[_codes_df["theme_name"] == _theme_row.theme_name].copy()
        _theme_codes = _theme_codes.sort_values(["question_id", "code_order", "code_id"], ascending=[True, True, True])
        _header = f"{_theme_row.theme_name} ({int(_theme_row.code_count)} codes · {int(_theme_row.question_count)} questions)"
        with st.expander(_header, expanded=False):
            if _theme_row.description:
                st.markdown(_theme_row.description)

            _question_ids = [qid for qid in _theme_codes["question_id"].dropna().unique().tolist() if qid]
            if _question_ids:
                _query_links: List[str] = []
                for _question_id in _question_ids:
                    _query_obj = corpus.queries.get(_question_id)
                    _label = (_query_obj.text if _query_obj else _question_id).strip()
                    if len(_label) > 96:
                        _label = _label[:93] + "..."
                    _qs = urllib.parse.urlencode({
                        "exp": _cur_exp,
                        "qid": _question_id,
                        "page": "explorer",
                    })
                    _query_links.append(f"<a href='?{_qs}'>{_html.escape(_label)}</a>")
                st.markdown(
                    "<div style='font-size:12px;color:#64748b;margin:6px 0 10px;'>"
                    "Source questions: " + " &nbsp;·&nbsp; ".join(_query_links) + "</div>",
                    unsafe_allow_html=True,
                )

            _display_df = _theme_codes.rename(columns={
                "question_id": "Question ID",
                "code_id": "Code ID",
                "code_name": "Code Name",
                "tag": "Tag",
                "code_description": "Code Description",
                "representative_excerpt": "Representative Excerpt",
            })[[
                "Question ID",
                "Code ID",
                "Code Name",
                "Tag",
                "Code Description",
                "Representative Excerpt",
            ]]
            st.dataframe(_display_df, use_container_width=True, hide_index=True)

    st.stop()

if _PAGE == "🗂️ Aggregated Tags":
    st.markdown("### Aggregated Tags")
    st.caption(
        "Stacked extracted tag snippets grouped by Version × ChatBot × Tag. "
        "Open a card to view full content."
    )

    _agg: Dict[tuple, List[dict]] = {}
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _tagged_map = _load_tagged_answers(_exp)
        _exp_metrics = _load_metrics_index(_exp)  # cached — no recalculation

        for _aid, _payload in _tagged_map.items():
            _ans = _exp_corpus.answers.get(_aid)
            if _ans is None:
                continue
            if not _ans.product or _ans.product.startswith("_"):
                continue

            _provider_label = PRODUCT_META.get(_ans.product, _DMETA)["label"]
            # Prefer query_id from the tagged payload (always populated); fall back to corpus
            _qid = _payload.get("query_id") or (_ans.query_id or "")
            _query_obj = _exp_corpus.queries.get(_qid)
            _query_text = (_query_obj.text if _query_obj else "") or _qid
            _lines = _payload.get("lines") or []
            _snips_by_cat = _extract_tag_snippets(_lines)
            for _cat in TAG_PARTITION_CATEGORIES:
                _snips = _snips_by_cat.get(_cat, [])
                if not _snips:
                    continue
                _k = (_exp, _provider_label, _cat)
                _agg.setdefault(_k, []).extend(
                    {
                        "text": _s,
                        "answer_id": _aid,
                        "query_id": _qid,
                        "query_text": _query_text,
                        "mentioned": bool(_exp_metrics.get(_aid, {}).get("mentioned", False)),
                    }
                    for _s in _snips
                )

    if not _agg:
        st.info("No aggregated tags available.")
    else:
        _mention_filter = st.radio(
            "Filter by Hydromea mention",
            ["All", "Mentioned only", "Not mentioned only"],
            horizontal=True,
            key="agg_mention_filter",
            label_visibility="collapsed",
        )
        st.caption(
            "**Mentioned only** — snippets from answers where Hydromea is cited &nbsp;·&nbsp; "
            "**Not mentioned only** — snippets from answers where Hydromea is not cited"
        )
        for _exp in EXPERIMENTS:
            st.markdown(f"**{_exp}**")
            _exp_keys = [k for k in _agg.keys() if k[0] == _exp]
            if not _exp_keys:
                st.caption("No tagged snippets for this version.")
                continue

            def _apply_mention_filter(items):
                if _mention_filter == "Mentioned only":
                    return [x for x in items if x.get("mentioned")]
                if _mention_filter == "Not mentioned only":
                    return [x for x in items if not x.get("mentioned")]
                return items

            def _render_snippet_expander(title, items, exp):
                if not items:
                    return
                _section_text = "\n".join(x["text"] for x in items)
                with st.expander(title, expanded=False):
                    st.code(_section_text, language="text")
                    _snip_parts: List[str] = []
                    for _it in items:
                        _qs = urllib.parse.urlencode({
                            "exp": exp,
                            "qid": _it["query_id"],
                            "page": "explorer",
                        })
                        _href = f"?{_qs}"
                        _tip  = _it["query_text"].replace("'", "&#39;").replace('"', "&quot;")
                        _txt  = _html.escape(_it["text"])
                        _snip_parts.append(
                            f"<div style='margin:3px 0;line-height:1.6;'>"
                            f"<a href='{_href}' data-tip='{_tip}' "
                            f"style='color:#111827;text-decoration:none;"
                            f"border-bottom:1px dotted #94a3b8;cursor:pointer;'>"
                            f"{_txt}</a>"
                            f"</div>"
                        )
                    st.markdown(
                        """<style>
a[data-tip]{position:relative;}
a[data-tip]::after{
  content:attr(data-tip);
  display:none;
  position:absolute;
  bottom:calc(100% + 4px);
  left:0;
  background:#1e293b;
  color:#f1f5f9;
  padding:5px 10px;
  border-radius:6px;
  font-size:11px;
  white-space:pre-wrap;
  max-width:420px;
  z-index:9999;
  box-shadow:0 2px 8px rgba(0,0,0,.25);
  pointer-events:none;
}
a[data-tip]:hover::after{display:block;}
a[data-tip]:hover{color:#3b82f6;}
</style>
<div style='font-size:11px;color:#94a3b8;margin:4px 0;'>"""
                        "Hover for source query &nbsp;·&nbsp; Click to open in Explorer"
                        "</div>"
                        + "".join(_snip_parts),
                        unsafe_allow_html=True,
                    )

            # ── All Chatbots combined ─────────────────────────────────────────
            st.markdown(
                "<div style='font-size:12px;font-weight:700;color:#64748b;"
                "letter-spacing:.5px;margin:6px 0 2px;'>🌐 All Chatbots</div>",
                unsafe_allow_html=True,
            )
            for _cat in TAG_PARTITION_CATEGORIES:
                _combined = []
                for _k in _exp_keys:
                    if _k[2] == _cat:
                        _combined.extend(_agg[_k])
                _combined = _apply_mention_filter(_combined)
                _render_snippet_expander(
                    f"All · {_cat} ({len(_combined)} tags)", _combined, _exp
                )

            # ── Per-chatbot breakdown ─────────────────────────────────────────
            st.markdown(
                "<div style='font-size:12px;font-weight:700;color:#64748b;"
                "letter-spacing:.5px;margin:10px 0 2px;'>🤖 By Chatbot</div>",
                unsafe_allow_html=True,
            )
            _providers = sorted({k[1] for k in _exp_keys})
            for _prov in _providers:
                for _cat in TAG_PARTITION_CATEGORIES:
                    _k = (_exp, _prov, _cat)
                    if _k not in _agg:
                        continue
                    _items = _apply_mention_filter([x for x in _agg[_k] if x.get("text")])
                    _render_snippet_expander(
                        f"{_prov} · {_cat} ({len(_items)} tags)", _items, _exp
                    )

            st.divider()

    st.stop()

if _PAGE == "📊 Hydromea Stats":
    df = build_stats_df(corpus, answer_metrics_index)
    products_all = sorted(p for p in corpus.by_product if not p.startswith("_"))

    # ── Baseline V0 reference ─────────────────────────────────────────────────
    _BL_EXP = "Baseline V0"
    _is_baseline = (_cur_exp == _BL_EXP)
    _baseline_counts: dict = {}   # provider → {spec_key → count, citation_count_sum → int}
    if not _is_baseline and _BL_EXP in EXPERIMENTS:
        _bl_corpus = _load(_BL_EXP)
        _bl_df = build_stats_df(_bl_corpus, _load_metrics_index(_BL_EXP))
        for _bp in products_all:
            _bpc: dict = {}
            for _bspec in FILTER_SPECS:
                _bcol = f"{_bp}__{_bspec['key']}"
                _bpc[_bspec["key"]] = int(_bl_df[_bcol].sum()) if _bcol in _bl_df.columns else 0
            _bcc_col = f"{_bp}__citation_count"
            _bpc["citation_count_sum"] = int(_bl_df[_bcc_col].sum()) if _bcc_col in _bl_df.columns else 0
            _baseline_counts[_bp] = _bpc
    _show_delta = not _is_baseline and bool(_baseline_counts)

    st.markdown(
        f'<h1 style="margin-bottom:4px;">📊 Hydromea Visibility</h1>'
        f'<p style="color:#64748b;font-size:15px;margin-top:0;">'
        f'Experiment: <strong>{_cur_exp}</strong> &nbsp;·&nbsp; '
        f'Tracks how often <strong>hydromea.com / hydromea.ch</strong> '
        'appears as a <em>source</em> and is explicitly <em>cited</em> in AI answers.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ How metrics are computed", expanded=False):
        _info_rows = [
            "| Metric | Group | Definition |",
            "|:---|:---|:---|",
            "| ✅ **Sourced** | Hydromea | `hydromea.com` / `hydromea.ch` appears in the `sources` list (hostname or URL field) |",
            "| 💬 **Hydromea cited** | Hydromea | The word `hydromea` appears anywhere in the response text (case-insensitive) |",
            "| 🔢 **Citation Count** | Hydromea | Number of distinct sources per answer where a Hydromea domain appears |",
        ]
        for _s in FILTER_SPECS:
            if _s["group"] == "Products — Mentioned":
                _info_rows.append(f"| {_s['icon']} **{_s['label']}** | {_s['group']} | Regex match in the response text |") 
            elif _s["group"] == "Products — Page Sourced":
                _info_rows.append(f"| {_s['icon']} **{_s['label']}** | {_s['group']} | Product-specific page URL found in the answer's source list |")
        _info_rows += [
            "",
            "**Baseline comparison**: when an experiment other than *Baseline V0* is selected, "
            "each metric card shows `+N / −N vs Baseline` and charts gain a second panel with the absolute change.",
        ]
        st.markdown("\n".join(_info_rows))

    # ── Answer Length Statistics ──────────────────────────────────────────────
    st.markdown("### Answer Length Statistics")
    st.caption("Paragraph = non-empty block separated by a blank line. Word = whitespace-separated token.")

    def _count_paragraphs(text: str) -> int:
        return sum(1 for p in text.split("\n\n") if p.strip())

    def _count_words(text: str) -> int:
        return len(text.split())

    _all_answer_texts: list[str] = []
    for _stat_exp in EXPERIMENTS:
        _stat_corpus = _load(_stat_exp)
        for _stat_ans in _stat_corpus.answers.values():
            _stat_resp = (_stat_ans.response or "").strip()
            if _stat_resp:
                _all_answer_texts.append(_stat_resp)

    if _all_answer_texts:
        _stat_concat = "\n\n".join(_all_answer_texts)
        _stat_paras  = [_count_paragraphs(t) for t in _all_answer_texts]
        _stat_words  = [_count_words(t) for t in _all_answer_texts]

        _len_stats_df = pd.DataFrame([
            {
                "Scope": f"All {len(_all_answer_texts)} answers (concatenated)",
                "Total Number of Paragraphs": _count_paragraphs(_stat_concat),
                "Total Number of Words":      _count_words(_stat_concat),
            },
            {
                "Scope": "Min (across individual answers)",
                "Total Number of Paragraphs": min(_stat_paras),
                "Total Number of Words":      min(_stat_words),
            },
            {
                "Scope": "Max (across individual answers)",
                "Total Number of Paragraphs": max(_stat_paras),
                "Total Number of Words":      max(_stat_words),
            },
        ])

        st.dataframe(
            _len_stats_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Scope":                        st.column_config.TextColumn("Scope",                        width="large"),
                "Total Number of Paragraphs":   st.column_config.NumberColumn("Total Number of Paragraphs", width="medium"),
                "Total Number of Words":        st.column_config.NumberColumn("Total Number of Words",      width="medium"),
            },
        )
    else:
        st.info("No answers loaded.")

    st.markdown("---")

    # ── Build per-provider summary from FILTER_SPECS ──────────────────────────
    st.markdown("### Provider snapshot")
    summary = []
    for p in products_all:
        pm    = PRODUCT_META.get(p, _DMETA)
        total = int(df[f"{p}__total"].sum())
        counts = {}
        for spec in FILTER_SPECS:
            col = f"{p}__{spec['key']}"
            counts[spec["key"]] = int(df[col].sum()) if col in df.columns else 0
        summary.append({
            "provider": p, "label": pm["label"], "color": pm["color"],
            "bg": pm["bg"], "icon": pm["icon"],
            "total": total, "counts": counts,
            "citation_count_sum": int(df[f"{p}__citation_count"].sum()) if f"{p}__citation_count" in df.columns else 0,
            "citation_count_avg": round(float(df[f"{p}__citation_count"].mean()), 2) if f"{p}__citation_count" in df.columns else 0.0,
        })

    snap_cols = st.columns(len(summary), gap="large")
    for col, r in zip(snap_cols, summary):
        with col:
            st.markdown(
                f'<div style="border:2px solid {r["color"]}44;border-radius:12px;'
                f'padding:14px 16px;background:{r["bg"]};margin-bottom:8px;">'
                f'<div style="font-size:15px;font-weight:800;color:{r["color"]};'
                f'margin-bottom:6px;">{r["icon"]} {r["label"]}</div>'
                f'<div style="font-size:12px;color:#64748b;">{r["total"]} answers</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            # One metric() per spec, grouped
            _last_grp = None
            for spec in FILTER_SPECS:
                if spec["group"] != _last_grp:
                    st.markdown(
                        f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;'
                        f'color:#94a3b8;margin:8px 0 2px;">{spec["group"].upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    _last_grp = spec["group"]
                cnt = r["counts"].get(spec["key"], 0)
                if _show_delta and r["provider"] in _baseline_counts:
                    _d = cnt - _baseline_counts[r["provider"]].get(spec["key"], 0)
                    _metric_delta = f"{"+" if _d >= 0 else ""}{_d} vs Baseline"
                else:
                    _metric_delta = None
                st.metric(f'{spec["icon"]} {spec["label"]}', cnt, _metric_delta)
            st.markdown(
                '<div style="font-size:10px;font-weight:700;letter-spacing:1px;'
                'color:#94a3b8;margin:8px 0 2px;">NUMERIC</div>',
                unsafe_allow_html=True,
            )
            if _show_delta and r["provider"] in _baseline_counts:
                _cc_d = r["citation_count_sum"] - _baseline_counts[r["provider"]].get("citation_count_sum", 0)
                _cc_delta = f"{"+" if _cc_d >= 0 else ""}{_cc_d} vs Baseline"
            else:
                _cc_delta = None
            st.metric("🔢 Citation Count", r["citation_count_sum"], _cc_delta)

    st.markdown("---")

    # ── Tag Partition Analysis ────────────────────────────────────────────────
    st.markdown("### Tag Partition Analysis")
    st.caption(
        "Distribution of tagging categories across providers, themes, and mention cohorts. "
        "Includes Baseline V0 comparison when a non-baseline experiment is selected."
    )

    tag_current_df, tag_current_meta = _build_tag_partition_rows(corpus, _cur_exp)
    if _BL_EXP in EXPERIMENTS:
        if _is_baseline:
            tag_baseline_df = tag_current_df.copy()
            tag_baseline_meta = tag_current_meta
        else:
            _tag_bl_corpus = _load(_BL_EXP)
            tag_baseline_df, tag_baseline_meta = _build_tag_partition_rows(_tag_bl_corpus, _BL_EXP)
    else:
        tag_baseline_df = pd.DataFrame()
        tag_baseline_meta = {"n_rows": 0, "n_missing_tagged": 0, "n_missing_query": 0}

    _cohort_col, _cat_col, _prov_col = st.columns([1.2, 2.2, 1.8], gap="large")
    with _cohort_col:
        cohort_scope = st.radio(
            "Cohort scope",
            ["All queries", "Hydromea Mentioned", "Global Mentioned", "Direct Relevance to page changes"],
            index=0,
            horizontal=False,
        )
    with _cat_col:
        selected_tag_categories = st.multiselect(
            "Tag categories",
            TAG_PARTITION_CATEGORIES,
            default=TAG_PARTITION_CATEGORIES,
        )
    with _prov_col:
        provider_options = sorted(tag_current_df["provider_label"].unique().tolist()) if not tag_current_df.empty else []
        selected_providers = st.multiselect(
            "Providers",
            provider_options,
            default=provider_options,
        )

    _page_change_sourced_keys = [
        "product_diskdrive_sourced",
        "product_luma_sourced",
        "product_exray_sourced",
    ]

    if not selected_tag_categories:
        st.info("Select at least one tag category to render partition charts.")
    else:
        if cohort_scope == "Hydromea Mentioned" and not tag_current_df.empty:
            # Mentioned-only comparison in each version independently.
            tag_current_view = tag_current_df[tag_current_df["hydromea_mentioned"]].copy()
            tag_baseline_view = tag_baseline_df[tag_baseline_df["hydromea_mentioned"]].copy()
        elif cohort_scope == "Global Mentioned" and not tag_current_df.empty:
            tag_current_view = tag_current_df[tag_current_df["global_mentioned"]].copy()
            tag_baseline_view = tag_baseline_df[tag_baseline_df["global_mentioned"]].copy()
        elif cohort_scope == "Direct Relevance to page changes":
            _cur_metrics = _load_metrics_index(_cur_exp)
            _base_metrics = _load_metrics_index(_BL_EXP) if _BL_EXP in EXPERIMENTS else {}

            def _is_page_change_relevant(_m: dict) -> bool:
                return any(bool(_m.get(_k, False)) for _k in _page_change_sourced_keys)

            _cur_ids = [
                _aid for _aid in tag_current_df["answer_id"].tolist()
                if _is_page_change_relevant(_cur_metrics.get(_aid, {}))
            ]
            _base_ids = [
                _aid for _aid in tag_baseline_df["answer_id"].tolist()
                if _is_page_change_relevant(_base_metrics.get(_aid, {}))
            ]
            tag_current_view = tag_current_df[tag_current_df["answer_id"].isin(_cur_ids)].copy()
            tag_baseline_view = tag_baseline_df[tag_baseline_df["answer_id"].isin(_base_ids)].copy()
        else:
            tag_current_view = tag_current_df.copy()
            tag_baseline_view = tag_baseline_df.copy()

        if selected_providers:
            tag_current_view = tag_current_view[tag_current_view["provider_label"].isin(selected_providers)]
            tag_baseline_view = tag_baseline_view[tag_baseline_view["provider_label"].isin(selected_providers)]

        if cohort_scope == "Hydromea Mentioned":
            st.info(
                "Hydromea Mentioned compares current answers where hydromea is mentioned "
                "against baseline answers where hydromea is mentioned."
            )
        elif cohort_scope == "Global Mentioned":
            st.info(
                "Global Mentioned compares current answers whose global score is greater than 0 "
                "against baseline answers whose global score is greater than 0."
            )
        elif cohort_scope == "Direct Relevance to page changes":
            st.info(
                "Direct Relevance to page changes keeps only answers whose sources include at least one "
                "of these pages: DiskDrive, Luma, or Exray."
            )

        # Ensure tag categories are numeric
        for cat in TAG_PARTITION_CATEGORIES:
            if cat in tag_current_view.columns:
                tag_current_view[cat] = pd.to_numeric(tag_current_view[cat], errors='coerce').fillna(0).astype(int)
            if cat in tag_baseline_view.columns:
                tag_baseline_view[cat] = pd.to_numeric(tag_baseline_view[cat], errors='coerce').fillna(0).astype(int)

        # Analysis bucket semantics depend on selected cohort scope.
        if cohort_scope == "Hydromea Mentioned":
            # Already filtered to answers where hydromea was mentioned, so all are "Mentioned"
            tag_current_view["analysis_mention_bucket"] = "Mentioned"
            tag_baseline_view["analysis_mention_bucket"] = "Mentioned"
        elif cohort_scope == "Global Mentioned":
            tag_current_view["analysis_mention_bucket"] = "Global Mentioned"
            tag_baseline_view["analysis_mention_bucket"] = "Global Mentioned"
        else:
            tag_current_view["analysis_mention_bucket"] = tag_current_view["global_mention_bucket"]
            tag_baseline_view["analysis_mention_bucket"] = tag_baseline_view["global_mention_bucket"]

        _k1, _k2, _k3 = st.columns(3)
        _k1.metric("Current answers included", int(len(tag_current_view)))
        _k2.metric("Baseline answers included", int(len(tag_baseline_view)))
        _k3.metric("Current missing tagged payload", int(tag_current_meta.get("n_missing_tagged", 0)))

        st.divider()

        # ── Overall distribution (not broken down by any dimension) ───────
        st.markdown("**Overall Distribution**")
        
        # Compute overall sums across all answers
        overall_counts = {}
        for cat in selected_tag_categories:
            overall_counts[cat] = int(tag_current_view[cat].sum())
        
        overall_current = pd.DataFrame([
            {"Category": cat, "Count": count} for cat, count in overall_counts.items()
        ])
        
        overall_baseline = pd.DataFrame()
        if _show_delta and not _is_baseline and not tag_baseline_view.empty:
            baseline_counts = {}
            for cat in selected_tag_categories:
                baseline_counts[cat] = int(tag_baseline_view[cat].sum())
            overall_baseline = pd.DataFrame([
                {"Category": cat, "Count": count} for cat, count in baseline_counts.items()
            ])
        
        if _show_delta and not _is_baseline and not overall_baseline.empty:
            _ocol, _ocol_delta = st.columns(2, gap="large")
        else:
            _ocol, _ocol_delta = st.container(), None
        
        with _ocol:
            st.markdown("Absolute counts")
            fig_overall = px.bar(
                overall_current,
                x="Category",
                y="Count",
                color="Category",
                color_discrete_map=TAG_PARTITION_COLORS,
                height=280,
                text="Count",
            )
            fig_overall.update_traces(textposition="outside")
            fig_overall.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Tag span count",
            )
            fig_overall.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_overall, use_container_width=True)
            
            st.markdown("Proportion (%)")
            overall_current["Count"] = pd.to_numeric(overall_current["Count"], errors='coerce').fillna(0).astype(int)
            total_count = overall_current["Count"].sum()
            if total_count > 0:
                overall_current["Proportion"] = pd.to_numeric(
                    (overall_current["Count"] / total_count * 100),
                    errors='coerce'
                ).fillna(0).astype(float).round(1)
            else:
                overall_current["Proportion"] = 0.0
            
            fig_overall_prop = px.bar(
                overall_current,
                x="Category",
                y="Proportion",
                color="Category",
                color_discrete_map=TAG_PARTITION_COLORS,
                height=280,
                text="Proportion",
            )
            fig_overall_prop.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_overall_prop.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="",
                yaxis_title="Share of total (%)",
            )
            fig_overall_prop.update_xaxes(tickangle=-25)
            st.plotly_chart(fig_overall_prop, use_container_width=True)
        
        if _show_delta and not _is_baseline and _ocol_delta is not None and not overall_baseline.empty:
            with _ocol_delta:
                st.markdown("Change vs Baseline V0")
                overall_delta = overall_current[["Category", "Count"]].merge(
                    overall_baseline[["Category", "Count"]],
                    on="Category",
                    how="outer",
                    suffixes=("_cur", "_base")
                ).fillna(0)
                overall_delta["Delta"] = overall_delta["Count_cur"] - overall_delta["Count_base"]
                
                fig_overall_delta = px.bar(
                    overall_delta,
                    x="Category",
                    y="Delta",
                    color="Category",
                    color_discrete_map=TAG_PARTITION_COLORS,
                    height=280,
                    text="Delta",
                )
                fig_overall_delta.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
                fig_overall_delta.update_traces(textposition="outside")
                fig_overall_delta.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="",
                    yaxis_title="Delta tag span count",
                )
                fig_overall_delta.update_xaxes(tickangle=-25)
                st.plotly_chart(fig_overall_delta, use_container_width=True)
                
                st.markdown("Change in Proportion vs Baseline V0 (%)")
                if not overall_baseline.empty:
                    overall_delta_prop = overall_current[["Category", "Count"]].merge(
                        overall_baseline[["Category", "Count"]],
                        on="Category",
                        how="outer",
                        suffixes=("_cur", "_base")
                    ).fillna(0)
                    # Ensure Count columns are numeric
                    overall_delta_prop["Count_cur"] = pd.to_numeric(overall_delta_prop["Count_cur"], errors='coerce').fillna(0).astype(int)
                    overall_delta_prop["Count_base"] = pd.to_numeric(overall_delta_prop["Count_base"], errors='coerce').fillna(0).astype(int)
                    # Relative percentage change on absolute counts
                    overall_delta_prop["DeltaProportion"] = pd.to_numeric(
                        (overall_delta_prop["Count_cur"] - overall_delta_prop["Count_base"]) / 
                        overall_delta_prop["Count_base"].replace(0, pd.NA) * 100,
                        errors='coerce'
                    ).fillna(0).astype(float).round(1)
                    
                    fig_overall_delta_prop = px.bar(
                        overall_delta_prop,
                        x="Category",
                        y="DeltaProportion",
                        color="Category",
                        color_discrete_map=TAG_PARTITION_COLORS,
                        height=280,
                        text="DeltaProportion",
                    )
                    fig_overall_delta_prop.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
                    fig_overall_delta_prop.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_overall_delta_prop.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="",
                        yaxis_title="Relative % change in count",
                    )
                    fig_overall_delta_prop.update_xaxes(tickangle=-25)
                    st.plotly_chart(fig_overall_delta_prop, use_container_width=True)
        
        st.divider()

        breakdown_specs = [
            ("By Chatbot", "provider_label", "Chatbot"),
            ("By Question Theme", "theme", "Theme"),
            ("By Hydromea Mention Cohort", "analysis_mention_bucket", "Mention Cohort"),
        ]

        summary_rows: List[dict] = []
        for title, key_col, key_label in breakdown_specs:
            st.markdown(f"**{title}**")

            cur_breakdown = tag_current_view
            base_breakdown = tag_baseline_view

            cur_long = _aggregate_tag_partition(cur_breakdown, key_col, selected_tag_categories)
            base_long = _aggregate_tag_partition(base_breakdown, key_col, selected_tag_categories)

            abs_rows = cur_long.copy()

            delta_long = _compute_tag_delta(cur_breakdown, base_breakdown, key_col, selected_tag_categories)

            _cur_totals = cur_long.groupby(key_col, dropna=False)["Count"].sum().reset_index().rename(
                columns={"Count": "CurrentBucketTotal"}
            )
            _base_totals = base_long.groupby(key_col, dropna=False)["Count"].sum().reset_index().rename(
                columns={"Count": "BaselineBucketTotal"}
            )

            cur_prop = cur_long.merge(_cur_totals, on=[key_col], how="left")
            cur_prop["Count"] = pd.to_numeric(cur_prop["Count"], errors='coerce').fillna(0).astype(int)
            cur_prop["Proportion"] = pd.to_numeric(
                (cur_prop["Count"] / cur_prop["CurrentBucketTotal"].replace(0, pd.NA)) * 100,
                errors='coerce'
            ).fillna(0).astype(float).round(1)

            base_prop = base_long.merge(_base_totals, on=[key_col], how="left")
            base_prop["Count"] = pd.to_numeric(base_prop["Count"], errors='coerce').fillna(0).astype(int)
            base_prop["BaselineProportion"] = pd.to_numeric(
                (base_prop["Count"] / base_prop["BaselineBucketTotal"].replace(0, pd.NA)) * 100,
                errors='coerce'
            ).fillna(0).astype(float).round(1)

            delta_prop = cur_long[[key_col, "Category", "Count"]].merge(
                base_long[[key_col, "Category", "Count"]],
                on=[key_col, "Category"],
                how="outer",
                suffixes=("_cur", "_base")
            ).fillna(0)
            # Ensure Count columns are numeric before calculation
            delta_prop["Count_cur"] = pd.to_numeric(delta_prop["Count_cur"], errors='coerce').fillna(0).astype(int)
            delta_prop["Count_base"] = pd.to_numeric(delta_prop["Count_base"], errors='coerce').fillna(0).astype(int)
            # Relative percentage change on absolute counts
            delta_prop["DeltaProportion"] = pd.to_numeric(
                (delta_prop["Count_cur"] - delta_prop["Count_base"]) / 
                delta_prop["Count_base"].replace(0, pd.NA) * 100,
                errors='coerce'
            ).fillna(0).astype(float).round(1)

            if _show_delta and not _is_baseline:
                _lcol, _rcol = st.columns(2, gap="large")
            else:
                _lcol, _rcol = st.container(), None

            with _lcol:
                st.markdown("Absolute counts")
                fig_abs = px.bar(
                    abs_rows,
                    x=key_col,
                    y="Count",
                    color="Category",
                    barmode="stack",
                    color_discrete_map=TAG_PARTITION_COLORS,
                    height=320,
                )
                fig_abs.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="",
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title=key_label,
                    yaxis_title="Tag span count",
                )
                fig_abs.update_xaxes(tickangle=-25)
                st.plotly_chart(fig_abs, use_container_width=True)

                st.markdown("Proportion (%)")
                fig_prop = px.bar(
                    cur_prop,
                    x=key_col,
                    y="Proportion",
                    color="Category",
                    barmode="stack",
                    color_discrete_map=TAG_PARTITION_COLORS,
                    height=320,
                    text="Proportion",
                )
                fig_prop.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                fig_prop.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="",
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title=key_label,
                    yaxis_title="Share within bucket (%)",
                )
                fig_prop.update_xaxes(tickangle=-25)
                st.plotly_chart(fig_prop, use_container_width=True)

            if _show_delta and not _is_baseline and _rcol is not None:
                with _rcol:
                    st.markdown("Change vs Baseline V0")
                    fig_delta = px.bar(
                        delta_long,
                        x=key_col,
                        y="Delta",
                        color="Category",
                        barmode="group",
                        color_discrete_map=TAG_PARTITION_COLORS,
                        height=320,
                        text="Delta",
                    )
                    fig_delta.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
                    fig_delta.update_traces(textposition="outside")
                    fig_delta.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend_title_text="",
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title=key_label,
                        yaxis_title="Delta tag span count",
                    )
                    fig_delta.update_xaxes(tickangle=-25)
                    st.plotly_chart(fig_delta, use_container_width=True)

                    st.markdown("Change in Proportion vs Baseline V0 (%)")
                    fig_delta_prop = px.bar(
                        delta_prop,
                        x=key_col,
                        y="DeltaProportion",
                        color="Category",
                        barmode="group",
                        color_discrete_map=TAG_PARTITION_COLORS,
                        height=320,
                        text="DeltaProportion",
                    )
                    fig_delta_prop.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
                    fig_delta_prop.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_delta_prop.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend_title_text="",
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title=key_label,
                        yaxis_title="Relative % change in count",
                    )
                    fig_delta_prop.update_xaxes(tickangle=-25)
                    st.plotly_chart(fig_delta_prop, use_container_width=True)

            _summary_frame = cur_long.merge(
                base_long.rename(columns={"Count": "BaselineCount"}),
                on=[key_col, "Category"],
                how="left",
            ).merge(
                delta_long,
                on=[key_col, "Category"],
                how="left",
            )

            _summary_frame = _summary_frame.merge(_cur_totals, on=[key_col], how="left")
            _summary_frame = _summary_frame.merge(_base_totals, on=[key_col], how="left")

            _summary_frame["Count"] = pd.to_numeric(_summary_frame["Count"], errors="coerce").fillna(0).astype(int)
            _summary_frame["BaselineCount"] = pd.to_numeric(_summary_frame["BaselineCount"], errors="coerce").fillna(0).astype(int)
            _summary_frame["Delta"] = pd.to_numeric(_summary_frame["Delta"], errors="coerce").fillna(0).astype(int)
            _summary_frame["CurrentBucketTotal"] = pd.to_numeric(_summary_frame["CurrentBucketTotal"], errors="coerce").fillna(0)
            _summary_frame["BaselineBucketTotal"] = pd.to_numeric(_summary_frame["BaselineBucketTotal"], errors="coerce").fillna(0)

            _summary_frame[f"Proportion ({_cur_exp}) %"] = (
                (_summary_frame["Count"] / _summary_frame["CurrentBucketTotal"].replace(0, pd.NA)) * 100
            ).fillna(0).round(1)
            _summary_frame["Proportion (Baseline V0) %"] = (
                (_summary_frame["BaselineCount"] / _summary_frame["BaselineBucketTotal"].replace(0, pd.NA)) * 100
            ).fillna(0).round(1)
            _summary_frame["Delta Proportion (pp)"] = (
                _summary_frame[f"Proportion ({_cur_exp}) %"] - _summary_frame["Proportion (Baseline V0) %"]
            ).round(1)

            for _, _row in _summary_frame.iterrows():
                summary_rows.append(
                    {
                        "Breakdown": title,
                        "Bucket": _row[key_col],
                        "Category": _row["Category"],
                        f"Count ({_cur_exp})": _row["Count"],
                        f"Proportion ({_cur_exp}) %": _row[f"Proportion ({_cur_exp}) %"],
                        "Count (Baseline V0)": _row["BaselineCount"],
                        "Proportion (Baseline V0) %": _row["Proportion (Baseline V0) %"],
                        "Delta vs Baseline": _row["Delta"],
                        "Delta Proportion (pp)": _row["Delta Proportion (pp)"],
                    }
                )

        st.markdown("**Tag Partition Summary Table**")
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.info("No rows available for the current tag-partition filters.")

    st.markdown("---")

    # ── Comparison charts (one chart per group) ───────────────────────────────
    st.markdown("### Provider comparison")
    for grp in _METRIC_GROUPS:
        grp_specs = [s for s in FILTER_SPECS if s["group"] == grp]
        st.markdown(f"**{grp}**")

        abs_rows, delta_rows, cmap = [], [], {}
        for spec in grp_specs:
            cmap[spec["label"]] = spec["color"]
            for r in summary:
                cnt = r["counts"].get(spec["key"], 0)
                abs_rows.append({"Provider": r["label"], "Metric": spec["label"], "Value": cnt})
                if _show_delta:
                    bl_cnt = _baseline_counts.get(r["provider"], {}).get(spec["key"], 0)
                    delta_rows.append({"Provider": r["label"], "Metric": spec["label"], "Δ vs Baseline": cnt - bl_cnt})

        if _show_delta:
            cl, cr = st.columns(2, gap="large")
        else:
            cl, cr = st.container(), None

        with cl:
            st.markdown("Absolute counts")
            fig = px.bar(pd.DataFrame(abs_rows), x="Provider", y="Value",
                         color="Metric", barmode="group", color_discrete_map=cmap,
                         height=300, text="Value")
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              legend_title_text="", margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        if _show_delta and cr is not None:
            with cr:
                st.markdown("Change vs Baseline V0")
                fig2 = px.bar(pd.DataFrame(delta_rows), x="Provider", y="Δ vs Baseline",
                              color="Metric", barmode="group", color_discrete_map=cmap,
                              height=300, text="Δ vs Baseline")
                fig2.update_traces(textposition="outside")
                fig2.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
                fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   legend_title_text="", margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Δ Count")
                st.plotly_chart(fig2, use_container_width=True)

    # ── Numeric metrics chart ──────────────────────────────────────────────────
    st.markdown("**Sourced Count**")
    _num_rows, _num_delta_rows = [], []
    for r in summary:
        _num_rows.append({"Provider": r["label"], "Metric": "Total Sourced Count", "Value": r["citation_count_sum"]})
        if _show_delta:
            _bl_cc = _baseline_counts.get(r["provider"], {}).get("citation_count_sum", 0)
            _num_delta_rows.append({"Provider": r["label"], "Metric": "Total Sourced Count", "Δ vs Baseline": r["citation_count_sum"] - _bl_cc})
    if _show_delta:
        _nc_l, _nc_r = st.columns(2, gap="large")
    else:
        _nc_l, _nc_r = st.container(), None
    with _nc_l:
        st.markdown("Absolute counts")
        _fig_num = px.bar(pd.DataFrame(_num_rows), x="Provider", y="Value",
                          color="Metric", barmode="group",
                          color_discrete_map={"Total Sourced Count": "#6366f1"},
                          height=300, text="Value")
        _fig_num.update_traces(textposition="outside")
        _fig_num.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               legend_title_text="", margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Count")
        st.plotly_chart(_fig_num, use_container_width=True)
    if _show_delta and _nc_r is not None:
        with _nc_r:
            st.markdown("Change vs Baseline V0")
            _fig_nd = px.bar(pd.DataFrame(_num_delta_rows), x="Provider", y="Δ vs Baseline",
                             color="Metric", barmode="group",
                             color_discrete_map={"Total Sourced Count": "#6366f1"},
                             height=300, text="Δ vs Baseline")
            _fig_nd.update_traces(textposition="outside")
            _fig_nd.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
            _fig_nd.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  legend_title_text="", margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Δ Count")
            st.plotly_chart(_fig_nd, use_container_width=True)

    # ── Answers Corpus Table ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Answers Corpus")
    st.caption(
        "One row per Version × AI Model combination. "
        "**All 3** is a virtual chatbot that concatenates the answers of all three models."
    )

    _corpus_rows: list = []
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _exp_products = sorted(p for p in _exp_corpus.by_product if not p.startswith("_"))

        _all3_parts: list[str] = []
        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            _parts: list[str] = []
            for _aid in _exp_corpus.by_product.get(_prod, []):
                _resp = _exp_corpus.answers[_aid].response
                if _resp and _resp.strip():
                    _parts.append(_resp.strip())
            _joined = "\n<--><--><-->\n".join(_parts)
            _all3_parts.extend(_parts)
            _corpus_rows.append({
                "Version":  _exp,
                "AI Model": _pm["label"],
                "Answers":  _joined,
            })

        # Virtual "All 3" row
        _corpus_rows.append({
            "Version":  _exp,
            "AI Model": "All 3",
            "Answers":  "\n<--><--><-->\n".join(_all3_parts),
        })

    _corpus_df = pd.DataFrame(_corpus_rows, columns=["Version", "AI Model", "Answers"])
    st.dataframe(
        _corpus_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version":  st.column_config.TextColumn("Version",  width="small"),
            "AI Model": st.column_config.TextColumn("AI Model", width="small"),
            "Answers":  st.column_config.TextColumn("Answers",  width="large"),
        },
    )

    # ── Excel export ───────────────────────────────────────────────────────────
    import io as _io
    _xl_buf = _io.BytesIO()
    with pd.ExcelWriter(_xl_buf, engine="openpyxl") as _xl_writer:
        _corpus_df.to_excel(_xl_writer, index=False, sheet_name="Answers Corpus")
        _ws = _xl_writer.sheets["Answers Corpus"]
        _ws.column_dimensions["A"].width = 14
        _ws.column_dimensions["B"].width = 16
        _ws.column_dimensions["C"].width = 120
        for _row in _ws.iter_rows(min_row=2):
            _row[2].alignment = __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
    st.download_button(
        label="⬇️ Export as Excel",
        data=_xl_buf.getvalue(),
        file_name="answers_corpus.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Answers by Global Mention ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Answers by Global Mention")
    st.caption(
        "One row per Version × AI Model. Answers are split by whether the computed global score "
        "is greater than 0."
    )

    _mention_rows: list = []
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _exp_metrics = _load_metrics_index(_exp)
        _exp_products = sorted(p for p in _exp_corpus.by_product if not p.startswith("_"))

        _all3_global_mentioned_parts: list[str] = []
        _all3_global_not_mentioned_parts: list[str] = []

        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            _global_mentioned_parts: list[str] = []
            _global_not_mentioned_parts: list[str] = []

            for _aid in _exp_corpus.by_product.get(_prod, []):
                _ans = _exp_corpus.answers[_aid]
                _resp = (_ans.response or "").strip()
                if not _resp:
                    continue
                if _exp_metrics[_aid].get("global_mentioned"):
                    _global_mentioned_parts.append(_resp)
                    _all3_global_mentioned_parts.append(_resp)
                else:
                    _global_not_mentioned_parts.append(_resp)
                    _all3_global_not_mentioned_parts.append(_resp)

            _mention_rows.append({
                "Version": _exp,
                "AI Model": _pm["label"],
                "Global Mentioned": "\n<--><--><-->\n".join(_global_mentioned_parts),
                "Global Not Mentioned": "\n<--><--><-->\n".join(_global_not_mentioned_parts),
            })

        # Virtual "All 3" row — combines all chatbots for this version
        _mention_rows.append({
            "Version": _exp,
            "AI Model": "All 3",
            "Global Mentioned": "\n<--><--><-->\n".join(_all3_global_mentioned_parts),
            "Global Not Mentioned": "\n<--><--><-->\n".join(_all3_global_not_mentioned_parts),
        })

    _mention_df = pd.DataFrame(_mention_rows, columns=["Version", "AI Model", "Global Mentioned", "Global Not Mentioned"])
    st.dataframe(
        _mention_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version": st.column_config.TextColumn("Version", width="small"),
            "AI Model": st.column_config.TextColumn("AI Model", width="small"),
            "Global Mentioned": st.column_config.TextColumn("Global Mentioned", width="large"),
            "Global Not Mentioned": st.column_config.TextColumn("Global Not Mentioned", width="large"),
        },
    )

    # ── Per-Question × Chatbot × Version detail table ─────────────────────────
    st.markdown("---")
    st.markdown("### Per-Question Detail Table")
    st.caption(
        "One row per Question × AI Model × Version. "
        "All signals read from the cached metrics index — no recomputation."
    )

    # Build product-level column specs once from FILTER_SPECS
    _pq_prod_specs = [
        s for s in FILTER_SPECS
        if s["group"] in ("Products — Mentioned", "Products — Page Sourced")
    ]

    _pq_rows: list = []
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _exp_metrics = _load_metrics_index(_exp)
        _exp_products = sorted(p for p in _exp_corpus.by_product if not p.startswith("_"))

        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            for _aid in _exp_corpus.by_product.get(_prod, []):
                _ans = _exp_corpus.answers[_aid]
                _q   = _exp_corpus.queries.get(_ans.query_id or "")
                _m   = _exp_metrics[_aid]
                _row: dict = {
                    "Version":            _exp,
                    "AI Model":           _pm["label"],
                    "Question":           _q.text if _q else "",
                    "Answer":             (_ans.response or "").strip(),
                    "Hydromea Mentioned": 1 if _m["mentioned"] else 0,
                    "Hydromea Sourced":   1 if _m["sourced"]   else 0,
                }
                for _s in _pq_prod_specs:
                    _row[_s["label"]] = 1 if _s["fn"](_m) else 0
                _pq_rows.append(_row)

    _pq_fixed_cols  = ["Version", "AI Model", "Question", "Answer",
                        "Hydromea Mentioned", "Hydromea Sourced"]
    _pq_prod_cols   = [s["label"] for s in _pq_prod_specs]
    _pq_all_cols    = _pq_fixed_cols + _pq_prod_cols
    _pq_df = pd.DataFrame(_pq_rows, columns=_pq_all_cols)

    st.dataframe(
        _pq_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version":            st.column_config.TextColumn("Version",  width="small"),
            "AI Model":           st.column_config.TextColumn("AI Model", width="small"),
            "Question":           st.column_config.TextColumn("Question", width="medium"),
            "Answer":             st.column_config.TextColumn("Answer",   width="large"),
            "Hydromea Mentioned": st.column_config.TextColumn("Hydromea Cited",   width="small"),
            "Hydromea Sourced":   st.column_config.TextColumn("Hydromea Sourced", width="small"),
            **{s["label"]: st.column_config.TextColumn(s["label"], width="small") for s in _pq_prod_specs},
        },
    )

    # Excel export
    import io as _io2
    _pq_xl_buf = _io2.BytesIO()
    with pd.ExcelWriter(_pq_xl_buf, engine="openpyxl") as _pq_xl_writer:
        _pq_df.to_excel(_pq_xl_writer, index=False, sheet_name="Per-Question Detail")
        _pq_ws = _pq_xl_writer.sheets["Per-Question Detail"]
        _pq_ws.column_dimensions["A"].width = 14   # Version
        _pq_ws.column_dimensions["B"].width = 14   # AI Model
        _pq_ws.column_dimensions["C"].width = 50   # Question
        _pq_ws.column_dimensions["D"].width = 100  # Answer
        for _pq_row in _pq_ws.iter_rows(min_row=2):
            _pq_row[3].alignment = __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
    st.download_button(
        label="⬇️ Export as Excel",
        data=_pq_xl_buf.getvalue(),
        file_name="per_question_detail.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Per-Question Features Table ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Per-Question Features Table")
    st.caption(
        "One row per Question × AI Model × Version. "
        "Focuses on the shared per-answer Hydromea/product count metrics."
    )

    _feat_rows: list = []
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _exp_metrics = _load_metrics_index(_exp)
        _exp_products = sorted(p for p in _exp_corpus.by_product if not p.startswith("_"))

        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            for _aid in _exp_corpus.by_product.get(_prod, []):
                _ans = _exp_corpus.answers[_aid]
                _q   = _exp_corpus.queries.get(_ans.query_id or "")
                _m   = _exp_metrics[_aid]

                _feat_rows.append({
                    "Version":                          _exp,
                    "AI Model":                         _pm["label"],
                    "Question":                         _q.text if _q else "",
                    "Answer":                           (_ans.response or "").strip(),
                    "Hydromea Mention Count":           _m.get("hydromea_mention_count", 0),
                    "# Products Mentioned":             _m.get("product_mention_count", 0),
                    "Hydromea URI/Source Count":        _m.get("hydromea_text_reference_count", 0),
                    "Global Score":                     _m.get("global_score", 0),
                })

    _feat_cols = [
        "Version", "AI Model", "Question", "Answer",
        "Hydromea Mention Count",
        "# Products Mentioned",
        "Hydromea URI/Source Count",
        "Global Score",
    ]
    _feat_df = pd.DataFrame(_feat_rows, columns=_feat_cols)

    st.dataframe(
        _feat_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version":                    st.column_config.TextColumn("Version",                    width="small"),
            "AI Model":                   st.column_config.TextColumn("AI Model",                   width="small"),
            "Question":                   st.column_config.TextColumn("Question",                   width="medium"),
            "Answer":                     st.column_config.TextColumn("Answer",                     width="large"),
            "Hydromea Mention Count":     st.column_config.NumberColumn("Hydromea Mention Count",  width="small"),
            "# Products Mentioned":       st.column_config.NumberColumn("# Products Mentioned",    width="small"),
            "Hydromea URI/Source Count":  st.column_config.NumberColumn("Hydromea URI/Source",     width="small"),
            "Global Score":               st.column_config.NumberColumn("Global Score",              width="small"),
        },
    )

    import io as _io3
    _feat_xl_buf = _io3.BytesIO()
    with pd.ExcelWriter(_feat_xl_buf, engine="openpyxl") as _feat_xl_writer:
        _feat_df.to_excel(_feat_xl_writer, index=False, sheet_name="Per-Question Features")
        _feat_ws = _feat_xl_writer.sheets["Per-Question Features"]
        _feat_ws.column_dimensions["A"].width = 14
        _feat_ws.column_dimensions["B"].width = 14
        _feat_ws.column_dimensions["C"].width = 50
        _feat_ws.column_dimensions["D"].width = 100
        _feat_ws.column_dimensions["E"].width = 22
        _feat_ws.column_dimensions["F"].width = 22
        _feat_ws.column_dimensions["G"].width = 24
        _feat_ws.column_dimensions["H"].width = 18
        _opx = __import__("openpyxl").styles.Alignment(wrap_text=True, vertical="top")
        for _feat_row in _feat_ws.iter_rows(min_row=2):
            _feat_row[3].alignment = _opx
            _feat_row[7].alignment = _opx
    st.download_button(
        label="⬇️ Export as Excel",
        data=_feat_xl_buf.getvalue(),
        file_name="per_question_features.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Answers by Category Table ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Answers by Category")
    st.caption(
        "One row per Version × AI Model. Each category column contains the concatenation "
        "of all answers for questions belonging to that theme."
    )

    # Ordered theme names: ["T1 – …", "T2 – …", ...]
    _theme_col_names: list[str] = [
        f"{code} – {name}" for code, name in _THEME_CODES.items()
    ]

    _cat_rows: list = []
    for _exp in EXPERIMENTS:
        _exp_corpus = _load(_exp)
        _exp_products = sorted(p for p in _exp_corpus.by_product if not p.startswith("_"))

        # Build a lookup: answer_id → column key ("Tx – Full name" or "❓ Unmatched")
        _UNMATCHED_COL = "❓ Unmatched"
        _all_cat_cols = _theme_col_names + [_UNMATCHED_COL]

        def _answer_theme(_corpus, _aid: str) -> str:
            _a = _corpus.answers[_aid]
            _q = _corpus.queries.get(_a.query_id or "", None)
            if _q is None:
                return _UNMATCHED_COL
            _full_name = _TEXT_TO_THEME.get(_q.text.strip().lower(), "")
            for _c, _n in _THEME_CODES.items():
                if _n == _full_name:
                    return f"{_c} – {_n}"
            return _UNMATCHED_COL  # query exists but not in theme mapping

        _all3_by_theme: dict[str, list[str]] = {t: [] for t in _all_cat_cols}

        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            _by_theme: dict[str, list[str]] = {t: [] for t in _all_cat_cols}

            for _aid in _exp_corpus.by_product.get(_prod, []):
                _t = _answer_theme(_exp_corpus, _aid)
                _resp = _exp_corpus.answers[_aid].response
                if _resp and _resp.strip():
                    _by_theme[_t].append(_resp.strip())
                    _all3_by_theme[_t].append(_resp.strip())

            _cat_row: dict = {
                "Version": _exp,
                "AI Model": _pm["label"],
            }
            for _t in _all_cat_cols:
                _cat_row[_t] = "\n<--><--><-->\n".join(_by_theme[_t])
            _cat_rows.append(_cat_row)

        # "All 3" virtual row
        _all3_row: dict = {"Version": _exp, "AI Model": "All 3"}
        for _t in _all_cat_cols:
            _all3_row[_t] = "\n<--><--><-->\n".join(_all3_by_theme[_t])
        _cat_rows.append(_all3_row)

    _all_cat_cols_final = _theme_col_names + ["❓ Unmatched"]
    _cat_df = pd.DataFrame(_cat_rows, columns=["Version", "AI Model"] + _all_cat_cols_final)
    st.dataframe(
        _cat_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version":  st.column_config.TextColumn("Version",  width="small"),
            "AI Model": st.column_config.TextColumn("AI Model", width="small"),
            **{t: st.column_config.TextColumn(t, width="large") for t in _all_cat_cols_final},
        },
    )

    # Excel export for category table
    _xl_cat_buf = _io.BytesIO()
    with pd.ExcelWriter(_xl_cat_buf, engine="openpyxl") as _xl_cat_writer:
        _cat_df.to_excel(_xl_cat_writer, index=False, sheet_name="By Category")
        _ws2 = _xl_cat_writer.sheets["By Category"]
        _ws2.column_dimensions["A"].width = 14
        _ws2.column_dimensions["B"].width = 16
        _openpyxl_styles = __import__("openpyxl").styles
        _col_letters = [
            __import__("openpyxl").utils.get_column_letter(i + 3)
            for i in range(len(_all_cat_cols_final))
        ]
        for _cl in _col_letters:
            _ws2.column_dimensions[_cl].width = 80
        for _row2 in _ws2.iter_rows(min_row=2):
            for _cell in _row2[2:]:
                _cell.alignment = _openpyxl_styles.Alignment(wrap_text=True, vertical="top")
    st.download_button(
        label="⬇️ Export by Category as Excel",
        data=_xl_cat_buf.getvalue(),
        file_name="answers_by_category.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ─────────────────────────────────────────────────────────────────────────────
# Word Cloud page
# ─────────────────────────────────────────────────────────────────────────────
if _PAGE == "☁️ Word Cloud":
    import re as _re
    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from wordcloud import WordCloud as _WC, STOPWORDS as _WC_STOPS
    try:
        from nltk.corpus import stopwords as _nltk_sw
        _NLTK_STOPS = set(_nltk_sw.words("english"))
    except Exception:
        _NLTK_STOPS = set()

    _CUSTOM_STOPS = {
        "can", "also", "use", "used", "using", "well", "one", "two", "three",
        "many", "much", "often", "provide", "provides", "provided", "offering",
        "offer", "offers", "include", "includes", "including", "example",
        "examples", "typically", "usually", "generally", "such", "may", "might",
        "need", "needs", "required", "requires", "allow", "allows", "ensure",
        "ensures", "help", "helps", "make", "makes", "made", "note", "however",
        "therefore", "additionally", "furthermore", "key", "important", "based",
        "specific", "especially", "various", "several", "certain", "different",
        "available", "designed", "system", "systems", "solution", "solutions",
        "technology", "technologies", "high", "low", "large", "small", "long",
        "short", "query", "user", "summary", "sources", "source", "information",
        "first", "second", "third", "fourth", "fifth",
    }
    _ALL_STOPS = _NLTK_STOPS | _WC_STOPS | _CUSTOM_STOPS

    def _wc_clean(text: str) -> str:
        text = _re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        text = _re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = _re.sub(r"https?://\S+", " ", text)
        text = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = _re.sub(r"^#{1,6}\s*", " ", text, flags=_re.MULTILINE)
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\d+", " ", text)
        text = _re.sub(r"[^a-zA-Z\s]", " ", text)
        text = _re.sub(r"\s+", " ", text).strip().lower()
        return " ".join(w for w in text.split() if w not in _ALL_STOPS and len(w) > 2)

    _SPHINX_CMAP    = _LSC.from_list("sphinx_blue",  ["#1565c0", "#1e88e5", "#42a5f5", "#90caf9"])
    _NO_GLOBAL_CMAP = _LSC.from_list("sphinx_slate", ["#475569", "#64748b", "#94a3b8", "#cbd5e1"])

    # Circular mask (1000×1000 square canvas)
    _sz = 1000
    _yy, _xx = _np.ogrid[:_sz, :_sz]
    _cc = _sz // 2
    _circle_mask = _np.full((_sz, _sz), 255, dtype=_np.uint8)
    _circle_mask[(_xx - _cc) ** 2 + (_yy - _cc) ** 2 <= (_cc - 4) ** 2] = 0

    def _make_wc(text, cmap, max_words):
        return _WC(
            mask=_circle_mask,
            background_color="white",
            stopwords=_ALL_STOPS,
            max_words=max_words,
            min_font_size=9,
            max_font_size=110,
            collocations=False,
            colormap=cmap,
            prefer_horizontal=1.0,
            relative_scaling=0.45,
            random_state=42,
        ).generate(text)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="margin-bottom:2px;">☁️ Word Cloud — All Versions</h1>'
        '<p style="color:#64748b;font-size:15px;margin-top:0;">'
        'Aggregated chatbot answers across all providers · '
        '<strong style="color:#1565c0;">top row = Global Mentioned</strong> · '
        '<strong style="color:#475569;">bottom row = Not Mentioned</strong></p>',
        unsafe_allow_html=True,
    )

    _max_words = st.slider("Max words per cloud", 10, 150, 30, 5)
    st.divider()

    # ── Build text blobs: experiment → {True: str, False: str} ───────────────
    _texts: dict = {}
    for _vexp in EXPERIMENTS:
        _v_corpus  = _load(_vexp)
        _v_metrics = _load_metrics_index(_vexp)
        _yes, _no  = [], []
        for _vans in _v_corpus.answers.values():
            _vm = _v_metrics.get(_vans.answer_id, {})
            (_yes if _vm.get("global_mentioned", False) else _no).append(_vans.response or "")
        _texts[_vexp] = {
            True:  _wc_clean(" ".join(_yes)),
            False: _wc_clean(" ".join(_no)),
        }

    # ── Render one row of 4 clouds ────────────────────────────────────────────
    def _render_row(label: str, color: str, flag: bool, cmap) -> None:
        st.markdown(
            f'<h3 style="color:{color};margin-bottom:6px;">{label}</h3>',
            unsafe_allow_html=True,
        )
        _cols = st.columns(len(EXPERIMENTS))
        for _col, _vexp in zip(_cols, EXPERIMENTS):
            with _col:
                _blob = _texts[_vexp][flag]
                if not _blob.strip():
                    st.caption(f"**{_vexp}** — no data")
                    continue
                _wc_obj = _make_wc(_blob, cmap, _max_words)
                _fig, _ax = plt.subplots(figsize=(5, 5), facecolor="white")
                _ax.imshow(_wc_obj, interpolation="bilinear")
                _ax.axis("off")
                _ax.set_title(_vexp, fontsize=13, fontweight="bold", color=color, pad=8)
                plt.tight_layout()
                st.pyplot(_fig, use_container_width=True)
                plt.close(_fig)
                _word_counts = {}
                for _w in _blob.split():
                    _word_counts[_w] = _word_counts.get(_w, 0) + 1
                _freq_df = pd.DataFrame(
                    [{"Word": w, "Frequency": c} for w, c in _word_counts.items()
                     if w in _wc_obj.words_],
                ).sort_values("Frequency", ascending=False).reset_index(drop=True)
                _freq_df.index += 1
                st.dataframe(_freq_df, use_container_width=True, height=250)

    _render_row("🌐 Global Mentioned  (score > 0)", "#1565c0", True,  _SPHINX_CMAP)
    st.divider()
    _render_row("⬜ Global Not Mentioned  (score = 0)", "#1565c0", False, _SPHINX_CMAP)
    st.stop()  # Don't render the explorer below

# ─────────────────────────────────────────────────────────────────────────────
# Main — guard (Explorer page)
# ─────────────────────────────────────────────────────────────────────────────
qid = st.session_state.get("qid")
if not qid or qid not in corpus.queries:
    st.info("Select a query from the sidebar to begin.")
    st.stop()

query = corpus.queries[qid]

# ─────────────────────────────────────────────────────────────────────────────
# Query header
# ─────────────────────────────────────────────────────────────────────────────
intent_css = _THEME_NAME_TO_CSS.get(query_theme(query.text), "badge-researching")
_theme_display = query_theme(query.text) or "—"
branded_html = '<span class="badge badge-branded">Branded</span>' if query.metadata.get("branded") else ""
created_date = (query.created_at or "")[:10]

answer_ids_for_q = corpus.by_query.get(qid, [])
_query_text_escaped = _html.escape(query.text)
# Build meta badges as a single string — a blank interpolated variable inside
# a multiline st.markdown block would create an empty line that breaks the
# CommonMark HTML-block parser, causing subsequent tags to render as raw text.
_meta_html = (
    f'<span class="badge {intent_css}">{_html.escape(_theme_display)}</span>'
    + (f' <span class="badge badge-branded">Branded</span>' if query.metadata.get("branded") else '')
    + f' <span class="badge badge-id">{query.query_id}</span>'
    + f' <span style="font-size:.78rem;color:#9ca3af;margin-left:auto">🕐 {created_date}</span>'
)
st.markdown(
    f'<div class="query-card">'
    f'<div class="query-text">💬 {_query_text_escaped}</div>'
    f'<div class="query-meta">{_meta_html}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Collect answers
# ─────────────────────────────────────────────────────────────────────────────
answers_by_product: Dict[str, Answer] = {}
for aid in corpus.by_query.get(qid, []):
    a = corpus.answers[aid]
    if a.product and not a.product.startswith("_"):
        answers_by_product[a.product] = a

# ─────────────────────────────────────────────────────────────────────────────
# Side-by-side columns
# ─────────────────────────────────────────────────────────────────────────────
active_products = [p for p in products if p in answers_by_product]
# Fall back to all known products if no answers loaded yet (edge case)
if not active_products:
    active_products = products

cols = st.columns(max(len(active_products), 1), gap="large")

for col, product in zip(cols, active_products):
    ans   = answers_by_product.get(product)
    m     = PRODUCT_META.get(product, _DMETA)
    label = m["label"]
    icon  = m["icon"]
    hdr   = f"provider-{product.lower()}"

    with col:
        # ── Provider header ───────────────────────────────────────────────
        st.markdown(
            f'<div class="provider-header {hdr}">{icon} {label}</div>',
            unsafe_allow_html=True,
        )

        if ans is None:
            st.markdown('<div class="answer-card no-answer">No answer available for this provider.</div>', unsafe_allow_html=True)
            continue

        # ── Hydromea visibility ────────────────────────────────────────────
        _m  = answer_metrics_index[ans.answer_id]
        _bi = _m["brand_idxs"]

        # Auto-render one badge per metric group from FILTER_SPECS
        _badges_by_group: dict = {}
        for spec in FILTER_SPECS:
            grp = spec["group"]
            val = spec["fn"](_m)
            badge_html = (
                f'<span class="vis-badge" style="background:{spec["color"]}22;'
                f'color:{spec["color"]};border:1px solid {spec["color"]}55;">'
                f'{spec["icon"]} {spec["label"]}</span>'
                if val else
                f'<span class="vis-badge vis-uncited">'
                f'— {spec["label"]}</span>'
            )
            _badges_by_group.setdefault(grp, []).append(badge_html)

        for grp, badges in _badges_by_group.items():
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;letter-spacing:1px;'
                f'color:#64748b;margin-bottom:4px;">{grp.upper()}</div>'
                + " ".join(badges),
                unsafe_allow_html=True,
            )
            st.markdown("<div style='margin-bottom:6px;'></div>", unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:4px;'></div>", unsafe_allow_html=True)

        # ── Stats row ─────────────────────────────────────────────────────
        total_src    = _m["n_sources"]
        source_position_raw = _m["source_position"]
        source_position_str = "N/A" if source_position_raw in (None, -1) else str(source_position_raw)
        citation_count      = _m["citation_count"]
        global_mentioned_str = "Yes" if _m.get("global_mentioned") else "No"
        ans_date     = (ans.timing.get("created_at") or "")[:10] or "—"
        report_key   = ans.run_context.get("visibility_report_key", "—")

        st.markdown(f"""
<div class="answer-card">
  <div class="stat-row">
    <div class="stat-pill">📅 <strong>{ans_date}</strong></div>
    <div class="stat-pill">📚 <strong>{total_src}</strong> sources</div>
    <div class="stat-pill">📍 Source Position <strong>{source_position_str}</strong></div>
    <div class="stat-pill">🔢 Sourced Count <strong>{citation_count}</strong></div>
        <div class="stat-pill">🌐 Global Mentioned <strong>{global_mentioned_str}</strong></div>
  </div>
  <hr style="border:none;border-top:1px solid #f3f4f6;margin:.6rem 0 .9rem">
  <div class="response-body">
""", unsafe_allow_html=True)

        # ── Response (rendered markdown) ──────────────────────────────────
        st.markdown(_clean_response(ans.response))

        # ── Tagged response overlay ───────────────────────────────────────
        if show_tagging:
            tagged_payload = tagged_answers.get(ans.answer_id)
            if tagged_payload:
                tagged_lines = tagged_payload.get("lines") or []
                if tagged_lines:
                    summary_counts = (tagged_payload.get("summary") or {}).get("span_counts") or _compute_span_counts(tagged_lines)
                    legend_html: List[str] = []
                    for cat, cnt in summary_counts.items():
                        if not cnt:
                            continue
                        style = _TAG_STYLE.get(cat, _TAG_STYLE_DEFAULT)
                        legend_html.append(
                            f'<span class="tag-chip {style["chip"]}">{_html.escape(cat)} · {cnt}</span>'
                        )

                    hint_line = (
                        '<div class="tagging-hint">Hover highlighted spans to inspect category and character offsets.</div>'
                        if show_tagging_hints else
                        '<div class="tagging-hint">Highlighted snippets indicate tagged evidence spans per category.</div>'
                    )

                    tagged_html = _render_tagged_lines(tagged_lines)
                    st.markdown(
                        '<div class="tagging-panel">'
                        '<div class="tagging-title">Tagged Answer View</div>'
                        f'{hint_line}'
                        f'<div class="tag-legend">{"".join(legend_html)}</div>'
                        f'<div class="tagged-response">{tagged_html}</div>'
                        '</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Tagging: no matching tagged JSON entry found for this answer in the selected experiment.")

        st.markdown("</div></div>", unsafe_allow_html=True)

        # ── Sources expander ──────────────────────────────────────────────
        _n_brand  = len(_bi)
        _src_lbl  = f"📚 Sources · {len(ans.sources)}"
        if _n_brand:
            _src_lbl += f" · {_n_brand} Hydromea 🔵"
        if ans.sources:
            with st.expander(_src_lbl, expanded=False):
                _brand_set = set(_bi)
                _src_html  = ['<ol style="padding-left:18px;font-size:13px;line-height:1.9;">']
                for i, src in enumerate(ans.sources, 1):
                    _url   = src.get("url", "#")
                    _title = _html.escape(src.get("title") or src.get("hostname") or _url)
                    _host  = _html.escape(src.get("hostname", ""))
                    _is_b  = i in _brand_set
                    _rbg   = "background:#eff6ff;border-radius:4px;padding:1px 4px;" if _is_b else ""
                    _bbdg  = (' <span style="background:#dbeafe;color:#1e40af;border-radius:4px;'
                              'padding:1px 6px;font-size:11px;font-weight:700;">🔵 Hydromea</span>'
                              if _is_b else "")
                    _src_html.append(
                        f'<li style="{_rbg}margin-bottom:3px;">'
                        f'<a href="{_url}" target="_blank" rel="noopener noreferrer">{_title}</a>'
                        f'{_bbdg}<span style="color:#94a3b8;font-size:11px;margin-left:6px;">{_host}</span>'
                        f'</li>'
                    )
                _src_html.append('</ol>')
                st.markdown("".join(_src_html), unsafe_allow_html=True)
        else:
            st.caption("No sources attached.")

# ─────────────────────────────────────────────────────────────────────────────
# Word Cloud page
# ─────────────────────────────────────────────────────────────────────────────
if _PAGE == "☁️ Word Cloud":
    import re as _re
    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap as _LSC
    from wordcloud import WordCloud as _WC, STOPWORDS as _WC_STOPS
    try:
        from nltk.corpus import stopwords as _nltk_sw
        _NLTK_STOPS = set(_nltk_sw.words("english"))
    except Exception:
        _NLTK_STOPS = set()

    _CUSTOM_STOPS = {
        "can", "also", "use", "used", "using", "well", "one", "two", "three",
        "many", "much", "often", "provide", "provides", "provided", "offering",
        "offer", "offers", "include", "includes", "including", "example",
        "examples", "typically", "usually", "generally", "such", "may", "might",
        "need", "needs", "required", "requires", "allow", "allows", "ensure",
        "ensures", "help", "helps", "make", "makes", "made", "note", "however",
        "therefore", "additionally", "furthermore", "key", "important", "based",
        "specific", "especially", "various", "several", "certain", "different",
        "available", "designed", "system", "systems", "solution", "solutions",
        "technology", "technologies", "high", "low", "large", "small", "long",
        "short", "query", "user", "summary", "sources", "source", "information",
        "first", "second", "third", "fourth", "fifth",
    }
    _ALL_STOPS = _NLTK_STOPS | _WC_STOPS | _CUSTOM_STOPS

    def _wc_clean(text: str) -> str:
        text = _re.sub(r"!\[.*?\]\(.*?\)", " ", text)
        text = _re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = _re.sub(r"https?://\S+", " ", text)
        text = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
        text = _re.sub(r"^#{1,6}\s*", " ", text, flags=_re.MULTILINE)
        text = _re.sub(r"<[^>]+>", " ", text)
        text = _re.sub(r"\d+", " ", text)
        text = _re.sub(r"[^a-zA-Z\s]", " ", text)
        text = _re.sub(r"\s+", " ", text).strip().lower()
        return " ".join(w for w in text.split() if w not in _ALL_STOPS and len(w) > 2)

    _SPHINX_CMAP    = _LSC.from_list("sphinx_blue",  ["#1565c0", "#1e88e5", "#42a5f5", "#90caf9"])
    _NO_GLOBAL_CMAP = _LSC.from_list("sphinx_slate", ["#475569", "#64748b", "#94a3b8", "#cbd5e1"])

    # Circular mask (1000×1000 square canvas)
    _sz = 1000
    _yy, _xx = _np.ogrid[:_sz, :_sz]
    _cc = _sz // 2
    _circle_mask = _np.full((_sz, _sz), 255, dtype=_np.uint8)
    _circle_mask[(_xx - _cc) ** 2 + (_yy - _cc) ** 2 <= (_cc - 4) ** 2] = 0

    def _make_wc(text, cmap, max_words):
        return _WC(
            mask=_circle_mask,
            background_color="white",
            stopwords=_ALL_STOPS,
            max_words=max_words,
            min_font_size=9,
            max_font_size=110,
            collocations=False,
            colormap=cmap,
            prefer_horizontal=1.0,
            relative_scaling=0.45,
            random_state=42,
        ).generate(text)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        '<h1 style="margin-bottom:2px;">☁️ Word Cloud — All Versions</h1>'
        '<p style="color:#64748b;font-size:15px;margin-top:0;">'
        'Aggregated chatbot answers across all providers · '
        '<strong style="color:#1565c0;">top row = Global Mentioned</strong> · '
        '<strong style="color:#475569;">bottom row = Not Mentioned</strong></p>',
        unsafe_allow_html=True,
    )

    _max_words = st.slider("Max words per cloud", 10, 150, 30, 5)
    st.divider()

    # ── Build text blobs: experiment → {True: str, False: str} ───────────────
    _texts: dict = {}
    for _vexp in EXPERIMENTS:
        _v_corpus  = _load(_vexp)
        _v_metrics = _load_metrics_index(_vexp)
        _yes, _no  = [], []
        for _vans in _v_corpus.answers.values():
            _vm = _v_metrics.get(_vans.answer_id, {})
            (_yes if _vm.get("global_mentioned", False) else _no).append(_vans.response or "")
        _texts[_vexp] = {
            True:  _wc_clean(" ".join(_yes)),
            False: _wc_clean(" ".join(_no)),
        }

    # ── Render one row of 4 clouds ────────────────────────────────────────────
    def _render_row(label: str, color: str, flag: bool, cmap) -> None:
        st.markdown(
            f'<h3 style="color:{color};margin-bottom:6px;">{label}</h3>',
            unsafe_allow_html=True,
        )
        _cols = st.columns(len(EXPERIMENTS))
        for _col, _vexp in zip(_cols, EXPERIMENTS):
            with _col:
                _blob = _texts[_vexp][flag]
                if not _blob.strip():
                    st.caption(f"**{_vexp}** — no data")
                    continue
                _wc_obj = _make_wc(_blob, cmap, _max_words)
                _fig, _ax = plt.subplots(figsize=(5, 5), facecolor="white")
                _ax.imshow(_wc_obj, interpolation="bilinear")
                _ax.axis("off")
                _ax.set_title(_vexp, fontsize=13, fontweight="bold", color=color, pad=8)
                plt.tight_layout()
                st.pyplot(_fig, use_container_width=True)
                plt.close(_fig)
                _word_counts = {}
                for _w in _blob.split():
                    _word_counts[_w] = _word_counts.get(_w, 0) + 1
                _freq_df = pd.DataFrame(
                    [{"Word": w, "Frequency": c} for w, c in _word_counts.items()
                     if w in _wc_obj.words_],
                ).sort_values("Frequency", ascending=False).reset_index(drop=True)
                _freq_df.index += 1
                st.dataframe(_freq_df, use_container_width=True, height=250)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"corpus loaded: **{len(corpus.queries)}** queries · "
    f"**{len(corpus.answers)}** answers · "
    f"report warnings: **{len(corpus.report.warnings)}**"
)
