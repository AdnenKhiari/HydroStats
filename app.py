# app.py  –  AI Answer Comparison + Hydromea Visibility  (Streamlit)
from __future__ import annotations
import difflib
import html as _html
import json
import pathlib
import re
from typing import Dict, List
import pandas as pd
import plotly.express as px
import streamlit as st
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
    except Exception:
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

def compute_answer_metrics(ans: Answer) -> dict:
    """
    Single source of truth for all per-answer metrics.
    Auto-populates `product_<key>` for every entry in _PRODUCT_PATTERNS.
    """
    _bi   = brand_idxs(ans)
    _text = ans.response or ""
    metrics: dict = {
        "sourced":         bool(_bi),
        "mentioned":       "hydromea" in _text.lower(),
        "source_position": _bi[0] if _bi else -1,
        "citation_count":  len(_bi),
        "n_sources":       ans.run_context.get("stats", {}).get("totalSources", len(ans.sources)),
        "brand_idxs":      _bi,
    }
    for p in _PRODUCT_PATTERNS:
        metrics[f'product_{p["key"]}'] = bool(p["pattern"].search(_text))
        if "source_url" in p:
            needle = p["source_url"].lower()
            metrics[f'product_{p["key"]}_sourced'] = any(
                needle in s.get("url", "").lower()
                for s in ans.sources
            )
    return metrics

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


def build_stats_df(corpus: LinkedCorpus) -> pd.DataFrame:
    """
    Wide dataframe: one row per query, one column per (product × FILTER_SPEC).
    Adding a new metric to FILTER_SPECS automatically adds its column here.
    """
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
                m = compute_answer_metrics(ans_list[0])
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
    """Called by the selectbox on_change — clears stale query id before the rerun."""
    st.session_state.pop("qid", None)

# Resolve current experiment from session state before any rendering
_cur_exp = st.session_state.get("sel_exp", EXPERIMENTS[0] if EXPERIMENTS else "")
corpus   = _load(_cur_exp)
products = sorted(p for p in corpus.by_product if not p.startswith("_"))


def _clean_response(text: str) -> str:
    """Strip the repeated 'Query: User query:' header OpenAI sometimes prepends."""
    return re.sub(r"^\*{0,2}Query:\s*User query:\s*\*{0,2}", "", text.strip())


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
        index=EXPERIMENTS.index(_cur_exp) if _cur_exp in EXPERIMENTS else 0,
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
        "nav", ["📋 Explorer", "📊 Hydromea Stats"],
        key="nav_page",
        label_visibility="collapsed",
    )
    st.divider()

    search_q = st.text_input("🔍 Search", placeholder="keyword in query…", label_visibility="collapsed")

    intents = sorted({query_theme(q.text) for q in corpus.queries.values() if query_theme(q.text)})
    intent_sel = st.selectbox("Filter by theme", ["All themes"] + intents, label_visibility="collapsed")

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
            result = any(spec["fn"](compute_answer_metrics(a)) for a in ans_for_q)
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

# ─────────────────────────────────────────────────────────────────────────────
# Stats page (shown when _PAGE == "📊 Hydromea Stats")
# ─────────────────────────────────────────────────────────────────────────────
if _PAGE == "📊 Hydromea Stats":
    df = build_stats_df(corpus)
    products_all = sorted(p for p in corpus.by_product if not p.startswith("_"))

    # ── Baseline V0 reference ─────────────────────────────────────────────────
    _BL_EXP = "Baseline V0"
    _is_baseline = (_cur_exp == _BL_EXP)
    _baseline_counts: dict = {}   # provider → {spec_key → count, citation_count_sum → int}
    if not _is_baseline and _BL_EXP in EXPERIMENTS:
        _bl_corpus = _load(_BL_EXP)
        _bl_df = build_stats_df(_bl_corpus)
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
            _joined = "\n\n---\n\n".join(_parts)
            _all3_parts.extend(_parts)
            _corpus_rows.append({
                "Version":  _exp,
                "AI Model": f'{_pm["icon"]} {_pm["label"]}',
                "Answers":  _joined,
            })

        # Virtual "All 3" row
        _corpus_rows.append({
            "Version":  _exp,
            "AI Model": "🤖 All 3",
            "Answers":  "\n\n---\n\n".join(_all3_parts),
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

        # Build a lookup: answer_id → theme name (via its query text)
        def _answer_theme(_corpus, _aid: str) -> str:
            _a = _corpus.answers[_aid]
            _q = _corpus.queries.get(_a.query_id or "", None)
            if _q is None:
                return ""
            _code = _TEXT_TO_THEME.get(_q.text.strip().lower(), "")
            # Convert full name back to "Tx – Full name" format
            for _c, _n in _THEME_CODES.items():
                if _n == _code:
                    return f"{_c} – {_n}"
            return _code  # fallback (already full name or empty)

        _all3_by_theme: dict[str, list[str]] = {t: [] for t in _theme_col_names}

        for _prod in _exp_products:
            _pm = PRODUCT_META.get(_prod, _DMETA)
            _by_theme: dict[str, list[str]] = {t: [] for t in _theme_col_names}

            for _aid in _exp_corpus.by_product.get(_prod, []):
                _t = _answer_theme(_exp_corpus, _aid)
                _resp = _exp_corpus.answers[_aid].response
                if _t in _by_theme and _resp and _resp.strip():
                    _by_theme[_t].append(_resp.strip())
                    _all3_by_theme[_t].append(_resp.strip())

            _cat_row: dict = {"Version": _exp, "AI Model": f'{_pm["icon"]} {_pm["label"]}'}
            for _t in _theme_col_names:
                _cat_row[_t] = "\n\n---\n\n".join(_by_theme[_t])
            _cat_rows.append(_cat_row)

        # "All 3" virtual row
        _all3_row: dict = {"Version": _exp, "AI Model": "🤖 All 3"}
        for _t in _theme_col_names:
            _all3_row[_t] = "\n\n---\n\n".join(_all3_by_theme[_t])
        _cat_rows.append(_all3_row)

    _cat_df = pd.DataFrame(_cat_rows, columns=["Version", "AI Model"] + _theme_col_names)
    st.dataframe(
        _cat_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Version":  st.column_config.TextColumn("Version",  width="small"),
            "AI Model": st.column_config.TextColumn("AI Model", width="small"),
            **{t: st.column_config.TextColumn(t, width="large") for t in _theme_col_names},
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
            for i in range(len(_theme_col_names))
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
answers_by_product: dict[str, object] = {}
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
        _m  = compute_answer_metrics(ans)
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
        ans_date     = (ans.timing.get("created_at") or "")[:10] or "—"
        report_key   = ans.run_context.get("visibility_report_key", "—")

        st.markdown(f"""
<div class="answer-card">
  <div class="stat-row">
    <div class="stat-pill">📅 <strong>{ans_date}</strong></div>
    <div class="stat-pill">📚 <strong>{total_src}</strong> sources</div>
    <div class="stat-pill">📍 Source Position <strong>{source_position_str}</strong></div>
    <div class="stat-pill">🔢 Sourced Count <strong>{citation_count}</strong></div>
  </div>
  <hr style="border:none;border-top:1px solid #f3f4f6;margin:.6rem 0 .9rem">
  <div class="response-body">
""", unsafe_allow_html=True)

        # ── Response (rendered markdown) ──────────────────────────────────
        st.markdown(_clean_response(ans.response))

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
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"corpus loaded: **{len(corpus.queries)}** queries · "
    f"**{len(corpus.answers)}** answers · "
    f"report warnings: **{len(corpus.report.warnings)}**"
)
