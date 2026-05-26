#!/usr/bin/env python3
"""Debug script to check tag delta inconsistency."""
import sys
sys.path.insert(0, '/Users/adnenkhiari/Documents/nawnaw')

from main import build_corpus, list_experiments
from app import (
    _build_tag_partition_rows,
    _load_tagged_answers,
    TAG_PARTITION_CATEGORIES,
    query_theme,
)
import pandas as pd

# Load experiments
EXPERIMENTS = list_experiments()
_cur_exp = "V1"
_BL_EXP = "Baseline V0"

print("=" * 80)
print(f"Comparing {_cur_exp} vs {_BL_EXP}")
print("=" * 80)

# Load current and baseline
corpus_cur = build_corpus(experiment=_cur_exp)
corpus_bl = build_corpus(experiment=_BL_EXP)

# Build tag partition rows
tag_current_df, tag_current_meta = _build_tag_partition_rows(corpus_cur, _cur_exp)
tag_baseline_df, tag_baseline_meta = _build_tag_partition_rows(corpus_bl, _BL_EXP)

print(f"\nCurrent ({_cur_exp}): {len(tag_current_df)} answers")
print(f"Baseline ({_BL_EXP}): {len(tag_baseline_df)} answers")

# Identify baseline-mentioned and current-mentioned queries (union)
baseline_mentioned_qids = set(
    tag_baseline_df.loc[tag_baseline_df["hydromea_mentioned"], "query_id"].tolist()
)
current_mentioned_qids = set(
    tag_current_df.loc[tag_current_df["hydromea_mentioned"], "query_id"].tolist()
)
hydromea_mentioned_qids = baseline_mentioned_qids | current_mentioned_qids

print(f"Baseline-mentioned query IDs: {len(baseline_mentioned_qids)}")
print(f"Current-mentioned query IDs: {len(current_mentioned_qids)}")
print(f"Union (Hydromea Mentioned): {len(hydromea_mentioned_qids)}")

# Filter to mentioned queries
tag_current_mentioned = tag_current_df[tag_current_df["query_id"].isin(hydromea_mentioned_qids)].copy()
tag_baseline_mentioned = tag_baseline_df[tag_baseline_df["query_id"].isin(hydromea_mentioned_qids)].copy()

print(f"Current answers in mentioned queries: {len(tag_current_mentioned)}")
print(f"Baseline answers in mentioned queries: {len(tag_baseline_mentioned)}")

# Calculate counts for ALL QUERIES
print("\n" + "=" * 80)
print("ALL QUERIES")
print("=" * 80)
for cat in TAG_PARTITION_CATEGORIES:
    cur_count = int(tag_current_df[cat].sum())
    bl_count = int(tag_baseline_df[cat].sum())
    delta = cur_count - bl_count
    pct_change = (delta / bl_count * 100) if bl_count > 0 else 0
    print(f"{cat:25s}: BL={bl_count:4d}, CUR={cur_count:4d}, Delta={delta:4d} ({pct_change:+.1f}%)")

# Calculate counts for BASELINE-MENTIONED QUERIES only
print("\n" + "=" * 80)
print("BASELINE-MENTIONED QUERIES ONLY")
print("=" * 80)
for cat in TAG_PARTITION_CATEGORIES:
    cur_count = int(tag_current_mentioned[cat].sum())
    bl_count = int(tag_baseline_mentioned[cat].sum())
    delta = cur_count - bl_count
    pct_change = (delta / bl_count * 100) if bl_count > 0 else 0
    print(f"{cat:25s}: BL={bl_count:4d}, CUR={cur_count:4d}, Delta={delta:4d} ({pct_change:+.1f}%)")

print("\n" + "=" * 80)
print("COMPARISON: Strong Recommendation")
print("=" * 80)
sr_all_bl = int(tag_baseline_df["Strong Recommendation"].sum())
sr_all_cur = int(tag_current_df["Strong Recommendation"].sum())
sr_all_delta = sr_all_cur - sr_all_bl

sr_ment_bl = int(tag_baseline_mentioned["Strong Recommendation"].sum())
sr_ment_cur = int(tag_current_mentioned["Strong Recommendation"].sum())
sr_ment_delta = sr_ment_cur - sr_ment_bl

print(f"All queries:        BL={sr_all_bl}, CUR={sr_all_cur}, Delta={sr_all_delta}")
print(f"Mentioned queries:  BL={sr_ment_bl}, CUR={sr_ment_cur}, Delta={sr_ment_delta}")
print(f"\nDifference in delta: {sr_ment_delta} - {sr_all_delta} = {sr_ment_delta - sr_all_delta}")
