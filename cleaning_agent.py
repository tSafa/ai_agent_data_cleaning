"""
llm_agent/cleaning_agent.py  v11
==================================
FIXES v11:
  - Fix 1: client_name and attribution columns use fill_constant(UNKNOWN)
            not fill_mode to avoid inflating one entity in dashboards
  - Fix 2: foreign key columns (user_id, whatsapp_user_id, *_id suffix)
            use fill_constant(-1) not fill_median
  - Fix 3: payment_date and semantic date columns get fix_datetime with
            LOW severity and a note that nulls are intentional
  - Fix 4: fill_constant is type-safe for numeric and string columns
  - Syntax error in _ATTRIBUTION_PAT fixed (broken string concatenation)
  - No icons anywhere in output
"""

import os
import sys
import json
import re
import argparse
import unicodedata
import warnings
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LLM_PROVIDER      = os.getenv("LLM_PROVIDER",   "ollama").lower()
AGENT_MODE        = os.getenv("AGENT_MODE",      "HUMAN").upper()
OLLAMA_HOST       = os.getenv("OLLAMA_HOST",     "http://localhost:11434")
OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY",    "")

BASE_DIR      = Path(__file__).parent.parent
RULES_DB_PATH = Path(os.getenv(
    "RULES_DB_PATH", str(BASE_DIR / "rules_db" / "approved_rules.json")
))
PROFILES_DIR  = BASE_DIR / "profiler" / "profiles"


# =====================================================================
#  PLAIN-ENGLISH ACTION EXPLAINER
# =====================================================================

_ACTION_PLAIN_ENGLISH = {
    "drop_column": (
        "Remove the entire column '{col}' from the dataset. "
        "It has too many missing values to be useful."
    ),
    "drop_rows_if_null": (
        "Delete any row where '{col}' is empty. "
        "Rows with no value in this column cannot be used for analysis."
    ),
    "drop_duplicates": (
        "Remove rows that are exact copies of another row. "
        "Duplicates inflate counts and distort totals in reports."
    ),
    "fill_median": (
        "Fill empty values in '{col}' with the median (middle) value of the column. "
        "The median is used instead of the average because it is not skewed "
        "by extreme outliers."
    ),
    "fill_mode": (
        "Fill empty values in '{col}' with the most common value in that column. "
        "This is the standard approach for text/category columns with missing entries."
    ),
    "fill_constant": (
        "Fill empty values in '{col}' with a fixed placeholder value "
        "(e.g. 'UNKNOWN' or 0). "
        "Use this when you want all gaps filled with the same label."
    ),
    "clip_outliers": (
        "Pull extreme values in '{col}' back within normal bounds. "
        "Values that are far too high or far too low are capped at the fence limits "
        "so they do not distort averages and KPIs."
    ),
    "flag_outliers": (
        "Add a new column '{col}_is_outlier' that marks suspicious values with 1 "
        "(outlier) or 0 (normal). The original values are kept untouched. "
        "Use this when you are not sure whether the extreme values are errors "
        "or genuinely rare."
    ),
    "lowercase": (
        "Convert all text in '{col}' to lowercase and remove extra spaces. "
        "This prevents the same word written differently (e.g. 'Paris' vs 'PARIS') "
        "from being counted as two separate categories."
    ),
    "cast_to_numeric": (
        "Convert '{col}' from text to a number. "
        "The column contains numbers stored as text, which prevents calculations "
        "like SUM or AVG."
    ),
    "replace_sentinel": (
        "Replace placeholder text in '{col}' (such as 'N/A', 'None', 'Unknown', '-') "
        "with a true empty value. "
        "These strings look like data but actually mean the value is missing."
    ),
    "fix_datetime": (
        "Parse '{col}' as a proper date/time column. "
        "Right now the dates are stored as plain text, which means you cannot "
        "filter by date range or compute durations."
    ),
    "normalize_unicode": (
        "Standardize special characters in '{col}' (e.g. accented letters, "
        "curly quotes, non-breaking spaces). This prevents invisible character "
        "differences from causing mismatches in joins and group-bys."
    ),
}


def plain_english_explanation(action: str, col: str, params: dict = None) -> str:
    template = _ACTION_PLAIN_ENGLISH.get(action)
    if template:
        text = template.format(col=col)
        if action == "clip_outliers" and params:
            lo = params.get("lower")
            hi = params.get("upper")
            if lo is not None and hi is not None:
                text += f" The allowed range is [{round(lo,2)} to {round(hi,2)}]."
        if action == "fill_constant" and params:
            text += f" The fill value will be: '{params.get('value', 'UNKNOWN')}'."
        return text
    return f"Apply '{action}' to column '{col}'."


# =====================================================================
#  PREVIEW ENGINE
# =====================================================================

def preview_rule(rule: dict, df: pd.DataFrame) -> None:
    action = rule.get("action", "")
    col    = rule.get("column", "")
    params = rule.get("params", {}) or {}

    print("\n  --- PREVIEW (5 sample rows) ---")

    try:
        if action == "drop_column":
            if col not in df.columns:
                print(f"  Column '{col}' not found in dataset.")
                return
            sample = df[[col]].head(5)
            print(f"  The column '{col}' will be completely removed.")
            print(f"  Current values (first 5 rows):")
            for i, val in enumerate(sample[col].tolist()):
                print(f"    Row {i}: {val}")

        elif action == "drop_rows_if_null":
            if col == "__global__":
                n = int(df.isnull().all(axis=1).sum())
                print(f"  {n} fully-empty rows will be removed from the dataset.")
            elif col in df.columns:
                null_mask = df[col].isna()
                n         = int(null_mask.sum())
                sample    = df[null_mask].head(5)
                print(f"  {n} rows will be deleted because '{col}' is empty.")
                print(f"  Sample of rows that will be deleted:")
                cols_to_show = [c for c in df.columns[:4] if c != col] + [col]
                print("    " + " | ".join(f"{c:>15}" for c in cols_to_show[:4]))
                for _, row in sample.head(5).iterrows():
                    print("    " + " | ".join(
                        f"{str(row.get(c,''))[:15]:>15}" for c in cols_to_show[:4]
                    ))

        elif action == "drop_duplicates":
            n_dups = int(df.duplicated().sum())
            print(f"  {n_dups} duplicate rows will be removed.")
            if n_dups > 0:
                sample    = df[df.duplicated(keep=False)].head(4)
                cols_show = list(df.columns[:4])
                print(f"  Example of duplicated rows:")
                print("    " + " | ".join(f"{c:>15}" for c in cols_show))
                for _, row in sample.iterrows():
                    print("    " + " | ".join(
                        f"{str(row[c])[:15]:>15}" for c in cols_show
                    ))

        elif action in ("fill_median", "fill_mode", "fill_constant"):
            if col not in df.columns:
                print(f"  Column '{col}' not found.")
                return
            null_mask       = df[col].isna()
            n               = int(null_mask.sum())
            non_null_sample = df[~null_mask].head(3)
            null_sample     = df[null_mask].head(3)

            if action == "fill_median":
                fill_val = round(pd.to_numeric(df[col], errors="coerce").median(), 4)
                label    = f"median = {fill_val}"
            elif action == "fill_mode":
                mode_vals = df[col].mode()
                fill_val  = mode_vals.iloc[0] if len(mode_vals) > 0 else "UNKNOWN"
                label     = f"most common value = '{fill_val}'"
            else:
                fill_val = params.get("value", "UNKNOWN")
                label    = f"constant = '{fill_val}'"

            print(f"  '{col}' has {n} empty cells. "
                  f"They will be filled with the {label}.")
            print(f"  Before (rows with empty '{col}'):")
            for _, row in null_sample.iterrows():
                print(f"    {col}: [EMPTY]  ->  will become: {fill_val}")
            print(f"  Existing non-empty values (unchanged):")
            for _, row in non_null_sample.iterrows():
                print(f"    {col}: {row[col]}")

        elif action in ("clip_outliers", "flag_outliers"):
            if col not in df.columns:
                print(f"  Column '{col}' not found.")
                return
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            lo = params.get("lower")
            hi = params.get("upper")
            if lo is None or hi is None:
                q1  = numeric_col.quantile(0.25)
                q3  = numeric_col.quantile(0.75)
                iqr = q3 - q1
                lo  = float(q1 - 1.5 * iqr)
                hi  = float(q3 + 1.5 * iqr)
            outlier_mask = (numeric_col < lo) | (numeric_col > hi)
            n_outliers   = int(outlier_mask.sum())
            sample       = df[outlier_mask].head(5)

            print(f"  '{col}' has {n_outliers} values outside the normal range "
                  f"[{round(lo,2)} to {round(hi,2)}].")
            if action == "clip_outliers":
                print(f"  These values will be pulled back to the boundary "
                      f"(not deleted).")
            else:
                print(f"  A new column '{col}_is_outlier' will be added "
                      f"(1=outlier, 0=normal).")
            print(f"  Sample outlier values:")
            for _, row in sample.iterrows():
                orig = row[col]
                if action == "clip_outliers":
                    new_val = (max(lo, min(hi, float(orig)))
                               if pd.notna(orig) else orig)
                    print(f"    {col}: {orig}  ->  {round(new_val, 4)}")
                else:
                    print(f"    {col}: {orig}  ->  flagged as outlier")

        elif action == "lowercase":
            if col not in df.columns:
                print(f"  Column '{col}' not found.")
                return
            sample = df[col].dropna().head(5)
            print(f"  Text values in '{col}' will be lowercased and stripped "
                  f"of extra spaces.")
            print(f"  Before  ->  After:")
            for val in sample:
                cleaned = str(val).strip().lower()
                marker  = "  [CHANGED]" if cleaned != str(val) else "  [no change]"
                print(f"    '{val}'  ->  '{cleaned}'{marker}")

        elif action == "cast_to_numeric":
            if col not in df.columns:
                print(f"  Column '{col}' not found.")
                return
            sample = df[col].head(5)
            print(f"  Values in '{col}' will be converted from text to numbers.")
            print(f"  Before  ->  After:")
            for val in sample:
                converted = pd.to_numeric(val, errors="coerce")
                print(f"    '{val}'  ->  {converted}")

        elif action == "replace_sentinel":
            if col == "__global__":
                print(f"  Placeholder text (N/A, None, Unknown, '-', etc.) will be")
                print(f"  replaced with truly empty values across all text columns.")
            elif col in df.columns:
                sentinels = {
                    "n/a", "na", "none", "null", "unknown",
                    "missing", "-", "--", "nd", "tbd", "",
                }
                str_col = df[col].astype(str).str.strip().str.lower()
                mask    = str_col.isin(sentinels)
                n       = int(mask.sum())
                sample  = df[mask].head(5)
                print(f"  '{col}' contains {n} placeholder values that disguise "
                      f"missing data.")
                print(f"  They will be replaced with true empty (NaN) values.")
                for _, row in sample.head(5).iterrows():
                    print(f"    '{row[col]}'  ->  [EMPTY]")

        elif action == "fix_datetime":
            if col not in df.columns and col != "__global__":
                print(f"  Column '{col}' not found.")
                return
            target_cols = [col] if col != "__global__" else []
            if col == "__global__":
                target_cols = [
                    c for c in df.select_dtypes(include="object").columns
                    if df[c].dropna().astype(str).str.match(
                        r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}"
                    ).mean() >= 0.7
                ]
            for tc in target_cols[:2]:
                sample = df[tc].dropna().head(3)
                print(f"  Column '{tc}': text dates will be parsed to proper "
                      f"datetime format.")
                for val in sample:
                    parsed = pd.to_datetime(val, errors="coerce")
                    print(f"    '{val}'  ->  {parsed}")

        elif action == "normalize_unicode":
            if col not in df.columns:
                print(f"  Column '{col}' not found.")
                return
            sample  = df[col].dropna().head(5)
            print(f"  Special/accented characters in '{col}' will be standardized.")
            changed = 0
            for val in sample:
                normalized = unicodedata.normalize("NFKC", str(val))
                if normalized != str(val):
                    print(f"    '{val}'  ->  '{normalized}'  [CHANGED]")
                    changed += 1
            if changed == 0:
                print(f"  (No visible changes in the first 5 rows, "
                      f"but hidden characters may be fixed.)")

        else:
            print(f"  No preview available for action '{action}'.")

    except Exception as e:
        print(f"  Preview failed: {e}")

    print("  --- END PREVIEW ---\n")


# =====================================================================
#  LAYER 1 -- DETERMINISTIC RULE GENERATOR
# =====================================================================

def generate_deterministic_rules(profile: dict) -> list:
    columns = profile.get("columns", {})
    dup     = profile.get("duplicate_analysis", {})
    rules   = []

    # ── Global: drop duplicates ────────────────────────────────────────────
    if dup.get("business_duplicate_count", 0) > 0:
        rules.append({
            "rule_id":    "det_drop_duplicates",
            "column":     "__global__",
            "action":     "drop_duplicates",
            "params":     {},
            "severity":   "HIGH",
            "reason":     (
                f"{dup['business_duplicate_count']} duplicate rows "
                f"({dup.get('business_duplicate_rate',0)*100:.1f}%) "
                f"corrupt KPI counts"
            ),
            "confidence": 1.0,
            "order":      1,
            "source":     "deterministic",
        })

    # ── Column-level patterns ──────────────────────────────────────────────

    # Phone/barcode identifier pattern - never cast to numeric
    _PHONE_ID_PATTERN = re.compile(
        r"(phone|mobile|telephone|whatsapp|tel$|fax|contact|barcode|bar_code)",
        re.IGNORECASE,
    )

    # GPS/coordinate columns - never treat as datetime
    _GPS_COL_PATTERN = re.compile(
        r"(^gps$|coord|coordinates|^location$|^loc$|"
        r"latitude|longitude|^lat$|^lon$)",
        re.IGNORECASE,
    )

    # Semantic null columns - null carries business meaning, never fill
    _SEMANTIC_NULL_COLS = re.compile(
        r"(^validated_by$|^payment_date$|^paid_at$|^validated_at$|"
        r"^approved_at$|^rejected_at$|^confirmed_at$|^closed_at$|"
        r"^deleted_at$)",
        re.IGNORECASE,
    )

    # Semantic date columns - parse as datetime but null is intentional
    _SEMANTIC_DATE_COLS = re.compile(
        r"(^payment_date$|^paid_at$|^validated_at$|^approved_at$|"
        r"^rejected_at$|^confirmed_at$|^closed_at$|^deleted_at$)",
        re.IGNORECASE,
    )

    # Balance columns - fill with 0 not median
    _BALANCE_COLS = re.compile(
        r"(^balance$|^solde$|^points$|^score$|^credits$)",
        re.IGNORECASE,
    )

    # Attribution columns - fill with UNKNOWN not mode
    # client_name included: filling with mode('Cafe des Amis') inflates
    # one client in every dashboard
    _ATTRIBUTION_PAT = re.compile(
        r"(reseller_name|vendor_name|agent_name|salesperson|"
        r"supplier_name|partner_name|distributor|broker_name|"
        r"assigned_to|handled_by|manager_name|"
        r"client_name|customer_name|contact_name)",
        re.IGNORECASE,
    )

    # Foreign key columns - never impute with median/mode
    # FK null = referential integrity gap, not a missing measurement
    _FK_NAME_PATTERN = re.compile(
        r"(_id$|^fk_|_key$|^user_id$|^whatsapp_user_id$)",
        re.IGNORECASE,
    )

    drop_order = 2

    for col, info in columns.items():
        null_rate = info.get("null_rate", 0)
        col_type  = info.get("detected_type", "unknown")
        issues    = {i["type"] for i in info.get("issues", [])}
        safe_id   = re.sub(r"[^a-z0-9_]", "_", col.lower())[:30]

        # ── Drop high-null columns ─────────────────────────────────────────
        if null_rate > 0.60:
            rules.append({
                "rule_id":    f"det_drop_{safe_id}",
                "column":     col,
                "action":     "drop_column",
                "params":     {},
                "severity":   "HIGH",
                "reason":     (
                    f"{null_rate*100:.1f}% nulls -- exceeds 60% drop threshold"
                ),
                "confidence": 1.0,
                "order":      drop_order,
                "source":     "deterministic",
            })
            drop_order += 1
            continue

        # ── Replace sentinel strings ───────────────────────────────────────
        if "sentinel_value_detected" in issues:
            sentinels = [s["value"] for s in info.get("sentinel_values", [])]
            rules.append({
                "rule_id":    f"det_sentinel_{safe_id[:25]}",
                "column":     col,
                "action":     "replace_sentinel",
                "params":     {},
                "severity":   "MEDIUM",
                "reason":     (
                    f"Sentinel strings {sentinels[:3]} disguise nulls as strings"
                ),
                "confidence": 1.0,
                "order":      3,
                "source":     "deterministic",
            })

        # ── Replace zero-as-null ───────────────────────────────────────────
        if "zero_as_null" in issues and col_type in ("numeric", "ordinal_numeric"):
            rules.append({
                "rule_id":    f"det_zero_null_{safe_id[:22]}",
                "column":     col,
                "action":     "replace_sentinel",
                "params":     {"value": 0},
                "severity":   "HIGH",
                "reason":     (
                    f"Zero values in '{col}' are physically impossible -- "
                    f"replace with NaN"
                ),
                "confidence": 1.0,
                "order":      3,
                "source":     "deterministic",
            })

        # ── Cast numeric-stored-as-string ──────────────────────────────────
        if "numeric_stored_as_string" in issues and not _PHONE_ID_PATTERN.search(col):
            rules.append({
                "rule_id":    f"det_cast_{safe_id[:25]}",
                "column":     col,
                "action":     "cast_to_numeric",
                "params":     {},
                "severity":   "HIGH",
                "reason":     (
                    f"'{col}' stores numbers as text -- "
                    f"breaks SUM/AVG calculations"
                ),
                "confidence": 1.0,
                "order":      3,
                "source":     "deterministic",
            })

        # ── Null imputation ────────────────────────────────────────────────
        if null_rate > 0.05:

            # Guard 1: semantic null - null = business state (pending/unpaid)
            if _SEMANTIC_NULL_COLS.match(col):
                pass  # never fill - null is meaningful

            # Guard 2: foreign key - null = referential gap, not missing data
            elif col_type == "foreign_key" or _FK_NAME_PATTERN.search(col):
                rules.append({
                    "rule_id":    f"det_flag_fk_{safe_id[:20]}",
                    "column":     col,
                    "action":     "fill_constant",
                    "params":     {"value": -1},
                    "severity":   "MEDIUM" if null_rate > 0.20 else "LOW",
                    "reason":     (
                        f"{null_rate*100:.1f}% null in FK column '{col}' -- "
                        f"fill with -1 sentinel to mark referential gap. "
                        f"Median of an ID is meaningless."
                    ),
                    "confidence": 1.0,
                    "order":      5,
                    "source":     "deterministic",
                })

            # Guard 3: numeric columns
            elif col_type in ("numeric", "ordinal_numeric"):
                if _BALANCE_COLS.match(col):
                    rules.append({
                        "rule_id":    f"det_fill_zero_{safe_id[:20]}",
                        "column":     col,
                        "action":     "fill_constant",
                        "params":     {"value": 0},
                        "severity":   "MEDIUM" if null_rate > 0.20 else "LOW",
                        "reason":     (
                            f"{null_rate*100:.1f}% empty values in balance column -- "
                            f"fill with 0 (technicians start with 0 points)"
                        ),
                        "confidence": 0.95,
                        "order":      5,
                        "source":     "deterministic",
                    })
                else:
                    rules.append({
                        "rule_id":    f"det_fill_median_{safe_id[:20]}",
                        "column":     col,
                        "action":     "fill_median",
                        "params":     {},
                        "severity":   "MEDIUM" if null_rate > 0.20 else "LOW",
                        "reason":     (
                            f"{null_rate*100:.1f}% empty values in numeric column -- "
                            f"fill with median"
                        ),
                        "confidence": 0.90,
                        "order":      5,
                        "source":     "deterministic",
                    })

            # Guard 4: categorical/text columns
            elif col_type in ("categorical", "boolean", "low_cardinality_text"):
                if _ATTRIBUTION_PAT.search(col):
                    rules.append({
                        "rule_id":    f"det_fill_unk_{safe_id[:20]}",
                        "column":     col,
                        "action":     "fill_constant",
                        "params":     {"value": "UNKNOWN"},
                        "severity":   "MEDIUM" if null_rate > 0.20 else "LOW",
                        "reason":     (
                            f"{null_rate*100:.1f}% empty values in attribution "
                            f"column -- fill with UNKNOWN to avoid inflating any "
                            f"single entity in dashboards"
                        ),
                        "confidence": 0.95,
                        "order":      5,
                        "source":     "deterministic",
                    })
                else:
                    rules.append({
                        "rule_id":    f"det_fill_mode_{safe_id[:20]}",
                        "column":     col,
                        "action":     "fill_mode",
                        "params":     {},
                        "severity":   "MEDIUM" if null_rate > 0.20 else "LOW",
                        "reason":     (
                            f"{null_rate*100:.1f}% empty values in text column -- "
                            f"fill with most common value"
                        ),
                        "confidence": 0.90,
                        "order":      5,
                        "source":     "deterministic",
                    })

        # ── Casing standardization ─────────────────────────────────────────
        if "inconsistent_casing" in issues and col_type == "categorical":
            rules.append({
                "rule_id":    f"det_lower_{safe_id[:25]}",
                "column":     col,
                "action":     "lowercase",
                "params":     {},
                "severity":   "LOW",
                "reason":     (
                    f"Case variants in '{col}' create duplicate categories in reports"
                ),
                "confidence": 1.0,
                "order":      7,
                "source":     "deterministic",
            })

        # ── Unicode normalization ──────────────────────────────────────────
        if "non_ascii_characters" in issues:
            rules.append({
                "rule_id":    f"det_unicode_{safe_id[:23]}",
                "column":     col,
                "action":     "normalize_unicode",
                "params":     {},
                "severity":   "LOW",
                "reason":     (
                    f"Special characters in '{col}' cause mismatches in joins "
                    f"and group-bys"
                ),
                "confidence": 1.0,
                "order":      7,
                "source":     "deterministic",
            })

    # ── Fix datetime for date-string columns ───────────────────────────────
    # Semantic date columns (payment_date etc.) are parsed but with LOW severity
    # and a note that nulls are intentional (null = not yet paid/validated).
    for col, info in columns.items():
        if info.get("detected_type") != "datetime_string":
            continue
        if info.get("null_rate", 1) > 0.60:
            continue
        if _GPS_COL_PATTERN.search(col):
            continue

        safe_id = re.sub(r"[^a-z0-9_]", "_", col.lower())[:22]

        if _SEMANTIC_DATE_COLS.match(col):
            # Parse as datetime but flag that nulls are intentional
            rules.append({
                "rule_id":    f"det_datetime_{safe_id}",
                "column":     col,
                "action":     "fix_datetime",
                "params":     {},
                "severity":   "LOW",
                "reason":     (
                    f"'{col}' contains dates stored as text -- parse as datetime. "
                    f"NOTE: null values here mean not yet paid or validated -- "
                    f"they are intentional, do NOT impute them."
                ),
                "confidence": 1.0,
                "order":      8,
                "source":     "deterministic",
            })
        else:
            rules.append({
                "rule_id":    f"det_datetime_{safe_id}",
                "column":     col,
                "action":     "fix_datetime",
                "params":     {},
                "severity":   "MEDIUM",
                "reason":     (
                    f"'{col}' contains dates stored as text -- "
                    f"needs to be parsed as real dates"
                ),
                "confidence": 1.0,
                "order":      8,
                "source":     "deterministic",
            })

    # ── Dedup rules by (column, action) ───────────────────────────────────
    seen, unique = set(), []
    for r in sorted(rules, key=lambda x: x["order"]):
        key = (r["column"], r["action"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# =====================================================================
#  LAYER 2 -- PROFILE COMPRESSOR
# =====================================================================

_COLS_FULL_THRESHOLD = 15
_COLS_SLIM_THRESHOLD = 30


def compress_profile_for_llm(profile: dict, skip_columns: set = None) -> dict:
    skip_columns = skip_columns or set()
    n_cols       = len(profile.get("columns", {}))

    if n_cols <= _COLS_FULL_THRESHOLD:
        cols_payload = _compress_columns_full(profile.get("columns", {}))
    elif n_cols <= _COLS_SLIM_THRESHOLD:
        cols_payload = _compress_columns_slim(profile.get("columns", {}))
    else:
        cols_payload = None

    all_flags      = profile.get("summary_flags", [])
    filtered_flags = [
        f for f in all_flags if f.get("column", "") not in skip_columns
    ]
    top_flags = _top_n_flags(filtered_flags, n=20)

    result = {
        "table":           profile.get("table_name", "unknown"),
        "n_rows":          profile.get("n_rows", 0),
        "n_columns":       n_cols,
        "dataset_quality": profile.get("dataset_quality", {}),
        "duplicate_rate":  profile.get("duplicate_rate", 0),
        "schema_drift": {
            "has_drift": profile.get("schema_drift", {}).get("has_drift", False),
            "new_columns": profile.get("schema_drift", {}).get("new_columns", []),
            "removed_columns": profile.get("schema_drift", {}).get(
                "removed_columns", []
            ),
        },
        "summary_flags":   top_flags,
        "cross_column": {
            "correlated_null_groups": profile.get("cross_column", {}).get(
                "correlated_null_groups", []
            ),
            "derived_columns": profile.get("cross_column", {}).get(
                "possible_derived_columns", []
            )[:5],
            "additive_formulas": [
                {"formula": r["formula"], "note": r["note"]}
                for r in profile.get("cross_column", {}).get(
                    "additive_relationships", []
                )
            ],
        },
        "logic_conflicts": profile.get("logic_consistency_issues", []),
    }
    if cols_payload is not None:
        result["columns"] = {
            col: info
            for col, info in cols_payload.items()
            if col not in skip_columns
        }
    return result


def _top_n_flags(flags: list, n: int = 20) -> list:
    sev_order    = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
    sorted_flags = sorted(
        flags,
        key=lambda f: (
            sev_order.get(f.get("severity", "INFO"), 9),
            f.get("quality_score") or 100,
        ),
    )[:n]
    result = []
    for f in sorted_flags:
        rec = f.get("recommended_action") or {}
        entry = {
            "column":           f.get("column"),
            "detected_type":    f.get("detected_type", "unknown"),
            "issue_type":       f.get("type"),
            "severity":         f.get("severity"),
            "issue_detail":     str(f.get("detail", ""))[:150],
            "null_rate":        f.get("null_rate"),
            "quality_score":    f.get("quality_score"),
            "suggested_action": (
                rec.get("action", "") if isinstance(rec, dict) else str(rec)
            ),
        }
        if f.get("iqr_outlier_rate") is not None:
            entry.update({
                "iqr_rate":        f.get("iqr_outlier_rate"),
                "consensus_rate":  f.get("consensus_outlier_rate"),
                "iqr_range_ratio": f.get("iqr_to_range_ratio"),
                "outlier_sample":  f.get("outlier_sample", [])[:3],
            })
        result.append(entry)
    return result


def _compress_columns_full(columns: dict) -> dict:
    out = {}
    for col, info in columns.items():
        entry = {
            "detected_type": info.get("detected_type"),
            "null_rate":     info.get("null_rate", 0),
            "quality_score": info.get("quality_score"),
            "is_derived":    info.get("is_derived_column", False),
            "issues": [
                {"type": i["type"], "severity": i["severity"]}
                for i in info.get("issues", [])
            ],
        }
        if stats := info.get("stats"):
            entry["stats"] = {
                k: stats[k]
                for k in (
                    "mean", "median", "std", "min", "max",
                    "skewness", "zero_rate",
                )
                if k in stats
            }
        if oa := info.get("outlier_analysis"):
            entry["outlier"] = {
                "iqr_rate":    oa.get("iqr_outlier_rate", 0),
                "consensus":   oa.get("consensus_outlier_rate", 0),
                "lower_fence": oa.get("iqr_lower_fence"),
                "upper_fence": oa.get("iqr_upper_fence"),
                "narrow_iqr":  oa.get("narrow_iqr", False),
            }
        if (info.get("detected_type") == "categorical"
                and info.get("top_values")):
            entry["top_values"] = dict(list(info["top_values"].items())[:5])
        if sv := info.get("sentinel_values"):
            entry["sentinel_values"] = [s["value"] for s in sv[:3]]
        out[col] = entry
    return out


def _compress_columns_slim(columns: dict) -> dict:
    out = {}
    for col, info in columns.items():
        out[col] = {
            "t":  info.get("detected_type", "?"),
            "nr": info.get("null_rate", 0),
            "qs": info.get("quality_score", 100),
            "i": [
                {"t": i["type"], "s": i["severity"]}
                for i in info.get("issues", [])
            ],
        }
        if oa := info.get("outlier_analysis"):
            if oa.get("iqr_outlier_rate", 0) > 0.02:
                out[col]["fence"] = [
                    round(oa.get("iqr_lower_fence", 0), 4),
                    round(oa.get("iqr_upper_fence", 0), 4),
                    oa.get("narrow_iqr", False),
                ]
    return out


# =====================================================================
#  PROMPT BUILDER
# =====================================================================

SYSTEM_PROMPT = (
    "You are a JSON-output assistant. Respond with valid JSON only. No prose."
)

_RULES = """\
ISSUE -> ACTION (use detected_type to choose):
  high_null_rate (null_rate>0.60)                                -> drop_column        order=2
  moderate_null_rate + detected_type IN [numeric,ordinal_numeric] -> fill_median       order=5
  moderate_null_rate + detected_type IN [categorical,boolean]    -> fill_mode          order=5
  low_null_rate (0.05-0.20) + numeric                           -> fill_median         order=5
  low_null_rate (0.05-0.20) + categorical                       -> fill_mode           order=5
  outliers + narrow_iqr=false -> clip_outliers params.lower and params.upper           order=6
  outliers + narrow_iqr=true  -> flag_outliers                   order=6
  high_skew                   -> flag_outliers                   order=6
  zero_as_null                -> replace_sentinel params={"value":0} order=3
  business_duplicate_rows     -> drop_duplicates column=__global__ order=1
  constant_column             -> drop_column                     order=2

NEVER:
  - fill_median/clip_outliers/flag_outliers on categorical, text, id, boolean
  - Impute foreign_key columns
  - Generate duplicate (column, action) pairs
  - Invent column names not in DATA QUALITY FLAGS
  - Output text outside the JSON array

REQUIRED FIELDS: rule_id, column, action, params, severity, reason, confidence, order
ALLOWED ACTIONS: drop_column drop_rows_if_null drop_duplicates fill_median fill_mode
  fill_constant clip_outliers flag_outliers lowercase cast_to_numeric
  replace_sentinel fix_datetime normalize_unicode"""


def _build_dynamic_few_shot(flags: list, fences: dict) -> str:
    if not flags:
        return ""

    example_in  = []
    example_out = []
    used_order  = set()

    for flag in flags[:3]:
        col       = flag.get("column", "col")
        dtype     = flag.get("detected_type", "numeric")
        issue     = flag.get("issue_type", "outliers")
        severity  = flag.get("severity", "MEDIUM")
        null_rate = flag.get("null_rate", 0.0)
        safe_col  = re.sub(r"[^a-z0-9_]", "_", col.lower())[:20]

        flag_entry = {
            "column":        col,
            "detected_type": dtype,
            "issue_type":    issue,
            "severity":      severity,
            "null_rate":     null_rate,
        }
        if flag.get("iqr_rate") is not None:
            flag_entry["iqr_rate"]        = flag.get("iqr_rate")
            flag_entry["consensus_rate"]  = flag.get("consensus_rate")
            flag_entry["iqr_range_ratio"] = flag.get("iqr_range_ratio")
        example_in.append(flag_entry)

        if issue == "high_null_rate":
            action = "drop_column"; params = {}; order = 2
        elif issue in ("moderate_null_rate", "low_null_rate"):
            if dtype in ("numeric", "ordinal_numeric"):
                action = "fill_median"; params = {}; order = 5
            else:
                action = "fill_mode"; params = {}; order = 5
        elif issue == "outliers":
            fence  = fences.get(col, {})
            lo, hi = fence.get("lower"), fence.get("upper")
            narrow = fence.get("narrow", True)
            if narrow or lo is None:
                action = "flag_outliers"; params = {}; order = 6
            else:
                action = "clip_outliers"
                params = {"lower": lo, "upper": hi}
                order  = 6
        elif issue == "zero_as_null":
            action = "replace_sentinel"; params = {"value": 0}; order = 3
        elif issue == "constant_column":
            action = "drop_column"; params = {}; order = 2
        elif issue == "inconsistent_casing":
            action = "lowercase"; params = {}; order = 7
        else:
            action = "flag_outliers"; params = {}; order = 6

        while order in used_order:
            order += 1
        used_order.add(order)

        example_out.append({
            "rule_id":    f"{action[:8]}_{safe_col}",
            "column":     col,
            "action":     action,
            "params":     params,
            "severity":   severity,
            "reason":     f"{issue} on column '{col}'",
            "confidence": 0.90,
            "order":      order,
        })

    if not example_in:
        return ""

    return (
        f"EXAMPLE INPUT (from a DIFFERENT dataset -- "
        f"do NOT copy these column names):\n"
        f"{json.dumps(example_in, indent=2)}\n\n"
        f"EXAMPLE OUTPUT (exact schema required -- start with [ end with ]):\n"
        f"{json.dumps(example_out, indent=2)}"
    )


def build_user_prompt(compressed: dict) -> str:
    flags      = compressed.get("summary_flags", [])
    table_name = compressed["table"]
    dq         = compressed.get("dataset_quality", {})
    high_count = dq.get("high_issues", 0)

    if not flags:
        return ""

    fences = {}
    for col, info in compressed.get("columns", {}).items():
        oa = info.get("outlier") if isinstance(info, dict) else None
        if oa and isinstance(oa, dict):
            lo = oa.get("lower_fence")
            hi = oa.get("upper_fence")
            if lo is not None and hi is not None:
                fences[col] = {
                    "lower":  round(lo, 4),
                    "upper":  round(hi, 4),
                    "narrow": oa.get("narrow_iqr", False),
                }

    enriched_flags = []
    for flag in flags:
        f   = dict(flag)
        col = f.get("column", "")
        if col in fences and f.get("issue_type") == "outliers":
            f["iqr_lower_fence"] = fences[col]["lower"]
            f["iqr_upper_fence"] = fences[col]["upper"]
            f["narrow_iqr"]      = fences[col]["narrow"]
            f["clip_action"]     = (
                "flag_outliers" if fences[col]["narrow"] else "clip_outliers"
            )
        enriched_flags.append(f)

    severity_note = ""
    if high_count > 20:
        severity_note = (
            f"NOTE: {high_count} HIGH issues total. "
            f"Showing top {len(flags)} most critical.\n\n"
        )

    focused = {
        "table":         table_name,
        "n_rows":        compressed["n_rows"],
        "summary_flags": enriched_flags,
    }
    if fences:
        focused["outlier_fences"] = fences

    few_shot = _build_dynamic_few_shot(enriched_flags, fences)
    payload  = json.dumps(focused, indent=2)

    return (
        f"TASK: Output a JSON array of data cleaning rules "
        f"for table '{table_name}'.\n"
        f"IMPORTANT: Use ONLY column names that appear in summary_flags below.\n"
        f"Do NOT copy column names from the example.\n\n"
        f"{few_shot}\n\n"
        f"DECISION RULES:\n{_RULES}\n\n"
        f"{severity_note}"
        f"DATA QUALITY FLAGS for '{table_name}':\n{payload}\n\n"
        f"Output ONLY the JSON array for '{table_name}'. "
        f"Start with [ and end with ]:\n"
    )


def build_retry_prompt(compressed: dict, error: str) -> str:
    base       = build_user_prompt(compressed)
    table_name = compressed["table"]
    return (
        f"PREVIOUS ATTEMPT FAILED: {error[:100]}\n"
        f"CRITICAL: Use ONLY column names from the DATA QUALITY FLAGS "
        f"for '{table_name}'.\n"
        f"Do NOT invent column names. "
        f"Output ONLY a JSON array starting with [ and ending with ].\n"
        f"Each element must have: "
        f"rule_id, column, action, params, severity, reason, confidence, order\n\n"
        + base
    )


# =====================================================================
#  LLM CONNECTORS
# =====================================================================

def _call_ollama(system: str, user: str, model: str = None) -> str:
    try:
        import ollama
    except ImportError:
        raise ImportError("Run: pip install ollama")
    target = model or OLLAMA_MODEL
    client = ollama.Client(host=OLLAMA_HOST)
    try:
        available   = client.list()
        model_names = [m.model for m in available.models]
        if not model_names:
            raise RuntimeError(
                f"No models installed. Run: ollama pull {target}"
            )
        if target not in model_names:
            prefix = target.split(":")[0]
            hit    = next(
                (m for m in model_names if m.startswith(prefix)), None
            )
            if hit:
                print(f"  '{target}' not found -> using '{hit}'")
                target = hit
            else:
                print(f"  '{target}' not found. Available: {model_names}")
                target = model_names[0]
    except Exception as e:
        if any(w in str(e).lower() for w in ("connection", "refused", "connect")):
            raise ConnectionError(
                f"Cannot connect to Ollama at {OLLAMA_HOST}\n"
                "Start with: ollama serve\n"
                "Pull model : ollama pull llama3.1:8b"
            ) from e
        raise
    resp = client.chat(
        model    = target,
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options  = {"temperature": 0, "num_predict": 4096, "top_p": 0.1},
    )
    return resp.message.content


def _call_anthropic(system: str, user: str, model: str = None) -> str:
    try:
        import anthropic
    except ImportError:
        raise ImportError("Run: pip install anthropic")
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in .env")
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg    = client.messages.create(
        model      = "claude-sonnet-4-5",
        max_tokens = 4096,
        system     = system,
        messages   = [{"role": "user", "content": user}],
        temperature= 0,
    )
    return msg.content[0].text


def _call_openai(system: str, user: str, model: str = None) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY not set in .env")
    resp = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model           = "gpt-4o",
        messages        = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        response_format = {"type": "json_object"},
        temperature     = 0,
    )
    return resp.choices[0].message.content


def _call_gemini(system: str, user: str, model: str = None) -> str:
    try:
        from google import genai
    except ImportError:
        raise ImportError("Run: pip install google-genai")
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")
    client   = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model    = "gemini-2.0-flash",
        contents = f"{system}\n\n---\n\n{user}",
    )
    return response.text


LLM_CALLERS = {
    "ollama":    _call_ollama,
    "anthropic": _call_anthropic,
    "openai":    _call_openai,
    "gemini":    _call_gemini,
}


def call_llm(system: str, user: str,
             provider: str = None, model: str = None) -> str:
    p      = (provider or LLM_PROVIDER).lower()
    caller = LLM_CALLERS.get(p)
    if not caller:
        raise ValueError(
            f"Unknown provider '{p}'. Valid: ollama|anthropic|openai|gemini"
        )
    labels = {
        "ollama":    model or OLLAMA_MODEL,
        "anthropic": "claude-sonnet-4-5",
        "openai":    "gpt-4o",
        "gemini":    "gemini-2.0-flash",
    }
    print(f"  Calling {p.upper()} ({labels.get(p,'?')})...")
    return caller(system, user, model)


# =====================================================================
#  RESPONSE PARSER
# =====================================================================

VALID_ACTIONS = {
    "drop_column", "drop_rows_if_null", "drop_duplicates",
    "fill_median", "fill_mode", "fill_constant",
    "clip_outliers", "flag_outliers",
    "lowercase", "cast_to_numeric", "replace_sentinel",
    "fix_datetime", "normalize_unicode",
}
REQUIRED_FIELDS = {
    "rule_id", "column", "action", "severity", "reason", "confidence"
}


def parse_llm_response(raw: str) -> list:
    cleaned = raw.strip().lstrip("\ufeff")
    cleaned = re.sub(r"```(?:json|JSON)?", "", cleaned).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()

    def _fix(s: str) -> str:
        s = re.sub(r",\s*]", "]", s)
        s = re.sub(r",\s*}", "}", s)
        return s

    if cleaned.startswith("["):
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                return json.loads(_fix(cleaned))
            except json.JSONDecodeError:
                pass

    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return json.loads(_fix(cleaned[start:end + 1]))
        except json.JSONDecodeError:
            pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(_fix(cleaned[start:end + 1]))
            for key in ("rules", "cleaning_rules", "actions", "data", "result"):
                if key in obj and isinstance(obj[key], list):
                    return obj[key]
            if "action" in obj and "rule_id" in obj:
                return [obj]
        except json.JSONDecodeError:
            pass

    objects = re.findall(r'\{[^{}]*"rule_id"[^{}]*\}', cleaned, re.DOTALL)
    if objects:
        rules = []
        for s in objects:
            try:
                rules.append(json.loads(_fix(s)))
            except json.JSONDecodeError:
                continue
        if rules:
            print(f"  Regex fallback extracted {len(rules)} rules from malformed JSON")
            return rules

    raise ValueError(
        f"Cannot parse JSON array from LLM response.\n"
        f"First 400 chars:\n{raw[:400]}"
    )


def validate_rule(rule: dict, known_columns: set,
                  profile: dict = None) -> tuple:
    if not isinstance(rule, dict):
        return False, f"Not a dict ({type(rule).__name__})"
    missing = REQUIRED_FIELDS - set(rule.keys())
    if missing:
        return False, f"Missing fields: {missing}"
    if rule["action"] not in VALID_ACTIONS:
        return False, f"Invalid action: '{rule['action']}'"
    try:
        conf = float(rule["confidence"])
        if not 0.0 <= conf <= 1.0:
            return False, f"confidence={conf} not in [0,1]"
    except (TypeError, ValueError):
        return False, f"confidence not a number: {rule.get('confidence')}"

    col = rule.get("column", "")
    if col != "__global__" and col not in known_columns:
        return False, f"Column '{col}' not in profile"
    if col == "__global__" and rule.get("action") not in (
        "drop_duplicates", "replace_sentinel", "drop_rows_if_null"
    ):
        return False, (
            f"Action '{rule.get('action')}' cannot target __global__"
        )
    if not str(rule.get("reason", "")).strip():
        rule["reason"] = f"{rule['action']} on {col}"

    # GPS columns never get fix_datetime
    _GPS_COLS = re.compile(
        r"(^gps$|^coord$|^coordinates$|^location$|^loc$|"
        r"^latitude$|^longitude$|^lat$|^lon$)",
        re.IGNORECASE,
    )
    if rule.get("action") == "fix_datetime" and _GPS_COLS.match(col):
        return False, (
            f"Rejected fix_datetime on '{col}': GPS/coordinate columns are "
            f"never datetime."
        )

    # Phone/barcode columns never get cast_to_numeric
    _PHONE_BLOCK = re.compile(
        r"(^phone$|^phone_number$|^client_phone$|^whatsapp_number$|"
        r"^whatsapp_num$|_phone$|^mobile$|^telephone$|^tel$|^fax$|"
        r"^bar_code$|^bar_code_number$|^bar_code_indoor$|^bar_code_outdoor$|"
        r"^barcode$|^qr_code$)",
        re.IGNORECASE,
    )
    if rule.get("action") == "cast_to_numeric" and _PHONE_BLOCK.match(col):
        return False, (
            f"Rejected cast_to_numeric on '{col}': phone/barcode columns "
            f"are identifiers."
        )

    # ID and FK columns are never dropped
    if rule.get("action") == "drop_column" and col != "__global__" and profile:
        col_type = (profile.get("detected_schema") or {}).get(col, "")
        if col_type in ("id", "foreign_key"):
            return False, (
                f"Rejected drop_column on '{col}': column is '{col_type}' -- "
                f"identifier columns are never dropped."
            )

    # Semantic null columns are never imputed
    _SEMANTIC_NULL_VALIDATE = re.compile(
        r"^(validated_by|payment_date|paid_at|validated_at|"
        r"approved_at|rejected_at|confirmed_at|closed_at|deleted_at)$",
        re.IGNORECASE,
    )
    if rule.get("action") in ("fill_median", "fill_mode", "fill_constant"):
        if _SEMANTIC_NULL_VALIDATE.match(col):
            return False, (
                f"Rejected {rule.get('action')} on '{col}': null values are "
                f"semantically meaningful here -- null means pending or not yet "
                f"assigned. Add a boolean flag instead."
            )

    # ID and FK columns are never imputed
    if rule.get("action") in ("fill_median", "fill_mode", "fill_constant") and profile:
        col_type = (profile.get("detected_schema") or {}).get(col, "")
        if col_type in ("id", "foreign_key"):
            return False, (
                f"Rejected {rule.get('action')} on '{col}': column is "
                f"'{col_type}' -- identifier columns cannot be imputed."
            )

    # Financial balance columns never get clip_outliers
    _BALANCE_CLIP_BLOCK = re.compile(
        r"(^balance$|^solde_after$|^solde_before$|^amount$|^amount_changed$|"
        r"^solde$|^points$|^credits$|^revenue$|^total_amount$)",
        re.IGNORECASE,
    )
    if rule.get("action") == "clip_outliers" and _BALANCE_CLIP_BLOCK.match(col):
        return False, (
            f"Rejected clip_outliers on '{col}': financial balance columns "
            f"have no upper cap -- use flag_outliers instead."
        )

    sev = str(rule.get("severity", "")).upper().strip()
    rule["severity"] = sev if sev in ("HIGH", "MEDIUM", "LOW") else "MEDIUM"
    rule.setdefault("order",  99)
    rule.setdefault("params", {})

    # Hallucination guards
    if profile and col != "__global__" and col in profile.get("columns", {}):
        col_info  = profile["columns"][col]
        null_rate = col_info.get("null_rate", 0)
        reason    = rule.get("reason", "").lower()
        action    = rule.get("action", "")

        if (action == "drop_column"
                and "null" in reason
                and null_rate < 0.05):
            return False, (
                f"Hallucination rejected: LLM claims '{col}' has high nulls "
                f"but actual null_rate = {null_rate*100:.1f}%."
            )

        if (action == "drop_column"
                and ("constant" in reason or "variance" in reason)
                and col_info.get("unique_count", 99) > 1):
            return False, (
                f"Hallucination rejected: LLM claims '{col}' is constant "
                f"but it has {col_info['unique_count']} unique values."
            )

        if (action in ("fill_median", "fill_mode", "fill_constant")
                and null_rate == 0):
            return False, (
                f"Hallucination rejected: LLM proposes '{action}' on '{col}' "
                f"but it has 0% nulls -- nothing to fill."
            )

        if action == "drop_column" and profile:
            schema   = profile.get("detected_schema", {})
            num_cols = [
                c for c, t in schema.items()
                if t in ("numeric", "ordinal_numeric") and c != col
            ]
            if len(num_cols) == 0:
                return False, (
                    f"Safety guard: refusing to drop '{col}' because it is "
                    f"the only numeric column."
                )

    return True, ""


# =====================================================================
#  HUMAN REVIEW
# =====================================================================

def human_review(rules: list, df: pd.DataFrame = None) -> list:
    print("\n" + "=" * 67)
    print("  AI CLEANING AGENT -- PROPOSED RULES FOR YOUR REVIEW")
    print("  y = approve    n = reject    s = approve all remaining")
    print("  p = preview what this rule will do on 5 sample rows")
    print("=" * 67)

    approved, auto_rest = [], False
    sorted_rules = sorted(rules, key=lambda r: r.get("order", 99))

    for i, rule in enumerate(sorted_rules, 1):
        if auto_rest:
            rule.update({
                "approved_by": "human_bulk",
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })
            approved.append(rule)
            continue

        action = rule.get("action", "?")
        col    = rule.get("column", "?")
        src    = (
            " [automatic]" if rule.get("source") == "deterministic"
            else " [AI suggested]"
        )

        print(f"\n  Rule {i} of {len(sorted_rules)}  |  "
              f"Severity: {rule.get('severity','MEDIUM')}{src}")
        print(f"  Rule ID : {rule['rule_id']}")
        print(f"  Column  : {col}")
        print(f"  Action  : {action}", end="")
        if rule.get("params"):
            print(f"  (parameters: {rule['params']})", end="")
        print()
        print(f"  Certainty : {float(rule.get('confidence', 0.75)):.0%}")
        print(f"  Run order : {rule.get('order','?')}")
        print(f"  Reason    : {rule.get('reason', 'no reason provided')}")
        print()

        reason_text = rule.get("reason", "").lower()
        if action == "drop_column" and "constant" in reason_text:
            explanation = (
                f"Remove the entire column '{col}' from the dataset. "
                f"This column has only one unique value -- it carries zero "
                f"information and would cause problems in ML models."
            )
        elif action == "drop_column" and "variance" in reason_text:
            explanation = (
                f"Remove the entire column '{col}' from the dataset. "
                f"It has zero variance -- every row has the same value."
            )
        else:
            explanation = plain_english_explanation(
                action, col, rule.get("params") or {}
            )

        print(f"  What this does:")
        words = explanation.split()
        line  = "    "
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

        while True:
            try:
                c = input("\n  Decision [y / n / s / p]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Interrupted.")
                return approved

            if c == "p":
                if df is not None:
                    preview_rule(rule, df)
                else:
                    print("  Preview not available -- no dataset was loaded.")
                continue

            if c in ("y", "n", "s"):
                break
            print("  Please type  y  n  s  or  p  and press Enter.")

        if c in ("y", "s"):
            rule.update({
                "approved_by": "human",
                "approved_at": datetime.now(timezone.utc).isoformat(),
            })
            approved.append(rule)
            if c == "s":
                auto_rest = True
                print("  Approved -- approving all remaining rules automatically.")
            else:
                print("  Approved.")
        else:
            print("  Rejected.")

    print(f"\n  {len(approved)} of {len(sorted_rules)} rules approved.\n")
    return approved


# =====================================================================
#  RULES DATABASE
# =====================================================================

def load_rules_db() -> dict:
    if not RULES_DB_PATH.exists():
        return {}
    try:
        content = RULES_DB_PATH.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        if isinstance(data, list):
            print("  Migrating rules_db to table-keyed format")
            grouped: dict = {}
            for r in data:
                t = r.get("table", "unknown")
                grouped.setdefault(t, []).append(r)
            return grouped
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  Rules DB corrupted ({e}) -- starting fresh")
        return {}


def save_rules_db(new_rules: list, table_name: str) -> None:
    RULES_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db  = load_rules_db()
    now = datetime.now(timezone.utc).isoformat()
    stamped        = [
        {**r, "table": table_name, "created_at": now} for r in new_rules
    ]
    db[table_name] = stamped
    total = sum(len(v) for v in db.values())
    RULES_DB_PATH.write_text(json.dumps(db, indent=2), encoding="utf-8")
    print(
        f"  Saved -> {RULES_DB_PATH}  "
        f"({len(stamped)} rules for '{table_name}', "
        f"{total} total across {len(db)} table(s))"
    )


# =====================================================================
#  MAIN AGENT
# =====================================================================

class DataCleaningAgent:

    def __init__(self, provider: str = None, model: str = None,
                 min_confidence: float = 0.55):
        self.provider       = provider or LLM_PROVIDER
        self.model          = model
        self.min_confidence = min_confidence

    def run(self, profile: dict, df: pd.DataFrame = None) -> list:
        table_name    = profile.get("table_name", "unknown")
        known_columns = set(profile.get("columns", {}).keys())

        print(f"\n  Generating automatic rules for '{table_name}'...")
        det_rules = generate_deterministic_rules(profile)
        det_pairs = {(r["column"], r["action"]) for r in det_rules}
        print(
            f"  {len(det_rules)} automatic rules generated "
            f"(null handling, casing, sentinels)"
        )

        handled_cols = {
            r["column"] for r in det_rules if r["column"] != "__global__"
        }
        compressed = compress_profile_for_llm(profile, skip_columns=handled_cols)
        llm_flags  = compressed.get("summary_flags", [])

        llm_rules = []
        if llm_flags:
            print(
                f"\n  {len(llm_flags)} complex issues (outliers, skew) "
                f"-> sending to AI model"
            )
            user_prompt = build_user_prompt(compressed)
            if user_prompt:
                raw = call_llm(
                    SYSTEM_PROMPT, user_prompt,
                    provider=self.provider, model=self.model,
                )
                print("  AI model response received")
                try:
                    llm_rules = parse_llm_response(raw)
                except ValueError as e1:
                    print(f"  Parse attempt 1 failed: {e1}")
                    print("  Retrying with stricter prompt...")
                    raw2 = call_llm(
                        SYSTEM_PROMPT,
                        build_retry_prompt(compressed, str(e1)),
                        provider=self.provider,
                        model=self.model,
                    )
                    try:
                        llm_rules = parse_llm_response(raw2)
                    except ValueError:
                        print("  AI failed after retry -- using automatic rules only")
                        _save_debug(raw2, table_name)
                        llm_rules = []
        else:
            print(f"\n  All issues covered automatically -- no AI call needed")

        llm_valid = []
        for rule in llm_rules:
            if not isinstance(rule, dict):
                continue
            ok, reason = validate_rule(rule, known_columns, profile=profile)
            if not ok:
                print(f"  Skipping AI rule '{rule.get('rule_id','?')}': {reason}")
                continue

            col = rule.get("column", "")
            if (rule.get("action") == "drop_column"
                    and col in profile.get("columns", {})):
                col_info  = profile["columns"][col]
                null_rate = col_info.get("null_rate", 0)
                issues    = {i["type"] for i in col_info.get("issues", [])}
                if null_rate < 0.60 and "constant_column" not in issues:
                    print(
                        f"  Skipping LLM drop_column '{col}': "
                        f"null_rate={null_rate:.0%} is below 60% threshold. "
                        f"LLM reason: '{rule.get('reason','')}' -- "
                        f"likely hallucinated."
                    )
                    continue

            llm_valid.append(rule)

        added = 0
        for rule in llm_valid:
            pair = (rule.get("column", ""), rule.get("action", ""))
            if pair not in det_pairs:
                det_rules.append(rule)
                det_pairs.add(pair)
                added += 1
        if added:
            print(f"  {added} AI rules added on top of automatic rules")

        seen, valid = set(), []
        for rule in sorted(det_rules, key=lambda r: r.get("order", 99)):
            pair = (rule.get("column", ""), rule.get("action", ""))
            if pair not in seen:
                seen.add(pair)
                valid.append(rule)

        dropped_cols = {
            r["column"] for r in valid
            if r["action"] == "drop_column" and r["column"] != "__global__"
        }
        if dropped_cols:
            before_contra = len(valid)
            valid = [
                r for r in valid
                if not (
                    r["column"] in dropped_cols
                    and r["action"] != "drop_column"
                )
            ]
            removed = before_contra - len(valid)
            if removed:
                print(
                    f"  Removed {removed} contradictory rule(s) "
                    f"(action on already-dropped column)"
                )

        for rule in valid:
            if rule.get("action") == "clip_outliers":
                lo  = (rule.get("params") or {}).get("lower")
                hi  = (rule.get("params") or {}).get("upper")
                col = rule.get("column", "")
                if (lo is None or hi is None) and col in profile.get("columns", {}):
                    oa = profile["columns"][col].get("outlier_analysis", {})
                    lo = oa.get("iqr_lower_fence")
                    hi = oa.get("iqr_upper_fence")
                    if lo is not None and hi is not None:
                        rule["params"] = {
                            "lower": round(lo, 4),
                            "upper": round(hi, 4),
                        }

                if (lo is not None and hi is not None
                        and col in profile.get("columns", {})):
                    col_stats = profile["columns"][col].get("stats", {})
                    data_min  = col_stats.get("min")
                    data_max  = col_stats.get("max")
                    if (data_min is not None and data_max is not None
                            and (data_max - data_min) > 0):
                        fence_range = hi - lo
                        data_range  = data_max - data_min
                        if fence_range / data_range < 0.05:
                            print(
                                f"  Note: clip_outliers on '{col}' has very "
                                f"narrow fences ({round(fence_range,2)} vs "
                                f"range {round(data_range,2)}) "
                                f"-> downgraded to flag_outliers"
                            )
                            rule["action"] = "flag_outliers"
                            rule["params"] = {}

        before = len(valid)
        valid  = [
            r for r in valid
            if float(r.get("confidence", 0)) >= self.min_confidence
        ]
        if before > len(valid):
            print(
                f"  {before - len(valid)} rules filtered "
                f"(below {self.min_confidence:.0%} certainty threshold)"
            )

        if not valid:
            print("  No valid rules produced.")
            return []

        print(f"  {len(valid)} total rules ready for review")

        if AGENT_MODE == "HUMAN":
            approved = human_review(valid, df=df)
        else:
            now      = datetime.now(timezone.utc).isoformat()
            approved = [
                {**r, "approved_by": "AUTO", "approved_at": now}
                for r in valid
            ]
            print(f"  AUTO mode: {len(approved)} rules approved automatically")

        if approved:
            save_rules_db(approved, table_name)
        return approved


def _save_debug(raw: str, table_name: str) -> None:
    path = BASE_DIR / "llm_agent" / f"{table_name}_debug.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw, encoding="utf-8")
    print(f"  Debug output saved -> {path}")


# =====================================================================
#  PROFILE AUTO-DISCOVERY
# =====================================================================

def find_latest_profile(table_name: str = None, sheet_name: str = None,
                        profiles_dir: Path = None) -> Path:
    import json as _json

    pdir = profiles_dir or PROFILES_DIR
    if not pdir.exists():
        raise FileNotFoundError(
            f"Profiles directory not found: {pdir}\n"
            "Run: python3 data_quality_agent/profiler.py --csv file.csv"
        )

    if table_name and sheet_name:
        candidates = (
            list(pdir.glob(f"{table_name}__{sheet_name}_latest_profile.json"))
            or list(pdir.glob(f"{table_name}__{sheet_name}_profile_*.json"))
        )
        if not candidates:
            raise FileNotFoundError(
                f"No profile found for table='{table_name}' "
                f"sheet='{sheet_name}' in {pdir}"
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)

    if table_name:
        exact = list(pdir.glob(f"{table_name}_latest_profile.json"))
        if exact:
            return exact[0]
        sheet_latests = list(pdir.glob(f"{table_name}__*_latest_profile.json"))
        if sheet_latests:
            def _nrows(p):
                try:
                    with open(p) as f:
                        return _json.load(f).get("n_rows", 0)
                except Exception:
                    return 0
            return max(sheet_latests, key=_nrows)
        candidates = list(pdir.glob(f"{table_name}*_profile_*.json"))
        if not candidates:
            raise FileNotFoundError(
                f"No profile found for table '{table_name}' in {pdir}"
            )
        return max(candidates, key=lambda p: p.stat().st_mtime)

    non_sheet = [
        p for p in pdir.glob("*_latest_profile.json")
        if "__" not in p.stem
    ]
    if non_sheet:
        return max(non_sheet, key=lambda p: p.stat().st_mtime)

    sheet_latests = list(pdir.glob("*_latest_profile.json"))
    if sheet_latests:
        def _nrows(p):
            try:
                with open(p) as f:
                    return _json.load(f).get("n_rows", 0)
            except Exception:
                return 0
        return max(sheet_latests, key=_nrows)

    candidates = list(pdir.glob("*_profile_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No profile JSON found in {pdir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def list_ollama_models() -> None:
    try:
        import ollama
        models = ollama.Client(host=OLLAMA_HOST).list()
        if not models.models:
            print(f"  No models. Run: ollama pull llama3.1:8b")
            return
        print(f"\n  Ollama models at {OLLAMA_HOST}:")
        for m in models.models:
            size = (getattr(m, "size", 0) or 0) / 1e9
            print(f"    - {m.model:<40}  {size:.1f} GB")
    except Exception as e:
        print(f"  Cannot connect to Ollama: {e}\n  Start with: ollama serve")


# =====================================================================
#  CLI
# =====================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Data Cleaning AI Agent v11 -- "
            "automatic rules + AI for complex cases"
        )
    )
    p.add_argument("--profile",      type=str, default=None)
    p.add_argument("--table",        type=str, default=None)
    p.add_argument("--sheet",        type=str, default=None)
    p.add_argument("--csv",          type=str, default=None,
                   help="Original data file for the 'p' preview option")
    p.add_argument("--auto",         action="store_true",
                   help="Approve all rules without interactive review")
    p.add_argument("--det-only",     action="store_true",
                   help="Use automatic rules only, skip the AI model call")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--provider",     type=str, default=None,
                   choices=["ollama", "anthropic", "openai", "gemini"])
    p.add_argument("--model",        type=str, default=None)
    p.add_argument("--confidence",   type=float, default=0.55)
    p.add_argument("--profiles-dir", type=str, default=None)
    p.add_argument("--list-models",  action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.list_models:
        list_ollama_models()
        sys.exit(0)

    if args.auto:
        os.environ["AGENT_MODE"] = "AUTO"

    provider     = args.provider or LLM_PROVIDER
    model        = args.model or (OLLAMA_MODEL if provider == "ollama" else None)
    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else None

    if args.sheet and not args.table and args.csv:
        inferred_table = Path(args.csv).stem
        print(
            f"  Note: --table not provided with --sheet '{args.sheet}'. "
            f"Inferring --table='{inferred_table}' from CSV filename."
        )
        args.table = inferred_table

    if args.sheet and not args.profile:
        pdir       = profiles_dir or PROFILES_DIR
        table_stem = args.table or (Path(args.csv).stem if args.csv else None)
        if table_stem:
            pattern   = f"{table_stem}__*_latest_profile.json"
            available = list(pdir.glob(pattern))
            sheets    = [
                re.search(r'__(.+)_latest_profile', p.stem).group(1)
                for p in available
                if re.search(r'__(.+)_latest_profile', p.stem)
            ]
            if sheets and args.sheet not in sheets:
                print(
                    f"\n  ERROR: Sheet '{args.sheet}' not found "
                    f"for table '{table_stem}'."
                )
                print(f"  Available sheets: {sheets}")
                sys.exit(1)

    if args.profile:
        profile_path = Path(args.profile)
    else:
        profile_path = find_latest_profile(
            table_name   = args.table,
            sheet_name   = args.sheet,
            profiles_dir = profiles_dir,
        )

    print(f"\n  Profile  : {profile_path.name}")
    if not profile_path.exists():
        print(f"  Not found: {profile_path}")
        sys.exit(1)

    with open(profile_path, encoding="utf-8") as f:
        profile = json.load(f)

    source_field    = profile.get("source", "")
    sheet_in_source = ""
    sm              = re.search(r'\[(.+)\]$', source_field)
    if sm:
        sheet_in_source       = sm.group(1)
        effective_table       = (
            profile.get("table_name", "table") + "__" + sheet_in_source
        )
        profile["table_name"] = effective_table
        print(
            f"  Sheet    : {sheet_in_source}  ->  "
            f"rules saved as '{effective_table}'"
        )

    dq = profile.get("dataset_quality", {})
    print(f"  Table    : {profile.get('table_name','?')}")
    print(f"  Rows     : {profile.get('n_rows','?'):,}")
    print(
        f"  Score    : {dq.get('dataset_readiness_score','?')}/100  "
        f"[{dq.get('readiness_interpretation','?')}]"
    )
    print(
        f"  Issues   : {dq.get('high_issues',0)} HIGH  "
        f"{dq.get('medium_issues',0)} MEDIUM  "
        f"{dq.get('low_issues',0)} LOW"
    )

    df_for_preview = None
    if args.csv:
        file_path = Path(args.csv)
        if file_path.exists():
            try:
                suffix = file_path.suffix.lower()
                if suffix in (".xlsx", ".xls"):
                    sheet_to_load  = sheet_in_source or args.sheet or 0
                    df_for_preview = pd.read_excel(
                        file_path, sheet_name=sheet_to_load
                    )
                    print(
                        f"  Dataset  : {len(df_for_preview):,} rows loaded "
                        f"(sheet: {sheet_to_load or 'first'})"
                    )
                elif suffix == ".parquet":
                    df_for_preview = pd.read_parquet(file_path)
                elif suffix == ".json":
                    try:
                        df_for_preview = pd.read_json(file_path)
                    except ValueError:
                        df_for_preview = pd.read_json(file_path, lines=True)
                else:
                    sys.path.insert(0, str(BASE_DIR / "data_quality_agent"))
                    from profiler import _read_csv_safe
                    df_for_preview = _read_csv_safe(file_path)
                    print(
                        f"  Dataset  : {len(df_for_preview):,} rows loaded "
                        f"for preview"
                    )
            except Exception as e:
                print(f"  Note: Could not load file for preview: {e}")
        else:
            print(f"  Note: File not found at {file_path} -- preview unavailable")

    if args.dry_run:
        det_rules    = generate_deterministic_rules(profile)
        handled_cols = {
            r["column"] for r in det_rules if r["column"] != "__global__"
        }
        compressed = compress_profile_for_llm(profile, skip_columns=handled_cols)
        payload    = build_user_prompt(compressed)
        print(f"\n  Automatic rules: {len(det_rules)}")
        print(f"  Flags for AI   : {len(compressed.get('summary_flags',[]))}")
        print(
            "\n" + "=" * 67
            + "\n  DRY RUN -- AI model payload\n"
            + "=" * 67
        )
        print(
            (payload or "[No AI payload -- all issues covered automatically]")
            [:3000]
        )
        print("=" * 67)
        sys.exit(0)

    if args.det_only:
        print(f"\n  Automatic-only mode -- no AI model call")
        det_rules = generate_deterministic_rules(profile)
        print(f"  {len(det_rules)} automatic rules generated")
        if AGENT_MODE == "HUMAN":
            approved = human_review(det_rules, df=df_for_preview)
        else:
            now      = datetime.now(timezone.utc).isoformat()
            approved = [
                {**r, "approved_by": "deterministic", "approved_at": now}
                for r in det_rules
            ]
        if approved:
            save_rules_db(approved, profile.get("table_name", "unknown"))
    else:
        print(
            f"  Provider : {provider.upper()} "
            f"({'local -- free' if provider == 'ollama' else 'cloud API'})"
        )
        if provider == "ollama":
            print(f"  Model    : {model}")
        agent    = DataCleaningAgent(
            provider       = provider,
            model          = model,
            min_confidence = args.confidence,
        )
        approved = agent.run(profile, df=df_for_preview)

    if approved:
        print(f"\n  {len(approved)} rules approved\n")
        print(
            f"  {'Order':<7} {'Severity':<10} {'Column':<28} "
            f"{'Action':<22} Type"
        )
        print("  " + "-" * 72)
        for r in sorted(approved, key=lambda x: x.get("order", 99)):
            src = r.get("source", "ai")
            print(
                f"  {str(r.get('order','?')):<7} {r['severity']:<10} "
                f"{r['column']:<28} {r['action']:<22} {src}"
            )
        tname = profile.get("table_name", "table")
        print(f"\n  Next step:")
        print(
            f"    python3 spark_cleaner/dynamic_cleaner.py "
            f"--csv your_file.csv --table {tname}"
        )
    else:
        print("\n  No rules were approved.")
