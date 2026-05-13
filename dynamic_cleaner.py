"""
spark_cleaner/dynamic_cleaner.py  v2
======================================
FIXES v2:
  - fill_constant is type-safe: numeric value fills numeric column,
    string value fills string column, no dtype mismatch errors
  - fill_constant(-1) for FK columns works correctly on integer columns
  - All other actions unchanged from v1
  - No icons anywhere in output
"""

import os
import sys
import json
import argparse
import unicodedata
import warnings
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR      = Path(__file__).parent.parent
RULES_DB_PATH = Path(os.getenv(
    "RULES_DB_PATH", str(BASE_DIR / "rules_db" / "approved_rules.json")
))


class RuleExecutor:

    def execute(self, df, rule):
        action  = rule.get("action", "")
        col     = rule.get("column", "")
        params  = rule.get("params", {}) or {}
        handler = getattr(
            self, f"_action_{action.replace('-', '_')}", None
        )
        if handler is None:
            return df, 0, f"  Unknown action '{action}' -- skipped"
        try:
            return handler(df, col, params)
        except Exception as e:
            return df, 0, f"  Error in rule '{rule.get('rule_id')}': {e}"

    def _action_drop_column(self, df, col, params):
        if col == "__global__":
            return df, 0, (
                "drop_column with __global__ is a no-op -- specify a column name"
            )
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        df = df.drop(columns=[col])
        return df, 1, f"Dropped column '{col}'"

    def _action_drop_rows_if_null(self, df, col, params):
        before = len(df)
        if col == "__global__":
            df    = df.dropna(how="all")
            after = len(df)
            return df, before - after, (
                f"Dropped {before - after} fully-empty rows"
            )
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        df    = df.dropna(subset=[col])
        after = len(df)
        return df, before - after, (
            f"Dropped {before - after} rows where '{col}' is null"
        )

    def _action_drop_duplicates(self, df, col, params):
        before = len(df)
        df     = df.drop_duplicates(keep="first")
        after  = len(df)
        return df, before - after, f"Dropped {before - after} duplicate rows"

    def _action_fill_median(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        null_mask   = df[col].isna()
        n           = int(null_mask.sum())
        if n == 0:
            return df, 0, f"No nulls in '{col}'"
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().sum() < n * 0.5:
            mode_vals = df[col].mode()
            if len(mode_vals) == 0:
                return df, 0, (
                    f"Cannot compute fill value for '{col}' -- skipped"
                )
            mode    = mode_vals.iloc[0]
            df[col] = df[col].fillna(mode)
            return df, n, (
                f"Filled {n} nulls in '{col}' with mode='{mode}' "
                f"(fill_median auto-downgraded: column is string type)"
            )
        median  = numeric_col.median()
        df[col] = numeric_col.fillna(median)
        return df, n, (
            f"Filled {n} nulls in '{col}' with median={round(median, 4)}"
        )

    def _action_fill_mode(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        null_mask = df[col].isna()
        n         = int(null_mask.sum())
        if n == 0:
            return df, 0, f"No nulls in '{col}'"
        mode_vals = df[col].mode()
        if len(mode_vals) == 0:
            return df, 0, f"Cannot compute mode for '{col}' -- skipped"
        mode    = mode_vals.iloc[0]
        df[col] = df[col].fillna(mode)
        return df, n, f"Filled {n} nulls in '{col}' with mode='{mode}'"

    def _action_fill_constant(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        value     = params.get("value", "UNKNOWN")
        null_mask = df[col].isna()
        n         = int(null_mask.sum())
        if n == 0:
            return df, 0, f"No nulls in '{col}'"

        # Type-safe fill
        # Case 1: numeric column + numeric value -> preserve dtype
        # Case 2: string column + numeric value -> convert value to string
        # Case 3: anything else -> let pandas handle it
        try:
            col_is_numeric = pd.api.types.is_numeric_dtype(df[col])
            val_is_numeric = isinstance(value, (int, float))

            if col_is_numeric and val_is_numeric:
                # Fill numeric column with numeric sentinel (-1, 0, etc.)
                df[col] = df[col].fillna(value)
            elif not col_is_numeric and val_is_numeric:
                # Fill string column with string representation of value
                df[col] = df[col].fillna(str(value))
            else:
                # Standard fill (string column with string value)
                df[col] = df[col].fillna(value)
        except Exception:
            # Absolute fallback
            df[col] = df[col].fillna(value)

        return df, n, (
            f"Filled {n} nulls in '{col}' with constant='{value}'"
        )

    def _action_clip_outliers(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = numeric_col.notna().mean()
        if valid_ratio < 0.5:
            return df, 0, (
                f"clip_outliers skipped for '{col}' -- "
                f"column is non-numeric ({valid_ratio:.0%} parseable)."
            )
        lower = params.get("lower")
        upper = params.get("upper")
        if lower is None and upper is None:
            q1    = numeric_col.quantile(0.25)
            q3    = numeric_col.quantile(0.75)
            iqr   = q3 - q1
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
        before_mask = (numeric_col < lower) | (numeric_col > upper)
        n           = int(before_mask.sum())
        df[col]     = numeric_col.clip(lower=lower, upper=upper)
        return df, n, (
            f"Clipped {n} outliers in '{col}' to "
            f"[{round(lower, 4)}, {round(upper, 4)}]"
        )

    def _action_flag_outliers(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = numeric_col.notna().mean()
        if valid_ratio < 0.5:
            return df, 0, (
                f"flag_outliers skipped for '{col}' -- "
                f"column is non-numeric ({valid_ratio:.0%} parseable)."
            )
        lower = params.get("lower")
        upper = params.get("upper")
        if lower is None and upper is None:
            q1    = numeric_col.quantile(0.25)
            q3    = numeric_col.quantile(0.75)
            iqr   = q3 - q1
            lower = float(q1 - 1.5 * iqr)
            upper = float(q3 + 1.5 * iqr)
        flag_col     = f"{col}_is_outlier"
        df[flag_col] = (
            (numeric_col < lower) | (numeric_col > upper)
        ).astype(int)
        n = int(df[flag_col].sum())
        return df, n, (
            f"Added flag column '{flag_col}' -- {n} outliers marked"
        )

    def _action_lowercase(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        before_unique = df[col].nunique()
        df[col]       = df[col].astype(str).str.strip().str.lower()
        df.loc[df[col] == "nan", col] = np.nan
        after_unique  = df[col].nunique()
        reduced       = before_unique - after_unique
        return df, reduced, (
            f"Lowercased '{col}' -- reduced unique values from "
            f"{before_unique} to {after_unique}"
        )

    def _action_cast_to_numeric(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        before_nulls = int(df[col].isna().sum())
        df[col]      = pd.to_numeric(df[col], errors="coerce")
        after_nulls  = int(df[col].isna().sum())
        new_nulls    = after_nulls - before_nulls
        return df, 1, (
            f"Cast '{col}' to numeric -- "
            f"{new_nulls} new NaN values from unparseable strings"
        )

    def _action_replace_sentinel(self, df, col, params):
        SENTINELS = {
            "n/a", "na", "n/d", "nd", "none", "null", "unknown",
            "missing", "undefined", "not available", "not applicable",
            "tbd", "to be determined", "-", "--", "---", "",
        }
        value = params.get("value")

        if col == "__global__":
            total, affected = 0, []
            if value is not None:
                for c in df.columns:
                    before = int((df[c] == value).sum())
                    if before > 0:
                        df[c]  = df[c].replace(value, np.nan)
                        total += before
                        affected.append(c)
                return df, total, (
                    f"Replaced value {value!r} with NaN in "
                    f"{len(affected)} column(s): {affected[:5]}"
                )
            else:
                for c in df.select_dtypes(include="object").columns:
                    str_col = df[c].astype(str).str.strip().str.lower()
                    mask    = str_col.isin(SENTINELS)
                    n       = int(mask.sum())
                    if n > 0:
                        df.loc[mask, c] = np.nan
                        total += n
                        affected.append(c)
                return df, total, (
                    f"Replaced {total} sentinels across "
                    f"{len(affected)} string column(s)"
                )

        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        if value is not None:
            before  = int((df[col] == value).sum())
            df[col] = df[col].replace(value, np.nan)
            return df, before, (
                f"Replaced {before} occurrences of {value!r} "
                f"with NaN in '{col}'"
            )
        else:
            str_col = df[col].astype(str).str.strip().str.lower()
            mask    = str_col.isin(SENTINELS)
            n       = int(mask.sum())
            df.loc[mask, col] = np.nan
            return df, n, (
                f"Replaced {n} sentinel strings with NaN in '{col}'"
            )

    def _action_fix_datetime(self, df, col, params):
        if col == "__global__":
            parsed_cols = []
            for c in df.select_dtypes(include="object").columns:
                sample       = df[c].dropna().astype(str).head(20)
                date_pattern = sample.str.match(
                    r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}", na=False
                )
                if date_pattern.mean() >= 0.7:
                    try:
                        df[c] = pd.to_datetime(
                            df[c], errors="coerce", format="mixed"
                        )
                        parsed_cols.append(c)
                    except Exception:
                        try:
                            df[c] = pd.to_datetime(df[c], errors="coerce")
                            parsed_cols.append(c)
                        except Exception:
                            pass
            if parsed_cols:
                return df, len(parsed_cols), (
                    f"Auto-parsed {len(parsed_cols)} date column(s): "
                    f"{parsed_cols}"
                )
            return df, 0, (
                "No date-like string columns detected for __global__ fix_datetime"
            )

        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"
        before_nulls = int(df[col].isna().sum())
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except TypeError:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        after_nulls = int(df[col].isna().sum())
        new_nulls   = after_nulls - before_nulls
        return df, 1, (
            f"Parsed '{col}' as datetime -- "
            f"{new_nulls} values became NaT (unparseable)"
        )

    def _action_normalize_unicode(self, df, col, params):
        if col not in df.columns:
            return df, 0, f"Column '{col}' not found -- skipped"

        def _norm(v):
            if pd.isna(v):
                return v
            return unicodedata.normalize("NFKC", str(v))

        df[col] = df[col].apply(_norm)
        return df, 1, (
            f"Applied NFKC unicode normalization to '{col}'"
        )


class DynamicCleaner:

    def __init__(self, rules_db_path=RULES_DB_PATH):
        self.executor      = RuleExecutor()
        self.rules_db_path = rules_db_path

    def load_rules(self, table_name):
        if not self.rules_db_path.exists():
            raise FileNotFoundError(
                f"Rules DB not found: {self.rules_db_path}\n"
                "Run the cleaning agent first: "
                "python3 llm_agent/cleaning_agent.py"
            )
        with open(self.rules_db_path, encoding="utf-8") as f:
            db = json.load(f)

        if isinstance(db, dict):
            table_rules = db.get(table_name, [])
            if not table_rules:
                lower_name = table_name.lower()
                for key, rules in db.items():
                    if key.lower() == lower_name:
                        print(
                            f"  Case-insensitive match: "
                            f"'{table_name}' -> '{key}'"
                        )
                        table_rules = rules
                        break
        elif isinstance(db, list):
            table_rules = [
                r for r in db
                if r.get("table", "").lower() == table_name.lower()
            ]
        else:
            table_rules = []

        if not table_rules:
            print(f"    No rules found for table '{table_name}'.")
            if isinstance(db, dict):
                print(f"    Available tables: {list(db.keys())}")
            print(f"    Run: python3 llm_agent/cleaning_agent.py")
            return []

        seen, unique_rules = set(), []
        for r in sorted(table_rules, key=lambda r: r.get("order", 99)):
            key = (r.get("column", ""), r.get("action", ""))
            if key not in seen:
                seen.add(key)
                unique_rules.append(r)

        print(f"    Loaded {len(unique_rules)} rules for '{table_name}'")
        return unique_rules

    def clean(self, df, table_name, dry_run=False):
        rules = self.load_rules(table_name)
        print(
            f"\n   Applying {len(rules)} rules to '{table_name}' "
            f"({len(df):,} rows x {len(df.columns)} cols)"
        )

        log      = []
        df_clean = df.copy()

        for rule in rules:
            rule_id = rule.get("rule_id", "?")
            action  = rule.get("action", "?")
            col     = rule.get("column", "?")
            order   = rule.get("order", "?")

            if dry_run:
                print(
                    f"    [DRY] order={order}  "
                    f"{action:<22} column={col}"
                )
                log.append({
                    "rule_id": rule_id, "action": action,
                    "column": col, "dry_run": True,
                })
                continue

            before_rows = len(df_clean)
            before_cols = len(df_clean.columns)
            df_clean, n_affected, description = self.executor.execute(
                df_clean, rule
            )
            after_rows  = len(df_clean)
            after_cols  = len(df_clean.columns)

            is_warn = (
                "skipped" in description.lower()
                or "error" in description.lower()
            )
            status = "  WARN" if is_warn else "    "
            print(
                f"  {status} [{order}] {action:<22} "
                f"{col:<22} -> {description}"
            )

            log.append({
                "rule_id":     rule_id,
                "action":      action,
                "column":      col,
                "order":       order,
                "n_affected":  n_affected,
                "description": description,
                "rows_before": before_rows,
                "rows_after":  after_rows,
                "cols_before": before_cols,
                "cols_after":  after_cols,
                "applied_at":  datetime.now(timezone.utc).isoformat(),
            })

        cols_removed = sum(
            1 for e in log
            if e.get("action") == "drop_column"
            and e.get("cols_after", 0) < e.get("cols_before", 0)
        )
        cols_added = sum(
            1 for e in log
            if e.get("action") == "flag_outliers"
            and e.get("cols_after", 0) > e.get("cols_before", 0)
        )
        summary = {
            "table":         table_name,
            "rules_applied": len(rules),
            "rows_before":   len(df),
            "rows_after":    len(df_clean),
            "cols_before":   len(df.columns),
            "cols_after":    len(df_clean.columns),
            "rows_removed":  len(df) - len(df_clean),
            "cols_removed":  cols_removed,
            "cols_added":    cols_added,
            "rule_log":      log,
            "completed_at":  datetime.now(timezone.utc).isoformat(),
        }
        return df_clean, summary


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Dynamic Cleaner -- applies AI-approved rules to any dataset"
        )
    )
    p.add_argument("--csv",      type=str, required=True,
                   help="Path to input file (CSV / Excel / Parquet / JSON)")
    p.add_argument("--table",    type=str, default=None,
                   help="Table name for rule matching (defaults to filename stem)")
    p.add_argument("--sheet",    type=str, default=None,
                   help="Excel sheet name to load")
    p.add_argument("--output",   type=str, default=None,
                   help="Output directory for cleaned file")
    p.add_argument("--dry-run",  action="store_true",
                   help="Show what would be applied without changing data")
    p.add_argument("--save-log", action="store_true",
                   help="Save cleaning log as JSON alongside output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    file_path  = Path(args.csv)
    table_name = args.table or file_path.stem

    if not file_path.exists():
        print(f"   File not found: {file_path}")
        sys.exit(1)

    sheet_name = args.sheet
    if sheet_name is None and "__" in table_name:
        parts      = table_name.split("__", 1)
        sheet_name = parts[1]

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=enc)
                if enc != "utf-8":
                    print(
                        f"  Note: loaded '{file_path.name}' "
                        f"with encoding '{enc}'"
                    )
                break
            except UnicodeDecodeError:
                continue
        else:
            print(
                f"  Could not decode '{file_path.name}' "
                f"with any known encoding."
            )
            sys.exit(1)

    elif suffix in (".xlsx", ".xls"):
        if sheet_name:
            available = pd.ExcelFile(file_path).sheet_names
            if sheet_name not in available:
                print(
                    f"  ERROR: Sheet '{sheet_name}' not found "
                    f"in '{file_path.name}'."
                )
                print(f"  Available sheets: {available}")
                sys.exit(1)
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            print(
                f"  Note: loaded sheet '{sheet_name}' "
                f"from '{file_path.name}'"
            )
        else:
            df = pd.read_excel(file_path)

    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)

    elif suffix == ".json":
        try:
            df = pd.read_json(file_path)
        except ValueError:
            df = pd.read_json(file_path, lines=True)

    else:
        print(
            f"  Unsupported format: {suffix}  "
            f"(supported: .csv .xlsx .xls .parquet .json)"
        )
        sys.exit(1)

    print(
        f"\n  Loaded '{file_path.name}' -> "
        f"{len(df):,} rows x {len(df.columns)} cols"
    )

    cleaner           = DynamicCleaner()
    df_clean, summary = cleaner.clean(df, table_name, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"\n  Cleaning summary:")
        print(
            f"      Rows : {summary['rows_before']:,} -> "
            f"{summary['rows_after']:,} "
            f"({summary['rows_removed']} removed)"
        )
        removed = summary.get("cols_removed", 0)
        added   = summary.get("cols_added", 0)
        parts   = []
        if removed:
            parts.append(f"{removed} removed")
        if added:
            parts.append(f"{added} added as flags")
        if not parts:
            parts.append("no change")
        print(
            f"      Cols : {summary['cols_before']} -> "
            f"{summary['cols_after']} ({', '.join(parts)})"
        )

        out_dir = (
            Path(args.output) if args.output
            else file_path.parent / "cleaned"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".json":
            out_path = out_dir / f"{table_name}_cleaned.json"
            df_clean.to_json(
                out_path, orient="records", indent=2, force_ascii=False
            )
        else:
            out_path = out_dir / f"{table_name}_cleaned.csv"
            df_clean.to_csv(out_path, index=False)

        print(f"      Saved -> {out_path}")

        if args.save_log:
            log_path = out_dir / f"{table_name}_cleaning_log.json"
            with open(log_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"      Log   -> {log_path}")
