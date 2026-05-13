import json
import math
import re
import argparse
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  FILE LOADER — supports CSV, Excel (all sheets), JSON, Parquet, TSV
#  FIX v7: multi-encoding fallback for CSVs with non-UTF-8 characters
# ══════════════════════════════════════════════════════════════════════════════

def _detect_encoding(path: Path) -> str:
    """
    Best-effort encoding detection.
    Priority: chardet (if installed) -> latin-1 (always decodes) fallback.
    """
    try:
        import chardet
        with open(path, "rb") as f:
            raw = f.read(min(65536, path.stat().st_size))   # sample up to 64 KB
        result = chardet.detect(raw)
        enc    = result.get("encoding") or "latin-1"
        # chardet sometimes returns 'ascii' for files that have a BOM — normalise
        if enc.lower() in ("ascii", ""):
            enc = "utf-8"
        return enc
    except ImportError:
        return "latin-1"   # latin-1 decodes every byte 0x00-0xFF without error


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Try encodings in order until one works.
    Covers: UTF-8, UTF-8 with BOM, Latin-1, Windows-1252, and chardet guess.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    # Prepend chardet guess so it gets priority
    detected = _detect_encoding(path)
    if detected.lower() not in [e.lower() for e in encodings_to_try]:
        encodings_to_try.insert(0, detected)

    last_err = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            if enc != "utf-8":
                print(f"  Note: loaded '{path.name}' with encoding '{enc}'")
            return df
        except (UnicodeDecodeError, LookupError) as e:
            last_err = e
            continue

    raise ValueError(
        f"Cannot decode '{path.name}' with any of {encodings_to_try}.\n"
        f"Last error: {last_err}\n"
        "Install chardet for better auto-detection: pip install chardet"
    )


def load_file(path: str | Path) -> dict:
    path   = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return {"data": _read_csv_safe(path)}

    elif suffix in (".xlsx", ".xls"):
        xls    = pd.ExcelFile(path)
        print(f"  Excel sheets detected: {xls.sheet_names}")
        sheets = {s: pd.read_excel(path, sheet_name=s) for s in xls.sheet_names}
        return {
            "data":        sheets[xls.sheet_names[0]],
            "all_sheets":  sheets,
            "sheet_names": xls.sheet_names,
        }

    elif suffix == ".json":
        try:
            return {"data": pd.read_json(path)}
        except ValueError:
            return {"data": pd.read_json(path, lines=True)}

    elif suffix == ".parquet":
        return {"data": pd.read_parquet(path)}

    elif suffix in (".txt", ".tsv"):
        return {"data": _read_csv_safe(path) if suffix == ".txt"
                else pd.read_csv(path, sep="\t", low_memory=False)}

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# ══════════════════════════════════════════════════════════════════════════════
#  JSON SERIALIZATION SAFETY
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    # FIX: Timestamp check MUST come before pd.NA / pd.NaT
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Paths
BASE_DIR     = Path(__file__).parent.parent
PROFILES_DIR = BASE_DIR / "profiler" / "profiles"
BASELINE_DIR = BASE_DIR / "profiler" / "baselines"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN TYPE DETECTOR
#  FIX v7: numeric columns are NEVER classified as temporal.
#  The old code checked TEMPORAL_KW on column name BEFORE checking the dtype,
#  so HR columns like "MonthlyIncome", "HourlyRate", "YearsAtCompany" were
#  misclassified as temporal. Now numeric dtype is checked first.
# ══════════════════════════════════════════════════════════════════════════════

class ColumnTypeDetector:

    ORDINAL_MAX              = 25
    ID_CARDINALITY_THRESHOLD = 0.90
    DATETIME_SAMPLE          = 50

    FK_NAME_PATTERNS = re.compile(
        r"(_id|_key|_fk|_code|_ref|_num|id_|fk_)$", re.IGNORECASE
    )
    # FIX 1: Phone identifier patterns — anchored to avoid false matches.
    # "phone" alone would match "PhoneService", "Smartphone" etc.
    # Pattern requires: standalone "phone", _phone suffix, or full known names.
    PHONE_ID_PATTERNS = re.compile(
        r"(^phone$|^phone_number$|^client_phone$|^whatsapp_number$|^whatsapp_num$|_phone$|_telephone$|_mobile$|^mobile$|^telephone$|^tel$|^fax$|^bar_code$|^bar_code_number$|^bar_code_indoor$|^bar_code_outdoor$|^bar_code_indoor_number$|^bar_code_outdoor_number$|^barcode$|^qr_code$)",
        re.IGNORECASE
    )
    TEXT_NAME_PATTERNS = re.compile(
        r"(review|comment|description|note|text|feedback|message|body|content|remark|detail|summary)",
        re.IGNORECASE,
    )
    # Words in column names that suggest a temporal column
    TEMPORAL_KW = re.compile(
        r"(date|time|timestamp|created_at|updated_at|modified_at|"
        r"birth_date|death_date|start_date|end_date|open_date|close_date|"
        r"registered_at|scheduled_at|_dt$|^dt_)",
        re.IGNORECASE,
    )
    # These words look temporal but are NOT when the column is numeric
    NUMERIC_NOT_TEMPORAL = re.compile(
        r"(rate|income|salary|wage|cost|price|amount|score|"
        r"year|month|week|day|hour|minute|second|age|duration|"
        r"period|tenure|years|months|days|hours)",
        re.IGNORECASE,
    )
    # BUG H FIX: GPS and coordinate columns are never datetime
    GPS_NON_TEMPORAL = re.compile(
        r"(^gps$|^coord$|^coordinates$|^location$|^loc$|^latitude$|^longitude$|^lat$|^lon$|^long$)",
        re.IGNORECASE,
    )

    DATETIME_PATTERNS = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}",
        r"^\d{2}/\d{2}/\d{4}$",
        r"^\d{2}-\d{2}-\d{4}$",
        r"^\d{4}/\d{2}/\d{2}$",
        r"^\d{2}\s+\w+\s+\d{4}$",
        r"^\w+\s+\d{2},?\s+\d{4}$",
    ]

    @classmethod
    def detect(cls, series: pd.Series) -> str:
        series_clean = series.dropna()
        n            = len(series_clean)
        col_name     = str(getattr(series, "name", "") or "")

        if n == 0:
            return "empty"

        # FIX 1: Phone/barcode columns — skip everything, return "id"
        if cls.PHONE_ID_PATTERNS.match(col_name):
            return "id"

        # FIX 3: GPS/coordinate columns — short-circuit BEFORE any branch.
        # Without this, the string branch calls _looks_like_datetime() on
        # GPS strings like "36.8189,10.1658" and misclassifies them.
        if hasattr(cls, "GPS_NON_TEMPORAL") and cls.GPS_NON_TEMPORAL.match(col_name):
            # GPS is categorical — it groups by zone but is not a date
            unique_count = series_clean.nunique()
            n_rows       = len(series_clean)
            unique_rate  = unique_count / n_rows if n_rows > 0 else 0
            if unique_rate > 0.80:
                return "text"
            return "categorical"

        if cls._is_boolean(series_clean):
            return "boolean"

        # Already parsed datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # --- Numeric branch (checked BEFORE any keyword-name checks) ----------
        if pd.api.types.is_numeric_dtype(series):
            unique_count = series_clean.nunique()
            unique_rate  = unique_count / n

            # Primary key: near-100% unique + sequential integers
            if (unique_rate >= 0.99
                    and pd.api.types.is_integer_dtype(series)
                    and cls._looks_like_numeric_id(series_clean)):
                return "id"

            # Foreign key: name ends in _id/_key etc. + moderate uniqueness
            if (cls.FK_NAME_PATTERNS.search(col_name)
                    and pd.api.types.is_integer_dtype(series)
                    and 0.01 <= unique_rate <= 0.80):
                return "foreign_key"

            return "ordinal_numeric" if unique_count <= cls.ORDINAL_MAX else "numeric"

        # --- String / object branch -------------------------------------------
        dtype = series.dtype
        is_string_col = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or str(dtype) in ("object", "string", "str")
        ) and not pd.api.types.is_numeric_dtype(dtype)

        if is_string_col:
            if cls._is_mixed_type(series_clean):
                return "mixed"

            unique_count = series_clean.nunique()
            unique_rate  = unique_count / n

            # Temporal string check (e.g. "2023-01-15" stored as object)
            if cls._looks_like_datetime(series_clean):
                return "datetime_string"

            if unique_rate >= cls.ID_CARDINALITY_THRESHOLD:
                return "id" if cls._looks_like_id(series_clean) else "text"

            if (cls.TEXT_NAME_PATTERNS.search(col_name)
                    and unique_count < 20 and unique_count >= 2):
                return "low_cardinality_text"

            return "categorical"

        return "unknown"

    @staticmethod
    def _is_boolean(series_clean: pd.Series) -> bool:
        if pd.api.types.is_bool_dtype(series_clean):
            return True
        unique_vals = set(series_clean.unique())
        bool_sets = [
            {True, False}, {0, 1}, {0.0, 1.0}, {"0", "1"},
            {"true", "false"}, {"True", "False"},
            {"yes", "no"}, {"Yes", "No"}, {"YES", "NO"},
            {"y", "n"}, {"Y", "N"},
        ]
        return unique_vals in bool_sets

    @classmethod
    def _looks_like_datetime(cls, series_clean: pd.Series) -> bool:
        sample = series_clean.head(cls.DATETIME_SAMPLE).astype(str)
        for pattern in cls.DATETIME_PATTERNS:
            try:
                if sample.str.match(pattern).mean() >= 0.70:
                    return True
            except Exception:
                pass
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed.notna().mean() >= 0.70:
                return True
        except Exception:
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.70:
                    return True
            except Exception:
                pass
        return False

    @staticmethod
    def _looks_like_id(series_clean: pd.Series) -> bool:
        sample = series_clean.head(30).astype(str)
        if sample.str.match(r"^\d+$").mean() > 0.8:
            return True
        if sample.str.match(r"^[0-9a-fA-F\-]{8,}$").mean() > 0.5:
            return True
        return False

    @staticmethod
    def _looks_like_numeric_id(series_clean: pd.Series) -> bool:
        col_name = str(getattr(series_clean, "name", "") or "").lower()
        if col_name.endswith("_id") or col_name == "id" or col_name.endswith("_key"):
            return True
        vals = series_clean.dropna().sort_values().values
        if len(vals) < 2:
            return False
        diffs = np.diff(vals.astype(float))
        return bool(np.all(diffs == diffs[0]) and diffs[0] > 0)

    @staticmethod
    def _is_mixed_type(series_clean: pd.Series) -> bool:
        if not pd.api.types.is_object_dtype(series_clean):
            return False
        sample  = series_clean.head(200)
        non_str = {type(v).__name__ for v in sample} - {"str"}
        return len(non_str) > 1


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE SCHEMA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class BaselineSchemaManager:
    def __init__(self, table_name: str, sheet_name: str = "",
                 baseline_dir: Path | None = None):
        self.table_name = table_name
        self.sheet_name = sheet_name
        bdir            = baseline_dir or BASELINE_DIR
        bdir.mkdir(parents=True, exist_ok=True)
        suffix             = f"__{sheet_name}" if sheet_name else ""
        self.baseline_path = bdir / f"{table_name}{suffix}_baseline.json"

    def has_baseline(self) -> bool:
        return self.baseline_path.exists()

    def save(self, schema: dict) -> None:
        with open(self.baseline_path, "w") as f:
            json.dump({"table_name": self.table_name,
                       "saved_at":   _utcnow_iso(),
                       "schema":     schema}, f, indent=2)

    def load(self) -> dict:
        with open(self.baseline_path) as f:
            return json.load(f)["schema"]

    def detect_drift(self, current_schema: dict) -> dict:
        if not self.has_baseline():
            return {"has_drift": False, "status": "no_baseline",
                    "new_columns": [], "removed_columns": [], "type_changes": []}
        baseline      = self.load()
        baseline_cols = set(baseline.keys())
        current_cols  = set(current_schema.keys())
        new_cols      = list(current_cols - baseline_cols)
        removed_cols  = list(baseline_cols - current_cols)
        type_changes  = [
            {"column": col, "was": baseline[col], "now": current_schema[col]}
            for col in baseline_cols & current_cols
            if baseline[col] != current_schema[col]
        ]
        has_drift = bool(new_cols or removed_cols or type_changes)
        return {
            "has_drift":         has_drift,
            "status":            "drift_detected" if has_drift else "ok",
            "new_columns":       new_cols,
            "removed_columns":   removed_cols,
            "type_changes":      type_changes,
            "baseline_saved_at": self._load_meta_date(),
        }

    def _load_meta_date(self) -> str:
        try:
            with open(self.baseline_path) as f:
                return json.load(f).get("saved_at", "unknown")
        except Exception:
            return "unknown"

    def reset(self) -> None:
        if self.baseline_path.exists():
            self.baseline_path.unlink()
            print(f"  Baseline reset for '{self.table_name}'")


# ══════════════════════════════════════════════════════════════════════════════
#  PATTERN MINER
# ══════════════════════════════════════════════════════════════════════════════

class PatternMiner:
    GENERALIZERS = [
        (re.compile(r"\d+"), "N"),
        (re.compile(r"[a-z]+"), "a"),
        (re.compile(r"[A-Z]+"), "A"),
        (re.compile(r"\s+"), " "),
    ]

    @classmethod
    def mine(cls, series: pd.Series, top_n: int = 8) -> dict:
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {}
        patterns = clean.apply(cls._generalize)
        vc       = patterns.value_counts()
        result   = {
            "pattern_count":      int(vc.nunique()),
            "top_patterns":       {str(k): int(v) for k, v in vc.head(top_n).items()},
            "has_mixed_patterns": bool(vc.nunique() > 3),
            "whitespace_issues":  {
                "leading_spaces":  bool(clean.str.startswith(" ").any()),
                "trailing_spaces": bool(clean.str.endswith(" ").any()),
                "tabs":            bool(clean.str.contains("\t").any()),
                "newlines":        bool(clean.str.contains("\n").any()),
            },
        }
        try:
            result["has_non_ascii"] = bool(clean.apply(lambda x: not x.isascii()).any())
        except Exception:
            result["has_non_ascii"] = False
        if result["has_non_ascii"]:
            try:
                mask = ~clean.apply(lambda x: x.isascii())
                result["non_ascii_count"]    = int(mask.sum())
                result["non_ascii_examples"] = clean[mask].head(3).tolist()
            except Exception:
                pass
        num_rate = round(float(pd.to_numeric(clean, errors="coerce").notna().mean()), 4)
        if num_rate > 0.80:
            result["numeric_stored_as_string"] = True
            result["numeric_as_string_rate"]   = num_rate
        return result

    @classmethod
    def _generalize(cls, value: str) -> str:
        s = value
        for pattern, replacement in cls.GENERALIZERS:
            s = pattern.sub(replacement, s)
        return s[:50]


# ══════════════════════════════════════════════════════════════════════════════
#  FORMAT CONSISTENCY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class FormatConsistencyAnalyzer:
    @classmethod
    def analyze(cls, series: pd.Series) -> dict:
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return {"format_issues": [], "has_format_issues": False}
        issues = []
        vc     = clean.value_counts()
        unmatched = cls._find_unmatched_brackets(vc)
        if unmatched:
            issues.append({"type": "unmatched_brackets",
                            "detail": f"Values with unmatched brackets/parens: {unmatched}",
                            "count": sum(vc.get(v, 0) for v in unmatched)})
        sep_variants = cls._find_separator_variants(vc)
        if sep_variants:
            issues.append({"type": "mixed_separators",
                            "detail": f"Same value with different separators: {sep_variants}"})
        naming = cls._find_mixed_naming(vc)
        if naming:
            issues.append({"type": "mixed_naming_conventions", "detail": naming})
        return {"format_issues": issues, "has_format_issues": len(issues) > 0}

    @staticmethod
    def _find_unmatched_brackets(vc: pd.Series) -> list:
        bad = []
        for val in vc.head(50).index:
            s = str(val)
            if s.count("(") + s.count("[") != s.count(")") + s.count("]"):
                bad.append(val)
        return bad

    @staticmethod
    def _find_separator_variants(vc: pd.Series) -> list:
        groups = defaultdict(list)
        for val in vc.head(100).index:
            normalized = re.sub(r"\s*([x*,/])\s*", r"\1", str(val).lower().strip())
            groups[normalized].append(val)
        return [v for v in groups.values() if len(v) > 1]

    @staticmethod
    def _find_mixed_naming(vc: pd.Series) -> str:
        has_named   = any(any(kw in str(v).lower() for kw in
                              ["hd", "full", "ultra", "qhd", "wqhd", "4k", "2k"])
                          for v in vc.head(30).index)
        has_numeric = any(bool(re.match(r"^\d+\s*[x*]\s*\d+", str(v)))
                          for v in vc.head(30).index)
        if has_named and has_numeric:
            return ("Column mixes human-readable labels (e.g. 'Full HD') with raw numeric formats "
                    "(e.g. '1920 x 1080'). Standardize to one convention for consistent grouping.")
        return ""


# ══════════════════════════════════════════════════════════════════════════════
#  ZERO-AS-NULL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class ZeroAsNullDetector:
    IMPLAUSIBLE_ZERO_KEYWORDS = [
        "weight", "kg", "lb", "price", "cost", "amount", "revenue", "salary",
        "income", "inch", "size", "height", "width", "length", "diameter",
        "speed", "freq", "battery", "watt", "age", "duration", "distance",
        "area", "volume", "quantity", "rate", "score", "rating",
    ]

    @classmethod
    def check(cls, col_name: str, series: pd.Series) -> dict:
        clean      = series.dropna()
        zero_count = int((clean == 0).sum())
        if zero_count == 0:
            return {}
        zero_rate = round(float((clean == 0).mean()), 4)
        if any(kw in str(col_name).lower() for kw in cls.IMPLAUSIBLE_ZERO_KEYWORDS):
            return {
                "zero_likely_null": True,
                "zero_count":       zero_count,
                "zero_rate":        zero_rate,
                "recommendation":   (
                    f"'{col_name}' has {zero_count} zero values ({zero_rate*100:.1f}%). "
                    "Domain analysis: zero is physically impossible for this column. "
                    "Replace zeros with NaN before imputation: df[col].replace(0, np.nan)"
                ),
            }
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#  SENTINEL VALUE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

SENTINEL_PATTERNS = re.compile(
    r"^(no\s+(data|info|storage|value|record|entry|result|content|information)"
    r"|none|n/?a|unknown|missing|undefined|null|not\s+available|not\s+applicable"
    r"|n/d|nd|-+|tbd|to\s+be\s+determined|not\s+specified|unspecified|na)$",
    re.IGNORECASE,
)
SENTINEL_NO_X     = re.compile(r"^no\s+\w+(\s+\w+)*$", re.IGNORECASE)
SENTINEL_ALLOWLIST = re.compile(
    r"^(north|norway|nokia|novel|node|normal|notebook|non[-\s])", re.IGNORECASE
)


def detect_sentinel_values(series: pd.Series) -> list:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return []
    clean     = series.dropna().astype(str)
    vc        = clean.value_counts()
    sentinels = []
    for val, count in vc.items():
        v = str(val).strip()
        if SENTINEL_ALLOWLIST.match(v):
            continue
        if SENTINEL_PATTERNS.match(v) or SENTINEL_NO_X.match(v):
            sentinels.append({"value": v, "count": int(count)})
    return sentinels


# ══════════════════════════════════════════════════════════════════════════════
#  OUTLIER ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class OutlierAnalyzer:
    @staticmethod
    def analyze(clean: pd.Series) -> dict:
        n = len(clean)
        if n < 10:
            return {}
        arr      = clean.values.astype(float)
        q1, q3   = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr      = q3 - q1
        iqr_mask = (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)
        mean     = float(np.mean(arr))
        std      = float(np.std(arr))
        z_mask   = np.abs((arr - mean) / std) > 3 if std > 0 else np.zeros(n, dtype=bool)
        median   = float(np.median(arr))
        mad      = float(np.median(np.abs(arr - median)))
        mz_mask  = ((0.6745 * np.abs(arr - median) / mad) > 3.5
                    if mad > 0 else np.zeros(n, dtype=bool))
        consensus = (iqr_mask.astype(int) + z_mask.astype(int) + mz_mask.astype(int)) >= 2
        col_range = float(arr.max()) - float(arr.min())
        itr       = round(iqr / col_range, 4) if col_range > 0 else None
        return {
            "iqr_outlier_rate":       round(float(iqr_mask.mean()), 4),
            "zscore_outlier_rate":    round(float(z_mask.mean()), 4),
            "modified_zscore_rate":   round(float(mz_mask.mean()), 4),
            "consensus_outlier_rate": round(float(consensus.mean()), 4),
            "iqr_lower_fence":        round(q1 - 1.5 * iqr, 4),
            "iqr_upper_fence":        round(q3 + 1.5 * iqr, 4),
            "iqr_to_range_ratio":     itr,
            "narrow_iqr":             bool(itr is not None and itr < 0.15),
            "outlier_sample":         sorted([round(float(v), 4) for v in arr[consensus][:10]]),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-COLUMN ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class CrossColumnAnalyzer:
    NULL_CORR_THRESHOLD = 0.80
    NUM_CORR_THRESHOLD  = 0.95
    MAX_PAIRS           = 200

    @classmethod
    def analyze(cls, df: pd.DataFrame, detected_schema: dict) -> dict:
        return {
            "correlated_null_groups":    cls._correlated_nulls(df),
            "high_correlation_pairs":    cls._numeric_correlations(df, detected_schema),
            "possible_derived_columns":  cls._derived_columns(df, detected_schema),
            "additive_relationships":    cls._additive_relationships(df, detected_schema),
            "nonlinear_derived_columns": cls._nonlinear_derived(df, detected_schema),
        }

    @classmethod
    def _correlated_nulls(cls, df: pd.DataFrame) -> list:
        df_partial = df[~df.isnull().all(axis=1)]
        null_cols  = [c for c in df_partial.columns if df_partial[c].isna().any()]
        if len(null_cols) < 2:
            return []
        null_matrix = df_partial[null_cols].isna().astype(int)
        groups, visited = [], set()
        cols = null_matrix.columns.tolist()
        for i in range(len(cols)):
            if cols[i] in visited:
                continue
            group = [cols[i]]
            for j in range(i + 1, len(cols)):
                if cols[j] in visited:
                    continue
                try:
                    corr = float(null_matrix[cols[i]].corr(null_matrix[cols[j]]))
                except Exception:
                    corr = 0.0
                if abs(corr) >= cls.NULL_CORR_THRESHOLD:
                    group.append(cols[j])
                    visited.add(cols[j])
            if len(group) > 1:
                visited.add(cols[i])
                same_rows = bool(
                    (null_matrix[group].sum(axis=1) == len(group)).sum() > 0
                )
                groups.append({
                    "columns":           group,
                    "same_rows_missing": same_rows,
                    "null_count":        int(df_partial[group[0]].isna().sum()),
                    "interpretation":    (
                        "Same-source join gap — missing together. Treat as a unit."
                        if same_rows else
                        "Correlated but not identical null pattern — conditional dependency."
                    ),
                })
        return groups

    @classmethod
    def _numeric_correlations(cls, df: pd.DataFrame, detected_schema: dict) -> list:
        num_cols = [c for c, t in detected_schema.items()
                    if t in ("numeric", "ordinal_numeric") and c in df.columns]
        pairs, checked = [], 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if checked >= cls.MAX_PAIRS:
                    break
                c1, c2 = num_cols[i], num_cols[j]
                try:
                    corr = float(df[[c1, c2]].dropna().corr().iloc[0, 1])
                except Exception:
                    corr = 0.0
                if abs(corr) >= cls.NUM_CORR_THRESHOLD:
                    pairs.append({"col_a": c1, "col_b": c2,
                                  "pearson_corr": round(corr, 4),
                                  "recommendation": (
                                      "Near-perfect correlation — one column may be derived from the other."
                                  )})
                checked += 1
        return sorted(pairs, key=lambda x: abs(x["pearson_corr"]), reverse=True)[:20]

    @classmethod
    def _derived_columns(cls, df: pd.DataFrame, detected_schema: dict) -> list:
        num_cols = [c for c, t in detected_schema.items()
                    if t in ("numeric", "ordinal_numeric") and c in df.columns]
        derived, checked = [], 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if checked >= cls.MAX_PAIRS:
                    break
                c1, c2  = num_cols[i], num_cols[j]
                sub     = df[[c1, c2]].dropna()
                if len(sub) < 10:
                    continue
                a, b    = sub[c1].values.astype(float), sub[c2].values.astype(float)
                nonzero = b != 0
                if nonzero.sum() < 10:
                    continue
                ratios = a[nonzero] / b[nonzero]
                mean_r = np.mean(ratios)
                cv     = float(np.std(ratios) / mean_r) if mean_r != 0 else 999
                if cv < 0.01:
                    derived.append({
                        "col_a":          c1,
                        "col_b":          c2,
                        "ratio":          round(float(mean_r), 6),
                        "cv":             round(cv, 6),
                        "recommendation": (
                            f"'{c1}' approx {round(float(mean_r), 4)} x '{c2}'. "
                            "Consider dropping one."
                        ),
                    })
                checked += 1
        return derived

    @classmethod
    def _additive_relationships(cls, df: pd.DataFrame, detected_schema: dict) -> list:
        num_cols = [c for c, t in detected_schema.items()
                    if t in ("numeric", "ordinal_numeric") and c in df.columns]
        if len(num_cols) < 3:
            return []
        results, checked = [], 0
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                for k in range(len(num_cols)):
                    if k in (i, j):
                        continue
                    if checked >= 500:
                        break
                    ca, cb, cc = num_cols[i], num_cols[j], num_cols[k]
                    sub = df[[ca, cb, cc]].dropna()
                    if len(sub) < 20:
                        continue
                    residuals = np.abs(sub[ca].values + sub[cb].values - sub[cc].values)
                    if residuals.mean() < 0.01 and (residuals > 0.01).mean() < 0.01:
                        results.append({
                            "formula":    f"{ca} + {cb} = {cc}",
                            "col_a":      ca, "col_b": cb, "col_result": cc,
                            "match_rate": round(float((residuals <= 0.01).mean()), 4),
                            "note":       (
                                f"'{cc}' is derived as '{ca}' + '{cb}'. "
                                "Do NOT drop source columns based on correlation alone."
                            ),
                        })
                    checked += 1
        return results

    @classmethod
    def _nonlinear_derived(cls, df: pd.DataFrame, detected_schema: dict) -> list:
        results = []
        needed  = ["X_res", "Y_res", "Inches", "ppi"]
        if all(c in df.columns for c in needed):
            sub = df[needed].dropna()
            if len(sub) >= 10:
                calc = (
                    np.sqrt(sub["X_res"] ** 2 + sub["Y_res"] ** 2) / sub["Inches"]
                ).round(2)
                match_rate = round(float((abs(calc - sub["ppi"]) <= 0.5).mean()), 4)
                if match_rate >= 0.95:
                    results.append({
                        "col_result":  "ppi",
                        "formula":     "sqrt(X_res^2 + Y_res^2) / Inches",
                        "source_cols": ["X_res", "Y_res", "Inches"],
                        "match_rate":  match_rate,
                        "note":        (
                            "'ppi' is a verified derived column (pixels per inch). "
                            "Its 'outliers' reflect real high-DPI screens, not errors. "
                            "Do NOT cap or impute ppi — fix source columns instead."
                        ),
                    })
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-COLUMN LOGIC CHECKER
# ══════════════════════════════════════════════════════════════════════════════

class CrossColumnLogicChecker:
    HEURISTIC_RULES = [
        {
            "rule_name":       "integrated_gpu_but_dedicated_flag",
            "string_col_kw":   ["gpu", "graphics"],
            "string_patterns": [
                r"Intel Iris", r"Intel UHD",
                r"Intel HD Graphics", r"Intel Graphics \(\d+-Cores\)",
            ],
            "bool_col_kw":    ["dedicated", "discrete"],
            "expected_bool":  0,
            "severity":       "HIGH",
            "detail_template": (
                "{n} rows have integrated Intel GPU in '{scol}' but {bcol}=1. "
                "Integrated GPUs (Iris/UHD/HD) are not dedicated. Likely mislabeled."
            ),
            "fix_template": "Set {bcol}=0 where {scol} matches integrated GPU pattern",
        },
        {
            "rule_name":      "storage_type_contradicts_hdd",
            "string_col_kw":  ["storage_type", "storagetype"],
            "string_patterns": [r"^SSD Only$", r"^SSD only$"],
            "bool_col_kw":    ["hhd", "hdd", "hard"],
            "expected_bool":  0,
            "severity":       "MEDIUM",
            "detail_template": (
                "{n} rows have Storage_Type='SSD Only' but {bcol} > 0. "
                "Inconsistent storage metadata."
            ),
            "fix_template": "Set {bcol}=0 where {scol}='SSD Only'",
        },
    ]

    @classmethod
    def check(cls, df: pd.DataFrame, detected_schema: dict) -> list:
        issues    = []
        col_lower = {c: c.lower() for c in df.columns}
        for rule in cls.HEURISTIC_RULES:
            str_col = cls._find_col(df.columns, rule["string_col_kw"], col_lower)
            if str_col is None:
                continue
            if detected_schema.get(str_col) not in ("categorical", "text"):
                continue
            bool_col = cls._find_col(df.columns, rule["bool_col_kw"], col_lower)
            if bool_col is None:
                continue
            pattern_mask = pd.Series(False, index=df.index)
            for pat in rule["string_patterns"]:
                try:
                    pattern_mask |= df[str_col].astype(str).str.contains(
                        pat, na=False, regex=True)
                except Exception:
                    pass
            if not pattern_mask.any():
                continue
            bool_vals     = pd.to_numeric(df[bool_col], errors="coerce")
            conflict_mask = pattern_mask & (
                bool_vals > 0 if rule["expected_bool"] == 0 else bool_vals == 0
            )
            n_conflicts = int(conflict_mask.sum())
            if n_conflicts == 0:
                continue
            issues.append({
                "rule_name":               rule["rule_name"],
                "severity":                rule["severity"],
                "string_column":           str_col,
                "boolean_column":          bool_col,
                "conflict_count":          n_conflicts,
                "conflict_rate":           round(n_conflicts / max(pattern_mask.sum(), 1), 4),
                "detail":                  rule["detail_template"].format(
                                               n=n_conflicts, scol=str_col, bcol=bool_col),
                "fix":                     rule["fix_template"].format(
                                               scol=str_col, bcol=bool_col),
                "sample_conflict_indices": df.index[conflict_mask].tolist()[:5],
            })
        return issues

    @staticmethod
    def _find_col(columns, keywords: list, col_lower: dict):
        for kw in keywords:
            for col in columns:
                if kw in col_lower[col]:
                    return col
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  DUPLICATE ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class DuplicateAnalyzer:
    @classmethod
    def analyze(cls, df: pd.DataFrame) -> dict:
        n                = len(df)
        fully_empty_mask = df.isnull().all(axis=1)
        empty_count      = int(fully_empty_mask.sum())
        df_data          = df[~fully_empty_mask]
        biz_dups         = int(df_data.duplicated().sum())
        biz_rate         = round(biz_dups / n, 4) if n > 0 else 0.0
        dup_sample       = []
        if biz_dups > 0:
            dup_rows = df_data[df_data.duplicated(keep=False)]
            for _, grp in dup_rows.groupby(list(df.columns[:3]), sort=False):
                if len(grp) > 1:
                    dup_sample.append(grp.index.tolist())
                if len(dup_sample) >= 3:
                    break
        return {
            "total_rows":               n,
            "fully_empty_rows":         empty_count,
            "fully_empty_indices":      df.index[fully_empty_mask].tolist(),
            "business_duplicate_count": biz_dups,
            "business_duplicate_rate":  biz_rate,
            "duplicate_sample_indices": dup_sample,
            "note": (
                f"{empty_count} fully-empty row(s) detected. "
                f"Drop first. True business duplicates: {biz_dups}."
            ) if empty_count > 0 else None,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  COLUMN QUALITY SCORER
# ══════════════════════════════════════════════════════════════════════════════

def column_quality_score(info: dict) -> int:
    score     = 100
    null_rate = info.get("null_rate", 0)
    if   null_rate > 0.60: score -= 50
    elif null_rate > 0.20: score -= 25
    elif null_rate > 0.05: score -= 10
    elif null_rate > 0:    score -= 3
    for issue in info.get("issues", []):
        sev = issue.get("severity", "")
        if   sev == "HIGH":   score -= 30
        elif sev == "MEDIUM": score -= 15
        elif sev == "LOW":    score -= 5
    if info.get("zero_as_null", {}).get("zero_likely_null"):
        score -= 15
    return max(0, score)


# ══════════════════════════════════════════════════════════════════════════════
#  CLEANING ACTION RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

class ActionRecommender:
    @staticmethod
    def recommend(col: str, col_type: str, info: dict, cross_col: dict) -> list:
        actions   = []
        null_rate = info.get("null_rate", 0)
        stats     = info.get("stats", {})

        if col_type == "foreign_key":
            if null_rate > 0:
                actions.append({"action": "flag_nulls",
                                 "method": "Add is_missing indicator column. Do NOT impute FK values.",
                                 "priority": "HIGH",
                                 "rationale": "FK nulls indicate referential integrity failure."})
            return actions

        zan = info.get("zero_as_null", {})
        if zan.get("zero_likely_null"):
            actions.append({"action": "replace_zero_with_nan",
                             "method": "df[col].replace(0, np.nan) then impute",
                             "priority": "HIGH",
                             "rationale": zan["recommendation"]})

        if null_rate > 0.60:
            actions.append({"action": "drop_column", "priority": "HIGH",
                             "rationale": f"{null_rate*100:.1f}% nulls exceed 60% threshold"})
        elif null_rate > 0.20:
            if col_type in ("numeric", "ordinal_numeric"):
                actions.append({"action": "impute_nulls",
                                 "method": "median (robust to skew)",
                                 "priority": "MEDIUM", "rationale": f"{null_rate*100:.1f}% nulls"})
            elif col_type in ("categorical", "boolean", "low_cardinality_text"):
                actions.append({"action": "impute_nulls",
                                 "method": "mode or UNKNOWN sentinel",
                                 "priority": "MEDIUM", "rationale": f"{null_rate*100:.1f}% nulls"})
            else:
                actions.append({"action": "flag_nulls",
                                 "method": "add is_missing indicator",
                                 "priority": "MEDIUM",
                                 "rationale": "Datetime imputation is semantically incorrect"})
        elif null_rate > 0:
            method = ("median for skewed, mean for symmetric"
                      if col_type in ("numeric", "ordinal_numeric") else "mode or UNKNOWN sentinel")
            actions.append({"action": "impute_nulls", "method": method,
                             "priority": "LOW", "rationale": f"{null_rate*100:.1f}% nulls"})

        for fmt in info.get("format_analysis", {}).get("format_issues", []):
            t = fmt.get("type", "")
            if t == "unmatched_brackets":
                actions.append({"action": "fix_format_typos",
                                 "method": "str.rstrip(')')",
                                 "priority": "MEDIUM", "rationale": str(fmt.get("detail", ""))})
            elif t == "mixed_separators":
                actions.append({"action": "standardize_separators",
                                 "method": "re.sub(r'\\s*([xX*])\\s*', ' x ', val)",
                                 "priority": "MEDIUM",
                                 "rationale": "Mixed separators cause silent groupby mismatches"})
            elif t == "mixed_naming_conventions":
                actions.append({"action": "standardize_naming_convention",
                                 "method": "Build lookup dict mapping named labels to raw numeric format",
                                 "priority": "MEDIUM", "rationale": str(fmt.get("detail", ""))})

        pa = info.get("pattern_analysis", {})
        if pa.get("has_non_ascii"):
            actions.append({"action": "normalize_unicode",
                             "method": "unicodedata.normalize('NFKC', val)",
                             "priority": "LOW",
                             "rationale": f"{pa.get('non_ascii_count', '?')} non-ASCII values"})

        for issue in info.get("issues", []):
            t = issue.get("type", "")
            if t == "outliers":
                oa     = info.get("outlier_analysis", {})
                narrow = issue.get("narrow_iqr", False)
                if info.get("is_derived_column"):
                    actions.append({"action": "do_not_treat_outliers",
                                     "method": "Fix source columns instead.",
                                     "priority": "INFO", "rationale": "Derived column"})
                elif narrow:
                    actions.append({"action": "investigate_outliers",
                                     "method": "DO NOT cap blindly — narrow IQR, define valid domain range.",
                                     "priority": "MEDIUM",
                                     "rationale": f"Narrow IQR. Consensus={oa.get('consensus_outlier_rate',0)*100:.1f}%"})
                elif oa.get("consensus_outlier_rate", 0) > 0.05:
                    actions.append({"action": "cap_outliers",
                                     "method": "Winsorize at 1st/99th percentile",
                                     "priority": "MEDIUM",
                                     "rationale": f"{oa.get('consensus_outlier_rate',0)*100:.1f}% consensus outliers"})
                else:
                    actions.append({"action": "review_outliers",
                                     "method": "Low consensus — verify domain knowledge before removing",
                                     "priority": "LOW", "rationale": "Only IQR flags these"})
            elif t == "high_skew":
                zr = stats.get("zero_rate", 0)
                actions.append({"action": "consider_transform",
                                 "method": f"{'log1p()' if zr > 0 else 'log() or Box-Cox'}",
                                 "priority": "LOW",
                                 "rationale": f"Skewness = {stats.get('skewness','?')}"})
            elif t == "inconsistent_casing":
                actions.append({"action": "standardize_casing",
                                 "method": "str.strip().str.lower()",
                                 "priority": "LOW",
                                 "rationale": "Case variants create duplicate categories"})
            elif t == "near_constant_categorical":
                actions.append({"action": "evaluate_usefulness",
                                 "method": "Keep if domain-meaningful; drop for ML",
                                 "priority": "LOW",
                                 "rationale": f">{info.get('dominant_rate',0)*100:.0f}% dominant value"})
            elif t == "zero_inflation":
                actions.append({"action": "handle_zero_inflation",
                                 "method": "Create binary indicator has_value = col > 0",
                                 "priority": "MEDIUM",
                                 "rationale": "High zero rate likely means 'not applicable'"})
            elif t == "sentinel_value_detected":
                actions.append({"action": "handle_sentinel_value",
                                 "method": "df[col].replace({'No Data': np.nan}), then impute",
                                 "priority": "MEDIUM",
                                 "rationale": str(issue.get("detail", ""))})
            elif t == "mixed_type_column":
                actions.append({"action": "coerce_or_split_column",
                                 "method": "pd.to_numeric(col, errors='coerce')",
                                 "priority": "HIGH",
                                 "rationale": "Mixed types cause silent pipeline errors"})
            elif t == "numeric_stored_as_string":
                actions.append({"action": "cast_to_numeric",
                                 "method": "pd.to_numeric(df[col], errors='coerce')",
                                 "priority": "HIGH",
                                 "rationale": "Numeric-as-string treated as categories by ML"})
            elif t == "constant_column":
                actions.append({"action": "drop_column",
                                 "method": "df.drop(columns=[col])",
                                 "priority": "MEDIUM",
                                 "rationale": "Zero variance"})
            elif t == "datetime_parse_failures":
                actions.append({"action": "fix_datetime_format",
                                 "method": "Inspect value_counts() of unparseable rows",
                                 "priority": "MEDIUM",
                                 "rationale": "Mixed formats -> silent NaT values"})
            elif t == "low_cardinality_text_warning":
                actions.append({"action": "investigate_text_column",
                                 "method": "Rename and treat as categorical or investigate source.",
                                 "priority": "MEDIUM",
                                 "rationale": "Free-text columns with < 20 unique values are suspicious."})
            elif t == "empty_string_values":
                actions.append({"action": "replace_empty_strings",
                                 "method": "df[col].replace('', np.nan)",
                                 "priority": "LOW",
                                 "rationale": "Empty strings bypass null checks silently"})

        for pair in cross_col.get("possible_derived_columns", []):
            if pair["col_a"] == col or pair["col_b"] == col:
                other = pair["col_b"] if pair["col_a"] == col else pair["col_a"]
                actions.append({"action": "evaluate_drop_redundant",
                                 "method": f"Keep one of ('{col}', '{other}')",
                                 "priority": "MEDIUM",
                                 "rationale": "Redundant — avoid multicollinearity"})
        return actions


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY FLAGS ACTION MATCHER
# ══════════════════════════════════════════════════════════════════════════════

_ISSUE_TO_ACTION_KEYWORDS = {
    "outliers":                        ("outlier", "cap", "investigate", "review", "derived"),
    "high_skew":                       ("transform",),
    "zero_inflation":                  ("zero_inflation",),
    "zero_as_null":                    ("replace_zero",),
    "mixed_negative_values":           ("negative",),
    "inconsistent_casing":             ("casing",),
    "whitespace_issues":               ("whitespace", "strip"),
    "near_constant_categorical":       ("usefulness",),
    "mixed_type_column":               ("coerce", "split"),
    "numeric_stored_as_string":        ("cast",),
    "constant_column":                 ("drop_column",),
    "high_null_rate":                  ("drop_column",),
    "moderate_null_rate":              ("impute",),
    "low_null_rate":                   ("impute",),
    "datetime_parse_failures":         ("datetime", "fix_datetime"),
    "future_dates":                    ("future",),
    "imbalanced_boolean":              ("evaluate",),
    "sentinel_value_detected":         ("sentinel",),
    "format_unmatched_brackets":       ("fix_format", "typo"),
    "format_mixed_separators":         ("separator",),
    "format_mixed_naming_conventions": ("naming", "convention"),
    "non_ascii_characters":            ("unicode", "ascii"),
    "empty_string_values":             ("strip", "empty", "replace_empty"),
    "low_cardinality_text_warning":    ("investigate_text",),
}


def _build_action_type_map(recommended_actions: list) -> dict:
    type_map = {}
    for issue_type, keywords in _ISSUE_TO_ACTION_KEYWORDS.items():
        for action in recommended_actions:
            action_str = (action.get("action", "") + " " + action.get("method", "")).lower()
            if any(kw in action_str for kw in keywords):
                type_map[issue_type] = action
                break
    return type_map


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE RANGE DAYS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_range_days(clean: pd.Series) -> int | None:
    try:
        delta = clean.max() - clean.min()
        return int(delta.days)
    except (OverflowError, Exception):
        try:
            max_dt = clean.max().to_pydatetime()
            min_dt = clean.min().to_pydatetime()
            return (max_dt - min_dt).days
        except Exception:
            return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class DataProfiler:
    def __init__(
        self,
        table_name:            str   = "table",
        null_drop_threshold:   float = 0.60,
        reset_baseline:        bool  = False,
        iqr_outlier_threshold: float = 0.02,
        sample_values_n:       int   = 5,
        sheet_name:            str   = "",
        profiles_dir:          Path | None = None,
        baseline_dir:          Path | None = None,
    ):
        self.table_name            = table_name
        self.sheet_name            = sheet_name
        self.null_drop_threshold   = null_drop_threshold
        self.iqr_outlier_threshold = iqr_outlier_threshold
        self.sample_values_n       = sample_values_n
        self._profiles_dir         = profiles_dir or PROFILES_DIR
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_mgr          = BaselineSchemaManager(
            table_name, sheet_name=sheet_name, baseline_dir=baseline_dir
        )
        if reset_baseline:
            self.baseline_mgr.reset()

    def profile_pandas_df(self, df: pd.DataFrame, source_name: str = "pandas") -> dict:
        return self._build_report(df, source_name)

    def profile_spark_df(self, spark_df, source_name: str = "spark",
                         sample_frac: float = 0.10, max_sample: int = 20_000) -> dict:
        total       = spark_df.count()
        sample_size = min(int(total * sample_frac), max_sample) or 1_000
        pdf         = spark_df.limit(sample_size).toPandas()
        print(f"  [Spark] Sampled {sample_size:,} / {total:,} rows for profiling")
        return self._build_report(pdf, source_name)

    def _build_report(self, df: pd.DataFrame, source_name: str) -> dict:
        n_rows, n_cols  = df.shape
        detected_schema = {col: ColumnTypeDetector.detect(df[col]) for col in df.columns}
        drift           = self.baseline_mgr.detect_drift(detected_schema)

        if not self.baseline_mgr.has_baseline():
            self.baseline_mgr.save(detected_schema)
            print(f"  Baseline schema saved for '{self.table_name}'")

        print("  Computing cross-column relationships...", end=" ", flush=True)
        cross_col = CrossColumnAnalyzer.analyze(df, detected_schema)
        print("done")

        print("  Analyzing duplicates...", end=" ", flush=True)
        dup_analysis = DuplicateAnalyzer.analyze(df)
        print("done")

        print("  Checking cross-column logic consistency...", end=" ", flush=True)
        logic_conflicts = CrossColumnLogicChecker.check(df, detected_schema)
        print("done")

        derived_col_set = set()
        for rel in cross_col.get("additive_relationships", []):
            derived_col_set.add(rel["col_result"])
        for nd in cross_col.get("nonlinear_derived_columns", []):
            derived_col_set.add(nd["col_result"])

        report = {
            "table_name":              self.table_name,
            "source":                  source_name,
            "profiled_at":             _utcnow_iso(),
            "n_rows":                  n_rows,
            "n_columns":               n_cols,
            "detected_schema":         detected_schema,
            "schema_drift":            drift,
            "duplicate_analysis":      dup_analysis,
            "duplicate_rate":          dup_analysis["business_duplicate_rate"],
            "duplicate_count":         dup_analysis["business_duplicate_count"],
            "memory_usage_mb":         round(float(df.memory_usage(deep=True).sum()) / 1e6, 2),
            "cross_column":            cross_col,
            "logic_consistency_issues": logic_conflicts,
            "columns":                 {},
            "global_issues":           [],
            "summary_flags":           [],
            "dataset_quality":         {},
        }

        for col in df.columns:
            col_type = detected_schema[col]
            report["columns"][col] = self._profile_column(
                df[col], col_type, cross_col, is_derived=(col in derived_col_set)
            )

        report["global_issues"]   = self._global_issues(report)
        report["summary_flags"]   = self._build_summary_flags(report)
        report["dataset_quality"] = self._dataset_quality(report)
        return report

    def _profile_column(self, series: pd.Series, col_type: str,
                        cross_col: dict, is_derived: bool = False) -> dict:
        n           = len(series)
        null_count  = int(series.isna().sum())
        null_rate   = round(null_count / n, 4) if n > 0 else 0.0
        clean       = series.dropna()
        sample_vals = [sanitize_for_json(v) for v in clean.head(self.sample_values_n).tolist()]

        info = {
            "detected_type":       col_type,
            "dtype":               str(series.dtype),
            "null_count":          null_count,
            "null_rate":           null_rate,
            "unique_count":        int(series.nunique()),
            "unique_rate":         round(series.nunique() / n, 4) if n > 0 else 0.0,
            "sample_values":       sample_vals,
            "is_derived_column":   is_derived,
            "issues":              [],
            "recommended_actions": [],
        }

        self._add_null_issues(info)

        if info["unique_count"] <= 1 and n > 0:
            info["issues"].append({"type": "constant_column", "severity": "MEDIUM",
                                   "detail": "Column has only one unique value — zero variance"})

        if col_type == "numeric":
            self._profile_numeric(series, info)
        elif col_type == "ordinal_numeric":
            self._profile_ordinal_numeric(series, info)
        elif col_type == "categorical":
            self._profile_categorical(series, info)
        elif col_type in ("datetime", "datetime_string"):
            self._profile_datetime(series, col_type, info)
        elif col_type == "boolean":
            self._profile_boolean(series, info)
        elif col_type == "id":
            info["note"] = "High-cardinality primary key — statistical profiling skipped"
        elif col_type == "foreign_key":
            self._profile_foreign_key(series, info)
        elif col_type == "text":
            self._profile_text(series, info)
        elif col_type == "low_cardinality_text":
            self._profile_low_cardinality_text(series, info)
        elif col_type == "mixed":
            self._profile_mixed(series, info)

        is_string_type = col_type in (
            "categorical", "text", "low_cardinality_text", "mixed"
        ) or (col_type == "id" and not pd.api.types.is_numeric_dtype(series))

        if is_string_type:
            info["pattern_analysis"] = PatternMiner.mine(clean)
            pa = info["pattern_analysis"]
            if (pa.get("whitespace_issues", {}).get("leading_spaces") or
                    pa.get("whitespace_issues", {}).get("trailing_spaces")):
                info["issues"].append({"type": "whitespace_issues", "severity": "LOW",
                                       "detail": "Leading/trailing whitespace detected"})
            if pa.get("numeric_stored_as_string"):
                info["issues"].append({"type": "numeric_stored_as_string", "severity": "HIGH",
                                       "detail": f"{pa['numeric_as_string_rate']*100:.1f}% of values are numeric strings"})
            if pa.get("has_non_ascii") and pa.get("non_ascii_count", 0) > 0:
                info["issues"].append({"type": "non_ascii_characters", "severity": "LOW",
                                       "detail": (f"{pa['non_ascii_count']} values contain non-ASCII. "
                                                  f"Examples: {pa.get('non_ascii_examples', [])}.")})

        if col_type in ("categorical", "text", "low_cardinality_text", "mixed"):
            info["format_analysis"] = FormatConsistencyAnalyzer.analyze(clean)
            if info["format_analysis"]["has_format_issues"]:
                for fmt in info["format_analysis"]["format_issues"]:
                    sev = "LOW" if fmt["type"] == "mixed_naming_conventions" else "MEDIUM"
                    info["issues"].append({"type": f"format_{fmt['type']}",
                                           "severity": sev,
                                           "detail": str(fmt.get("detail", ""))})

        if col_type == "categorical":
            sentinels = detect_sentinel_values(series)
            if sentinels:
                info["sentinel_values"] = sentinels
                info["issues"].append({"type": "sentinel_value_detected", "severity": "MEDIUM",
                                       "detail": (f"Sentinel strings disguising nulls: "
                                                  f"{[s['value'] for s in sentinels]}. Replace with NaN.")})

        if col_type in ("numeric", "ordinal_numeric"):
            zan = ZeroAsNullDetector.check(str(series.name), clean)
            if zan:
                info["zero_as_null"] = zan
                if "zero_as_null" not in {i["type"] for i in info["issues"]}:
                    info["issues"].append({"type": "zero_as_null", "severity": "HIGH",
                                           "detail": zan["recommendation"]})

        info["quality_score"]       = column_quality_score(info)
        info["recommended_actions"] = ActionRecommender.recommend(
            col=str(series.name), col_type=col_type, info=info, cross_col=cross_col
        )
        return info

    def _add_null_issues(self, info: dict) -> None:
        r = info["null_rate"]
        if   r > self.null_drop_threshold:
            info["issues"].append({"type": "high_null_rate",    "severity": "HIGH",   "detail": f"{r*100:.1f}% nulls"})
        elif r > 0.20:
            info["issues"].append({"type": "moderate_null_rate", "severity": "MEDIUM", "detail": f"{r*100:.1f}% nulls"})
        elif r > 0.05:
            info["issues"].append({"type": "low_null_rate",      "severity": "LOW",    "detail": f"{r*100:.1f}% nulls"})

    def _profile_numeric(self, series: pd.Series, info: dict) -> None:
        clean = series.dropna()
        if len(clean) < 4:
            return
        oa        = OutlierAnalyzer.analyze(clean)
        skewness  = round(float(clean.skew()), 4)
        kurt      = round(float(clean.kurtosis()), 4)
        neg_rate  = round(float((clean < 0).mean()), 4)
        zero_rate = round(float((clean == 0).mean()), 4)
        info["stats"] = {
            "mean": round(float(clean.mean()), 4), "median": round(float(clean.median()), 4),
            "std":  round(float(clean.std()),  4), "min":    round(float(clean.min()),    4),
            "max":  round(float(clean.max()),  4), "q1":     round(float(clean.quantile(0.25)), 4),
            "q3":   round(float(clean.quantile(0.75)), 4),
            "p05":  round(float(clean.quantile(0.05)), 4),
            "p95":  round(float(clean.quantile(0.95)), 4),
            "skewness": skewness, "kurtosis": kurt,
            "negative_rate": neg_rate, "zero_rate": zero_rate,
        }
        info["outlier_analysis"] = oa
        iqr_rate  = oa.get("iqr_outlier_rate", 0)
        consensus = oa.get("consensus_outlier_rate", 0)
        narrow    = oa.get("narrow_iqr", False)
        if iqr_rate > self.iqr_outlier_threshold:
            severity   = "LOW" if info.get("is_derived_column") else ("HIGH" if iqr_rate >= 0.10 else "MEDIUM")
            extra_note = " — DERIVED COLUMN: fix source columns instead." if info.get("is_derived_column") else ""
            detail     = (f"IQR outliers: {iqr_rate*100:.1f}%  "
                          f"Z-score: {oa.get('zscore_outlier_rate',0)*100:.1f}%  "
                          f"Consensus: {consensus*100:.1f}%")
            if narrow:
                detail += f"  NOTE: IQR is only {oa['iqr_to_range_ratio']*100:.1f}% of full range."
            info["issues"].append({"type": "outliers", "severity": severity,
                                   "detail": detail + extra_note, "narrow_iqr": narrow})
        if abs(skewness) >= 2.0:
            info["issues"].append({"type": "high_skew", "severity": "MEDIUM",
                                   "detail": (f"Skewness = {skewness}  "
                                              f"({'right' if skewness > 0 else 'left'}-skewed).")})
        if zero_rate > 0.50:
            info["issues"].append({"type": "zero_inflation", "severity": "MEDIUM",
                                   "detail": f"{zero_rate*100:.1f}% zeros — may encode 'not applicable'."})
        if 0 < neg_rate < 1.0:
            info["issues"].append({"type": "mixed_negative_values", "severity": "MEDIUM",
                                   "detail": f"{neg_rate*100:.1f}% negative — verify domain allows negatives"})

    def _profile_ordinal_numeric(self, series: pd.Series, info: dict) -> None:
        clean     = series.dropna()
        vc        = clean.value_counts().sort_index()
        zero_rate = round(float((clean == 0).mean()), 4)
        info["value_distribution"] = {str(k): int(v) for k, v in vc.items()}
        if len(clean) >= 2:
            info["summary_stats"] = {
                "mean": round(float(clean.mean()), 4), "median": round(float(clean.median()), 4),
                "min":  round(float(clean.min()),  4), "max":    round(float(clean.max()),    4),
                "unique_count": int(clean.nunique()), "zero_rate": zero_rate,
            }
        if zero_rate > 0.50:
            info["issues"].append({"type": "zero_inflation", "severity": "MEDIUM",
                                   "detail": f"{zero_rate*100:.1f}% zeros — likely 'none/not applicable'."})
        # FIX 7: Detect extreme ratio outliers in ordinal columns.
        # e.g. amount_changed with values {0, 10, 10000} — the 10000 is 1000x normal.
        # Ordinal columns skip OutlierAnalyzer, so we add a ratio check here.
        if clean.nunique() >= 2:
            import numpy as np
            arr          = clean.values.astype(float)
            nonzero_vals = arr[arr > 0]
            if len(nonzero_vals) >= 2:
                median_nz = float(np.median(nonzero_vals))
                max_val   = float(arr.max())
                if median_nz > 0 and max_val / median_nz > 20:
                    info["issues"].append({
                        "type":     "outliers",
                        "severity": "HIGH",
                        "detail":   (
                            f"Extreme value detected: max={round(max_val,2)} is "
                            f"{round(max_val/median_nz,1)}x the median "
                            f"({round(median_nz,2)}). Likely data entry error or "
                            f"sentinel value. Check and replace if invalid."
                        ),
                        "narrow_iqr": False,
                    })

    def _profile_categorical(self, series: pd.Series, info: dict) -> None:
        clean = series.dropna().astype(str)
        if len(clean) == 0:
            return
        vc           = clean.value_counts()
        unique_lower = clean.str.strip().str.lower().nunique()
        unique_raw   = clean.nunique()
        unique_strip = clean.str.strip().nunique()
        info["top_values"]     = {str(k): int(v) for k, v in vc.head(15).items()}
        info["dominant_value"] = str(vc.index[0])
        info["dominant_rate"]  = round(float(vc.iloc[0] / len(clean)), 4)
        if unique_raw > 5:
            info["bottom_values"] = {str(k): int(v) for k, v in vc.tail(5).items()}
        probs           = (vc / len(clean)).values
        info["entropy"] = round(float(-np.sum(probs * np.log2(probs + 1e-12))), 4)
        if unique_raw > unique_lower:
            info["issues"].append({"type": "inconsistent_casing", "severity": "LOW",
                                   "detail": f"{unique_raw} raw unique values -> {unique_lower} after strip+lowercase."})
        elif unique_strip < unique_raw:
            info["issues"].append({"type": "whitespace_issues", "severity": "LOW",
                                   "detail": f"{unique_raw - unique_strip} extra categories from whitespace"})
        if info["dominant_rate"] > 0.95:
            info["issues"].append({"type": "near_constant_categorical", "severity": "LOW",
                                   "detail": f"'{info['dominant_value']}' = {info['dominant_rate']*100:.1f}%. Near-constant."})

    def _profile_foreign_key(self, series: pd.Series, info: dict) -> None:
        clean = series.dropna()
        vc    = clean.value_counts()
        info["fk_stats"] = {
            "unique_referenced_ids": int(clean.nunique()),
            "rows_per_id_mean":      round(float(len(clean) / max(clean.nunique(), 1)), 2),
            "rows_per_id_max":       int(vc.iloc[0]) if len(vc) > 0 else 0,
            "rows_per_id_min":       int(vc.iloc[-1]) if len(vc) > 0 else 0,
            "top_referenced":        {str(k): int(v) for k, v in vc.head(5).items()},
        }
        info["note"] = (f"Foreign key column — references {clean.nunique()} unique IDs. "
                        "Do NOT impute, transform, or treat as a numeric feature.")

    def _profile_low_cardinality_text(self, series: pd.Series, info: dict) -> None:
        clean = series.dropna().astype(str)
        vc    = clean.value_counts()
        info["top_values"]     = {str(k): int(v) for k, v in vc.head(15).items()}
        info["dominant_value"] = str(vc.index[0]) if len(vc) > 0 else ""
        info["dominant_rate"]  = round(float(vc.iloc[0] / len(clean)), 4) if len(clean) > 0 else 0
        probs           = (vc / len(clean)).values
        info["entropy"] = round(float(-np.sum(probs * np.log2(probs + 1e-12))), 4)
        info["issues"].append({"type": "low_cardinality_text_warning", "severity": "MEDIUM",
                               "detail": (f"Column name suggests free text but only {series.nunique()} unique values. "
                                          "Likely a label/category.")})

    def _profile_datetime(self, series: pd.Series, col_type: str, info: dict) -> None:
        try:
            if col_type == "datetime_string":
                try:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                except TypeError:
                    parsed = pd.to_datetime(series, errors="coerce")
            else:
                parsed = series
        except Exception:
            parsed = pd.to_datetime(series, errors="coerce")

        clean = parsed.dropna()
        if len(clean) == 0:
            return
        pfr = round(float(parsed.isna().sum()) / max(len(series), 1), 4)
        info["datetime_stats"] = {
            "min": str(clean.min()), "max": str(clean.max()),
            "range_days": _safe_range_days(clean), "parse_fail_rate": pfr,
        }
        if pfr > 0.05:
            info["issues"].append({"type": "datetime_parse_failures",
                                   "severity": "MEDIUM" if pfr > 0.10 else "LOW",
                                   "detail": f"{pfr*100:.1f}% values failed datetime parse"})
        now = pd.Timestamp.now()
        try:
            clean_naive = clean.dt.tz_localize(None) if clean.dt.tz is not None else clean
            future_rate = round(float((clean_naive > now).mean()), 4)
            if future_rate > 0.01:
                info["issues"].append({"type": "future_dates", "severity": "LOW",
                                       "detail": f"{future_rate*100:.1f}% values are in the future"})
        except Exception:
            pass

    def _profile_boolean(self, series: pd.Series, info: dict) -> None:
        clean = series.dropna()
        vc    = clean.value_counts(normalize=True)
        info["boolean_distribution"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        max_rate = float(vc.iloc[0])
        if max_rate > 0.95:
            info["issues"].append({"type": "imbalanced_boolean", "severity": "LOW",
                                   "detail": f"'{vc.index[0]}' = {max_rate*100:.1f}% — near-constant."})
        if len(vc) > 1:
            minority_rate = float(vc.iloc[-1])
            if 0.05 < minority_rate < 0.40:
                info["boolean_imbalance_note"] = f"Minority class = {minority_rate*100:.1f}%"

    def _profile_text(self, series: pd.Series, info: dict) -> None:
        clean   = series.dropna().astype(str)
        lengths = clean.str.len()
        words   = clean.str.split().str.len()
        info["text_stats"] = {
            "avg_char_length": round(float(lengths.mean()), 1),
            "min_char_length": int(lengths.min()),
            "max_char_length": int(lengths.max()),
            "avg_word_count":  round(float(words.mean()), 1) if words.notna().any() else None,
        }
        empty_rate = round(float((clean.str.strip() == "").mean()), 4)
        if empty_rate > 0.01:
            info["issues"].append({"type": "empty_string_values", "severity": "LOW",
                                   "detail": f"{empty_rate*100:.1f}% empty/whitespace strings"})

    def _profile_mixed(self, series: pd.Series, info: dict) -> None:
        type_counts = Counter(type(v).__name__ for v in series.dropna().head(200))
        info["type_mix"] = dict(type_counts)
        info["issues"].append({"type": "mixed_type_column", "severity": "HIGH",
                               "detail": f"Multiple Python types: {dict(type_counts)}."})

    def _global_issues(self, report: dict) -> list:
        issues = []
        dup    = report["duplicate_analysis"]
        if dup["fully_empty_rows"] > 0:
            issues.append({"type": "trailing_empty_rows", "severity": "HIGH",
                           "detail": (f"{dup['fully_empty_rows']} fully-empty row(s). Drop before analysis."),
                           "recommended_action": "df.dropna(how='all', inplace=True)"})
        if dup["business_duplicate_count"] > 0:
            rate = dup["business_duplicate_rate"]
            issues.append({"type": "business_duplicate_rows",
                           "severity": "HIGH" if rate > 0.05 else "LOW",
                           "detail": f"{dup['business_duplicate_count']} duplicate rows ({rate*100:.2f}%).",
                           "recommended_action": "df.drop_duplicates(keep='first')"})
        drift = report["schema_drift"]
        if drift.get("has_drift"):
            for col in drift.get("removed_columns", []):
                issues.append({"type": "schema_drift_removed_columns", "severity": "HIGH",
                               "detail": f"Column '{col}' in baseline but missing from current data"})
            for col in drift.get("new_columns", []):
                issues.append({"type": "schema_drift_new_columns", "severity": "MEDIUM",
                               "detail": f"New column '{col}' not in baseline"})
            for tc in drift.get("type_changes", []):
                issues.append({"type": "schema_drift_type_change", "severity": "HIGH",
                               "detail": f"Column '{tc['column']}' type: {tc['was']} -> {tc['now']}"})
        for group in report["cross_column"].get("correlated_null_groups", []):
            issues.append({"type": "correlated_null_group", "severity": "MEDIUM",
                           "detail": f"Columns {group['columns']} have correlated nulls. {group['interpretation']}"})
        for pair in report["cross_column"].get("high_correlation_pairs", [])[:5]:
            issues.append({"type": "high_correlation_pair", "severity": "LOW",
                           "detail": f"'{pair['col_a']}' <-> '{pair['col_b']}' Pearson r={pair['pearson_corr']}."})
        for pair in report["cross_column"].get("possible_derived_columns", []):
            issues.append({"type": "derived_column_detected", "severity": "MEDIUM",
                           "detail": pair["recommendation"]})
        for rel in report["cross_column"].get("additive_relationships", []):
            issues.append({"type": "additive_relationship_detected", "severity": "INFO",
                           "detail": f"Formula: {rel['formula']} (match rate: {rel['match_rate']*100:.1f}%). {rel['note']}"})
        for nd in report["cross_column"].get("nonlinear_derived_columns", []):
            issues.append({"type": "nonlinear_derived_column", "severity": "INFO",
                           "detail": f"'{nd['col_result']}' = {nd['formula']} ({nd['match_rate']*100:.1f}%). {nd['note']}"})
        for conflict in report.get("logic_consistency_issues", []):
            issues.append({"type": f"logic_conflict_{conflict['rule_name']}",
                           "severity": conflict["severity"], "detail": conflict["detail"],
                           "fix": conflict["fix"], "affected_rows": conflict["conflict_count"],
                           "sample_indices": conflict["sample_conflict_indices"]})
        return issues

    def _build_summary_flags(self, report: dict) -> list:
        flags = []
        for col, info in report["columns"].items():
            action_map   = _build_action_type_map(info.get("recommended_actions", []))
            first_action = info["recommended_actions"][0] if info.get("recommended_actions") else None
            for issue in info.get("issues", []):
                issue_type = issue.get("type", "")
                flag = {"column": col, "detected_type": info.get("detected_type"),
                        "null_rate": info.get("null_rate"),
                        "quality_score": info.get("quality_score"),
                        "is_derived": info.get("is_derived_column", False),
                        **issue}
                if stats := info.get("stats"):
                    flag.update({"skewness": stats.get("skewness"), "zero_rate": stats.get("zero_rate"),
                                 "mean": stats.get("mean"), "std": stats.get("std")})
                if oa := info.get("outlier_analysis"):
                    flag.update({"iqr_outlier_rate": oa.get("iqr_outlier_rate"),
                                 "consensus_outlier_rate": oa.get("consensus_outlier_rate"),
                                 "iqr_to_range_ratio": oa.get("iqr_to_range_ratio"),
                                 "outlier_sample": oa.get("outlier_sample")})
                if "dominant_rate" in info:
                    flag.update({"dominant_value": info.get("dominant_value"),
                                 "dominant_rate": info.get("dominant_rate")})
                flag["recommended_action"] = action_map.get(issue_type, first_action)
                flags.append(flag)
        for gi in report.get("global_issues", []):
            flags.append({"column": "__global__", **gi})
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
        flags.sort(key=lambda f: severity_order.get(f.get("severity", "INFO"), 9))
        return flags

    def _dataset_quality(self, report: dict) -> dict:
        col_scores = [info["quality_score"] for info in report["columns"].values()]
        avg_score  = round(float(np.mean(col_scores)), 1) if col_scores else 0
        n_high     = sum(1 for f in report["summary_flags"] if f.get("severity") == "HIGH")
        n_medium   = sum(1 for f in report["summary_flags"] if f.get("severity") == "MEDIUM")
        n_low      = sum(1 for f in report["summary_flags"] if f.get("severity") == "LOW")
        worst_cols = sorted(
            [(col, info["quality_score"]) for col, info in report["columns"].items()],
            key=lambda x: x[1])[:5]
        readiness = avg_score - n_high * 5 - report["duplicate_analysis"]["business_duplicate_rate"] * 50
        if report["duplicate_analysis"]["fully_empty_rows"] > 0:
            readiness -= 3
        readiness = max(0, min(100, round(readiness, 1)))
        return {
            "avg_column_quality_score": avg_score,
            "min_column_quality_score": min(col_scores) if col_scores else 0,
            "dataset_readiness_score":  readiness,
            "readiness_interpretation": ("READY" if readiness >= 80 else
                                         "NEEDS_CLEANING" if readiness >= 50 else "SEVERELY_DEGRADED"),
            "total_issues":            len(report["summary_flags"]),
            "high_issues":             n_high,
            "medium_issues":           n_medium,
            "low_issues":              n_low,
            "worst_columns":           [{"column": c, "score": s} for c, s in worst_cols],
            "cleaning_priority_order": [c for c, _ in worst_cols],
        }

    def save_report(self, report: dict, timestamp: str = "") -> Path:
        ts        = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sheet_sfx = f"__{self.sheet_name}" if self.sheet_name else ""
        filename  = f"{self.table_name}{sheet_sfx}_profile_{ts}.json"
        path      = self._profiles_dir / filename
        clean     = sanitize_for_json(report)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        latest = self._profiles_dir / f"{self.table_name}{sheet_sfx}_latest_profile.json"
        with open(latest, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        return path


# ══════════════════════════════════════════════════════════════════════════════
#  PRETTY PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    SEV = {"HIGH": "HIGH", "MEDIUM": "MEDIUM", "LOW": "LOW", "INFO": "INFO"}
    print("\n" + "=" * 72)
    print(f"  DATA PROFILING REPORT -- {report['table_name']} / {report['source']}")
    print("=" * 72)
    print(f"  Rows      : {report['n_rows']:,}   Columns: {report['n_columns']}   Memory: {report['memory_usage_mb']} MB")
    print(f"  Profiled  : {report['profiled_at']}")
    dup = report.get("duplicate_analysis", {})
    if dup.get("fully_empty_rows", 0) > 0:
        print(f"  Empty rows: {dup['fully_empty_rows']} fully-empty row(s) -- drop first")
    print(f"  Duplicates: {dup.get('business_duplicate_count', 0)} business duplicates ({dup.get('business_duplicate_rate', 0)*100:.2f}%)")
    drift = report["schema_drift"]
    print("  Schema    : " + (
        "Baseline saved (first run)" if drift["status"] == "no_baseline" else
        "DRIFT DETECTED"            if drift["has_drift"]                 else
        "Matches baseline"
    ))
    dq = report.get("dataset_quality", {})
    print(f"  Readiness : {dq.get('dataset_readiness_score','?')}/100  [{dq.get('readiness_interpretation','')}]")
    cc = report.get("cross_column", {})
    if cc.get("correlated_null_groups"):    print(f"  Corr.Null : {len(cc['correlated_null_groups'])} group(s)")
    if cc.get("high_correlation_pairs"):    print(f"  High Corr : {len(cc['high_correlation_pairs'])} pair(s)")
    if cc.get("additive_relationships"):    print(f"  Additive  : {len(cc['additive_relationships'])} formula(s) verified")
    if cc.get("nonlinear_derived_columns"): print(f"  Derived   : {len(cc['nonlinear_derived_columns'])} nonlinear derivation(s)")
    lc = report.get("logic_consistency_issues", [])
    if lc:
        print(f"  Logic     : {len(lc)} conflict(s) -- {sum(c['conflict_count'] for c in lc)} affected rows")
    print(f"\n  {'Column':<24} {'Type':<20} {'Null%':>6}  {'Score':>5}  Issues")
    print("  " + "-" * 72)
    for col, info in report["columns"].items():
        null_pct  = info["null_rate"] * 100
        score     = info.get("quality_score", "?")
        derived   = "[derived] " if info.get("is_derived_column") else ""
        null_bar  = "HIGH" if info["null_rate"] > 0.40 else "MED" if info["null_rate"] > 0.20 else "OK"
        issues_str = " | ".join(i["type"] for i in info.get("issues", [])) or "--"
        print(f"  [{null_bar}] {derived}{col:<20} {info['detected_type']:<20} {null_pct:>5.1f}%  {score:>5}  {issues_str[:50]}")
        if oa := info.get("outlier_analysis"):
            if oa.get("iqr_outlier_rate", 0) > 0:
                narrow = " [narrow IQR]" if oa.get("narrow_iqr") else ""
                print(f"       IQR={oa['iqr_outlier_rate']*100:.1f}%{narrow}  Z={oa['zscore_outlier_rate']*100:.1f}%  Consensus={oa['consensus_outlier_rate']*100:.1f}%")
    if report["global_issues"]:
        print("\n  Global issues:")
        for gi in report["global_issues"]:
            print(f"    [{gi.get('severity','?')}] {gi.get('type','?')}: {str(gi.get('detail',''))[:110]}")
    print(f"\n  Summary flags: {dq.get('total_issues',0)} total  ({dq.get('high_issues',0)} HIGH  {dq.get('medium_issues',0)} MEDIUM  {dq.get('low_issues',0)} LOW)")
    if dq.get("worst_columns"):
        print("  Worst columns (cleaning priority):")
        for entry in dq["worst_columns"]:
            print(f"    - {entry['column']:<28} score={entry['score']}")
    print("=" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(description="Data Profiler v7 -- LLM-ready data quality report")
    p.add_argument("--csv",               type=str,   help="Path to input file (CSV/Excel/JSON/Parquet/TSV)")
    p.add_argument("--table",             type=str,   default=None)
    p.add_argument("--sheet",             type=str,   default=None)
    p.add_argument("--null-threshold",    type=float, default=0.60)
    p.add_argument("--outlier-threshold", type=float, default=0.02)
    p.add_argument("--sample-n",          type=int,   default=5)
    p.add_argument("--reset-baseline",    action="store_true")
    p.add_argument("--no-save",           action="store_true")
    p.add_argument("--output-dir",        type=str,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.csv:
        file_path = Path(args.csv)
    else:
        mock_dir = BASE_DIR / "mock_data"
        csvs     = list(mock_dir.glob("*.*")) if mock_dir.exists() else []
        if not csvs:
            print("No file found. Usage: python profiler.py --csv path/to/file")
            raise SystemExit(1)
        file_path = csvs[0]
        print(f"  Auto-selected: {file_path}")

    if not file_path.exists():
        print(f"File not found: {file_path}")
        raise SystemExit(1)

    table_name   = args.table or file_path.stem
    profiles_dir = Path(args.output_dir) if args.output_dir else None
    loaded       = load_file(file_path)

    if "all_sheets" in loaded and len(loaded["all_sheets"]) > 1:
        all_sheets = loaded["all_sheets"]
        if args.sheet:
            if args.sheet not in all_sheets:
                print(f"  Sheet '{args.sheet}' not found. Available: {list(all_sheets.keys())}")
                raise SystemExit(1)
            all_sheets = {args.sheet: all_sheets[args.sheet]}

        print(f"\n  Loaded '{file_path.name}' -- {len(all_sheets)} sheet(s) to profile")
        shared_ts   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        all_reports = {}

        for sheet_name, df in all_sheets.items():
            print(f"\n  --- Sheet: '{sheet_name}' ({len(df):,} rows x {len(df.columns)} cols) ---")
            profiler = DataProfiler(
                table_name=table_name, sheet_name=sheet_name,
                null_drop_threshold=args.null_threshold,
                iqr_outlier_threshold=args.outlier_threshold,
                reset_baseline=args.reset_baseline, sample_values_n=args.sample_n,
                profiles_dir=profiles_dir,
            )
            report = profiler.profile_pandas_df(df, source_name=f"{file_path.name}[{sheet_name}]")
            all_reports[sheet_name] = report
            print_report(report)
            if not args.no_save:
                saved = profiler.save_report(report, timestamp=shared_ts)
                print(f"  Report saved -> {saved}\n")

        print("\n" + "=" * 72 + "\n  MULTI-SHEET SUMMARY\n" + "=" * 72)
        for sn, rep in all_reports.items():
            dq = rep.get("dataset_quality", {})
            print(f"  [{sn}] {rep['n_rows']:,}r x {rep['n_columns']}c  "
                  f"Readiness={dq.get('dataset_readiness_score','?')} "
                  f"Issues: {dq.get('high_issues',0)} HIGH  {dq.get('medium_issues',0)} MED  {dq.get('low_issues',0)} LOW")
        print("=" * 72 + "\n")

    else:
        df         = loaded["data"]
        sheet_name = ""
        if "sheet_names" in loaded:
            sheet_name = loaded["sheet_names"][0]

        print(f"\n  Loaded '{file_path.name}' -> {len(df):,} rows x {len(df.columns)} columns")
        profiler = DataProfiler(
            table_name=table_name, sheet_name=sheet_name,
            null_drop_threshold=args.null_threshold,
            iqr_outlier_threshold=args.outlier_threshold,
            reset_baseline=args.reset_baseline, sample_values_n=args.sample_n,
            profiles_dir=profiles_dir,
        )
        report = profiler.profile_pandas_df(df, source_name=file_path.name)
        print_report(report)
        if not args.no_save:
            saved = profiler.save_report(report)
            print(f"  Report saved -> {saved}")
            print(f"  Latest copy  -> {profiler._profiles_dir}/{table_name}_latest_profile.json\n")
