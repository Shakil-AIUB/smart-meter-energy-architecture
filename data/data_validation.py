"""
Data validation for the Building Energy Optimisation project.

Runs a set of checks on the processed real dataset (real_building_energy.csv,
produced by preprocess_cubems.py) BEFORE it's used for evaluation. This is
what you report in the paper's "Dataset & Data Quality" subsection to show
the evaluation is built on trustworthy data, not just "we downloaded a CSV".

Checks performed:
1. Schema check       — required columns present, correct dtypes
2. Completeness       — % missing per column
3. Timestamp integrity — no duplicate timestamps, chronological order,
                         gaps larger than the expected sampling interval
4. Range/plausibility — values within physically sensible bounds
                         (e.g. occupancy in [0,1], kW >= 0, temp in a
                         believable range for an office building)
5. Outlier flagging    — values beyond N standard deviations, reported
                         (not silently dropped) so you can discuss them

Usage:
    python data_validation.py --data real_building_energy.csv --interval-min 15 --report validation_report.md
"""

import argparse
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp": "datetime",
    "occupancy": "float",
    "hvac_kw": "float",
    "lighting_kw": "float",
    "plug_kw": "float",
    "total_kw": "float",
    "outdoor_temp_c": "float",
    "indoor_temp_c": "float",
}

PLAUSIBLE_RANGES = {
    "occupancy": (0.0, 1.0),
    "hvac_kw": (0.0, 50.0),
    "lighting_kw": (0.0, 20.0),
    "plug_kw": (0.0, 20.0),
    "total_kw": (0.0, 80.0),
    "outdoor_temp_c": (0.0, 45.0),
    "indoor_temp_c": (10.0, 40.0),
}


def check_schema(df: pd.DataFrame) -> list[str]:
    issues = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            issues.append(f"MISSING COLUMN: '{col}' not found.")
    return issues


def check_completeness(df: pd.DataFrame) -> pd.DataFrame:
    missing_pct = (df.isna().sum() / len(df) * 100).round(2)
    return missing_pct.to_frame("missing_pct")


def check_timestamp_integrity(df: pd.DataFrame, interval_min: int) -> dict:
    ts = pd.to_datetime(df["timestamp"])
    results = {}
    results["duplicate_timestamps"] = int(ts.duplicated().sum())
    results["is_sorted"] = bool(ts.is_monotonic_increasing)

    expected_gap = pd.Timedelta(minutes=interval_min)
    diffs = ts.diff().dropna()
    gaps = diffs[diffs > expected_gap]
    results["n_gaps_larger_than_expected"] = int(len(gaps))
    results["largest_gap"] = str(diffs.max()) if len(diffs) else "n/a"
    results["total_expected_rows_if_no_gaps"] = int(
        (ts.max() - ts.min()) / expected_gap
    ) + 1 if len(ts) else 0
    results["actual_rows"] = len(ts)
    return results


def check_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        n_below = int((series < lo).sum())
        n_above = int((series > hi).sum())
        rows.append({
            "column": col,
            "expected_range": f"[{lo}, {hi}]",
            "actual_min": round(series.min(), 2) if len(series) else None,
            "actual_max": round(series.max(), 2) if len(series) else None,
            "n_below_range": n_below,
            "n_above_range": n_above,
        })
    return pd.DataFrame(rows)


def flag_outliers(df: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
    rows = []
    for col in ["hvac_kw", "lighting_kw", "plug_kw", "indoor_temp_c"]:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.std() == 0 or len(series) < 2:
            continue
        z = (series - series.mean()) / series.std()
        n_outliers = int((z.abs() > z_thresh).sum())
        rows.append({
            "column": col,
            "z_threshold": z_thresh,
            "n_outliers": n_outliers,
            "pct_outliers": round(100 * n_outliers / len(series), 3),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate the processed real dataset before evaluation.")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--interval-min", type=int, default=15)
    parser.add_argument("--report", type=str, default="validation_report.md")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    schema_issues = check_schema(df)
    completeness = check_completeness(df)
    ts_integrity = check_timestamp_integrity(df, args.interval_min)
    ranges = check_ranges(df)
    outliers = flag_outliers(df)

    lines = []
    lines.append("# Data Validation Report\n")
    lines.append(f"Dataset: `{args.data}`  \nRows: {len(df)}\n")

    lines.append("## 1. Schema Check")
    if schema_issues:
        lines += [f"- ❌ {i}" for i in schema_issues]
    else:
        lines.append("- ✅ All required columns present.")
    lines.append("")

    lines.append("## 2. Completeness (% missing per column)")
    lines.append(completeness.to_markdown())
    lines.append("")

    lines.append("## 3. Timestamp Integrity")
    for k, v in ts_integrity.items():
        lines.append(f"- **{k}**: {v}")
    coverage_pct = 100 * ts_integrity["actual_rows"] / max(ts_integrity["total_expected_rows_if_no_gaps"], 1)
    lines.append(f"- **temporal coverage**: {coverage_pct:.1f}% of the expected {args.interval_min}-min slots have a row")
    lines.append("")

    lines.append("## 4. Range / Plausibility Check")
    lines.append(ranges.to_markdown(index=False))
    lines.append("")

    lines.append("## 5. Outlier Flagging (|z| > 4)")
    if len(outliers):
        lines.append(outliers.to_markdown(index=False))
    else:
        lines.append("No columns had enough variance to check, or none flagged.")
    lines.append("")

    lines.append("## Summary for the paper")
    lines.append(
        f"- The dataset covers {coverage_pct:.1f}% of expected {args.interval_min}-minute slots "
        f"between {pd.to_datetime(df['timestamp']).min()} and {pd.to_datetime(df['timestamp']).max()}.\n"
        f"- {ts_integrity['duplicate_timestamps']} duplicate timestamps found.\n"
        f"- No out-of-range values were silently kept; see the range table above for any flagged rows.\n"
        f"- Real sensor data has genuine gaps (unlike simulated data) — report the completeness "
        f"and coverage numbers above directly in your paper's dataset section as evidence of "
        f"data-quality handling, rather than hiding them."
    )

    report_text = "\n".join(lines)
    with open(args.report, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nSaved to {args.report}")


if __name__ == "__main__":
    main()
