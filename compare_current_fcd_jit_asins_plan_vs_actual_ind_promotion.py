"""Compare origin-time planned vs eval-time actual ind_promotion for one FCD.

This script is intentionally independent of model training.  It uses only the
ASINs in the supplied JIT cohort CSV and fixes the rolling dates to the current
experiment by default:

    data cut / plan snapshot: 2025-10-04
    FCD / H1:                 2025-10-05
    H2:                       2025-10-12
    H3:                       2025-10-19
    evaluation snapshot:      2025-10-25

For each ASIN and target week, planned_ind_promotion is 1 when at least one
origin-time deal interval covers that week; otherwise it is 0.  The choice
between multiple overlapping deals is irrelevant for this binary feature.
"""

from __future__ import annotations

import argparse
import io
import re
from pathlib import Path

import boto3
import numpy as np
import pandas as pd


S3_BUCKET = "amxl-asin-forecast590184089576"
MODEL_DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)
DEALS_PREFIX = "amxl-asin-forecast-intern/asin_deals/"

DEFAULT_DATA_CUT = "2025-10-04"
DEFAULT_ASIN_FILE_GLOB = "jit_asins_top90pct_demand*.csv"


def _normalize_asin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _find_asin_column(df: pd.DataFrame) -> str:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    if "asin" not in by_lower:
        raise KeyError(
            "The cohort CSV must contain an ASIN column. "
            f"Available columns: {list(df.columns)}"
        )
    return by_lower["asin"]


def _resolve_asin_csv(path: str | Path | None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"ASIN cohort CSV not found: {resolved}")
        return resolved

    matches = sorted(Path.cwd().glob(DEFAULT_ASIN_FILE_GLOB))
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise FileNotFoundError(
            "No JIT ASIN cohort CSV was found in the current directory. "
            "Pass asin_csv_path explicitly."
        )
    raise RuntimeError(
        "Multiple candidate JIT ASIN CSV files were found; pass "
        "asin_csv_path explicitly:\n  "
        + "\n  ".join(str(p.resolve()) for p in matches)
    )


def _list_s3_keys(bucket: str, prefix: str, s3_client) -> list[str]:
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)
    return keys


def _read_s3_csv(bucket: str, key: str, s3_client) -> pd.DataFrame:
    print(f"Reading s3://{bucket}/{key}", flush=True)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    df = pd.read_csv(io.BytesIO(body))
    print(f"  rows={len(df):,}, columns={len(df.columns):,}", flush=True)
    return df


def _find_eval_snapshot_key(
    data_cut: pd.Timestamp,
    bucket: str,
    data_prefix: str,
    s3_client,
) -> tuple[pd.Timestamp, str]:
    eval_cut = data_cut + pd.Timedelta(days=21)
    token = eval_cut.strftime("%Y-%m-%d")
    prefix = f"{data_prefix}{token}"
    pattern = re.compile(
        rf"df_head_body_add_holiday_{re.escape(token)}_?ETLM_[vV]3\.csv$"
    )
    matches = [
        key for key in _list_s3_keys(bucket, prefix, s3_client)
        if pattern.search(key)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one eval snapshot for {eval_cut.date()}, "
            f"found {len(matches)}: {matches}"
        )
    return eval_cut, matches[0]


def _find_deals_snapshot_key(
    data_cut: pd.Timestamp,
    bucket: str,
    deals_prefix: str,
    s3_client,
) -> str:
    token = data_cut.strftime("%Y%m%d")
    expected = f"{deals_prefix.rstrip('/')}/asin_deals_{token}.csv000"
    # Use the exact origin-time filename when it exists.  Listing also gives a
    # readable error if the S3 export used a slightly different suffix.
    keys = _list_s3_keys(
        bucket,
        f"{deals_prefix.rstrip('/')}/asin_deals_{token}",
        s3_client,
    )
    if expected in keys:
        return expected
    if len(keys) == 1:
        print(
            f"Expected {expected}, using the only matching key instead: {keys[0]}",
            flush=True,
        )
        return keys[0]
    raise RuntimeError(
        f"Could not uniquely resolve the deals snapshot for {data_cut.date()}. "
        f"Expected {expected}; matching keys={keys}"
    )


def _build_target_grid(
    asins: list[str],
    fcd: pd.Timestamp,
) -> pd.DataFrame:
    target_weeks = [
        fcd,
        fcd + pd.Timedelta(days=7),
        fcd + pd.Timedelta(days=14),
    ]
    grid = pd.MultiIndex.from_product(
        [asins, target_weeks],
        names=["asin", "order_week"],
    ).to_frame(index=False)
    horizon_map = {week: f"H{i + 1}" for i, week in enumerate(target_weeks)}
    grid["forecast_horizon"] = grid["order_week"].map(horizon_map)
    return grid


def _build_planned_ind_promotion(
    target_grid: pd.DataFrame,
    deals_raw: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "asin",
        "asin_promo_start_week",
        "asin_promo_end_week",
    }
    missing = sorted(required - set(deals_raw.columns))
    if missing:
        raise KeyError(
            "Deals snapshot is missing required columns: " + ", ".join(missing)
        )

    deals = deals_raw.copy()
    deals["asin"] = _normalize_asin(deals["asin"])
    deals["asin_promo_start_week"] = pd.to_datetime(
        deals["asin_promo_start_week"], errors="coerce"
    ).dt.normalize()
    deals["asin_promo_end_week"] = pd.to_datetime(
        deals["asin_promo_end_week"], errors="coerce"
    ).dt.normalize()

    selected_asins = set(target_grid["asin"])
    min_week = target_grid["order_week"].min()
    max_week = target_grid["order_week"].max()
    deals = deals[
        deals["asin"].isin(selected_asins)
        & deals["asin_promo_start_week"].notna()
        & deals["asin_promo_end_week"].notna()
        & (deals["asin_promo_start_week"] <= max_week)
        & (deals["asin_promo_end_week"] >= min_week)
    ].copy()

    candidates = target_grid.merge(
        deals[
            ["asin", "asin_promo_start_week", "asin_promo_end_week"]
        ],
        on="asin",
        how="left",
    )
    covered = (
        candidates["asin_promo_start_week"].notna()
        & (candidates["order_week"] >= candidates["asin_promo_start_week"])
        & (candidates["order_week"] <= candidates["asin_promo_end_week"])
    )
    matches = candidates.loc[covered].copy()

    if matches.empty:
        coverage = pd.DataFrame(
            columns=[
                "asin",
                "order_week",
                "matching_deal_count",
                "matched_start_week_min",
                "matched_start_week_max",
                "matched_end_week_min",
                "matched_end_week_max",
            ]
        )
    else:
        coverage = (
            matches.groupby(["asin", "order_week"], as_index=False)
            .agg(
                matching_deal_count=("asin_promo_start_week", "size"),
                matched_start_week_min=("asin_promo_start_week", "min"),
                matched_start_week_max=("asin_promo_start_week", "max"),
                matched_end_week_min=("asin_promo_end_week", "min"),
                matched_end_week_max=("asin_promo_end_week", "max"),
            )
        )

    plan = target_grid.merge(
        coverage,
        on=["asin", "order_week"],
        how="left",
        validate="one_to_one",
    )
    plan["matching_deal_count"] = (
        plan["matching_deal_count"].fillna(0).astype(int)
    )
    plan["planned_ind_promotion"] = (
        plan["matching_deal_count"] > 0
    ).astype(int)
    return plan


def _build_actual_ind_promotion(
    target_grid: pd.DataFrame,
    eval_raw: pd.DataFrame,
) -> pd.DataFrame:
    required = {"asin", "order_week", "ind_promotion"}
    missing = sorted(required - set(eval_raw.columns))
    if missing:
        raise KeyError(
            "Evaluation snapshot is missing required columns: "
            + ", ".join(missing)
        )

    actual = eval_raw[["asin", "order_week", "ind_promotion"]].copy()
    actual["asin"] = _normalize_asin(actual["asin"])
    actual["order_week"] = pd.to_datetime(
        actual["order_week"], errors="coerce"
    ).dt.normalize()
    actual["actual_eval_ind_promotion"] = pd.to_numeric(
        actual["ind_promotion"], errors="coerce"
    )
    actual = actual[
        actual["asin"].isin(set(target_grid["asin"]))
        & actual["order_week"].isin(set(target_grid["order_week"]))
    ].copy()

    # If the source unexpectedly contains duplicate ASIN-week rows, binary
    # actual promotion is 1 when any duplicate says promotion=1.  The duplicate
    # count remains visible in the output for auditing.
    actual_by_key = (
        actual.groupby(["asin", "order_week"], as_index=False)
        .agg(
            actual_eval_ind_promotion=(
                "actual_eval_ind_promotion",
                "max",
            ),
            eval_rows_for_asin_week=("ind_promotion", "size"),
        )
    )
    return target_grid[["asin", "order_week"]].merge(
        actual_by_key,
        on=["asin", "order_week"],
        how="left",
        validate="one_to_one",
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _summarize_binary_comparison(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", detail)] + [
        (h, detail[detail["forecast_horizon"] == h])
        for h in ["H1", "H2", "H3"]
    ]

    for horizon, sub_all in groups:
        sub = sub_all[sub_all["actual_eval_ind_promotion"].notna()].copy()
        planned = sub["planned_ind_promotion"].astype(int)
        actual = sub["actual_eval_ind_promotion"].gt(0).astype(int)

        tp = int(((planned == 1) & (actual == 1)).sum())
        tn = int(((planned == 0) & (actual == 0)).sum())
        fp = int(((planned == 1) & (actual == 0)).sum())
        fn = int(((planned == 0) & (actual == 1)).sum())
        n = len(sub)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)

        rows.append({
            "forecast_horizon": horizon,
            "expected_rows": len(sub_all),
            "rows_with_actual": n,
            "missing_actual_rows": int(
                sub_all["actual_eval_ind_promotion"].isna().sum()
            ),
            "planned_positive_count": int(planned.sum()),
            "actual_positive_count": int(actual.sum()),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "planned_deal_rate": _safe_div(int(planned.sum()), n),
            "actual_deal_rate": _safe_div(int(actual.sum()), n),
            "mismatch_rate": _safe_div(fp + fn, n),
            "agreement_accuracy": _safe_div(tp + tn, n),
            "precision": precision,
            "recall": recall,
            "f1": (
                _safe_div(2 * precision * recall, precision + recall)
                if np.isfinite(precision)
                and np.isfinite(recall)
                and precision + recall > 0
                else np.nan
            ),
        })
    return pd.DataFrame(rows)


def compare_current_fcd_ind_promotion(
    asin_csv_path: str | Path | None = None,
    output_dir: str | Path = "jit_current_fcd_promo_comparison",
    data_cut: str | pd.Timestamp = DEFAULT_DATA_CUT,
    bucket: str = S3_BUCKET,
    data_prefix: str = MODEL_DATA_PREFIX,
    deals_prefix: str = DEALS_PREFIX,
) -> dict[str, object]:
    """Run the planned-vs-actual comparison for the supplied JIT ASIN cohort."""
    data_cut = pd.Timestamp(data_cut).normalize()
    fcd = data_cut + pd.Timedelta(days=1)
    target_weeks = [
        fcd,
        fcd + pd.Timedelta(days=7),
        fcd + pd.Timedelta(days=14),
    ]

    asin_path = _resolve_asin_csv(asin_csv_path)
    cohort_raw = pd.read_csv(asin_path)
    asin_col = _find_asin_column(cohort_raw)
    asins = sorted(
        set(_normalize_asin(cohort_raw[asin_col].dropna())) - {"", "nan"}
    )
    if not asins:
        raise RuntimeError(f"No ASINs were found in {asin_path}")

    print("=" * 100)
    print("CURRENT-FCD PLANNED VS ACTUAL IND_PROMOTION")
    print("=" * 100)
    print(f"ASIN cohort CSV: {asin_path}")
    print(f"Unique ASINs: {len(asins):,}")
    print(f"Data cut / plan snapshot: {data_cut.date()}")
    print(f"FCD: {fcd.date()}")
    print(
        "Target weeks: "
        + ", ".join(
            f"H{i + 1}={week.date()}" for i, week in enumerate(target_weeks)
        )
    )

    s3_client = boto3.client("s3")
    deals_key = _find_deals_snapshot_key(
        data_cut, bucket, deals_prefix, s3_client
    )
    eval_cut, eval_key = _find_eval_snapshot_key(
        data_cut, bucket, data_prefix, s3_client
    )
    print(f"Evaluation snapshot: {eval_cut.date()}")

    deals_raw = _read_s3_csv(bucket, deals_key, s3_client)
    eval_raw = _read_s3_csv(bucket, eval_key, s3_client)

    target_grid = _build_target_grid(asins, fcd)
    plan = _build_planned_ind_promotion(target_grid, deals_raw)
    actual = _build_actual_ind_promotion(target_grid, eval_raw)
    detail = plan.merge(
        actual,
        on=["asin", "order_week"],
        how="left",
        validate="one_to_one",
    )
    detail["actual_eval_ind_promotion"] = pd.to_numeric(
        detail["actual_eval_ind_promotion"], errors="coerce"
    )
    detail["is_match"] = np.where(
        detail["actual_eval_ind_promotion"].isna(),
        np.nan,
        detail["planned_ind_promotion"].eq(
            detail["actual_eval_ind_promotion"].gt(0).astype(int)
        ),
    )
    detail["comparison_class"] = np.select(
        [
            detail["actual_eval_ind_promotion"].isna(),
            detail["planned_ind_promotion"].eq(1)
            & detail["actual_eval_ind_promotion"].gt(0),
            detail["planned_ind_promotion"].eq(0)
            & detail["actual_eval_ind_promotion"].le(0),
            detail["planned_ind_promotion"].eq(1)
            & detail["actual_eval_ind_promotion"].le(0),
            detail["planned_ind_promotion"].eq(0)
            & detail["actual_eval_ind_promotion"].gt(0),
        ],
        ["MISSING_ACTUAL", "TP", "TN", "FP", "FN"],
        default="UNEXPECTED",
    )
    detail["data_cut"] = data_cut
    detail["fcd"] = fcd
    detail["deals_s3_key"] = deals_key
    detail["eval_s3_key"] = eval_key

    expected_rows = len(asins) * 3
    if len(detail) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows:,} ASIN-week rows, got {len(detail):,}."
        )
    per_asin_week_count = detail.groupby("asin")["order_week"].nunique()
    if not per_asin_week_count.eq(3).all():
        raise AssertionError("At least one ASIN does not have exactly H1-H3.")
    observed_weeks = set(detail["order_week"].dropna().unique())
    if observed_weeks != set(pd.to_datetime(target_weeks).values):
        raise AssertionError(
            f"Order-week mismatch: observed={sorted(observed_weeks)}, "
            f"expected={target_weeks}"
        )

    summary = _summarize_binary_comparison(detail)
    summary.insert(0, "data_cut", data_cut)
    summary.insert(1, "fcd", fcd)

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = fcd.strftime("%Y-%m-%d")
    detail_path = output_dir / (
        f"jit_asins_planned_vs_actual_ind_promotion_detail_fcd_{token}.csv"
    )
    summary_path = output_dir / (
        f"jit_asins_planned_vs_actual_ind_promotion_summary_fcd_{token}.csv"
    )
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nBinary comparison summary:")
    print(summary.round(6).to_string(index=False))
    print("\nMismatch counts:")
    print(
        detail.groupby(
            ["forecast_horizon", "comparison_class"],
            dropna=False,
        ).size().unstack(fill_value=0).to_string()
    )
    print(f"\nSaved detail:  {detail_path}")
    print(f"Saved summary: {summary_path}")

    return {
        "summary": summary,
        "detail": detail,
        "asin_csv_path": str(asin_path),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "deals_s3_key": deals_key,
        "eval_s3_key": eval_key,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare planned and actual ind_promotion for the current FCD "
            "using ASINs from a local cohort CSV."
        )
    )
    parser.add_argument(
        "--asin-csv",
        default=None,
        help=(
            "Path to the JIT ASIN cohort CSV. If omitted, the script looks "
            f"for exactly one {DEFAULT_ASIN_FILE_GLOB!r} file in the current directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="jit_current_fcd_promo_comparison",
    )
    parser.add_argument("--data-cut", default=DEFAULT_DATA_CUT)
    args = parser.parse_args()
    compare_current_fcd_ind_promotion(
        asin_csv_path=args.asin_csv,
        output_dir=args.output_dir,
        data_cut=args.data_cut,
    )


if __name__ == "__main__":
    _main()


# Jupyter usage:
# result = compare_current_fcd_ind_promotion(
#     asin_csv_path="/path/to/jit_asins_top90pct_demand_....csv",
#     output_dir="jit_current_fcd_promo_comparison",
# )
