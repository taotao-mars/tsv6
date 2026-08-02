#!/usr/bin/env python3
"""
Standalone Chris ∩ JIT joint-cohort diagnostic for three promotion/pricing fields.

Goal
----
For every origin cut and ASIN:

1. PLAN
   Read H1/H2/H3 directly from the origin snapshot:
       H1 target week = origin cut + 1 day
       H2 target week = origin cut + 8 days
       H3 target week = origin cut + 15 days

2. ACTUAL
   Read the same target week from the next corresponding snapshot:
       H1 actual = same target week in next available cut
       H2 actual = same target week in second next available cut
       H3 actual = same target week in third next available cut

3. LAG1 BASELINE
   At the origin cut, take the latest non-null value whose order_week is at or
   before the origin cut, and repeat it for H1/H2/H3.

The script compares both PLAN and LAG1 against ACTUAL, so it can answer:
    "Is lag1 better than the planned value?"

No forecasting model is imported, trained, or executed.

Fields
------
- ind_promotion
- promotion_pricing_amount
- pricing_type

Main outputs
------------
1. promo_plan_lag1_vs_actual_detail.csv
   One row per ASIN × origin cut × horizon.

2. promo_plan_lag1_vs_actual_summary_by_cut_horizon.csv
   Metrics by origin cut and horizon.

3. promo_plan_lag1_vs_actual_summary_overall.csv
   Overall metrics by horizon.

4. promo_plan_lag1_vs_actual_winner_by_asin.csv
   ASIN-level comparison showing whether PLAN or LAG1 is better.

5. promo_plan_lag1_vs_actual_missingness.csv
   Missing-value diagnostics.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable, Optional

import boto3
import numpy as np
import pandas as pd


DEFAULT_BUCKET = "amxl-asin-forecast590184089576"
DEFAULT_DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)

PROMO_FIELDS = [
    "ind_promotion",
    "promotion_pricing_amount",
    "pricing_type",
]


def _list_s3_keys(bucket: str, prefix: str, s3_client) -> list[str]:
    paginator = s3_client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)
    return keys


def list_snapshot_cuts(
    bucket: str,
    data_prefix: str,
    s3_client,
) -> pd.DataFrame:
    pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )

    rows = []
    for key in _list_s3_keys(bucket, data_prefix, s3_client):
        match = pattern.search(key)
        if match:
            rows.append(
                {
                    "cut": pd.Timestamp(match.group(1)).normalize(),
                    "key": key,
                }
            )

    cuts = (
        pd.DataFrame(rows)
        .drop_duplicates("cut", keep="last")
        .sort_values("cut")
        .reset_index(drop=True)
    )
    if cuts.empty:
        raise RuntimeError("No snapshot files were found under the S3 prefix.")
    return cuts


def build_origin_next_cut_pairs(cuts: pd.DataFrame) -> pd.DataFrame:
    """
    Require three later snapshots. We deliberately use the next three available
    cuts rather than hard-coding +7/+14/+21 days, so a missing weekly snapshot
    is visible in the actual_cut columns instead of silently reading the wrong
    file.
    """
    rows = []
    for i in range(len(cuts) - 3):
        rows.append(
            {
                "data_cut": cuts.loc[i, "cut"],
                "origin_key": cuts.loc[i, "key"],
                "h1_actual_cut": cuts.loc[i + 1, "cut"],
                "h1_actual_key": cuts.loc[i + 1, "key"],
                "h2_actual_cut": cuts.loc[i + 2, "cut"],
                "h2_actual_key": cuts.loc[i + 2, "key"],
                "h3_actual_cut": cuts.loc[i + 3, "cut"],
                "h3_actual_key": cuts.loc[i + 3, "key"],
            }
        )
    return pd.DataFrame(rows)


def _read_s3_csv_columns(
    bucket: str,
    key: str,
    columns: Iterable[str],
    s3_client,
) -> pd.DataFrame:
    wanted = set(columns)
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(body), usecols=lambda c: c in wanted)


def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["asin"] = out["asin"].astype(str).str.strip()
    out["order_week"] = pd.to_datetime(
        out["order_week"], errors="coerce"
    ).dt.normalize()
    return out


def _normalize_string(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def _last_non_null(series: pd.Series):
    values = series.dropna()
    return values.iloc[-1] if len(values) else np.nan


def _load_optional_cohort(
    csv_path: Optional[str],
    excluded_asins: Optional[set[str]] = None,
) -> Optional[set[str]]:
    if not csv_path:
        return None

    df = pd.read_csv(csv_path, usecols=["asin"])
    asins = set(
        df["asin"].dropna().astype(str).str.strip().replace("", np.nan).dropna()
    )
    if excluded_asins:
        asins -= excluded_asins
    return asins


def _selected_asins(
    origin_asins: set[str],
    actual_asins_by_horizon: dict[int, set[str]],
    chris_set: Optional[set[str]],
    jit_set: Optional[set[str]],
) -> list[str]:
    """
    Keep the fixed ASIN cohort for each origin cut.

    Do NOT intersect with H1/H2/H3 actual availability. Missing plan or actual
    values remain as NaN on that ASIN × horizon row and are handled only by
    missingness/metric masks.
    """
    selected = set(origin_asins)
    if chris_set is not None:
        selected &= chris_set
    if jit_set is not None:
        selected &= jit_set
    return sorted(selected)


def _collapse_asin_week(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return (
        df.sort_values(["asin", "order_week"])
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            **{
                f"{prefix}_ind_promotion": (
                    "ind_promotion",
                    _last_non_null,
                ),
                f"{prefix}_promotion_pricing_amount": (
                    "promotion_pricing_amount",
                    _last_non_null,
                ),
                f"{prefix}_pricing_type": (
                    "pricing_type",
                    _last_non_null,
                ),
            }
        )
    )


def _numeric_errors(
    detail: pd.DataFrame,
    estimate_prefix: str,
    field: str,
) -> None:
    estimate = pd.to_numeric(
        detail[f"{estimate_prefix}_{field}"], errors="coerce"
    )
    actual = pd.to_numeric(detail[f"actual_{field}"], errors="coerce")
    both = estimate.notna() & actual.notna()

    detail[f"{estimate_prefix}_{field}_available"] = both
    detail[f"{estimate_prefix}_{field}_error"] = np.where(
        both, estimate - actual, np.nan
    )
    detail[f"{estimate_prefix}_{field}_abs_error"] = np.where(
        both, np.abs(estimate - actual), np.nan
    )
    detail[f"{estimate_prefix}_{field}_exact_match"] = np.where(
        both, np.isclose(estimate, actual, atol=1e-12), np.nan
    )


def _categorical_errors(
    detail: pd.DataFrame,
    estimate_prefix: str,
    field: str,
) -> None:
    estimate = _normalize_string(detail[f"{estimate_prefix}_{field}"])
    actual = _normalize_string(detail[f"actual_{field}"])
    both = estimate.notna() & actual.notna()

    detail[f"{estimate_prefix}_{field}_available"] = both
    detail[f"{estimate_prefix}_{field}_exact_match"] = np.where(
        both, estimate.eq(actual), np.nan
    )


def _add_plan_vs_lag1_winners(detail: pd.DataFrame) -> None:
    for field in ["ind_promotion", "promotion_pricing_amount"]:
        p = pd.to_numeric(
            detail[f"plan_{field}_abs_error"], errors="coerce"
        )
        l = pd.to_numeric(
            detail[f"lag1_{field}_abs_error"], errors="coerce"
        )
        comparable = p.notna() & l.notna()

        winner = np.full(len(detail), None, dtype=object)
        winner[comparable & (p < l)] = "PLAN"
        winner[comparable & (l < p)] = "LAG1"
        winner[comparable & np.isclose(p, l, atol=1e-12)] = "TIE"
        detail[f"{field}_winner"] = winner
        detail[f"{field}_lag1_improvement_vs_plan"] = np.where(
            comparable, p - l, np.nan
        )

    field = "pricing_type"
    p = detail[f"plan_{field}_exact_match"]
    l = detail[f"lag1_{field}_exact_match"]
    comparable = p.notna() & l.notna()

    winner = np.full(len(detail), None, dtype=object)
    winner[comparable & (p == True) & (l == False)] = "PLAN"
    winner[comparable & (l == True) & (p == False)] = "LAG1"
    winner[comparable & (p == l)] = "TIE"
    detail[f"{field}_winner"] = winner


def compare_one_origin(
    origin_raw: pd.DataFrame,
    actual_raw_by_horizon: dict[int, pd.DataFrame],
    data_cut: pd.Timestamp,
    actual_cut_by_horizon: dict[int, pd.Timestamp],
    chris_set: Optional[set[str]],
    jit_set: Optional[set[str]],
) -> pd.DataFrame:
    required = {"asin", "order_week", *PROMO_FIELDS}

    if required - set(origin_raw.columns):
        raise ValueError(
            "Origin snapshot is missing columns: "
            f"{sorted(required - set(origin_raw.columns))}"
        )

    origin = _normalize_frame(origin_raw)
    actuals = {
        h: _normalize_frame(df)
        for h, df in actual_raw_by_horizon.items()
    }

    for h, df in actuals.items():
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"H{h} actual snapshot is missing columns: {sorted(missing)}"
            )

    target_week_by_horizon = {
        1: data_cut + pd.Timedelta(days=1),
        2: data_cut + pd.Timedelta(days=8),
        3: data_cut + pd.Timedelta(days=15),
    }

    origin_hist = origin[
        origin["order_week"].notna()
        & (origin["order_week"] <= data_cut)
    ].copy()

    origin_plan_rows = origin[
        origin["order_week"].isin(target_week_by_horizon.values())
    ].copy()

    actual_rows_by_horizon = {}
    actual_asins_by_horizon = {}
    for h in [1, 2, 3]:
        target_week = target_week_by_horizon[h]
        rows = actuals[h][actuals[h]["order_week"] == target_week].copy()
        actual_rows_by_horizon[h] = rows
        actual_asins_by_horizon[h] = set(rows["asin"].dropna())

    selected = _selected_asins(
        origin_asins=set(origin["asin"].dropna()),
        actual_asins_by_horizon=actual_asins_by_horizon,
        chris_set=chris_set,
        jit_set=jit_set,
    )

    origin_hist = origin_hist[origin_hist["asin"].isin(selected)]
    origin_plan_rows = origin_plan_rows[
        origin_plan_rows["asin"].isin(selected)
    ]

    # PLAN: H1/H2/H3 values already stored in the origin snapshot.
    plan = _collapse_asin_week(origin_plan_rows, "plan")
    week_to_horizon = {
        week: horizon for horizon, week in target_week_by_horizon.items()
    }
    plan["horizon"] = plan["order_week"].map(week_to_horizon)

    # LAG1: latest non-null known value at/before origin cut.
    origin_hist = origin_hist.sort_values(["asin", "order_week"])
    lag1 = (
        origin_hist.groupby("asin", as_index=False)
        .agg(
            lag1_source_week=("order_week", "max"),
            lag1_ind_promotion=("ind_promotion", _last_non_null),
            lag1_promotion_pricing_amount=(
                "promotion_pricing_amount",
                _last_non_null,
            ),
            lag1_pricing_type=("pricing_type", _last_non_null),
        )
    )

    actual_parts = []
    for h in [1, 2, 3]:
        actual = _collapse_asin_week(
            actual_rows_by_horizon[h][
                actual_rows_by_horizon[h]["asin"].isin(selected)
            ],
            "actual",
        )
        actual["horizon"] = h
        actual["actual_cut"] = actual_cut_by_horizon[h]
        actual_parts.append(actual)
    actual = pd.concat(actual_parts, ignore_index=True)

    grid = pd.MultiIndex.from_product(
        [selected, [1, 2, 3]],
        names=["asin", "horizon"],
    ).to_frame(index=False)
    grid["target_week"] = grid["horizon"].map(target_week_by_horizon)

    detail = grid.merge(
        plan,
        left_on=["asin", "target_week", "horizon"],
        right_on=["asin", "order_week", "horizon"],
        how="left",
    ).drop(columns=["order_week"], errors="ignore")

    detail = detail.merge(lag1, on="asin", how="left")

    detail = detail.merge(
        actual,
        left_on=["asin", "target_week", "horizon"],
        right_on=["asin", "order_week", "horizon"],
        how="left",
    ).drop(columns=["order_week"], errors="ignore")

    detail.insert(0, "data_cut", data_cut)
    detail.insert(1, "cohort", "CHRIS_JIT_JOINT")

    for estimate in ["plan", "lag1"]:
        _numeric_errors(detail, estimate, "ind_promotion")
        _numeric_errors(
            detail,
            estimate,
            "promotion_pricing_amount",
        )
        _categorical_errors(detail, estimate, "pricing_type")

    _add_plan_vs_lag1_winners(detail)
    return detail


def _mean_or_nan(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else np.nan


def summarize(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for keys, group in detail.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["rows"] = len(group)
        row["unique_asins"] = group["asin"].nunique()

        for field in ["ind_promotion", "promotion_pricing_amount"]:
            for estimate in ["plan", "lag1"]:
                available = group[
                    f"{estimate}_{field}_available"
                ].fillna(False)
                exact = group.loc[
                    available,
                    f"{estimate}_{field}_exact_match",
                ]
                abs_error = group.loc[
                    available,
                    f"{estimate}_{field}_abs_error",
                ]

                row[f"{estimate}_{field}_n"] = int(available.sum())
                row[f"{estimate}_{field}_match_rate"] = _mean_or_nan(exact)
                row[f"{estimate}_{field}_mae"] = _mean_or_nan(abs_error)
                row[f"{estimate}_{field}_median_abs_error"] = (
                    float(pd.to_numeric(abs_error, errors="coerce").median())
                    if len(abs_error)
                    else np.nan
                )
                row[f"{estimate}_{field}_p90_abs_error"] = (
                    float(
                        pd.to_numeric(
                            abs_error, errors="coerce"
                        ).quantile(0.90)
                    )
                    if len(abs_error)
                    else np.nan
                )

            comparable = group[f"{field}_winner"].notna()
            winners = group.loc[comparable, f"{field}_winner"]
            row[f"{field}_plan_win_rate"] = (
                float((winners == "PLAN").mean()) if len(winners) else np.nan
            )
            row[f"{field}_lag1_win_rate"] = (
                float((winners == "LAG1").mean()) if len(winners) else np.nan
            )
            row[f"{field}_tie_rate"] = (
                float((winners == "TIE").mean()) if len(winners) else np.nan
            )
            row[f"{field}_mean_lag1_improvement_vs_plan"] = _mean_or_nan(
                group[f"{field}_lag1_improvement_vs_plan"]
            )

        field = "pricing_type"
        for estimate in ["plan", "lag1"]:
            available = group[
                f"{estimate}_{field}_available"
            ].fillna(False)
            exact = group.loc[
                available,
                f"{estimate}_{field}_exact_match",
            ]
            row[f"{estimate}_{field}_n"] = int(available.sum())
            row[f"{estimate}_{field}_match_rate"] = _mean_or_nan(exact)

        comparable = group[f"{field}_winner"].notna()
        winners = group.loc[comparable, f"{field}_winner"]
        row[f"{field}_plan_win_rate"] = (
            float((winners == "PLAN").mean()) if len(winners) else np.nan
        )
        row[f"{field}_lag1_win_rate"] = (
            float((winners == "LAG1").mean()) if len(winners) else np.nan
        )
        row[f"{field}_tie_rate"] = (
            float((winners == "TIE").mean()) if len(winners) else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_by_asin(detail: pd.DataFrame) -> pd.DataFrame:
    if detail is None or detail.empty:
        return pd.DataFrame(columns=["asin", "horizon"])

    required = {"asin", "horizon"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=["asin", "horizon"])

    summary = summarize(detail, ["asin", "horizon"])
    if summary.empty or not required.issubset(summary.columns):
        return pd.DataFrame(columns=["asin", "horizon"])

    return summary.sort_values(["asin", "horizon"]).reset_index(drop=True)


def missingness_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon, group in detail.groupby("horizon"):
        for field in PROMO_FIELDS:
            for estimate in ["plan", "lag1", "actual"]:
                col = f"{estimate}_{field}"
                missing = group[col].isna()
                rows.append(
                    {
                        "horizon": horizon,
                        "source": estimate,
                        "field": field,
                        "rows": len(group),
                        "missing_count": int(missing.sum()),
                        "missing_rate": float(missing.mean()),
                    }
                )
    return pd.DataFrame(rows)


# ================================================================
# NOTEBOOK CONFIG — paste this whole file into one Jupyter cell
# ================================================================
BUCKET = DEFAULT_BUCKET
DATA_PREFIX = DEFAULT_DATA_PREFIX

CHRIS_CSV = "asin_list_from_amxl_fcst_scot_to_chris_20260723.csv"
JIT_CSV = "jit_asin_list_from_Hrishi_20270727.csv"

# Analyze only the latest three origin cuts.
MAX_CUTS = 3


def _complete_case_mask(df: pd.DataFrame) -> pd.Series:
    """
    Keep only ASIN × horizon rows where PLAN, LAG1, and ACTUAL all exist for
    all three fields. Selection is performed independently for every cut.
    """
    required = [
        "plan_ind_promotion",
        "lag1_ind_promotion",
        "actual_ind_promotion",
        "plan_promotion_pricing_amount",
        "lag1_promotion_pricing_amount",
        "actual_promotion_pricing_amount",
        "plan_pricing_type",
        "lag1_pricing_type",
        "actual_pricing_type",
    ]
    return df[required].notna().all(axis=1)


def _compact_summary(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []

    for keys, g in detail.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["rows"] = len(g)
        row["asins"] = g["asin"].nunique()

        # Binary/numeric promotion indicator
        for field in ["ind_promotion", "promotion_pricing_amount"]:
            plan_abs = pd.to_numeric(
                g[f"plan_{field}_abs_error"], errors="coerce"
            )
            lag_abs = pd.to_numeric(
                g[f"lag1_{field}_abs_error"], errors="coerce"
            )
            winner = g[f"{field}_winner"]

            row[f"{field}_plan_mae"] = plan_abs.mean()
            row[f"{field}_lag1_mae"] = lag_abs.mean()
            row[f"{field}_plan_win_rate"] = (winner == "PLAN").mean()
            row[f"{field}_lag1_win_rate"] = (winner == "LAG1").mean()
            row[f"{field}_tie_rate"] = (winner == "TIE").mean()
            row[f"{field}_lag1_improvement"] = (plan_abs - lag_abs).mean()

        # Categorical pricing type
        plan_match = pd.to_numeric(
            g["plan_pricing_type_exact_match"], errors="coerce"
        )
        lag_match = pd.to_numeric(
            g["lag1_pricing_type_exact_match"], errors="coerce"
        )
        winner = g["pricing_type_winner"]

        row["pricing_type_plan_match_rate"] = plan_match.mean()
        row["pricing_type_lag1_match_rate"] = lag_match.mean()
        row["pricing_type_plan_win_rate"] = (winner == "PLAN").mean()
        row["pricing_type_lag1_win_rate"] = (winner == "LAG1").mean()
        row["pricing_type_tie_rate"] = (winner == "TIE").mean()

        rows.append(row)

    return pd.DataFrame(rows)


def _print_interpretation(summary: pd.DataFrame, label: str) -> None:
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)

    if summary.empty:
        print("没有完整可比较样本。")
        return

    show_cols = [
        c for c in [
            "data_cut",
            "horizon",
            "rows",
            "asins",
            "ind_promotion_plan_mae",
            "ind_promotion_lag1_mae",
            "ind_promotion_plan_win_rate",
            "ind_promotion_lag1_win_rate",
            "promotion_pricing_amount_plan_mae",
            "promotion_pricing_amount_lag1_mae",
            "promotion_pricing_amount_plan_win_rate",
            "promotion_pricing_amount_lag1_win_rate",
            "pricing_type_plan_match_rate",
            "pricing_type_lag1_match_rate",
            "pricing_type_plan_win_rate",
            "pricing_type_lag1_win_rate",
        ] if c in summary.columns
    ]

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(summary[show_cols].to_string(index=False))


def run_last_three_cut_analysis():
    s3_client = boto3.client("s3")

    cuts = list_snapshot_cuts(
        bucket=BUCKET,
        data_prefix=DATA_PREFIX,
        s3_client=s3_client,
    )
    pairs = build_origin_next_cut_pairs(cuts).sort_values("data_cut").tail(MAX_CUTS)

    if pairs.empty:
        raise RuntimeError("没有找到具备后三个 snapshot 的 origin cut。")

    chris_set = _load_optional_cohort(CHRIS_CSV)
    jit_set = _load_optional_cohort(
        JIT_CSV,
        excluded_asins={"B01FV0F13E", "B073H7VJ37"},
    )

    if not chris_set or not jit_set:
        raise RuntimeError("Chris 或 JIT cohort 文件为空/无法读取。")

    joint_set = chris_set & jit_set
    if not joint_set:
        raise RuntimeError("Chris ∩ JIT 为空。")

    print(
        f"Chris={len(chris_set):,} | JIT={len(jit_set):,} | "
        f"Joint={len(joint_set):,}"
    )
    print("只分析最近 3 个 origin cut；每个 cut 独立选择完整可比较行。")
    print(
        "完整行定义：PLAN、LAG1、ACTUAL 在三个字段上全部非空。"
    )

    read_columns = ["asin", "order_week", *PROMO_FIELDS]
    complete_parts = []

    for i, row in pairs.reset_index(drop=True).iterrows():
        data_cut = pd.Timestamp(row["data_cut"])
        actual_cut_by_horizon = {
            1: pd.Timestamp(row["h1_actual_cut"]),
            2: pd.Timestamp(row["h2_actual_cut"]),
            3: pd.Timestamp(row["h3_actual_cut"]),
        }

        origin_raw = _read_s3_csv_columns(
            BUCKET, row["origin_key"], read_columns, s3_client
        )
        actual_raw_by_horizon = {
            h: _read_s3_csv_columns(
                BUCKET,
                row[f"h{h}_actual_key"],
                read_columns,
                s3_client,
            )
            for h in [1, 2, 3]
        }

        detail = compare_one_origin(
            origin_raw=origin_raw,
            actual_raw_by_horizon=actual_raw_by_horizon,
            data_cut=data_cut,
            actual_cut_by_horizon=actual_cut_by_horizon,
            chris_set=chris_set,
            jit_set=jit_set,
        )

        before = len(detail)
        complete = detail.loc[_complete_case_mask(detail)].copy()
        after = len(complete)

        print("\n" + "-" * 100)
        print(
            f"CUT {i + 1}: origin={data_cut.date()} | "
            f"原始行={before:,} | 完整可比较行={after:,} | "
            f"完整ASIN={complete['asin'].nunique():,}"
        )

        counts = (
            complete.groupby("horizon")
            .agg(rows=("asin", "size"), asins=("asin", "nunique"))
            .reset_index()
        )
        print(counts.to_string(index=False))

        cut_summary = _compact_summary(
            complete,
            ["data_cut", "horizon"],
        )
        _print_interpretation(
            cut_summary,
            f"CUT {data_cut.date()} — PLAN vs LAG1",
        )

        complete_parts.append(complete)

    complete_all = pd.concat(complete_parts, ignore_index=True)

    overall = _compact_summary(complete_all, ["horizon"])
    _print_interpretation(
        overall,
        "最近三个 CUT 合并结果 — 按 H1/H2/H3",
    )

    total = _compact_summary(complete_all.assign(all="ALL"), ["all"])
    _print_interpretation(
        total,
        "最近三个 CUT 合并结果 — 总体",
    )

    print("\n判读规则：")
    print("1) MAE 越低越好。")
    print("2) match rate 越高越好。")
    print("3) PLAN win rate > LAG1 win rate：PLAN 更可靠。")
    print("4) LAG1 win rate > PLAN win rate：说明该字段存在明显 lag/staleness。")
    print("5) lag1_improvement = PLAN MAE - LAG1 MAE；正数代表 LAG1 更好。")

    return complete_all, overall, total


# Run immediately in the notebook cell. Nothing is written to disk.
complete_detail_df, overall_df, total_df = run_last_three_cut_analysis()
