# ================================================================
# CHECK CURRENT-CUT H1/H2/H3 PLAN VS FUTURE REALIZED VALUES
#
# Standalone analysis only. It does NOT import or run the forecasting model.
#
# Business semantics:
#   Origin cut i:
#     - the last 3 distinct order_week values in the origin snapshot are H1/H2/H3 plans
#     - the immediately preceding distinct order_week is the current/actual week
#
#   Realized values:
#     - H1 actual comes from cut i+1, matched by ASIN + original H1 order_week
#     - H2 actual comes from cut i+2, matched by ASIN + original H2 order_week
#     - H3 actual comes from cut i+3, matched by ASIN + original H3 order_week
#
# Cohort:
#   current origin snapshot ASINs ∩ Chris ∩ JIT
#
# Output:
#   Prints metrics and examples only. No files are saved.
# ================================================================

import io
import re
import warnings

import boto3
import numpy as np
import pandas as pd


# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
BUCKET = "amxl-asin-forecast590184089576"
DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)

CHRIS_CSV = "asin_list_from_amxl_fcst_scot_to_chris_20260723.csv"
JIT_CSV = "jit_asin_list_from_Hrishi_20270727.csv"

# Analyze only the first N eligible origin cuts returned by the cut list.
MAX_ORIGIN_CUTS = 3

# Number of largest-gap ASIN examples printed for each field/horizon.
TOP_ASINS_TO_PRINT = 10

JIT_EXCLUDED_ASINS = {"B01FV0F13E", "B073H7VJ37"}

PLAN_FIELDS = [
    "ind_promotion",
    "promotion_pricing_amount",
    "pricing_type",
]

# pricing_type is categorical. The other two are evaluated numerically.
CATEGORICAL_FIELDS = {"pricing_type"}

SCOT_OOS_CANDIDATES = [
    "scot_oos",
    "is_oos_dp",
    "oos_dp",
    "remove_oos_dp",
    "is_oos",
    "oos_flag",
]


# ----------------------------------------------------------------
# BASIC HELPERS
# ----------------------------------------------------------------
def normalize_string(series):
    out = series.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def normalize_frame(df):
    out = df.copy()

    required = {"asin", "order_week"}
    missing = sorted(required - set(out.columns))
    if missing:
        raise ValueError("Snapshot missing required columns: " + ", ".join(missing))

    out["asin"] = normalize_string(out["asin"])
    out["order_week"] = pd.to_datetime(
        out["order_week"], errors="coerce"
    ).dt.normalize()

    return out.dropna(subset=["asin", "order_week"])


def last_non_null(series):
    values = series.dropna()
    return values.iloc[-1] if len(values) else np.nan


def load_asin_set(path, excluded=None):
    df = pd.read_csv(path, usecols=["asin"])
    result = set(normalize_string(df["asin"]).dropna().astype(str))
    if excluded:
        result -= set(excluded)
    return result


def find_scot_oos_col(df):
    for col in SCOT_OOS_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "No SCOT OOS column found. Tried: "
        + ", ".join(SCOT_OOS_CANDIDATES)
    )


# ----------------------------------------------------------------
# S3 SNAPSHOT HELPERS
# ----------------------------------------------------------------
def list_s3_keys(bucket, prefix, s3):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)

    return keys


def list_snapshot_cuts(bucket, prefix, s3):
    pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )

    rows = []
    for key in list_s3_keys(bucket, prefix, s3):
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
        raise RuntimeError(
            "No snapshot files matched the expected V3 filename pattern."
        )

    if len(cuts) < 4:
        raise RuntimeError(
            f"Only {len(cuts)} snapshots found; at least 4 are required."
        )

    return cuts


def build_origin_candidates(cuts):
    """
    Every candidate needs three later cuts:
      origin i, H1 actual i+1, H2 actual i+2, H3 actual i+3.
    """
    rows = []

    for i in range(len(cuts) - 3):
        rows.append(
            {
                "origin_cut": cuts.loc[i, "cut"],
                "origin_key": cuts.loc[i, "key"],
                "h1_actual_cut": cuts.loc[i + 1, "cut"],
                "h1_actual_key": cuts.loc[i + 1, "key"],
                "h2_actual_cut": cuts.loc[i + 2, "cut"],
                "h2_actual_key": cuts.loc[i + 2, "key"],
                "h3_actual_cut": cuts.loc[i + 3, "cut"],
                "h3_actual_key": cuts.loc[i + 3, "key"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        "origin_cut", ascending=True
    ).reset_index(drop=True)


def read_s3_csv_columns(bucket, key, wanted_columns, s3):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    wanted = set(wanted_columns)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        return pd.read_csv(
            io.BytesIO(body),
            usecols=lambda c: c in wanted,
            low_memory=False,
        )


# ----------------------------------------------------------------
# SNAPSHOT COLLAPSE / TARGET-WEEK LOGIC
# ----------------------------------------------------------------
def collapse_asin_week(df, prefix):
    """
    Collapse duplicate ASIN-week rows by taking the final non-null value
    for each plan field.
    """
    output_cols = [
        "asin",
        "order_week",
        *[f"{prefix}_{field}" for field in PLAN_FIELDS],
    ]

    if df.empty:
        return pd.DataFrame(columns=output_cols)

    work = df.copy()
    for field in PLAN_FIELDS:
        if field not in work.columns:
            work[field] = np.nan

    aggregations = {
        f"{prefix}_{field}": (field, last_non_null)
        for field in PLAN_FIELDS
    }

    return (
        work.sort_values(["asin", "order_week"])
        .groupby(["asin", "order_week"], as_index=False)
        .agg(**aggregations)
    )


def infer_current_and_horizon_weeks(origin):
    """
    The origin snapshot already contains:
      ... historical/current weeks, H1 plan, H2 plan, H3 plan.

    Therefore:
      current week = fourth distinct order_week from the end
      H1/H2/H3     = final three distinct order_week values

    This deliberately does NOT compare order_week with the snapshot cut date,
    because their date anchors may differ.
    """
    weeks = sorted(origin["order_week"].dropna().unique())

    if len(weeks) < 4:
        return None

    return {
        "current": pd.Timestamp(weeks[-4]).normalize(),
        1: pd.Timestamp(weeks[-3]).normalize(),
        2: pd.Timestamp(weeks[-2]).normalize(),
        3: pd.Timestamp(weeks[-1]).normalize(),
    }


def determine_selected_asins(
    origin,
    current_order_week,
    chris_set,
    jit_set,
):
    """
    Use the joint ASIN cohort only:
      current origin snapshot ASINs ∩ Chris ∩ JIT

    No additional scot_oos filtering is applied.
    """
    origin_asins = set(origin["asin"].dropna().astype(str))
    selected = sorted(origin_asins & chris_set & jit_set)

    return selected, {
        "origin_asins": len(origin_asins),
        "chris_jit_joint": len(selected),
        "selected_asins": len(selected),
    }


# ----------------------------------------------------------------
# BUILD ASIN × HORIZON DETAIL
# ----------------------------------------------------------------
def build_detail_for_origin(
    origin_raw,
    actual_raw_by_h,
    origin_cut,
    actual_cut_by_h,
    chris_set,
    jit_set,
):
    origin = normalize_frame(origin_raw)
    actuals = {
        h: normalize_frame(df)
        for h, df in actual_raw_by_h.items()
    }

    week_map = infer_current_and_horizon_weeks(origin)
    if week_map is None:
        return None, {
            "skip_reason": "origin snapshot has fewer than 4 distinct order_week values"
        }

    current_week = week_map["current"]
    target_week_by_h = {h: week_map[h] for h in [1, 2, 3]}

    # H1 lag1 = current week; H2 lag1 = H1 plan week; H3 lag1 = H2 plan week.
    lag1_week_by_h = {
        1: current_week,
        2: target_week_by_h[1],
        3: target_week_by_h[2],
    }

    selected, cohort_info = determine_selected_asins(
        origin=origin,
        current_order_week=current_week,
        chris_set=chris_set,
        jit_set=jit_set,
    )

    if not selected:
        return None, {
            **cohort_info,
            "skip_reason": "current-cut ASINs ∩ Chris ∩ JIT cohort is empty",
        }

    # Current-cut PLAN values.
    plan_rows = origin[
        origin["asin"].isin(selected)
        & origin["order_week"].isin(target_week_by_h.values())
    ].copy()

    plan = collapse_asin_week(plan_rows, "plan")
    plan["horizon"] = plan["order_week"].map(
        {week: h for h, week in target_week_by_h.items()}
    )

    # Current-cut LAG1 values.
    lag_parts = []
    for h in [1, 2, 3]:
        lag_week = lag1_week_by_h[h]
        rows = origin[
            origin["asin"].isin(selected)
            & origin["order_week"].eq(lag_week)
        ].copy()

        part = collapse_asin_week(rows, "lag1")
        part = part.rename(columns={"order_week": "lag1_source_week"})
        part["horizon"] = h
        lag_parts.append(part)

    lag1 = pd.concat(lag_parts, ignore_index=True)

    # Future realized values.
    # Each horizon uses its corresponding future cut, but matches the ORIGINAL
    # target order_week exactly.
    actual_parts = []
    actual_match_rows = {}

    for h in [1, 2, 3]:
        target_week = target_week_by_h[h]

        rows = actuals[h][
            actuals[h]["asin"].isin(selected)
            & actuals[h]["order_week"].eq(target_week)
        ].copy()

        actual_match_rows[h] = len(rows)

        part = collapse_asin_week(rows, "actual")
        part["horizon"] = h
        part["actual_cut"] = actual_cut_by_h[h]
        actual_parts.append(part)

    actual = pd.concat(actual_parts, ignore_index=True)

    # Fixed grid preserves missing PLAN/ACTUAL/LAG1 rows.
    grid = pd.MultiIndex.from_product(
        [selected, [1, 2, 3]],
        names=["asin", "horizon"],
    ).to_frame(index=False)

    grid["target_week"] = grid["horizon"].map(target_week_by_h)

    detail = (
        grid
        .merge(
            plan,
            left_on=["asin", "target_week", "horizon"],
            right_on=["asin", "order_week", "horizon"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["order_week"], errors="ignore")
        .merge(
            lag1,
            on=["asin", "horizon"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            actual,
            left_on=["asin", "target_week", "horizon"],
            right_on=["asin", "order_week", "horizon"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["order_week"], errors="ignore")
    )

    detail.insert(0, "origin_cut", pd.Timestamp(origin_cut))
    detail.insert(1, "current_order_week", current_week)

    return detail, {
        **cohort_info,
        "target_week_by_h": target_week_by_h,
        "lag1_week_by_h": lag1_week_by_h,
        "actual_cut_by_h": actual_cut_by_h,
        "plan_source": "current origin snapshot",
        "actual_match_rows_by_h": actual_match_rows,
    }


# ----------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------
def numeric_metrics(group, field):
    plan = pd.to_numeric(group[f"plan_{field}"], errors="coerce")
    lag1 = pd.to_numeric(group[f"lag1_{field}"], errors="coerce")
    actual = pd.to_numeric(group[f"actual_{field}"], errors="coerce")

    plan_ok = plan.notna() & actual.notna()
    lag1_ok = lag1.notna() & actual.notna()
    both_ok = plan.notna() & lag1.notna() & actual.notna()

    plan_abs = (plan - actual).abs()
    lag1_abs = (lag1 - actual).abs()

    result = {
        "rows": len(group),
        "plan_missing": int(plan.isna().sum()),
        "actual_missing": int(actual.isna().sum()),
        "lag1_missing": int(lag1.isna().sum()),
        "plan_actual_compared": int(plan_ok.sum()),
        "lag1_actual_compared": int(lag1_ok.sum()),
        "both_compared": int(both_ok.sum()),
        "plan_mae": plan_abs[plan_ok].mean() if plan_ok.any() else np.nan,
        "lag1_mae": lag1_abs[lag1_ok].mean() if lag1_ok.any() else np.nan,
        "plan_median_abs_gap": (
            plan_abs[plan_ok].median() if plan_ok.any() else np.nan
        ),
        "lag1_median_abs_gap": (
            lag1_abs[lag1_ok].median() if lag1_ok.any() else np.nan
        ),
        "plan_p90_abs_gap": (
            plan_abs[plan_ok].quantile(0.90) if plan_ok.any() else np.nan
        ),
        "lag1_p90_abs_gap": (
            lag1_abs[lag1_ok].quantile(0.90) if lag1_ok.any() else np.nan
        ),
        "plan_exact_match_rate": (
            np.isclose(plan[plan_ok], actual[plan_ok]).mean()
            if plan_ok.any()
            else np.nan
        ),
    }

    if both_ok.any():
        p = plan_abs[both_ok]
        l = lag1_abs[both_ok]

        result.update(
            {
                "plan_win_rate": (p < l).mean(),
                "lag1_win_rate": (l < p).mean(),
                "tie_rate": np.isclose(p, l).mean(),
                # Positive means PLAN has lower error than LAG1.
                "mae_gain_vs_lag1": (l - p).mean(),
            }
        )
    else:
        result.update(
            {
                "plan_win_rate": np.nan,
                "lag1_win_rate": np.nan,
                "tie_rate": np.nan,
                "mae_gain_vs_lag1": np.nan,
            }
        )

    return result


def categorical_metrics(group, field):
    plan = normalize_string(group[f"plan_{field}"])
    lag1 = normalize_string(group[f"lag1_{field}"])
    actual = normalize_string(group[f"actual_{field}"])

    plan_ok = plan.notna() & actual.notna()
    lag1_ok = lag1.notna() & actual.notna()
    both_ok = plan.notna() & lag1.notna() & actual.notna()

    result = {
        "rows": len(group),
        "plan_missing": int(plan.isna().sum()),
        "actual_missing": int(actual.isna().sum()),
        "lag1_missing": int(lag1.isna().sum()),
        "plan_actual_compared": int(plan_ok.sum()),
        "lag1_actual_compared": int(lag1_ok.sum()),
        "both_compared": int(both_ok.sum()),
        "plan_match_rate": (
            plan[plan_ok].eq(actual[plan_ok]).mean()
            if plan_ok.any()
            else np.nan
        ),
        "lag1_match_rate": (
            lag1[lag1_ok].eq(actual[lag1_ok]).mean()
            if lag1_ok.any()
            else np.nan
        ),
    }

    if both_ok.any():
        p_match = plan[both_ok].eq(actual[both_ok])
        l_match = lag1[both_ok].eq(actual[both_ok])

        result.update(
            {
                "plan_win_rate": (p_match & ~l_match).mean(),
                "lag1_win_rate": (l_match & ~p_match).mean(),
                "tie_rate": (p_match == l_match).mean(),
            }
        )
    else:
        result.update(
            {
                "plan_win_rate": np.nan,
                "lag1_win_rate": np.nan,
                "tie_rate": np.nan,
            }
        )

    return result


def fmt_number(value):
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):.6f}"


def fmt_pct(value):
    if pd.isna(value):
        return "NA"
    return f"{100 * float(value):.2f}%"


# ----------------------------------------------------------------
# PRINTING
# ----------------------------------------------------------------
def print_numeric_summary(group, field):
    metrics = numeric_metrics(group, field)

    print(
        f"rows={metrics['rows']:,} | "
        f"PLAN missing={metrics['plan_missing']:,} | "
        f"ACTUAL missing={metrics['actual_missing']:,} | "
        f"LAG1 missing={metrics['lag1_missing']:,}"
    )
    print(
        f"PLAN↔ACTUAL compared={metrics['plan_actual_compared']:,} | "
        f"LAG1↔ACTUAL compared={metrics['lag1_actual_compared']:,} | "
        f"PLAN/LAG1/ACTUAL all present={metrics['both_compared']:,}"
    )
    print(
        "PLAN: "
        f"MAE={fmt_number(metrics['plan_mae'])} | "
        f"median abs gap={fmt_number(metrics['plan_median_abs_gap'])} | "
        f"P90 abs gap={fmt_number(metrics['plan_p90_abs_gap'])} | "
        f"exact match={fmt_pct(metrics['plan_exact_match_rate'])}"
    )
    print(
        "LAG1: "
        f"MAE={fmt_number(metrics['lag1_mae'])} | "
        f"median abs gap={fmt_number(metrics['lag1_median_abs_gap'])} | "
        f"P90 abs gap={fmt_number(metrics['lag1_p90_abs_gap'])}"
    )
    print(
        "PLAN vs LAG1: "
        f"PLAN wins={fmt_pct(metrics['plan_win_rate'])} | "
        f"ties={fmt_pct(metrics['tie_rate'])} | "
        f"LAG1 wins={fmt_pct(metrics['lag1_win_rate'])} | "
        f"mean absolute-error gain={fmt_number(metrics['mae_gain_vs_lag1'])}"
    )


def print_categorical_summary(group, field):
    metrics = categorical_metrics(group, field)

    print(
        f"rows={metrics['rows']:,} | "
        f"PLAN missing={metrics['plan_missing']:,} | "
        f"ACTUAL missing={metrics['actual_missing']:,} | "
        f"LAG1 missing={metrics['lag1_missing']:,}"
    )
    print(
        f"PLAN↔ACTUAL compared={metrics['plan_actual_compared']:,} | "
        f"LAG1↔ACTUAL compared={metrics['lag1_actual_compared']:,} | "
        f"PLAN/LAG1/ACTUAL all present={metrics['both_compared']:,}"
    )
    print(
        f"PLAN exact match={fmt_pct(metrics['plan_match_rate'])} | "
        f"LAG1 exact match={fmt_pct(metrics['lag1_match_rate'])}"
    )
    print(
        "PLAN vs LAG1: "
        f"PLAN wins={fmt_pct(metrics['plan_win_rate'])} | "
        f"ties={fmt_pct(metrics['tie_rate'])} | "
        f"LAG1 wins={fmt_pct(metrics['lag1_win_rate'])}"
    )


def print_examples(group, field, top_n):
    columns = [
        "origin_cut",
        "asin",
        "horizon",
        "target_week",
        f"plan_{field}",
        f"actual_{field}",
        f"lag1_{field}",
        "lag1_source_week",
        "actual_cut",
    ]

    work = group[columns].copy()

    if field in CATEGORICAL_FIELDS:
        plan = normalize_string(work[f"plan_{field}"])
        actual = normalize_string(work[f"actual_{field}"])

        valid = plan.notna() & actual.notna()
        work = work.loc[valid].copy()
        work["plan_matches_actual"] = plan[valid].eq(actual[valid])

        # Mismatches first.
        work = work.sort_values(
            ["plan_matches_actual", "origin_cut", "horizon", "asin"],
            ascending=[True, False, True, True],
        ).head(top_n)
    else:
        plan = pd.to_numeric(work[f"plan_{field}"], errors="coerce")
        actual = pd.to_numeric(work[f"actual_{field}"], errors="coerce")
        lag1 = pd.to_numeric(work[f"lag1_{field}"], errors="coerce")

        valid = plan.notna() & actual.notna()
        work = work.loc[valid].copy()

        work["plan_abs_gap"] = (plan[valid] - actual[valid]).abs()
        work["lag1_abs_gap"] = (lag1[valid] - actual[valid]).abs()

        work = work.sort_values(
            "plan_abs_gap", ascending=False
        ).head(top_n)

    if work.empty:
        print("No comparable examples.")
    else:
        print(work.to_string(index=False))


def print_cut_report(detail, info):
    origin_cut = pd.Timestamp(detail["origin_cut"].iloc[0])

    print("\n" + "#" * 118)
    print(f"ORIGIN CUT: {origin_cut.date()}")
    print("#" * 118)
    print(f"Current order_week: {info['target_week_by_h'][1] - pd.Timedelta(days=7)}")
    print(
        "Target weeks: "
        + " | ".join(
            f"H{h}={info['target_week_by_h'][h].date()}"
            for h in [1, 2, 3]
        )
    )
    print(
        "Actual cuts: "
        + " | ".join(
            f"H{h}={pd.Timestamp(info['actual_cut_by_h'][h]).date()}"
            for h in [1, 2, 3]
        )
    )
    print(
        f"Cohort: origin ASINs={info['origin_asins']:,} | "
        f"joint selected={info['selected_asins']:,}"
    )
    print(
        "Matched actual source rows: "
        + " | ".join(
            f"H{h}={info['actual_match_rows_by_h'][h]:,}"
            for h in [1, 2, 3]
        )
    )

    for field in PLAN_FIELDS:
        print("\n" + "=" * 118)
        print(f"FIELD: {field}")
        print("=" * 118)

        for h in [1, 2, 3]:
            print(f"\nH{h}")
            group = detail[detail["horizon"].eq(h)].copy()

            if field in CATEGORICAL_FIELDS:
                print_categorical_summary(group, field)
            else:
                print_numeric_summary(group, field)

        print("\nALL H1-H3 COMBINED")
        if field in CATEGORICAL_FIELDS:
            print_categorical_summary(detail, field)
        else:
            print_numeric_summary(detail, field)

        print(f"\nTop {TOP_ASINS_TO_PRINT} examples:")
        print_examples(detail, field, TOP_ASINS_TO_PRINT)


def print_combined_report(combined):
    print("\n" + "#" * 118)
    print("COMBINED ACROSS ALL VALID ORIGIN CUTS")
    print("#" * 118)

    for field in PLAN_FIELDS:
        print("\n" + "=" * 118)
        print(f"FIELD: {field}")
        print("=" * 118)

        for h in [1, 2, 3]:
            print(f"\nH{h}")
            group = combined[combined["horizon"].eq(h)].copy()

            if field in CATEGORICAL_FIELDS:
                print_categorical_summary(group, field)
            else:
                print_numeric_summary(group, field)

        print("\nALL H1-H3 COMBINED")
        if field in CATEGORICAL_FIELDS:
            print_categorical_summary(combined, field)
        else:
            print_numeric_summary(combined, field)

        print(f"\nTop {TOP_ASINS_TO_PRINT} examples across cuts:")
        print_examples(combined, field, TOP_ASINS_TO_PRINT)


# ----------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------
def main():
    print("=" * 118)
    print("FIRST 3 CUTS — JOINT ASINS — PLAN VS FUTURE REALIZED VALUE CHECK")
    print("=" * 118)
    print("Nothing will be written to disk.")

    chris_set = load_asin_set(CHRIS_CSV)
    jit_set = load_asin_set(
        JIT_CSV,
        excluded=JIT_EXCLUDED_ASINS,
    )

    print(
        f"Chris={len(chris_set):,} | "
        f"JIT after exclusions={len(jit_set):,} | "
        f"Chris∩JIT={len(chris_set & jit_set):,}"
    )

    s3 = boto3.client("s3")
    cuts = list_snapshot_cuts(BUCKET, DATA_PREFIX, s3)
    candidates = build_origin_candidates(cuts)

    wanted_columns = [
        "asin",
        "order_week",
        *PLAN_FIELDS,
    ]

    valid_details = []

    # Run only the first three eligible origin cuts.
    for _, row in candidates.head(MAX_ORIGIN_CUTS).iterrows():

        origin_cut = pd.Timestamp(row["origin_cut"])

        print(f"\n[READ] origin cut {origin_cut.date()}")

        origin_raw = read_s3_csv_columns(
            BUCKET,
            row["origin_key"],
            wanted_columns,
            s3,
        )

        actual_raw_by_h = {
            1: read_s3_csv_columns(
                BUCKET, row["h1_actual_key"], wanted_columns, s3
            ),
            2: read_s3_csv_columns(
                BUCKET, row["h2_actual_key"], wanted_columns, s3
            ),
            3: read_s3_csv_columns(
                BUCKET, row["h3_actual_key"], wanted_columns, s3
            ),
        }

        actual_cut_by_h = {
            1: pd.Timestamp(row["h1_actual_cut"]),
            2: pd.Timestamp(row["h2_actual_cut"]),
            3: pd.Timestamp(row["h3_actual_cut"]),
        }

        detail, info = build_detail_for_origin(
            origin_raw=origin_raw,
            actual_raw_by_h=actual_raw_by_h,
            origin_cut=origin_cut,
            actual_cut_by_h=actual_cut_by_h,
            chris_set=chris_set,
            jit_set=jit_set,
        )

        if detail is None:
            print(
                f"[SKIP] origin cut {origin_cut.date()}: "
                f"{info.get('skip_reason', 'invalid cut')}"
            )
            continue

        valid_details.append(detail)
        print_cut_report(detail, info)

    if not valid_details:
        raise RuntimeError(
            "No valid origin cuts were found. Check snapshot schema and cohort files."
        )

    if len(valid_details) < MAX_ORIGIN_CUTS:
        print(
            f"\n[WARNING] Requested the first {MAX_ORIGIN_CUTS} cuts, "
            f"but only {len(valid_details)} produced valid comparisons."
        )

    combined = pd.concat(valid_details, ignore_index=True)
    print_combined_report(combined)

    print("\n" + "=" * 118)
    print("DONE — printed only; no output files were saved.")
    print("=" * 118)

    return combined


# Run immediately when pasted into one notebook cell or executed as a script.
detail = main()
