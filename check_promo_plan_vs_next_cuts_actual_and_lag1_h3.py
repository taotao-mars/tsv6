# ================================================================
# JOINT Chris ∩ JIT ∩ scot_oos=0
# Latest 3 origin cuts: PLAN vs LAG1 vs ACTUAL
# Paste this whole file into ONE Jupyter cell and run.
# No CSV files are written. Results are printed only.
# ================================================================

import io
import re
import warnings

import boto3
import numpy as np
import pandas as pd


# -----------------------------
# CONFIG
# -----------------------------
BUCKET = "amxl-asin-forecast590184089576"
DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)

CHRIS_CSV = "asin_list_from_amxl_fcst_scot_to_chris_20260723.csv"
JIT_CSV = "jit_asin_list_from_Hrishi_20270727.csv"

MAX_CUTS = 3
TOP_ASINS_TO_PRINT = 10

JIT_EXCLUDED_ASINS = {"B01FV0F13E", "B073H7VJ37"}

FIELDS = [
    "ind_promotion",
    "promotion_pricing_amount",
    "pricing_type",
]

SCOT_OOS_CANDIDATES = [
    "scot_oos",
    "is_oos_dp",
    "oos_dp",
    "remove_oos_dp",
    "is_oos",
    "oos_flag",
]


# -----------------------------
# S3 / snapshot helpers
# -----------------------------
def list_s3_keys(bucket, prefix, s3):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj.get("Key"):
                keys.append(obj["Key"])
    return keys


def list_snapshot_cuts(bucket, prefix, s3):
    pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )

    rows = []
    for key in list_s3_keys(bucket, prefix, s3):
        m = pattern.search(key)
        if m:
            rows.append(
                {
                    "cut": pd.Timestamp(m.group(1)).normalize(),
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
        raise RuntimeError("No weekly snapshot files were found.")
    return cuts


def build_last3_origin_pairs(cuts):
    rows = []
    for i in range(len(cuts) - 3):
        rows.append(
            {
                "data_cut": cuts.loc[i, "cut"],
                "origin_key": cuts.loc[i, "key"],
                "h1_cut": cuts.loc[i + 1, "cut"],
                "h1_key": cuts.loc[i + 1, "key"],
                "h2_cut": cuts.loc[i + 2, "cut"],
                "h2_key": cuts.loc[i + 2, "key"],
                "h3_cut": cuts.loc[i + 3, "cut"],
                "h3_key": cuts.loc[i + 3, "key"],
            }
        )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise RuntimeError("Not enough snapshots to form 3 future cuts.")
    return pairs.sort_values("data_cut").tail(MAX_CUTS).reset_index(drop=True)


def read_s3_csv_columns(bucket, key, wanted_columns, s3):
    wanted = set(wanted_columns)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.DtypeWarning)
        return pd.read_csv(
            io.BytesIO(body),
            usecols=lambda c: c in wanted,
            low_memory=False,
        )


def normalize_frame(df):
    out = df.copy()
    out["asin"] = out["asin"].astype("string").str.strip()
    out["order_week"] = pd.to_datetime(
        out["order_week"], errors="coerce"
    ).dt.normalize()
    return out


def normalize_string(s):
    out = s.astype("string").str.strip()
    return out.mask(out.isin(["", "nan", "None", "<NA>"]))


def last_non_null(s):
    v = s.dropna()
    return v.iloc[-1] if len(v) else np.nan


def load_asin_set(path, excluded=None):
    df = pd.read_csv(path, usecols=["asin"])
    result = set(
        df["asin"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )
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


# -----------------------------
# One-cut construction
# -----------------------------
def collapse_asin_week(df, prefix):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "asin",
                "order_week",
                f"{prefix}_ind_promotion",
                f"{prefix}_promotion_pricing_amount",
                f"{prefix}_pricing_type",
            ]
        )

    return (
        df.sort_values(["asin", "order_week"])
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            **{
                f"{prefix}_ind_promotion": (
                    "ind_promotion",
                    last_non_null,
                ),
                f"{prefix}_promotion_pricing_amount": (
                    "promotion_pricing_amount",
                    last_non_null,
                ),
                f"{prefix}_pricing_type": (
                    "pricing_type",
                    last_non_null,
                ),
            }
        )
    )


def make_detail_for_cut(
    origin_raw,
    actual_raw_by_h,
    data_cut,
    actual_cut_by_h,
    chris_set,
    jit_set,
):
    origin = normalize_frame(origin_raw)
    actuals = {h: normalize_frame(df) for h, df in actual_raw_by_h.items()}

    scot_col = find_scot_oos_col(origin)

    # 1) Joint cohort.
    origin_asins = set(origin["asin"].dropna())
    joint = origin_asins & chris_set & jit_set

    # 2) Filter scot_oos=0 using latest row at or before origin cut.
    scot_hist = origin[
        origin["asin"].isin(joint)
        & origin["order_week"].notna()
        & (origin["order_week"] <= data_cut)
    ][["asin", "order_week", scot_col]].copy()

    scot_hist[scot_col] = pd.to_numeric(
        scot_hist[scot_col], errors="coerce"
    )

    scot_at_cut = (
        scot_hist.sort_values(["asin", "order_week"])
        .groupby("asin", as_index=False)
        .tail(1)
    )

    oos0_asins = set(
        scot_at_cut.loc[scot_at_cut[scot_col].eq(0), "asin"]
    )
    selected = sorted(joint & oos0_asins)

    # The three target order_weeks are the next three actual cut dates.
    # This keeps target_week aligned with the weekly snapshot/order_week date.
    target_week_by_h = {
        1: actual_cut_by_h[1],
        2: actual_cut_by_h[2],
        3: actual_cut_by_h[3],
    }

    # PLAN: values for those future target weeks in this origin snapshot.
    plan_rows = origin[
        origin["asin"].isin(selected)
        & origin["order_week"].isin(target_week_by_h.values())
    ].copy()

    plan = collapse_asin_week(plan_rows, "plan")
    plan["horizon"] = plan["order_week"].map(
        {week: h for h, week in target_week_by_h.items()}
    )

    # LAG1: latest available historical value at or before origin cut.
    hist = origin[
        origin["asin"].isin(selected)
        & origin["order_week"].notna()
        & (origin["order_week"] <= data_cut)
    ].sort_values(["asin", "order_week"])

    lag1 = (
        hist.groupby("asin", as_index=False)
        .agg(
            lag1_source_week=("order_week", "max"),
            lag1_ind_promotion=("ind_promotion", last_non_null),
            lag1_promotion_pricing_amount=(
                "promotion_pricing_amount",
                last_non_null,
            ),
            lag1_pricing_type=("pricing_type", last_non_null),
        )
    )

    # ACTUAL: same target week, read from corresponding later cut.
    actual_parts = []
    for h in [1, 2, 3]:
        target_week = target_week_by_h[h]
        rows = actuals[h][
            actuals[h]["asin"].isin(selected)
            & actuals[h]["order_week"].eq(target_week)
        ].copy()

        part = collapse_asin_week(rows, "actual")
        part["horizon"] = h
        part["actual_cut"] = actual_cut_by_h[h]
        actual_parts.append(part)

    actual = pd.concat(actual_parts, ignore_index=True)

    # Fixed ASIN × H1/H2/H3 grid. Missing values remain NaN.
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
        )
        .drop(columns=["order_week"], errors="ignore")
        .merge(lag1, on="asin", how="left")
        .merge(
            actual,
            left_on=["asin", "target_week", "horizon"],
            right_on=["asin", "order_week", "horizon"],
            how="left",
        )
        .drop(columns=["order_week"], errors="ignore")
    )

    detail.insert(0, "data_cut", data_cut)
    detail.insert(1, "scot_oos_filter", 0)

    return detail, {
        "origin_asins": len(origin_asins),
        "chris_jit_joint": len(joint),
        "scot_oos0": len(selected),
        "scot_col": scot_col,
    }


# -----------------------------
# Metrics / printing
# -----------------------------
def numeric_metrics(g, field):
    plan = pd.to_numeric(g[f"plan_{field}"], errors="coerce")
    lag1 = pd.to_numeric(g[f"lag1_{field}"], errors="coerce")
    actual = pd.to_numeric(g[f"actual_{field}"], errors="coerce")

    plan_ok = plan.notna() & actual.notna()
    lag_ok = lag1.notna() & actual.notna()
    both_ok = plan.notna() & lag1.notna() & actual.notna()

    plan_abs = (plan - actual).abs()
    lag_abs = (lag1 - actual).abs()

    result = {
        "rows": len(g),
        "plan_missing": int(plan.isna().sum()),
        "actual_missing": int(actual.isna().sum()),
        "lag1_missing": int(lag1.isna().sum()),
        "plan_actual_compared": int(plan_ok.sum()),
        "lag1_actual_compared": int(lag_ok.sum()),
        "both_compared": int(both_ok.sum()),
        "plan_mae": float(plan_abs[plan_ok].mean()) if plan_ok.any() else np.nan,
        "lag1_mae": float(lag_abs[lag_ok].mean()) if lag_ok.any() else np.nan,
        "plan_median_abs_gap": (
            float(plan_abs[plan_ok].median()) if plan_ok.any() else np.nan
        ),
        "lag1_median_abs_gap": (
            float(lag_abs[lag_ok].median()) if lag_ok.any() else np.nan
        ),
        "plan_p90_abs_gap": (
            float(plan_abs[plan_ok].quantile(0.90)) if plan_ok.any() else np.nan
        ),
        "lag1_p90_abs_gap": (
            float(lag_abs[lag_ok].quantile(0.90)) if lag_ok.any() else np.nan
        ),
    }

    if both_ok.any():
        p = plan_abs[both_ok]
        l = lag_abs[both_ok]
        result.update(
            {
                "plan_win_rate": float((p < l).mean()),
                "lag1_win_rate": float((l < p).mean()),
                "tie_rate": float(np.isclose(p, l).mean()),
                "lag1_improvement": float((p - l).mean()),
            }
        )
    else:
        result.update(
            {
                "plan_win_rate": np.nan,
                "lag1_win_rate": np.nan,
                "tie_rate": np.nan,
                "lag1_improvement": np.nan,
            }
        )

    return result


def categorical_metrics(g, field):
    plan = normalize_string(g[f"plan_{field}"])
    lag1 = normalize_string(g[f"lag1_{field}"])
    actual = normalize_string(g[f"actual_{field}"])

    plan_ok = plan.notna() & actual.notna()
    lag_ok = lag1.notna() & actual.notna()
    both_ok = plan.notna() & lag1.notna() & actual.notna()

    result = {
        "rows": len(g),
        "plan_missing": int(plan.isna().sum()),
        "actual_missing": int(actual.isna().sum()),
        "lag1_missing": int(lag1.isna().sum()),
        "plan_actual_compared": int(plan_ok.sum()),
        "lag1_actual_compared": int(lag_ok.sum()),
        "both_compared": int(both_ok.sum()),
        "plan_match_rate": (
            float(plan[plan_ok].eq(actual[plan_ok]).mean())
            if plan_ok.any()
            else np.nan
        ),
        "lag1_match_rate": (
            float(lag1[lag_ok].eq(actual[lag_ok]).mean())
            if lag_ok.any()
            else np.nan
        ),
    }

    if both_ok.any():
        p_match = plan[both_ok].eq(actual[both_ok])
        l_match = lag1[both_ok].eq(actual[both_ok])

        result.update(
            {
                "plan_win_rate": float((p_match & ~l_match).mean()),
                "lag1_win_rate": float((l_match & ~p_match).mean()),
                "tie_rate": float((p_match == l_match).mean()),
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


def fmt(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{float(x):.4f}"


def print_field_summary(g, field):
    print(f"\n  [{field}]")

    if field == "pricing_type":
        m = categorical_metrics(g, field)
        print(
            "    Missing counts: "
            f"PLAN={m['plan_missing']:,}, "
            f"ACTUAL={m['actual_missing']:,}, "
            f"LAG1={m['lag1_missing']:,}"
        )
        print(
            "    Comparable counts: "
            f"PLAN-vs-ACTUAL={m['plan_actual_compared']:,}, "
            f"LAG1-vs-ACTUAL={m['lag1_actual_compared']:,}, "
            f"all-three={m['both_compared']:,}"
        )
        print(
            "    Match rate: "
            f"PLAN={fmt(m['plan_match_rate'])}, "
            f"LAG1={fmt(m['lag1_match_rate'])}"
        )
        print(
            "    Winner rate on all-three rows: "
            f"PLAN={fmt(m['plan_win_rate'])}, "
            f"LAG1={fmt(m['lag1_win_rate'])}, "
            f"TIE={fmt(m['tie_rate'])}"
        )
    else:
        m = numeric_metrics(g, field)
        print(
            "    Missing counts: "
            f"PLAN={m['plan_missing']:,}, "
            f"ACTUAL={m['actual_missing']:,}, "
            f"LAG1={m['lag1_missing']:,}"
        )
        print(
            "    Comparable counts: "
            f"PLAN-vs-ACTUAL={m['plan_actual_compared']:,}, "
            f"LAG1-vs-ACTUAL={m['lag1_actual_compared']:,}, "
            f"all-three={m['both_compared']:,}"
        )
        print(
            "    MAE: "
            f"PLAN={fmt(m['plan_mae'])}, "
            f"LAG1={fmt(m['lag1_mae'])}"
        )
        print(
            "    Median abs gap: "
            f"PLAN={fmt(m['plan_median_abs_gap'])}, "
            f"LAG1={fmt(m['lag1_median_abs_gap'])}"
        )
        print(
            "    P90 abs gap: "
            f"PLAN={fmt(m['plan_p90_abs_gap'])}, "
            f"LAG1={fmt(m['lag1_p90_abs_gap'])}"
        )
        print(
            "    Winner rate on all-three rows: "
            f"PLAN={fmt(m['plan_win_rate'])}, "
            f"LAG1={fmt(m['lag1_win_rate'])}, "
            f"TIE={fmt(m['tie_rate'])}"
        )
        print(
            "    PLAN_MAE - LAG1_MAE equivalent improvement: "
            f"{fmt(m['lag1_improvement'])} "
            "(positive means LAG1 is better)"
        )


def print_top_asin_gaps(g, field, n=10):
    if field == "pricing_type":
        plan = normalize_string(g[f"plan_{field}"])
        lag1 = normalize_string(g[f"lag1_{field}"])
        actual = normalize_string(g[f"actual_{field}"])

        tmp = g[
            ["asin", "horizon", "target_week"]
        ].copy()
        tmp["plan"] = plan
        tmp["lag1"] = lag1
        tmp["actual"] = actual
        tmp = tmp[
            tmp["plan"].notna()
            & tmp["lag1"].notna()
            & tmp["actual"].notna()
        ]
        tmp["plan_match"] = tmp["plan"].eq(tmp["actual"])
        tmp["lag1_match"] = tmp["lag1"].eq(tmp["actual"])

        interesting = tmp[
            tmp["plan_match"].ne(tmp["lag1_match"])
        ].head(n)

        if len(interesting):
            print(f"    Example ASINs where PLAN and LAG1 differ (first {n}):")
            print(interesting.to_string(index=False))
        return

    plan = pd.to_numeric(g[f"plan_{field}"], errors="coerce")
    lag1 = pd.to_numeric(g[f"lag1_{field}"], errors="coerce")
    actual = pd.to_numeric(g[f"actual_{field}"], errors="coerce")

    tmp = g[["asin", "horizon", "target_week"]].copy()
    tmp["plan"] = plan
    tmp["lag1"] = lag1
    tmp["actual"] = actual
    tmp["plan_abs_gap"] = (plan - actual).abs()
    tmp["lag1_abs_gap"] = (lag1 - actual).abs()
    tmp["lag1_improvement"] = tmp["plan_abs_gap"] - tmp["lag1_abs_gap"]

    tmp = tmp[
        tmp["plan_abs_gap"].notna()
        & tmp["lag1_abs_gap"].notna()
    ]

    if tmp.empty:
        return

    print(f"    Largest PLAN gaps (top {n} ASIN rows):")
    print(
        tmp.sort_values("plan_abs_gap", ascending=False)
        .head(n)
        .to_string(index=False)
    )

    print(f"    Rows where LAG1 improves most over PLAN (top {n}):")
    print(
        tmp.sort_values("lag1_improvement", ascending=False)
        .head(n)
        .to_string(index=False)
    )


# -----------------------------
# Run
# -----------------------------
def run_analysis():
    s3 = boto3.client("s3")

    cuts = list_snapshot_cuts(BUCKET, DATA_PREFIX, s3)
    pairs = build_last3_origin_pairs(cuts)

    chris_set = load_asin_set(CHRIS_CSV)
    jit_set = load_asin_set(
        JIT_CSV,
        excluded=JIT_EXCLUDED_ASINS,
    )

    print("=" * 110)
    print("JOINT Chris ∩ JIT ∩ scot_oos=0 | PLAN vs LAG1 vs ACTUAL")
    print("=" * 110)
    print(
        f"Chris={len(chris_set):,} | "
        f"JIT(after exclusions)={len(jit_set):,} | "
        f"Chris∩JIT={len(chris_set & jit_set):,}"
    )
    print("Only latest 3 origin cuts. Nothing will be saved.")

    read_columns = [
        "asin",
        "order_week",
        *FIELDS,
        *SCOT_OOS_CANDIDATES,
    ]

    all_detail = []

    for cut_i, row in pairs.iterrows():
        data_cut = pd.Timestamp(row["data_cut"])
        actual_cut_by_h = {
            1: pd.Timestamp(row["h1_cut"]),
            2: pd.Timestamp(row["h2_cut"]),
            3: pd.Timestamp(row["h3_cut"]),
        }

        origin_raw = read_s3_csv_columns(
            BUCKET, row["origin_key"], read_columns, s3
        )

        actual_raw_by_h = {
            h: read_s3_csv_columns(
                BUCKET, row[f"h{h}_key"], read_columns, s3
            )
            for h in [1, 2, 3]
        }

        detail, cohort_info = make_detail_for_cut(
            origin_raw=origin_raw,
            actual_raw_by_h=actual_raw_by_h,
            data_cut=data_cut,
            actual_cut_by_h=actual_cut_by_h,
            chris_set=chris_set,
            jit_set=jit_set,
        )

        all_detail.append(detail)

        print("\n" + "#" * 110)
        print(
            f"CUT {cut_i + 1}: origin={data_cut.date()} | "
            f"H1={actual_cut_by_h[1].date()} | "
            f"H2={actual_cut_by_h[2].date()} | "
            f"H3={actual_cut_by_h[3].date()}"
        )
        print(
            f"Cohort: origin ASIN={cohort_info['origin_asins']:,} | "
            f"Chris∩JIT={cohort_info['chris_jit_joint']:,} | "
            f"{cohort_info['scot_col']}=0 => {cohort_info['scot_oos0']:,} ASIN"
        )

        for h in [1, 2, 3]:
            g = detail[detail["horizon"].eq(h)]
            print("\n" + "-" * 110)
            print(
                f"H{h} | target_week={actual_cut_by_h[h].date()} | "
                f"ASIN rows={len(g):,}"
            )

            for field in FIELDS:
                print_field_summary(g, field)

            # Print ASIN-level examples, but keep output readable.
            print("\n  ASIN-level largest differences:")
            for field in FIELDS:
                print(f"\n  >>> {field}")
                print_top_asin_gaps(
                    g,
                    field,
                    n=TOP_ASINS_TO_PRINT,
                )

    combined = pd.concat(all_detail, ignore_index=True)

    print("\n" + "=" * 110)
    print("COMBINED RESULTS ACROSS THE LATEST 3 CUTS")
    print("=" * 110)

    for h in [1, 2, 3]:
        g = combined[combined["horizon"].eq(h)]
        print("\n" + "-" * 110)
        print(f"Combined H{h} | rows={len(g):,} | ASIN={g['asin'].nunique():,}")
        for field in FIELDS:
            print_field_summary(g, field)

    print("\n" + "-" * 110)
    print(
        f"Combined ALL horizons | rows={len(combined):,} | "
        f"ASIN={combined['asin'].nunique():,}"
    )
    for field in FIELDS:
        print_field_summary(combined, field)

    print("\nInterpretation:")
    print("1) Numeric fields: lower MAE / median / P90 gap is better.")
    print("2) pricing_type: higher match rate is better.")
    print("3) PLAN win rate > LAG1 win rate means origin plan is more useful.")
    print("4) LAG1 win rate > PLAN win rate indicates plan lag/staleness.")
    print("5) Missing counts show how often PLAN or ACTUAL itself is unavailable.")

    return combined


detail_df = run_analysis()
