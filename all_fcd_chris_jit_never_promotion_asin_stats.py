"""Find latest-FCD test ASINs that never promote from first H1 to last H3.

Definition used here
--------------------
1. Discover every complete rolling pair:
      origin snapshot D
      SCOT FCD D + 1 day
      evaluation snapshot D + 21 days
2. Record the earliest matched FCD's H1 and the latest matched FCD's H3.
3. Build the candidate cohort only at the latest matched FCD:
      latest origin ∩ latest eval ∩ latest SCOT ∩ Chris ∩ JIT
4. Build every weekly order_week from the first H1 through the last H3.
5. Read the latest evaluation snapshot and keep only candidate ASINs that:
      - have an ind_promotion observation in every tested target week; and
      - have ind_promotion == 0 in every tested target week.

This is a cohort-statistics/oracle diagnostic.  It does not train a model.
"""

from pathlib import Path
import io
import re

import boto3
import numpy as np
import pandas as pd


S3_BUCKET = "amxl-asin-forecast590184089576"
DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)
SCOT_PREFIX = "amxl-asin-forecast-intern/scotforecast/"

CHRIS_ASIN_CSV = "asin_list_from_amxl_fcst_scot_to_chris_20260723.csv"
JIT_ASIN_CSV = None
JIT_ASIN_GLOB = "jit_asins_top90pct_demand*.csv"

OUTPUT_DIR = Path("all_fcd_chris_jit_never_promotion_stats")


def _normalize_asin(series):
    # pandas StringDtype preserves missing values instead of converting them
    # into the literal ASIN string "nan".
    return series.astype("string").str.strip()


def _find_asin_column(columns):
    by_lower = {str(c).strip().lower(): c for c in columns}
    if "asin" not in by_lower:
        raise KeyError(f"No ASIN column found. Columns={list(columns)}")
    return by_lower["asin"]


def _load_local_asin_set(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ASIN CSV not found: {path}")
    header = pd.read_csv(path, nrows=0)
    asin_col = _find_asin_column(header.columns)
    values = pd.read_csv(path, usecols=[asin_col])[asin_col]
    return set(_normalize_asin(values).replace("", np.nan).dropna()), path


def _resolve_jit_path(jit_path=None):
    if jit_path is not None:
        path = Path(jit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JIT ASIN CSV not found: {path}")
        return path

    roots = [Path.cwd(), Path.cwd() / "upload"]
    matches = sorted({
        p.resolve()
        for root in roots
        for p in root.glob(JIT_ASIN_GLOB)
        if p.is_file()
    })
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"No {JIT_ASIN_GLOB} found. Set JIT_ASIN_CSV explicitly."
        )
    raise RuntimeError(
        "Multiple JIT files found. Set JIT_ASIN_CSV explicitly:\n  "
        + "\n  ".join(str(p) for p in matches)
    )


def _list_s3_keys(s3, bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys.extend(
            obj["Key"]
            for obj in page.get("Contents", [])
            if obj.get("Key")
        )
    return keys


def discover_complete_fcd_pairs(s3, bucket=S3_BUCKET):
    data_pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )
    scot_pattern = re.compile(
        r"from(\d{4}-\d{2}-\d{2})_20weeks__"
        r"headbody_scot_fcst_(?:no_refresh|refresh)\.parquet$"
    )

    data_by_date = {}
    for key in _list_s3_keys(s3, bucket, DATA_PREFIX):
        match = data_pattern.search(key)
        if match:
            data_by_date[pd.Timestamp(match.group(1)).normalize()] = key

    scot_by_date = {}
    for key in _list_s3_keys(s3, bucket, SCOT_PREFIX):
        match = scot_pattern.search(key)
        if not match:
            continue
        fcd = pd.Timestamp(match.group(1)).normalize()
        previous = scot_by_date.get(fcd)
        if previous is None or (
            "no_refresh" in key and "no_refresh" not in previous
        ):
            scot_by_date[fcd] = key

    rows = []
    for data_cut, data_key in sorted(data_by_date.items()):
        scot_fcd = data_cut + pd.Timedelta(days=1)
        eval_cut = data_cut + pd.Timedelta(days=21)
        if eval_cut not in data_by_date or scot_fcd not in scot_by_date:
            continue
        rows.append({
            "data_cut": data_cut,
            "scot_fcd": scot_fcd,
            "eval_cut": eval_cut,
            "data_key": data_key,
            "eval_data_key": data_by_date[eval_cut],
            "scot_key": scot_by_date[scot_fcd],
        })

    pairs = pd.DataFrame(rows).sort_values("scot_fcd").reset_index(drop=True)
    if pairs.empty:
        raise RuntimeError("No complete origin/eval/SCOT FCD pairs were found.")
    return pairs


def _read_s3_csv(s3, bucket, key, wanted_columns):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    wanted = {c.lower() for c in wanted_columns}
    df = pd.read_csv(
        io.BytesIO(body),
        usecols=lambda c: str(c).strip().lower() in wanted,
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = sorted(wanted - set(df.columns))
    if missing:
        raise KeyError(f"S3 CSV {key} is missing columns: {missing}")
    return df


def _read_s3_scot_asins(s3, bucket, key):
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    df = pd.read_parquet(io.BytesIO(body), columns=["asin"])
    return set(_normalize_asin(df["asin"]).replace("", np.nan).dropna())


def run_all_fcd_never_promotion_stats(
    bucket=S3_BUCKET,
    chris_asin_csv=CHRIS_ASIN_CSV,
    jit_asin_csv=JIT_ASIN_CSV,
    output_dir=OUTPUT_DIR,
):
    """Run the all-FCD cohort statistic and save summary plus full ASIN list."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")

    chris_set, chris_path = _load_local_asin_set(chris_asin_csv)
    jit_path = _resolve_jit_path(jit_asin_csv)
    jit_set, _ = _load_local_asin_set(jit_path)
    pairs = discover_complete_fcd_pairs(s3, bucket=bucket)

    first_h1 = pd.Timestamp(pairs.iloc[0]["scot_fcd"]).normalize()
    latest_pair = pairs.iloc[-1]
    last_h3 = (
        pd.Timestamp(latest_pair["scot_fcd"]).normalize()
        + pd.Timedelta(days=14)
    )
    target_weeks = list(pd.date_range(first_h1, last_h3, freq="7D"))

    print("=" * 100)
    print("ALL-FCD NEVER-PROMOTION ASIN STATISTICS")
    print("=" * 100)
    print(f"Complete matched FCDs: {len(pairs):,}")
    print(f"First FCD / H1: {pairs.iloc[0]['scot_fcd'].date()}")
    print(f"Latest matched FCD: {latest_pair['scot_fcd'].date()}")
    print(f"Latest FCD H3: {last_h3.date()}")
    print(f"Weekly order_weeks checked: {len(target_weeks):,}")
    print(f"Chris ASINs: {len(chris_set):,} | file={chris_path}")
    print(f"JIT ASINs: {len(jit_set):,} | file={jit_path}")

    latest_origin = _read_s3_csv(
        s3, bucket, latest_pair["data_key"], ["asin"]
    )
    latest_eval_asins = _read_s3_csv(
        s3, bucket, latest_pair["eval_data_key"], ["asin"]
    )
    latest_origin_set = set(
        _normalize_asin(latest_origin["asin"]).replace("", np.nan).dropna()
    )
    latest_eval_set = set(
        _normalize_asin(latest_eval_asins["asin"]).replace("", np.nan).dropna()
    )
    latest_scot_set = _read_s3_scot_asins(
        s3, bucket, latest_pair["scot_key"]
    )
    latest_joint = (
        latest_origin_set
        & latest_eval_set
        & latest_scot_set
        & chris_set
        & jit_set
    )
    if not latest_joint:
        raise RuntimeError(
            "The latest-FCD Origin/Eval/SCOT/Chris/JIT intersection is empty."
        )
    print(
        "Latest-FCD joint cohort | "
        f"origin={len(latest_origin_set):,} | "
        f"eval={len(latest_eval_set):,} | "
        f"scot={len(latest_scot_set):,} | "
        f"final={len(latest_joint):,}"
    )

    # The latest evaluation snapshot contains the realized values through the
    # latest matched FCD's H3 and retains the earlier historical test weeks.
    latest_eval = _read_s3_csv(
        s3,
        bucket,
        latest_pair["eval_data_key"],
        ["asin", "order_week", "ind_promotion"],
    )
    latest_eval["asin"] = _normalize_asin(latest_eval["asin"])
    latest_eval["order_week"] = pd.to_datetime(
        latest_eval["order_week"], errors="coerce"
    ).dt.normalize()
    latest_eval["ind_promotion"] = pd.to_numeric(
        latest_eval["ind_promotion"], errors="coerce"
    )
    latest_eval = latest_eval[
        latest_eval["asin"].isin(latest_joint)
        & latest_eval["order_week"].isin(target_weeks)
    ].copy()

    by_asin_week = (
        latest_eval.groupby(["asin", "order_week"], as_index=False)
        .agg(ind_promotion=("ind_promotion", "max"))
    )
    by_asin = (
        by_asin_week.groupby("asin", as_index=False)
        .agg(
            observed_test_weeks=("order_week", "nunique"),
            promotion_week_count=("ind_promotion", lambda s: int((s > 0).sum())),
            max_ind_promotion=("ind_promotion", "max"),
            sum_ind_promotion=("ind_promotion", "sum"),
        )
    )
    by_asin["expected_test_weeks"] = len(target_weeks)
    by_asin["complete_test_week_coverage"] = (
        by_asin["observed_test_weeks"] == len(target_weeks)
    )
    by_asin["never_promotional"] = (
        by_asin["complete_test_week_coverage"]
        & by_asin["max_ind_promotion"].eq(0.0)
    )

    # ASINs with zero observed rows must appear in the audit table and must not
    # be classified as never-promotional.
    audit = pd.DataFrame({"asin": sorted(latest_joint)}).merge(
        by_asin,
        on="asin",
        how="left",
        validate="one_to_one",
    )
    audit["expected_test_weeks"] = len(target_weeks)
    audit["observed_test_weeks"] = audit["observed_test_weeks"].fillna(0).astype(int)
    audit["promotion_week_count"] = audit["promotion_week_count"].fillna(0).astype(int)
    audit["complete_test_week_coverage"] = (
        audit["observed_test_weeks"] == len(target_weeks)
    )
    audit["never_promotional"] = (
        audit["complete_test_week_coverage"]
        & audit["max_ind_promotion"].eq(0.0)
    )

    never_promo = audit[audit["never_promotional"]].copy()
    never_promo.insert(1, "first_test_week", first_h1)
    never_promo.insert(2, "last_test_week", last_h3)
    never_promo = never_promo.sort_values("asin").reset_index(drop=True)

    complete_coverage_count = int(audit["complete_test_week_coverage"].sum())
    never_promo_count = len(never_promo)
    latest_joint_count = len(latest_joint)

    summary = pd.DataFrame([{
        "complete_matched_fcd_count": len(pairs),
        "first_fcd_h1": first_h1,
        "latest_matched_fcd": latest_pair["scot_fcd"],
        "latest_fcd_h3": last_h3,
        "distinct_test_week_count": len(target_weeks),
        "latest_fcd_joint_asin_count": latest_joint_count,
        "complete_test_week_coverage_asin_count": complete_coverage_count,
        "never_promotional_asin_count": never_promo_count,
        "never_promotional_share_of_latest_joint": (
            never_promo_count / latest_joint_count
        ),
        "never_promotional_share_of_complete_coverage": (
            never_promo_count / max(complete_coverage_count, 1)
        ),
        "latest_eval_cut": latest_pair["eval_cut"],
        "latest_eval_s3_key": latest_pair["eval_data_key"],
    }])

    summary_path = output_dir / "all_fcd_never_promotion_summary.csv"
    list_path = output_dir / "all_fcd_never_promotion_asin_list.csv"
    audit_path = output_dir / "all_fcd_promotion_coverage_audit.csv"
    weeks_path = output_dir / "all_fcd_tested_target_weeks.csv"

    summary.to_csv(summary_path, index=False)
    never_promo.to_csv(list_path, index=False)
    audit.to_csv(audit_path, index=False)
    pd.DataFrame({"order_week": target_weeks}).to_csv(weeks_path, index=False)

    print("\n" + "=" * 100)
    print("FINAL STATISTICS")
    print("=" * 100)
    print(f"Test range: {first_h1.date()} through {last_h3.date()}")
    print(f"Latest-FCD joint ASINs: {latest_joint_count:,}")
    print(f"ASINs with every tested week present: {complete_coverage_count:,}")
    print(f"Never-promotional ASINs: {never_promo_count:,}")
    print(
        "Never-promotional share of latest-FCD test cohort: "
        f"{never_promo_count / latest_joint_count:.2%}"
    )
    print("\nFirst 50 never-promotional ASINs:")
    print(never_promo[["asin"]].head(50).to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved full ASIN list: {list_path}")
    print(f"Saved coverage audit: {audit_path}")
    print(f"Saved tested week list: {weeks_path}")

    return {
        "summary": summary,
        "never_promotion_asins": never_promo,
        "coverage_audit": audit,
        "tested_weeks": pd.DataFrame({"order_week": target_weeks}),
        "pairs": pairs,
    }


if __name__ == "__main__":
    all_fcd_never_promotion_results = run_all_fcd_never_promotion_stats()
