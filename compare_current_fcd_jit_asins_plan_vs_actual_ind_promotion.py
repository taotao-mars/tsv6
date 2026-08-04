import io
import re
from pathlib import Path

import boto3
import numpy as np
import pandas as pd


# ============================================================
# 1. 当前 FCD 设置
# ============================================================

S3_BUCKET = "amxl-asin-forecast590184089576"

DATA_CUT = pd.Timestamp("2025-10-04")
FCD = pd.Timestamp("2025-10-05")

TARGET_WEEKS = [
    FCD,                              # H1
    FCD + pd.Timedelta(days=7),       # H2
    FCD + pd.Timedelta(days=14),      # H3
]

HORIZON_MAP = {
    TARGET_WEEKS[0]: "H1",
    TARGET_WEEKS[1]: "H2",
    TARGET_WEEKS[2]: "H3",
}

DEALS_KEY = (
    "amxl-asin-forecast-intern/asin_deals/"
    "asin_deals_20251004.csv000"
)

EVAL_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_2025-10-25"
)

OUTPUT_DIR = Path("jit_current_fcd_promo_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 自动找到本地 JIT ASIN CSV
# ============================================================

def find_jit_asin_csv():
    patterns = [
        "jit_asins_top90pct_demand*.csv",
        "**/jit_asins_top90pct_demand*.csv",
    ]

    candidates = []

    search_roots = [
        Path.cwd(),
        Path("/home/sagemaker-user"),
    ]

    for root in search_roots:
        if not root.exists():
            continue

        for pattern in patterns:
            try:
                candidates.extend(root.glob(pattern))
            except Exception:
                pass

    candidates = sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file()
        }
    )

    if not candidates:
        raise FileNotFoundError(
            "找不到 jit_asins_top90pct_demand*.csv。"
            "请把 CSV 上传到当前 notebook 目录。"
        )

    # 如果存在多个同名版本，使用最后修改的一个
    selected = max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )

    print("JIT ASIN CSV:", selected)

    if len(candidates) > 1:
        print("发现多个候选文件，使用最后修改的文件。")
        for path in candidates:
            print("  ", path)

    return selected


ASIN_CSV_PATH = find_jit_asin_csv()


# ============================================================
# 3. S3 读取函数
# ============================================================

s3_client = boto3.client("s3")


def read_s3_csv(bucket, key):
    print(f"\nReading s3://{bucket}/{key}")

    body = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )["Body"].read()

    df = pd.read_csv(io.BytesIO(body))

    print(
        f"Loaded rows={len(df):,}, "
        f"columns={len(df.columns):,}"
    )

    return df


def list_s3_keys(bucket, prefix):
    keys = []

    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix=prefix,
    ):
        for obj in page.get("Contents", []):
            if obj.get("Key"):
                keys.append(obj["Key"])

    return keys


def find_eval_key():
    keys = list_s3_keys(
        S3_BUCKET,
        EVAL_PREFIX,
    )

    pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"2025-10-25_?ETLM_[vV]3\.csv$"
    )

    matches = [
        key
        for key in keys
        if pattern.search(key)
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "无法唯一确定 2025-10-25 eval 文件："
            f"{matches}"
        )

    return matches[0]


EVAL_KEY = find_eval_key()

print("\nDeals file:", DEALS_KEY)
print("Eval file:", EVAL_KEY)


# ============================================================
# 4. 读取 JIT cohort、planned deals、eval actual
# ============================================================

asin_df = pd.read_csv(ASIN_CSV_PATH)

asin_column_map = {
    str(column).strip().lower(): column
    for column in asin_df.columns
}

if "asin" not in asin_column_map:
    raise KeyError(
        f"ASIN CSV 中找不到 asin 列。现有列：{list(asin_df.columns)}"
    )

asin_col = asin_column_map["asin"]

jit_asins = sorted(
    set(
        asin_df[asin_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    - {"", "nan"}
)

print("\nUnique JIT ASINs:", len(jit_asins))

deals_raw = read_s3_csv(
    S3_BUCKET,
    DEALS_KEY,
)

eval_raw = read_s3_csv(
    S3_BUCKET,
    EVAL_KEY,
)


# ============================================================
# 5. 构造每个 ASIN 的 H1/H2/H3
# ============================================================

target_grid = (
    pd.MultiIndex.from_product(
        [
            jit_asins,
            TARGET_WEEKS,
        ],
        names=[
            "asin",
            "order_week",
        ],
    )
    .to_frame(index=False)
)

target_grid["forecast_horizon"] = (
    target_grid["order_week"]
    .map(HORIZON_MAP)
)

expected_rows = len(jit_asins) * 3

assert len(target_grid) == expected_rows

print("\nTarget weeks:")

for week, horizon in HORIZON_MAP.items():
    print(f"{horizon}: {week.date()}")

print("Expected ASIN-week rows:", expected_rows)


# ============================================================
# 6. 从 origin-time deals 构造 planned_ind_promotion
#
# 逻辑与老板的 SQL 一致：
# 只要至少一个 deal 覆盖该 asin + order_week，就等于 1。
# ============================================================

required_deal_cols = {
    "asin",
    "asin_promo_start_week",
    "asin_promo_end_week",
}

missing_deal_cols = (
    required_deal_cols
    - set(deals_raw.columns)
)

if missing_deal_cols:
    raise KeyError(
        f"Deals table 缺少字段：{missing_deal_cols}"
    )

deals = deals_raw.copy()

deals["asin"] = (
    deals["asin"]
    .astype(str)
    .str.strip()
)

deals["asin_promo_start_week"] = pd.to_datetime(
    deals["asin_promo_start_week"],
    errors="coerce",
).dt.normalize()

deals["asin_promo_end_week"] = pd.to_datetime(
    deals["asin_promo_end_week"],
    errors="coerce",
).dt.normalize()

# 只保留当前 JIT ASIN
deals = deals[
    deals["asin"].isin(set(jit_asins))
].copy()

# 只保留可能与 H1-H3 重叠的 deal
deals = deals[
    deals["asin_promo_start_week"].notna()
    & deals["asin_promo_end_week"].notna()
    & (
        deals["asin_promo_start_week"]
        <= max(TARGET_WEEKS)
    )
    & (
        deals["asin_promo_end_week"]
        >= min(TARGET_WEEKS)
    )
].copy()

# 按 ASIN join，然后检查目标周是否落在 deal 区间中
plan_candidates = target_grid.merge(
    deals[
        [
            "asin",
            "asin_promo_start_week",
            "asin_promo_end_week",
        ]
    ],
    on="asin",
    how="left",
)

covered_mask = (
    plan_candidates[
        "asin_promo_start_week"
    ].notna()
    & (
        plan_candidates["order_week"]
        >= plan_candidates[
            "asin_promo_start_week"
        ]
    )
    & (
        plan_candidates["order_week"]
        <= plan_candidates[
            "asin_promo_end_week"
        ]
    )
)

covered_deals = plan_candidates[
    covered_mask
].copy()

# 一个 ASIN-week 可以匹配多个 deal，
# 但是 ind_promotion 仍然只等于 1
planned_coverage = (
    covered_deals
    .groupby(
        [
            "asin",
            "order_week",
        ],
        as_index=False,
    )
    .agg(
        matching_deal_count=(
            "asin_promo_start_week",
            "size",
        ),
        earliest_matched_start=(
            "asin_promo_start_week",
            "min",
        ),
        latest_matched_start=(
            "asin_promo_start_week",
            "max",
        ),
        earliest_matched_end=(
            "asin_promo_end_week",
            "min",
        ),
        latest_matched_end=(
            "asin_promo_end_week",
            "max",
        ),
    )
)

planned = target_grid.merge(
    planned_coverage,
    on=[
        "asin",
        "order_week",
    ],
    how="left",
    validate="one_to_one",
)

planned["matching_deal_count"] = (
    planned["matching_deal_count"]
    .fillna(0)
    .astype(int)
)

planned["planned_ind_promotion"] = (
    planned["matching_deal_count"] > 0
).astype(int)


# ============================================================
# 7. 从 eval snapshot 读取 actual_ind_promotion
# ============================================================

required_eval_cols = {
    "asin",
    "order_week",
    "ind_promotion",
}

missing_eval_cols = (
    required_eval_cols
    - set(eval_raw.columns)
)

if missing_eval_cols:
    raise KeyError(
        f"Eval table 缺少字段：{missing_eval_cols}"
    )

actual = eval_raw[
    [
        "asin",
        "order_week",
        "ind_promotion",
    ]
].copy()

actual["asin"] = (
    actual["asin"]
    .astype(str)
    .str.strip()
)

actual["order_week"] = pd.to_datetime(
    actual["order_week"],
    errors="coerce",
).dt.normalize()

actual["actual_ind_promotion"] = pd.to_numeric(
    actual["ind_promotion"],
    errors="coerce",
)

actual = actual[
    actual["asin"].isin(set(jit_asins))
    & actual["order_week"].isin(
        set(TARGET_WEEKS)
    )
].copy()

# 如果出现重复 ASIN-week，只要任意一行为 1，
# actual_ind_promotion 就等于 1
actual_by_week = (
    actual
    .groupby(
        [
            "asin",
            "order_week",
        ],
        as_index=False,
    )
    .agg(
        actual_ind_promotion=(
            "actual_ind_promotion",
            "max",
        ),
        eval_row_count=(
            "ind_promotion",
            "size",
        ),
    )
)


# ============================================================
# 8. Planned 与 Actual 比较
# ============================================================

comparison = planned.merge(
    actual_by_week,
    on=[
        "asin",
        "order_week",
    ],
    how="left",
    validate="one_to_one",
)

comparison["actual_ind_promotion"] = pd.to_numeric(
    comparison["actual_ind_promotion"],
    errors="coerce",
)

comparison["comparison_class"] = np.select(
    [
        comparison[
            "actual_ind_promotion"
        ].isna(),

        (
            comparison[
                "planned_ind_promotion"
            ].eq(1)
            & comparison[
                "actual_ind_promotion"
            ].gt(0)
        ),

        (
            comparison[
                "planned_ind_promotion"
            ].eq(0)
            & comparison[
                "actual_ind_promotion"
            ].le(0)
        ),

        (
            comparison[
                "planned_ind_promotion"
            ].eq(1)
            & comparison[
                "actual_ind_promotion"
            ].le(0)
        ),

        (
            comparison[
                "planned_ind_promotion"
            ].eq(0)
            & comparison[
                "actual_ind_promotion"
            ].gt(0)
        ),
    ],
    [
        "MISSING_ACTUAL",
        "TP",
        "TN",
        "FP",
        "FN",
    ],
    default="UNEXPECTED",
)

comparison["is_match"] = comparison[
    "comparison_class"
].isin(["TP", "TN"])


# ============================================================
# 9. 计算 Accuracy、Precision、Recall、F1
# ============================================================

def safe_div(a, b):
    return a / b if b else np.nan


summary_rows = []

for horizon in [
    "ALL",
    "H1",
    "H2",
    "H3",
]:
    if horizon == "ALL":
        subset_all = comparison.copy()
    else:
        subset_all = comparison[
            comparison["forecast_horizon"]
            == horizon
        ].copy()

    subset = subset_all[
        subset_all[
            "actual_ind_promotion"
        ].notna()
    ].copy()

    planned_y = subset[
        "planned_ind_promotion"
    ].astype(int)

    actual_y = subset[
        "actual_ind_promotion"
    ].gt(0).astype(int)

    tp = int(
        (
            (planned_y == 1)
            & (actual_y == 1)
        ).sum()
    )

    tn = int(
        (
            (planned_y == 0)
            & (actual_y == 0)
        ).sum()
    )

    fp = int(
        (
            (planned_y == 1)
            & (actual_y == 0)
        ).sum()
    )

    fn = int(
        (
            (planned_y == 0)
            & (actual_y == 1)
        ).sum()
    )

    n = len(subset)

    precision = safe_div(
        tp,
        tp + fp,
    )

    recall = safe_div(
        tp,
        tp + fn,
    )

    f1 = (
        safe_div(
            2 * precision * recall,
            precision + recall,
        )
        if (
            np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
        )
        else np.nan
    )

    summary_rows.append(
        {
            "data_cut": DATA_CUT,
            "fcd": FCD,
            "forecast_horizon": horizon,

            "expected_rows": len(
                subset_all
            ),

            "rows_with_actual": n,

            "missing_actual_rows": int(
                subset_all[
                    "actual_ind_promotion"
                ].isna().sum()
            ),

            "planned_positive_count": int(
                planned_y.sum()
            ),

            "actual_positive_count": int(
                actual_y.sum()
            ),

            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,

            "planned_deal_rate": safe_div(
                planned_y.sum(),
                n,
            ),

            "actual_deal_rate": safe_div(
                actual_y.sum(),
                n,
            ),

            "mismatch_rate": safe_div(
                fp + fn,
                n,
            ),

            "agreement_accuracy": safe_div(
                tp + tn,
                n,
            ),

            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

summary = pd.DataFrame(summary_rows)


# ============================================================
# 10. 严格检查 order_week
# ============================================================

assert len(comparison) == expected_rows

asin_week_count = (
    comparison
    .groupby("asin")["order_week"]
    .nunique()
)

assert asin_week_count.eq(3).all(), (
    "至少一个 ASIN 没有完整的 H1/H2/H3"
)

observed_weeks = set(
    comparison["order_week"].unique()
)

assert observed_weeks == set(TARGET_WEEKS), (
    f"order_week 不匹配：{observed_weeks}"
)


# ============================================================
# 11. 保存并打印结果
# ============================================================

detail_path = (
    OUTPUT_DIR
    / "jit_asins_planned_vs_actual_ind_promotion_detail_fcd_2025-10-05.csv"
)

summary_path = (
    OUTPUT_DIR
    / "jit_asins_planned_vs_actual_ind_promotion_summary_fcd_2025-10-05.csv"
)

comparison.to_csv(
    detail_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 110)
print("PLANNED VS ACTUAL IND_PROMOTION SUMMARY")
print("=" * 110)

print(
    summary.round(6).to_string(
        index=False
    )
)

print("\nTP/TN/FP/FN by horizon:")

print(
    comparison.groupby(
        [
            "forecast_horizon",
            "comparison_class",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nLargest mismatch samples:")

mismatch_rows = comparison[
    comparison["comparison_class"].isin(
        [
            "FP",
            "FN",
            "MISSING_ACTUAL",
        ]
    )
]

if mismatch_rows.empty:
    print("No mismatches.")
else:
    print(
        mismatch_rows[
            [
                "asin",
                "order_week",
                "forecast_horizon",
                "planned_ind_promotion",
                "actual_ind_promotion",
                "comparison_class",
                "matching_deal_count",
                "earliest_matched_start",
                "latest_matched_start",
                "earliest_matched_end",
                "latest_matched_end",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

print("\nSaved detail:", detail_path)
print("Saved summary:", summary_path)

result = {
    "summary": summary,
    "detail": comparison,
    "detail_path": str(detail_path),
    "summary_path": str(summary_path),
}
