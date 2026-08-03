"""
Joint exposure and demand forecasting for rolling three-week horizons.

The pipeline:
  1. builds the eligible ASIN cohort for each rolling cut;
  2. constructs historical, future-known, graph, and DPH proxy features;
  3. trains a joint exposure-and-demand model;
  4. generates demand quantiles and exposure forecasts;
  5. aligns predictions with SCOT for evaluation;
  6. saves one prediction.csv for each completed rolling cut.

The three standardized WAPE functions are included at the top of this file:
  - calculate_wape_using_lp_oos2
  - quick_error_check
  - weekly_error_check

The rolling pipeline prepares their required columns without changing the scoring formulas.
"""

import os
import time
import multiprocessing as mp
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
np.random.seed(42)


# =============================================================================
# Standardized WAPE Evaluation Functions (P50 / P70 / P90)
#
# These functions are kept verbatim from the approved notebook evaluation cell,
# including the P90 extension. The model and rolling pipeline only prepare the
# expected columns and call these functions; the scoring formulas are unchanged.
# =============================================================================

def calculate_wape_using_lp_oos2(df, quantiles, remove_oos_dp=False, source='lp'):

    print(f"Shape when read in is {df.shape}")

    if remove_oos_dp:
        if source == 'lp':
            df = df[df['oos_status'] == 0]
        if source == 'amxl':
            df = df[df['amxl_oos'] == 0]
    else:
        df = df.copy(deep=True)

    print(f"Shape after remove oos is {df.shape}")

    for quantile in quantiles:

        print(f"Working on quantile {quantile}")

        amxl_col = f'p{int(quantile*100)}_amxl'
        scot_col = f'p{int(quantile*100)}_scot'

        amxl_overbias_col = f'p{int(quantile*100)}_amxl_overbias'
        scot_overbias_col = f'p{int(quantile*100)}_scot_overbias'

        amxl_underbias_col = f'p{int(quantile*100)}_amxl_underbias'
        scot_underbias_col = f'p{int(quantile*100)}_scot_underbias'

        amxl_penalty_col = f'p{int(quantile*100)}_amxl_penalty'
        scot_penalty_col = f'p{int(quantile*100)}_scot_penalty'

        amxl_ob_wape_col = f'p{int(quantile*100)}_amxl_ob_wape'
        scot_ob_wape_col = f'p{int(quantile*100)}_scot_ob_wape'

        amxl_ub_wape_col = f'p{int(quantile*100)}_amxl_ub_wape'
        scot_ub_wape_col = f'p{int(quantile*100)}_scot_ub_wape'

        amxl_wape_col = f'p{int(quantile*100)}_amxl_wape'
        scot_wape_col = f'p{int(quantile*100)}_scot_wape'

        delta_wape_col = f'p{int(quantile*100)}_delta_wape'

        df = df.dropna(subset=[amxl_col, scot_col])

        print(f"Shape after remove when SCOT has null fcst is {df.shape}")

        df[amxl_wape_col] = np.nan
        df[scot_wape_col] = np.nan

        df[amxl_overbias_col] = np.where(
            (df[amxl_col] >= df['fbi_demand']),
            abs(df[amxl_col] - df['fbi_demand']) * (1 - quantile),
            0
        )
        df[scot_overbias_col] = np.where(
            (df[scot_col] >= df['fbi_demand']),
            abs(df[scot_col] - df['fbi_demand']) * (1 - quantile),
            0
        )
        df[amxl_ob_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_overbias_col] / df['fbi_demand'],
            np.nan
        )
        df[scot_ob_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_overbias_col] / df['fbi_demand'],
            np.nan
        )

        df[amxl_underbias_col] = np.where(
            (df[amxl_col] < df['fbi_demand']),
            abs(df[amxl_col] - df['fbi_demand']) * quantile,
            0
        )
        df[scot_underbias_col] = np.where(
            (df[scot_col] < df['fbi_demand']),
            abs(df[scot_col] - df['fbi_demand']) * quantile,
            0
        )
        df[amxl_ub_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_underbias_col] / df['fbi_demand'],
            np.nan
        )
        df[scot_ub_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_underbias_col] / df['fbi_demand'],
            np.nan
        )

        df[amxl_penalty_col] = df[amxl_overbias_col] + df[amxl_underbias_col]
        df[scot_penalty_col] = df[scot_overbias_col] + df[scot_underbias_col]

        df[amxl_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[amxl_penalty_col] / df['fbi_demand'],
            np.nan
        )

        df[scot_wape_col] = np.where(
            df['fbi_demand'] != 0,
            df[scot_penalty_col] / df['fbi_demand'],
            np.nan
        )

        df[delta_wape_col] = df[amxl_wape_col] - df[scot_wape_col]

    return df


def quick_error_check(df, cols):
    original_output = df[cols].sum() / df['fbi_demand'].sum()

    if 'p50_amxl_penalty' in cols:
        penalty_diff = (df['p50_amxl_penalty'].sum() - df['p50_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()
    elif 'p70_amxl_penalty' in cols:
        penalty_diff = (df['p70_amxl_penalty'].sum() - df['p70_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()
    elif 'p90_amxl_penalty' in cols:
        penalty_diff = (df['p90_amxl_penalty'].sum() - df['p90_scot_penalty'].sum()) * 10000 / df['fbi_demand'].sum()

    return original_output, penalty_diff


def weekly_error_check(df, cols, cols_type):
    """
    Parameters:
    df: DataFrame
    cols: list of columns to analyze (unused, kept for signature compatibility)
    cols_type: 'p50', 'p70', or 'p90' -- selects which quantile's per-horizon breakdown to build
    """
    if df.shape[0] > 0:

        if cols_type == 'p50':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p50_amxl_penalty': 'sum',
                'p50_scot_penalty': 'sum',
                'p50_amxl_overbias': 'sum',
                'p50_scot_overbias': 'sum',
                'p50_amxl_underbias': 'sum',
                'p50_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p50_amxl_penalty'] < result['p50_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p50_amxl_overbias'] < result['p50_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p50_amxl_underbias'] < result['p50_scot_underbias'], 'win', 'lose')
            result['p50_amxl_wape'] = result['p50_amxl_penalty'] / result['fbi_demand']
            result['p50_scot_wape'] = result['p50_scot_penalty'] / result['fbi_demand']
            result['p50_diff_bps'] = (result['p50_amxl_penalty'] - result['p50_scot_penalty']) * 10000 / result['fbi_demand']

        if cols_type == 'p70':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p70_amxl_penalty': 'sum',
                'p70_scot_penalty': 'sum',
                'p70_amxl_overbias': 'sum',
                'p70_scot_overbias': 'sum',
                'p70_amxl_underbias': 'sum',
                'p70_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p70_amxl_penalty'] < result['p70_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p70_amxl_overbias'] < result['p70_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p70_amxl_underbias'] < result['p70_scot_underbias'], 'win', 'lose')
            result['p70_amxl_wape'] = result['p70_amxl_penalty'] / result['fbi_demand']
            result['p70_scot_wape'] = result['p70_scot_penalty'] / result['fbi_demand']
            result['penalty_diff_bps'] = (result['p70_amxl_penalty'] - result['p70_scot_penalty']) * 10000 / result['fbi_demand']

        if cols_type == 'p90':
            result = df.groupby(
                ['fcst_week_index', ]
            ).agg({
                'p90_amxl_penalty': 'sum',
                'p90_scot_penalty': 'sum',
                'p90_amxl_overbias': 'sum',
                'p90_scot_overbias': 'sum',
                'p90_amxl_underbias': 'sum',
                'p90_scot_underbias': 'sum',
                'fbi_demand': 'sum',
                'p50_amxl': 'sum',
                'p70_amxl': 'sum',
                'p90_amxl': 'sum',
                'p50_scot': 'sum',
                'p70_scot': 'sum',
                'p90_scot': 'sum'
            }).reset_index()

            result['penalty_win'] = np.where(result['p90_amxl_penalty'] < result['p90_scot_penalty'], 'win', 'lose')
            result['over_win'] = np.where(result['p90_amxl_overbias'] < result['p90_scot_overbias'], 'win', 'lose')
            result['under_win'] = np.where(result['p90_amxl_underbias'] < result['p90_scot_underbias'], 'win', 'lose')
            result['p90_amxl_wape'] = result['p90_amxl_penalty'] / result['fbi_demand']
            result['p90_scot_wape'] = result['p90_scot_penalty'] / result['fbi_demand']
            result['penalty_diff_bps'] = (result['p90_amxl_penalty'] - result['p90_scot_penalty']) * 10000 / result['fbi_demand']
        return result
    else:
        print('df has zero rows')
        return pd.DataFrame()


# =============================================================================
# Standardized SCOT Alignment and WAPE Evaluation
# =============================================================================

def _evaluate_standard_wape_against_scot(
    result,
    scot_df,
    data_raw1=None,
    asin_stats=None,
    remove_oos_dp=True,
    source="lp",
):
    """
    Align the joint-model forecasts with real SCOT P50/P70/P90 forecasts and
    run the three standardized WAPE functions above.

    The evaluator does not alter any WAPE formula. It only prepares the exact
    AMXL/SCOT columns required by the approved evaluation functions.
    """
    if not isinstance(result, dict) or "forecast_df" not in result:
        raise ValueError("result must be a dict containing forecast_df.")

    forecast_df = result["forecast_df"].copy()
    scot = scot_df.copy()

    forecast_df.columns = [str(c).strip() for c in forecast_df.columns]
    scot.columns = [str(c).strip() for c in scot.columns]

    required_forecast = {
        "asin", "order_week", "fcst_week_index", "fbi_demand",
        "p50_amxl", "p70_amxl", "p90_amxl",
    }
    missing_forecast = sorted(required_forecast - set(forecast_df.columns))
    if missing_forecast:
        raise ValueError(
            "Joint forecast is missing columns required by standardized WAPE: "
            + ", ".join(missing_forecast)
        )

    # Use the exact SCOT schema supplied by the production forecast file.
    required_scot = {
        "asin", "order_week",
        "forecast_qty_p50", "forecast_qty_p70", "forecast_qty_p90",
    }
    missing_scot = sorted(required_scot - set(scot.columns))
    if missing_scot:
        raise ValueError(
            "SCOT data is missing columns required by standardized WAPE: "
            + ", ".join(missing_scot)
        )

    selected_scot_cols = {
        "p50_scot": "forecast_qty_p50",
        "p70_scot": "forecast_qty_p70",
        "p90_scot": "forecast_qty_p90",
    }

    for frame in (forecast_df, scot):
        frame["asin"] = frame["asin"].astype(str).str.strip()
        frame["order_week"] = pd.to_datetime(frame["order_week"], errors="coerce")

    forecast_df = forecast_df.dropna(subset=["asin", "order_week"]).copy()
    scot = scot.dropna(subset=["asin", "order_week"]).copy()

    for target, source_col in selected_scot_cols.items():
        scot[target] = pd.to_numeric(scot[source_col], errors="coerce")

    scot_keep = (
        scot[["asin", "order_week", "p50_scot", "p70_scot", "p90_scot"]]
        .groupby(["asin", "order_week"], as_index=False)
        .agg({
            "p50_scot": "mean",
            "p70_scot": "mean",
            "p90_scot": "mean",
        })
    )

    forecast_df_scot_real = forecast_df.merge(
        scot_keep,
        on=["asin", "order_week"],
        how="inner",
        validate="many_to_one",
    )

    if forecast_df_scot_real.empty:
        raise RuntimeError(
            "No ASIN-week rows matched between joint forecasts and SCOT. "
            "Check ASIN formatting and order_week anchors."
        )

    forecast_df_scot_real["p70_scot"] = np.maximum(
        forecast_df_scot_real["p70_scot"],
        forecast_df_scot_real["p50_scot"],
    )
    forecast_df_scot_real["p90_scot"] = np.maximum(
        forecast_df_scot_real["p90_scot"],
        forecast_df_scot_real["p70_scot"],
    )

    print("\n" + "=" * 80)
    print("REAL SCOT ALIGNMENT")
    print("=" * 80)
    print(f"Joint forecast rows: {len(forecast_df):,}")
    print(f"Matched rows: {len(forecast_df_scot_real):,}")
    print(f"Matched ASINs: {forecast_df_scot_real['asin'].nunique():,}")
    print(
        "Matched weeks:",
        forecast_df_scot_real["order_week"].min(),
        "to",
        forecast_df_scot_real["order_week"].max(),
    )

    wape_df = calculate_wape_using_lp_oos2(
        forecast_df_scot_real,
        [0.5, 0.7, 0.9],
        remove_oos_dp=remove_oos_dp,
        source=source,
    )

    cols_by_quantile = {
        "p50": [
            "p50_amxl_penalty", "p50_scot_penalty",
            "p50_amxl_overbias", "p50_scot_overbias",
            "p50_amxl_underbias", "p50_scot_underbias",
            "fbi_demand",
        ],
        "p70": [
            "p70_amxl_penalty", "p70_scot_penalty",
            "p70_amxl_overbias", "p70_scot_overbias",
            "p70_amxl_underbias", "p70_scot_underbias",
            "fbi_demand",
        ],
        "p90": [
            "p90_amxl_penalty", "p90_scot_penalty",
            "p90_amxl_overbias", "p90_scot_overbias",
            "p90_amxl_underbias", "p90_scot_underbias",
            "fbi_demand",
        ],
    }

    outputs = {
        "forecast_df_scot_real": forecast_df_scot_real,
        "wape_df": wape_df,
    }

    for quantile, cols in cols_by_quantile.items():
        wape, penalty_diff = quick_error_check(wape_df, cols)
        horizon_wape = weekly_error_check(wape_df, cols, cols_type=quantile)
        outputs[f"{quantile}_wape"] = wape
        outputs[f"{quantile}_penalty_diff"] = penalty_diff
        outputs[f"h_wape_{quantile}"] = horizon_wape

        print("\n" + quantile.upper() + " WAPE")
        print(wape)
        print(f"{quantile.upper()} penalty diff AMXL - SCOT: {penalty_diff}")
        print(f"{quantile.upper()} by horizon:")
        print(horizon_wape.to_string(index=False))

    return outputs

# =====================================================
# Chris ASIN cohort — loaded once for the whole rolling run
# =====================================================
CHRIS_DF_PATH = "asin_list_from_amxl_fcst_scot_to_chris_20260723.csv"

if not os.path.exists(CHRIS_DF_PATH):
    raise FileNotFoundError(
        f"Chris ASIN cohort file not found: {CHRIS_DF_PATH}. "
        "Place the CSV in the current working directory."
    )

_chris_df = pd.read_csv(CHRIS_DF_PATH, usecols=["asin"])
CHRIS_ASIN_SET = set(
    _chris_df["asin"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)
del _chris_df

print(
    f"[CHRIS-BASELINE] loaded {len(CHRIS_ASIN_SET):,} unique ASINs "
    f"from {CHRIS_DF_PATH}",
    flush=True,
)

# =====================================================
# GPU / device
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[joint_h3_rolling] using device: {DEVICE}")


def _move_batch_to_device(batch, device):
    """Move every tensor value in a collate'd batch dict to `device`.

    Non-tensor entries (asin strings, target_week date lists, etc.) are
    left untouched.
    """
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


# =============================================================================
# External Evaluation Dependencies
#
# The rolling evaluation calls the standardized functions below from the
# notebook or parent runtime. They are deliberately not duplicated in this file.
#
# Required before calling the rolling runner:
#   calculate_wape_using_lp_oos2
#   quick_error_check
#   weekly_error_check
# =============================================================================


# =============================================================================
# Cohort Sampling and Sparsity Labels
#
# Utilities in this section sample ASINs, intersect them with SCOT eligibility,
# and calculate demand sparsity groups used by diagnostics.
# =============================================================================


def prepare_data_from_sample_scot_intersection(
    data_raw1,
    scot_df,
    n_asins=5000,
    seed=42,
):
    """
    Sample ASINs from data_raw1, then keep only ASINs also present in scot_df.

    n_asins=None means "use every ASIN" -- no sampling is done.
    """
    df = data_raw1.copy()
    scot = scot_df.copy()

    df["asin"] = df["asin"].astype(str)
    df["order_week"] = pd.to_datetime(df["order_week"])
    scot["asin"] = scot["asin"].astype(str)

    unique_asins = df["asin"].dropna().unique()

    if n_asins is None:
        sample_asins = unique_asins
    else:
        rng = np.random.default_rng(seed)
        sample_asins = rng.choice(
            unique_asins,
            size=min(n_asins, len(unique_asins)),
            replace=False,
        )

    sample_asin_set = set(sample_asins)
    scot_asin_set = set(scot["asin"].dropna().unique())
    intersect_asins = sorted(sample_asin_set & scot_asin_set)

    print("\n" + "=" * 80)
    print("SAMPLE-SCOT ASIN INTERSECTION")
    print("=" * 80)
    print("Sample ASINs:", len(sample_asin_set))
    print("SCOT ASINs:", len(scot_asin_set))
    print("Intersection ASINs:", len(intersect_asins))
    print("Sample ASINs missing in SCOT:", len(sample_asin_set - scot_asin_set))

    data_small = df[df["asin"].isin(intersect_asins)].copy()
    sample_asin_df = pd.DataFrame({"asin": list(sample_asins)})
    intersect_asin_df = pd.DataFrame({"asin": intersect_asins})

    print("Data rows after intersection:", len(data_small))
    print("Data ASINs after intersection:", data_small["asin"].nunique())

    return data_small, sample_asin_df, intersect_asin_df


def add_zero_rate_group(data_raw, zero_thresholds=(0.4, 0.7)):
    df = data_raw.copy()
    df["fbi_demand"] = pd.to_numeric(df["fbi_demand"], errors="coerce").fillna(0).clip(lower=0)
    asin_stats = (
        df.groupby("asin")
        .agg(
            zero_rate=("fbi_demand", lambda x: (x == 0).mean()),
            total_demand=("fbi_demand", "sum"),
            n_weeks=("fbi_demand", "count"),
        )
        .reset_index()
    )
    low, high = zero_thresholds
    def assign_group(z):
        if z < low: return "low_sparse"
        elif z < high: return "mid_sparse"
        else: return "high_sparse"
    asin_stats["zero_group"] = asin_stats["zero_rate"].apply(assign_group)
    df = df.merge(asin_stats[["asin", "zero_rate", "zero_group"]], on="asin", how="left")
    print("\nASIN counts by zero-rate group:")
    print(asin_stats.groupby("zero_group")["asin"].nunique().reset_index(name="n_asins"))
    return df, asin_stats


# =============================================================================
# Feature Engineering and ASIN Time-Series Construction
#
# Convert weekly raw records into one model-ready time series per ASIN.
# Outputs include historical encoder features, future-known covariates,
# forecast-origin-safe DPH proxies, targets, and diagnostic fields.
# =============================================================================


def _infer_pkg_dimension_cols(df):
    """
    Use the package-dimension columns from the current modeling schema.
    Diagnostic only; not used as model input.
    """
    required = {
        "height": "pkg_height",
        "length": "pkg_length",
        "width": "pkg_width",
    }

    missing = [col for col in required.values() if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required package-dimension columns: " + ", ".join(missing)
        )

    return required.copy()


def _get_1d_col(df, col):
    """
    Return one 1-D Series even if df has duplicate column names.
    """
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x



def _compute_total_dph_cap(df, q=0.995):
    """
    Compute a global cap from total_dph.

    For fast experiments, this uses the current modeling dataframe.
    For a stricter production backtest, compute this cap using training weeks only.
    """
    if "total_dph" not in df.columns:
        return np.inf

    s = pd.to_numeric(df["total_dph"], errors="coerce").fillna(0.0).clip(lower=0)

    if len(s) == 0 or s.sum() <= 0:
        return np.inf

    cap = float(s.quantile(q))

    if not np.isfinite(cap) or cap <= 0:
        return np.inf

    return cap


def _apply_dph_cap(df, cap):
    """
    Apply one total_dph-based cap to total_dph, buy_box_dph, and in_stock_dph.
    This stabilizes heavy-tailed exposure decoder targets.
    """
    for c in ["total_dph", "buy_box_dph", "in_stock_dph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
            if np.isfinite(cap):
                df[c] = df[c].clip(upper=cap)
    return df



def _select_stock_decoder_extra_cols(data_raw):
    """
    Select additional features to help the external exposure covariates.

    These are NOT true future in_stock_dph. They are product / popularity / price / promo
    / package features that can help predict future exposure.

    We keep a conservative list to avoid leakage-prone realized future outcomes.
    """
    # Top-3 ablation choice for the external exposure / in-stock decoder.
    # These are the only additional stock_extra__ features added to future_context.
    candidate_cols = [
        "ind_promotion",
        "promotion_pricing_amount",
        "pricing_type",
    ]

    # Avoid realized target / future outcome columns.
    exclude_cols = {
        "fbi_demand",
        "order_units",
        "scot_oos",
        "in_stock_dph",
        "asin",
        "order_week",
    }

    cols = [
        c for c in candidate_cols
        if c in data_raw.columns and c not in exclude_cols
    ]

    return cols


def _encode_stock_decoder_extra_features(df, extra_cols):
    """Encode the three fixed stock-decoder context fields."""
    out_cols = []

    for c in extra_cols:
        if c not in df.columns:
            continue

        new_c = f"stock_extra__{c}"
        raw = _get_1d_col(df, c)

        if c == "ind_promotion":
            val = pd.to_numeric(raw, errors="coerce").fillna(0.0)
            std = float(val.std()) if float(val.std()) > 1e-8 else 1.0
            mean = float(val.mean())
            df[new_c] = ((val - mean) / std).clip(-5, 5)

        elif c == "promotion_pricing_amount":
            val = pd.to_numeric(raw, errors="coerce").fillna(0.0)
            val = np.log1p(val.clip(lower=0))
            std = float(val.std()) if float(val.std()) > 1e-8 else 1.0
            mean = float(val.mean())
            df[new_c] = ((val - mean) / std).clip(-5, 5)

        elif c == "pricing_type":
            if pd.api.types.is_numeric_dtype(df[c]):
                val = pd.to_numeric(raw, errors="coerce").fillna(0.0)
                val = np.log1p(val.clip(lower=0))
                std = float(val.std()) if float(val.std()) > 1e-8 else 1.0
                mean = float(val.mean())
                df[new_c] = ((val - mean) / std).clip(-5, 5)
            else:
                codes, uniques = pd.factorize(raw.astype(str).fillna("MISSING"))
                denom = max(len(uniques) - 1, 1)
                df[new_c] = codes.astype(float) / denom

        else:
            raise ValueError(f"Unexpected stock decoder feature: {c}")

        out_cols.append(new_c)

    return df, out_cols



def _safe_numeric(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def _rolling_mean(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def _rolling_std(arr, window):
    return pd.Series(arr).rolling(window, min_periods=2).std().fillna(0).values


def _rolling_positive_mean(arr, window):
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = vals.mean() if len(vals) > 0 else 0.0
    return out


def _rolling_positive_quantile(arr, window, q):
    """
    FIX: arr[lo:i] not arr[lo:i+1]
    Excludes current timestep to prevent data leakage.
    """
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]          # ← FIX: exclude current step
        vals = vals[vals > 0]
        out[i] = np.quantile(vals, q) if len(vals) > 0 else 0.0
    return out


def _rolling_max_lag(arr, window):
    """Lag-safe rolling max excluding current step."""
    out = np.zeros(len(arr), dtype=np.float32)
    for i in range(len(arr)):
        lo = max(0, i - window)
        vals = arr[lo:i]
        out[i] = vals.max() if len(vals) > 0 else 0.0
    return out


def _zero_streak(active):
    out = np.zeros(len(active), dtype=np.float32)
    cur = 0
    for i, a in enumerate(active):
        if a > 0: cur = 0
        else: cur += 1
        out[i] = cur
    return out



# -----------------------------------------------------------------------------
# Four-process per-ASIN feature construction for load_real_data.
# The sorted DataFrame is inherited by forked workers through copy-on-write.
# Each worker receives only a contiguous row range, avoiding per-ASIN task queues.
# -----------------------------------------------------------------------------
_MP_LOAD_FEATURE_DF = None
_MP_LOAD_CONTEXT_COLS = None
_MP_LOAD_DPH_PROXY_COLS = None


def _build_one_asin_feature_record(asin, group, context_cols, dph_proxy_cols):
    """Build one ASIN record with the same feature logic as the serial version."""
    # The parent DataFrame is already sorted by ASIN and Week. A group slice has a
    # fresh positional index only when needed; most operations use NumPy directly.
    demand = group["Demand"].to_numpy(dtype=float, copy=False)
    oos = group["OOS"].to_numpy(dtype=float, copy=False)
    weeks = group["Week"].to_numpy(copy=False)
    t = group["t"].to_numpy(copy=False)
    T = len(demand)

    v_t = np.log1p(demand)
    b_t = (demand > 0).astype(float)

    d_t = np.zeros(T)
    last = -1
    for i in range(T):
        if b_t[i] > 0:
            last = i
        d_t[i] = (i - last) / 52.0 if last >= 0 else 1.0

    in_stock_lag = group["in_stock_dph"].to_numpy(dtype=float, copy=False)
    instock_raw = group["in_stock_dph"].to_numpy(dtype=float, copy=False)
    price_log = group["our_price"].to_numpy(dtype=float, copy=False)
    price_raw = group["our_price_raw"].to_numpy(dtype=float, copy=False)
    pkg_volume_raw = group["pkg_volume_raw"].to_numpy(dtype=float, copy=False)
    total_dph_raw = group["total_dph"].to_numpy(dtype=float, copy=False)
    buy_box_dph_raw = group["buy_box_dph"].to_numpy(dtype=float, copy=False)

    hist_nonzero_mean_52 = _rolling_positive_mean(demand, 52)
    hist_nonzero_p75_52 = _rolling_positive_quantile(demand, 52, 0.75)
    recent_peak_13 = _rolling_max_lag(demand, 13)

    active_rate_4 = _rolling_mean(b_t, 4)
    active_rate_13 = _rolling_mean(b_t, 13)
    oos_rate_4 = _rolling_mean(oos, 4)
    oos_rate_13 = _rolling_mean(oos, 13)
    instock_mean_4 = _rolling_mean(in_stock_lag, 4)
    instock_mean_13 = _rolling_mean(in_stock_lag, 13)

    total_dph_mean_4 = _rolling_mean(total_dph_raw, 4)
    total_dph_mean_13 = _rolling_mean(total_dph_raw, 13)
    buy_box_dph_mean_4 = _rolling_mean(buy_box_dph_raw, 4)
    buy_box_dph_mean_13 = _rolling_mean(buy_box_dph_raw, 13)

    buy_box_rate = np.clip(buy_box_dph_raw / (total_dph_raw + 1.0), 0.0, 10.0)
    in_stock_rate = np.clip(instock_raw / (total_dph_raw + 1.0), 0.0, 10.0)
    in_stock_given_buybox = np.clip(instock_raw / (buy_box_dph_raw + 1.0), 0.0, 10.0)

    zero_streak = _zero_streak(b_t) / 52.0
    positive_mean_4 = _rolling_positive_mean(demand, 4)
    positive_mean_13 = _rolling_positive_mean(demand, 13)
    positive_max_13 = _rolling_max_lag(demand, 13)
    positive_std_13 = _rolling_std(np.log1p(demand), 13)

    features = np.stack([
        v_t,
        b_t,
        d_t,
        np.sin(2 * np.pi * t / 52),
        np.cos(2 * np.pi * t / 52),
        group["promo_t"].to_numpy(dtype=float, copy=False),
        np.sin(2 * np.pi * t / 13),
        np.cos(2 * np.pi * t / 13),
        np.log1p(hist_nonzero_mean_52),
        np.log1p(hist_nonzero_p75_52),
        np.log1p(recent_peak_13),
        np.log1p(in_stock_lag),
        oos,
        active_rate_4,
        active_rate_13,
        oos_rate_4,
        oos_rate_13,
        np.log1p(instock_mean_4),
        np.log1p(instock_mean_13),
        zero_streak,
        price_log,
        np.log1p(positive_mean_4),
        np.log1p(positive_mean_13),
        np.log1p(positive_max_13),
        positive_std_13,
        np.log1p(total_dph_raw),
        np.log1p(buy_box_dph_raw),
        np.log1p(total_dph_mean_4),
        np.log1p(total_dph_mean_13),
        np.log1p(buy_box_dph_mean_4),
        np.log1p(buy_box_dph_mean_13),
        buy_box_rate,
        in_stock_rate,
        in_stock_given_buybox,
    ], axis=1).astype(np.float32)

    future_context = group[context_cols].to_numpy(dtype=np.float32, copy=True)
    return asin, {
        "features": features,
        "future_context": future_context,
        "demand": demand.astype(np.float32),
        "week": weeks,
        "oos": oos.astype(np.float32),
        "price_raw": price_raw.astype(np.float32),
        "pkg_volume_raw": pkg_volume_raw.astype(np.float32),
        "instock_raw": instock_raw.astype(np.float32),
        "total_dph_raw": total_dph_raw.astype(np.float32),
        "buy_box_dph_raw": buy_box_dph_raw.astype(np.float32),
        "dph_proxy_context_idx": {
            c: context_cols.index(c) for c in dph_proxy_cols if c in context_cols
        },
    }


def _mp_load_feature_chunk_worker(task):
    """Process one contiguous ASIN-aligned row chunk inherited through fork."""
    worker_id, row_start, row_end, expected_asins = task
    global _MP_LOAD_FEATURE_DF, _MP_LOAD_CONTEXT_COLS, _MP_LOAD_DPH_PROXY_COLS
    t0 = time.perf_counter()
    print(
        f"[LOAD-W{worker_id}] START | rows={row_end-row_start:,} | "
        f"expected_asins={expected_asins:,}",
        flush=True,
    )
    chunk = _MP_LOAD_FEATURE_DF.iloc[row_start:row_end]
    out = {}
    for i, (asin, group) in enumerate(chunk.groupby("ASIN", sort=False), start=1):
        asin_key, record = _build_one_asin_feature_record(
            asin, group, _MP_LOAD_CONTEXT_COLS, _MP_LOAD_DPH_PROXY_COLS
        )
        out[asin_key] = record
        if i == 1 or i % 1000 == 0 or i == expected_asins:
            elapsed = time.perf_counter() - t0
            rate = i / max(elapsed, 1e-9)
            eta = (expected_asins - i) / max(rate, 1e-9)
            print(
                f"[LOAD-W{worker_id}] {i:,}/{expected_asins:,} "
                f"({100*i/max(expected_asins,1):.1f}%) | "
                f"elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m",
                flush=True,
            )
    print(
        f"[LOAD-W{worker_id}] DONE | asins={len(out):,} | "
        f"elapsed={(time.perf_counter()-t0)/60:.2f}m",
        flush=True,
    )
    return worker_id, out

def load_real_data(data_raw, dph_cap_q=0.995, dph_cap_end_week=None):
    _stage_t0 = time.perf_counter()
    print(f"[STAGE] load_real_data START | rows={len(data_raw):,} | asins={data_raw['asin'].nunique() if 'asin' in data_raw.columns else 'NA'}", flush=True)
    """
    34 history features.
    Feature index map:
      0  log1p(demand)
      1  active indicator
      2  distance since last active / 52
      3  sin(2π t/52)
      4  cos(2π t/52)
      5  promo_t
      6  sin(2π t/13)
      7  cos(2π t/13)
      8  hist_nonzero_mean_52_log   ← lag-fixed
      9  hist_nonzero_p75_52_log    ← lag-fixed
      10 recent_peak_13_log         ← lag-fixed
      11 in_stock_dph_lag_log
      12 oos
      13 active_rate_4
      14 active_rate_13
      15 oos_rate_4
      16 oos_rate_13
      17 instock_mean_4_log
      18 instock_mean_13_log
      19 zero_streak_scaled
      20 price_log
      21 positive_mean_4_log        ← lag-fixed
      22 positive_mean_13_log       ← lag-fixed
      23 positive_max_13_log        ← lag-fixed
      24 positive_std_13

      Added historical DPH funnel features:
      25 total_dph_log
      26 buy_box_dph_log
      27 total_dph_mean_4_log
      28 total_dph_mean_13_log
      29 buy_box_dph_mean_4_log
      30 buy_box_dph_mean_13_log
      31 buy_box_rate
      32 in_stock_rate
      33 in_stock_given_buybox
    """
    holiday_cols = [c for c in data_raw.columns if c.startswith("holiday_indicator_")]
    distance_cols = [c for c in data_raw.columns if c.startswith("distance_")]
    stock_extra_raw_cols = _select_stock_decoder_extra_cols(data_raw)
    pkg_cols = _infer_pkg_dimension_cols(data_raw)

    # ------------------------------------------------------------
    # Future-known context features.
    # We add business seasonality and major shopping-event proximity
    # BEFORE keep_cols is created, so these columns truly enter future_context.
    # ------------------------------------------------------------
    data_raw = data_raw.copy()
    data_raw["order_week"] = pd.to_datetime(data_raw["order_week"], errors="coerce")
    data_raw["order_month"] = data_raw["order_week"].dt.month.astype(float)
    data_raw["month_sin"] = np.sin(2 * np.pi * data_raw["order_month"] / 12.0)
    data_raw["month_cos"] = np.cos(2 * np.pi * data_raw["order_month"] / 12.0)

    data_raw["season_winter"] = data_raw["order_month"].isin([12, 1, 2]).astype(float)
    data_raw["season_spring"] = data_raw["order_month"].isin([3, 4, 5]).astype(float)
    data_raw["season_summer"] = data_raw["order_month"].isin([6, 7, 8]).astype(float)
    data_raw["season_fall"] = data_raw["order_month"].isin([9, 10, 11]).astype(float)

    seasonal_cols = [
        "order_month",
        "month_sin",
        "month_cos",
        "season_winter",
        "season_spring",
        "season_summer",
        "season_fall",
    ]

    # Major shopping-event proximity from the fixed input schema.
    major_event_distance_cols = [
        "distance_blackfriday",
        "distance_cybermonday",
        "distance_primeday",
        "distance_christmasday",
        "distance_thanksgivingday",
        "distance_newyearseve",
        "distance_laborday",
        "distance_memorialday",
    ]
    proximity_cols = []
    for c in major_event_distance_cols:
        if c not in data_raw.columns:
            continue
        new_c = f"{c}_proximity"
        data_raw[new_c] = (
            1.0 - pd.to_numeric(data_raw[c], errors="coerce").fillna(0.0).abs()
        ).clip(0.0, 1.0)
        proximity_cols.append(new_c)

    # Include holiday indicators, raw distance features, explicit season features,
    # and major-event proximity features.
    context_cols = ["our_price"] + holiday_cols + distance_cols + seasonal_cols + proximity_cols
    context_cols = list(dict.fromkeys(context_cols))

    base_cols = ["asin", "order_week", "fbi_demand", "scot_oos"]

    # Keep in_stock_dph for history encoder only.
    # It is intentionally excluded from future_context.
    # Keep DPH variables for history-only safe proxy features.
    # They are not used as raw future context.
    history_only_cols = ["in_stock_dph", "total_dph", "buy_box_dph"]

    extra_diag_cols = [c for c in pkg_cols.values() if c is not None]

    keep_cols = [
        c for c in base_cols + context_cols + history_only_cols + extra_diag_cols + stock_extra_raw_cols
        if c in data_raw.columns
    ]

    # Remove duplicate column names. Duplicates can happen because package columns
    # are used both for total_size diagnostics and stock-decoder extra features.
    keep_cols = list(dict.fromkeys(keep_cols))

    df = data_raw[keep_cols].copy()

    # Encode additional product / popularity / promo / size features for stock decoder.
    df, stock_extra_cols = _encode_stock_decoder_extra_features(df, stock_extra_raw_cols)

    # Add encoded stock-extra columns to future_context.
    # These features help the external exposure covariates.
    context_cols = context_cols + stock_extra_cols

    # Forecast-origin-safe historical DPH proxy features.
    # These columns are placeholders here and are filled inside DemandDataset
    # using only history up to each forecast origin.
    dph_proxy_cols = [
        "hist_total_dph_last_log",
        "hist_total_dph_mean4_log",
        "hist_total_dph_mean13_log",
        "hist_buy_box_dph_last_log",
        "hist_buy_box_dph_mean4_log",
        "hist_buy_box_dph_mean13_log",
        "hist_instock_dph_last_log",
        "hist_instock_dph_mean4_log",
        "hist_instock_dph_mean13_log",
    ]
    for c in dph_proxy_cols:
        df[c] = 0.0

    context_cols = context_cols + dph_proxy_cols
    df = df.rename(columns={"asin":"ASIN","order_week":"Week","fbi_demand":"Demand","scot_oos":"OOS"})

    h_col = pkg_cols.get("height")
    l_col = pkg_cols.get("length")
    w_col = pkg_cols.get("width")

    if h_col is not None and l_col is not None and w_col is not None:
        pkg_h = pd.to_numeric(_get_1d_col(df, h_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_l = pd.to_numeric(_get_1d_col(df, l_col), errors="coerce").fillna(0).clip(lower=0)
        pkg_w = pd.to_numeric(_get_1d_col(df, w_col), errors="coerce").fillna(0).clip(lower=0)
        df["pkg_volume_raw"] = pkg_h * pkg_l * pkg_w
    else:
        df["pkg_volume_raw"] = np.nan

    df["Week"] = pd.to_datetime(df["Week"])
    df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce").fillna(0).clip(lower=0)
    df["OOS"] = pd.to_numeric(df["OOS"], errors="coerce").fillna(0)
    for c in context_cols:
        df = _safe_numeric(df, c, default=0.0)

    # Keep raw price for amount diagnostics, then use log price for model context.
    df["our_price_raw"] = df["our_price"].clip(lower=0)
    df["our_price"] = np.log1p(df["our_price_raw"])

    # Use historical in_stock_dph directly in the encoder; no lag shift.
    # Future in_stock_dph is not used in future_context.
    if "in_stock_dph" in df.columns:
        df["in_stock_dph"] = pd.to_numeric(df["in_stock_dph"], errors="coerce").fillna(0.0)
        df["in_stock_dph"] = df["in_stock_dph"].clip(lower=0)
    else:
        df["in_stock_dph"] = 0.0

    # Historical total_dph / buy_box_dph are used only as forecast-origin-safe summaries.
    for c in ["total_dph", "buy_box_dph"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(lower=0)
        else:
            df[c] = 0.0

    # Cap heavy-tailed DPH targets using a history-only exposure scale cap.
    # In rolling backtests, dph_cap_end_week is the last week observable at the
    # forecast origin. Future H1-H3 actual DPH rows remain available as targets
    # but do not participate in estimating the clipping threshold.
    if dph_cap_end_week is None:
        dph_cap_source = df
        dph_cap_source_desc = "all rows supplied to load_real_data"
    else:
        dph_cap_end_week = pd.Timestamp(dph_cap_end_week)
        dph_cap_source = df[df["Week"] <= dph_cap_end_week]
        dph_cap_source_desc = f"Week <= {dph_cap_end_week.date()}"
        if dph_cap_source.empty:
            raise ValueError(
                "No rows are available for DPH-cap estimation at or before "
                f"dph_cap_end_week={dph_cap_end_week.date()}."
            )

    dph_cap = _compute_total_dph_cap(dph_cap_source, q=dph_cap_q)
    df = _apply_dph_cap(df, dph_cap)
    for c in holiday_cols:
        df[c] = df[c].clip(lower=0, upper=1)

    # Distance-to-holiday features are future-known scalar calendar features.
    # Keep direction if raw values are signed: negative = before holiday, positive = after holiday.
    for c in distance_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        df[c] = df[c].clip(lower=-12, upper=12) / 12.0

    df = df.sort_values(["ASIN", "Week"]).reset_index(drop=True)

    if len(holiday_cols) > 0:
        holiday_window = np.zeros(len(df), dtype=np.float32)
        for c in holiday_cols:
            cur = df[c].values.astype(float)
            prev_window = np.roll(cur, -1); prev_window[-1] = 0
            holiday_window = np.maximum(holiday_window, np.maximum(cur, prev_window))
        df["promo_t"] = holiday_window
    else:
        df["promo_t"] = 0.0

    df["t"] = ((df["Week"] - df["Week"].min()).dt.days // 7).astype(int)

    data = {}
    _n_asins = int(df["ASIN"].nunique())
    _feat_t0 = time.perf_counter()
    _num_load_workers = min(4, max(1, _n_asins))
    print(
        f"[LOAD] per-ASIN feature construction START | {_n_asins:,} ASINs | "
        f"requested_workers={_num_load_workers}",
        flush=True,
    )

    # Since df is sorted by ASIN, compute ASIN-aligned contiguous row chunks.
    _boundary_t0 = time.perf_counter()
    _asin_values = df["ASIN"].to_numpy(copy=False)
    _change_pos = np.flatnonzero(_asin_values[1:] != _asin_values[:-1]) + 1
    _group_starts = np.r_[0, _change_pos]
    _group_ends = np.r_[_change_pos, len(df)]
    _asin_index_chunks = np.array_split(np.arange(_n_asins), _num_load_workers)
    _tasks = []
    for _wid, _idx_chunk in enumerate(_asin_index_chunks):
        if len(_idx_chunk) == 0:
            continue
        _row_start = int(_group_starts[int(_idx_chunk[0])])
        _row_end = int(_group_ends[int(_idx_chunk[-1])])
        _tasks.append((_wid, _row_start, _row_end, int(len(_idx_chunk))))
        print(
            f"[LOAD] worker {_wid}: rows={_row_end-_row_start:,} | "
            f"asins={len(_idx_chunk):,}",
            flush=True,
        )
    print(
        f"[LOAD] chunk planning DONE | elapsed={time.perf_counter()-_boundary_t0:.2f}s",
        flush=True,
    )

    _used_parallel = False
    if len(_tasks) > 1:
        try:
            _current_start = mp.get_start_method(allow_none=True)
            if _current_start == "spawn":
                raise RuntimeError(
                    "spawn start method would pickle the full DataFrame; using serial fallback"
                )
            global _MP_LOAD_FEATURE_DF, _MP_LOAD_CONTEXT_COLS, _MP_LOAD_DPH_PROXY_COLS
            _MP_LOAD_FEATURE_DF = df
            _MP_LOAD_CONTEXT_COLS = context_cols
            _MP_LOAD_DPH_PROXY_COLS = dph_proxy_cols
            print(
                f"[LOAD] launching {len(_tasks)} fork workers...",
                flush=True,
            )
            _pool_t0 = time.perf_counter()
            _ctx = mp.get_context("fork")
            with _ctx.Pool(processes=len(_tasks)) as _pool:
                _parts = _pool.map(_mp_load_feature_chunk_worker, _tasks)
            print(
                f"[LOAD] workers RETURNED | elapsed={(time.perf_counter()-_pool_t0)/60:.2f}m",
                flush=True,
            )

            _merge_t0 = time.perf_counter()
            print("[LOAD] merging worker outputs START", flush=True)
            for _wid, _part in sorted(_parts, key=lambda x: x[0]):
                data.update(_part)
                print(
                    f"[LOAD] merged worker {_wid} | part_asins={len(_part):,} | "
                    f"total={len(data):,}",
                    flush=True,
                )
            print(
                f"[LOAD] merging worker outputs DONE | elapsed={time.perf_counter()-_merge_t0:.2f}s",
                flush=True,
            )
            _used_parallel = True
        except Exception as _mp_err:
            print(
                f"[LOAD] 4-worker mode failed: {type(_mp_err).__name__}: {_mp_err}",
                flush=True,
            )
            print("[LOAD] falling back to SERIAL construction", flush=True)
        finally:
            _MP_LOAD_FEATURE_DF = None
            _MP_LOAD_CONTEXT_COLS = None
            _MP_LOAD_DPH_PROXY_COLS = None

    if not _used_parallel:
        _serial_t0 = time.perf_counter()
        for _asin_i, (asin, group) in enumerate(df.groupby("ASIN", sort=False), start=1):
            asin_key, record = _build_one_asin_feature_record(
                asin, group, context_cols, dph_proxy_cols
            )
            data[asin_key] = record
            if _asin_i == 1 or _asin_i % 2000 == 0 or _asin_i == _n_asins:
                _elapsed = time.perf_counter() - _serial_t0
                _rate = _asin_i / max(_elapsed, 1e-9)
                _eta = (_n_asins - _asin_i) / max(_rate, 1e-9)
                print(
                    f"[LOAD-SERIAL] {_asin_i:,}/{_n_asins:,} "
                    f"({100*_asin_i/max(_n_asins,1):.1f}%) | "
                    f"elapsed={_elapsed/60:.1f}m | ETA={_eta/60:.1f}m",
                    flush=True,
                )

    print(
        f"[LOAD] per-ASIN feature construction DONE | mode={'4-worker' if _used_parallel else 'serial'} | "
        f"asins={len(data):,} | elapsed={(time.perf_counter()-_feat_t0)/60:.2f}m",
        flush=True,
    )
    print(f"[STAGE] load_real_data DONE | {(time.perf_counter()-_stage_t0)/60:.2f} min", flush=True)
    print("History encoder dim: 34")
    print(f"Package dimension columns for total_size: {pkg_cols}")
    print("History in_stock_dph: raw historical value, no lag shift")
    print("Future context excludes in_stock_dph")
    print("Future context includes distance_* calendar features")
    print("External exposure safe mode: demand uses external predicted DPH hats only")
    print("Safe historical DPH proxies: total/buy_box/in_stock last/mean4/mean13")
    print("History encoder includes DPH funnel features")
    print(
        f"DPH cap q: {dph_cap_q} | cap value: {dph_cap} | "
        f"source: {dph_cap_source_desc} | source_rows: {len(dph_cap_source):,}",
        flush=True,
    )
    print(f"Context dim: {len(context_cols)}", flush=True)
    print("[PROFILE-C62] base load_real_data reached RETURN boundary", flush=True)
    _ret_t0 = time.perf_counter()
    _ret_value = (data, len(context_cols), context_cols)
    print(
        f"[PROFILE-C62] base return tuple ready | elapsed={time.perf_counter()-_ret_t0:.6f}s",
        flush=True,
    )
    return _ret_value


# =============================================================================
# Rolling Supervised Dataset
#
# Convert each ASIN time series into:
#   - a fixed historical window;
#   - future-known covariates for the forecast horizon;
#   - demand and exposure targets used during training and evaluation.
#
# Forecast-origin-safe proxy values are computed from history only.
# =============================================================================

class DemandDataset(Dataset):
    def __init__(self, data, history=52, horizon=3, mode="train", val_weeks=20):
        self.samples = []
        for asin, d in data.items():
            T = len(d["demand"])
            if mode == "train":
                starts = range(max(0, T - val_weeks - horizon - history + 1))
            else:
                s = T - history - horizon
                starts = [s] if s >= 0 else []

            for start in starts:
                self.samples.append({
                    "x": torch.tensor(d["features"][start:start+history], dtype=torch.float32),
                    "future_context": torch.tensor(
                        self._make_future_context_with_dph_proxies(
                            d=d,
                            start=start,
                            history=history,
                            horizon=horizon,
                        ),
                        dtype=torch.float32),
                    "y": torch.tensor(d["demand"][start+history:start+history+horizon], dtype=torch.float32),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in d["week"][start+history:start+history+horizon]],
                    "oos": torch.tensor(d["oos"][start+history:start+history+horizon], dtype=torch.float32),
                    "our_price": torch.tensor(
                        d["price_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "pkg_volume": torch.tensor(
                        d["pkg_volume_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_instock": torch.tensor(
                        d["instock_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_total_dph": torch.tensor(
                        d["total_dph_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                    "future_buy_box_dph": torch.tensor(
                        d["buy_box_dph_raw"][start+history:start+history+horizon],
                        dtype=torch.float32),
                })

    def _safe_hist_mean(self, arr, start, history, window):
        hist = arr[start:start+history]
        if len(hist) == 0:
            return 0.0
        hist = hist[-min(window, len(hist)):]
        return float(np.mean(hist))

    def _make_future_context_with_dph_proxies(self, d, start, history, horizon):
        """
        Fill historical DPH summary proxy features using only values up to forecast origin.
        These are repeated across the horizon and do not use future true DPH.
        """
        fc = d["future_context"][start+history:start+history+horizon].copy()
        idx = d.get("dph_proxy_context_idx", {})

        total_hist = d.get("total_dph_raw", None)
        buy_hist = d.get("buy_box_dph_raw", None)
        instock_hist = d.get("instock_raw", None)

        def fill(col, val):
            if col in idx:
                fc[:, idx[col]] = np.log1p(max(float(val), 0.0))

        if total_hist is not None:
            total_last = total_hist[start+history-1] if history > 0 else 0.0
            fill("hist_total_dph_last_log", total_last)
            fill("hist_total_dph_mean4_log", self._safe_hist_mean(total_hist, start, history, 4))
            fill("hist_total_dph_mean13_log", self._safe_hist_mean(total_hist, start, history, 13))

        if buy_hist is not None:
            buy_last = buy_hist[start+history-1] if history > 0 else 0.0
            fill("hist_buy_box_dph_last_log", buy_last)
            fill("hist_buy_box_dph_mean4_log", self._safe_hist_mean(buy_hist, start, history, 4))
            fill("hist_buy_box_dph_mean13_log", self._safe_hist_mean(buy_hist, start, history, 13))

        if instock_hist is not None:
            instock_last = instock_hist[start+history-1] if history > 0 else 0.0
            fill("hist_instock_dph_last_log", instock_last)
            fill("hist_instock_dph_mean4_log", self._safe_hist_mean(instock_hist, start, history, 4))
            fill("hist_instock_dph_mean13_log", self._safe_hist_mean(instock_hist, start, history, 13))

        return fc

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# =============================================================================
# Model Architecture
#
# The model is organized into four conceptual parts:
#   A. Historical encoder
#   B. Latent uncertainty components
#   C. Future-covariate decoder
#   D. Demand distribution decoder
# =============================================================================

# -----------------------------------------------------------------------------
# A. Historical Encoder
#
# Causal temporal convolutions and sparse peak-aware attention transform the
# historical feature window into sequence states and a final ASIN state.
# -----------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class SparsePeakAttention(nn.Module):
    def __init__(self, d_model=32, n_heads=4, beta_peak=1.0, soft_mask_scale=3.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.beta_peak = beta_peak
        self.soft_mask_scale = soft_mask_scale

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(0.1)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x, b_t, peak_score):
        B, T, D = x.shape
        q = self.q_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        k = self.k_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B,T,self.n_heads,self.d_head).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / np.sqrt(self.d_head)

        # Softly down-weight zero-demand weeks.
        sparse_mask = (b_t == 0) & ~(b_t == 0).all(dim=1, keepdim=True)
        scores = scores - self.soft_mask_scale * sparse_mask.float()[:, None, None, :]

        peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
        scores = scores + self.beta_peak * peak_norm[:, None, None, :]

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1,2).contiguous().view(B,T,D)
        out  = self.out_proj(out)
        return self.norm(x + out)


class TCNSparseAttnEncoder(nn.Module):
    def __init__(self, input_dim=34, d_model=32, horizon=3):
        super().__init__()
        self.horizon = horizon
        self.input_proj = nn.Linear(input_dim, d_model)

        # Dilations include quarterly and annual scales.
        dilations = [1, 2, 4, 8, 13, 26, 52]
        self.convs = nn.ModuleList([CausalConv1d(d_model, d_model, 2, d) for d in dilations])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in dilations])

        self.sparse_attn = SparsePeakAttention(d_model, n_heads=4, beta_peak=1.0)
        self.final_norm  = nn.LayerNorm(d_model)

        self.base_head  = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))
        self.alpha_head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Linear(64,horizon))

    def encode(self, x):
        """Return full encoder states and final state.

        H_enc: [B, T, d_model]
        h_t:   [B, d_model]
        b_t / peak_score are returned for decoder-side peak attention biases.
        """
        b_t        = x[:, :, 1]
        peak_score = torch.sqrt(torch.expm1(x[:,:,0]).clamp(min=0) + 1e-6)

        h = self.input_proj(x).permute(0,2,1)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h) + h
            h = h.permute(0,2,1)
            h = norm(h)
            h = F.gelu(h)
            h = h.permute(0,2,1)

        H_enc = self.sparse_attn(h.permute(0,2,1), b_t, peak_score)
        h_t   = self.final_norm(H_enc[:,-1,:])
        return H_enc, h_t, b_t, peak_score

    def forward(self, x):
        H_enc, h_t, b_t, peak_score = self.encode(x)
        mu    = F.softplus(self.base_head(h_t))
        alpha = F.softplus(self.alpha_head(h_t)) + 1e-4
        return mu, alpha, h_t



# -----------------------------------------------------------------------------
# B. Latent Uncertainty Components
#
# Generate context-conditioned latent variables and epistemic residuals used
# for Monte Carlo demand-distribution sampling.
# -----------------------------------------------------------------------------
class ContextZGenerator(nn.Module):
    def __init__(self, d_phi=32, context_dim=2, d_z=16, horizon=3):
        super().__init__()
        self.d_z = d_z
        self.net = nn.Sequential(
            nn.Linear(d_phi + horizon * context_dim, 64),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2 * d_z)
        )

    def forward(self, phi, future_context):
        B   = phi.shape[0]
        ctx = future_context.reshape(B, -1)
        out = self.net(torch.cat([phi, ctx], dim=-1))
        z_mean, z_logstd = out.chunk(2, dim=-1)
        z_std = F.softplus(z_logstd) + 1e-4
        return z_mean, z_std


class Epinet(nn.Module):
    def __init__(self, d_phi=32, d_z=16, horizon=3, prior_scale=0.3):
        super().__init__()
        self.d_z = d_z; self.horizon = horizon; self.prior_scale = prior_scale
        self.learnable = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 2*horizon*d_z)
        )
        self.prior = nn.Sequential(
            nn.Linear(d_z+d_phi,64), nn.ReLU(),
            nn.Linear(64, 2*horizon*d_z)
        )
        for p in self.prior.parameters(): p.requires_grad = False

    def forward(self, phi, z):
        inp = torch.cat([z, phi], dim=-1)
        sl  = self.learnable(inp).view(-1, 2*self.horizon, self.d_z)
        sl  = torch.einsum("bhd,bd->bh", sl, z)
        sp  = self.prior(inp).view(-1, 2*self.horizon, self.d_z)
        sp  = torch.einsum("bhd,bd->bh", sp, z) * self.prior_scale
        out = sl + sp
        return out[:,:self.horizon], out[:,self.horizon:]


# -----------------------------------------------------------------------------
# C. Future-Covariate Decoder
#
# Model interactions across forecast weeks and attend from future-known
# covariates to the historical encoder states.
# -----------------------------------------------------------------------------
class HorizonTCNBlock(nn.Module):
    """Lightweight horizon-axis TCN block, matching the exposure decoder idea.

    Input / output shape: [B, H, D].  It lets each future horizon build its
    query from the complete 20-week future-context pattern before attending
    to historical encoder states.
    """
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.10):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        z = x.transpose(1, 2)
        z = self.drop(F.gelu(self.conv1(z)))
        z = self.drop(F.gelu(self.conv2(z)))
        z = z.transpose(1, 2)
        m = min(z.shape[1], residual.shape[1])
        return self.norm(residual[:, :m, :] + z[:, :m, :])


class DecoderEpinet(nn.Module):
    """z-conditioned epistemic residual over stop-gradient decoder features.

    The base decoder feature is detached before entering this module. Therefore:
      * the decoder epinet learns from the demand loss;
      * z affects the joint 20-week prediction;
      * this epistemic branch cannot update the base future TCN / cross-attention.

    This mirrors the encoder epinet use of phi = h_t.detach().
    """
    def __init__(self, d_decoder=32, context_dim=2, d_z=16,
                 hidden=64, prior_scale=0.20):
        super().__init__()
        self.d_z = d_z
        self.prior_scale = prior_scale
        in_dim = d_decoder + context_dim + d_z

        # Output d_z coefficients for mu and alpha at every horizon.
        self.learnable = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(hidden, 2 * d_z),
        )
        self.prior = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_z),
        )
        for p in self.prior.parameters():
            p.requires_grad = False

        # Start the learned residual at zero; fixed prior still supplies variation.
        nn.init.zeros_(self.learnable[-1].weight)
        nn.init.zeros_(self.learnable[-1].bias)

    def forward(self, decoder_feature, future_context, z):
        B, H, _ = decoder_feature.shape
        z_h = z[:, None, :].expand(B, H, -1)

        # Critical stop-gradient: z branch reads decoder information but cannot
        # reshape the deterministic decoder representation.
        epi_in = torch.cat([
            decoder_feature.detach(),
            future_context.detach(),
            z_h,
        ], dim=-1)

        learned = self.learnable(epi_in).view(B, H, 2, self.d_z)
        prior = self.prior(epi_in).view(B, H, 2, self.d_z)

        learned = torch.einsum('bhkd,bd->bhk', learned, z)
        prior = torch.einsum('bhkd,bd->bhk', prior, z) * self.prior_scale
        out = learned + prior
        return out[:, :, 0], out[:, :, 1]


class DecoderPeakCrossAttention(nn.Module):
    """Deterministic future-TCN + peak-aware cross-attention demand decoder.

    Important ENN separation:
      * z does NOT enter the base future TCN or the base attention query;
      * the resulting decoder state is later passed through stop-gradient to
        DecoderEpinet together with z.

    Thus the decoder can inform the z branch, while the z branch cannot update
    or distort the deterministic decoder representation.
    """
    def __init__(self, d_model=32, context_dim=2, horizon=3, n_heads=4,
                 dropout=0.1, active_bias=0.50, peak_bias=0.75,
                 future_tcn_dilations=(1, 2, 4)):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.horizon = horizon
        self.active_bias = active_bias
        self.peak_bias = peak_bias

        self.future_proj = nn.Sequential(
            nn.Linear(context_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.horizon_emb = nn.Embedding(horizon, d_model)

        # Same core idea as the exposure decoder: future context is first
        # modeled along the horizon axis, then used as cross-attention queries.
        self.future_tcn = nn.ModuleList([
            HorizonTCNBlock(
                d_model=d_model,
                kernel_size=3,
                dilation=d,
                dropout=dropout,
            )
            for d in future_tcn_dilations
        ])

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _future_peak_score(self, future_context):
        # The last three columns are already log1p(exposure_hat), so do not
        # apply a second log1p here.
        if future_context.shape[-1] >= 3:
            return future_context[:, :, -3:].clamp(min=0).mean(dim=-1)
        return future_context.new_zeros(future_context.shape[:2])

    def forward(self, H_enc, future_context, b_t=None, peak_score=None,
                return_attn=False):
        B, T, D = H_enc.shape
        H = future_context.shape[1]
        device = H_enc.device

        horizon_ids = torch.arange(H, device=device).clamp(max=self.horizon - 1)
        q0 = self.future_proj(future_context)
        q0 = q0 + self.horizon_emb(horizon_ids)[None, :, :]
        for block in self.future_tcn:
            q0 = block(q0)

        q = self.q_proj(q0).view(B, H, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(H_enc).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(H_enc).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)

        if b_t is not None:
            scores = scores + self.active_bias * b_t.float()[:, None, None, :]

        if peak_score is not None:
            peak_norm = peak_score / (peak_score.max(dim=1, keepdim=True)[0] + 1e-6)
            f_peak = self._future_peak_score(future_context)
            f_peak = f_peak / (f_peak.max(dim=1, keepdim=True)[0] + 1e-6)
            scores = scores + (
                self.peak_bias
                * f_peak[:, None, :, None]
                * peak_norm[:, None, None, :]
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, H, D)
        out = self.out_proj(out)
        dec_h = self.norm(q0 + out)

        if return_attn:
            return dec_h, attn.mean(dim=1)
        return dec_h



# -----------------------------------------------------------------------------
# D. Demand Distribution Decoder
#
# Combine deterministic decoder states with stop-gradient epistemic residuals
# to parameterize the future demand distribution.
# -----------------------------------------------------------------------------


# =============================================================================
# Training Losses
#
# Negative-binomial demand loss and auxiliary bias-related loss components.
# The rolling runner controls which terms are active through its parameters.
# =============================================================================

def negbin_nll_elementwise(y, mu, alpha):
    eps = 1e-6
    r   = (1.0/alpha).clamp(min=eps)
    p   = (mu*alpha/(1+mu*alpha)).clamp(eps, 1-eps)
    return -(
        torch.lgamma(y+r) - torch.lgamma(r) - torch.lgamma(y+1)
        + r*torch.log(1-p) + y*torch.log(p)
    )


def tail_weighted_negbin_nll(y, mu, alpha, beta_tail=0.5):
    nll    = negbin_nll_elementwise(y, mu, alpha)
    weight = 1.0 + beta_tail * torch.log1p(y)
    return (nll * weight).sum() / weight.sum().clamp(min=1.0)


def active_underforecast_loss(y, mu, log_scale=True):
    """Penalize underforecasting only when the true future demand is active.

    This addresses the observed issue: occurrence is learned, but active magnitude
    is still too conservative.
    """
    active = (y > 0).float()
    if active.sum() <= 0:
        return y.new_tensor(0.0)
    if log_scale:
        under = torch.relu(torch.log1p(y.clamp(min=0)) - torch.log1p(mu.clamp(min=0)))
    else:
        under = torch.relu(y - mu)
    return (active * under).sum() / active.sum().clamp(min=1.0)


def active_overforecast_loss(y, mu, log_scale=True):
    """Penalize overforecasting only when the true future demand is active.

    Symmetric mirror of active_underforecast_loss -- same active (y > 0) mask,
    same log1p scale, just the opposite direction (mu > y instead of y > mu).
    Kept as a separate, independently-weighted term (rather than folded into
    a single symmetric loss) so lambda_under and lambda_over can be tuned or
    disabled independently for ablation.
    """
    active = (y > 0).float()
    if active.sum() <= 0:
        return y.new_tensor(0.0)
    if log_scale:
        over = torch.relu(torch.log1p(mu.clamp(min=0)) - torch.log1p(y.clamp(min=0)))
    else:
        over = torch.relu(mu - y)
    return (active * over).sum() / active.sum().clamp(min=1.0)


# =============================================================================
# Encoder and Training Diagnostics
#
# Optional research utilities for checking whether the historical encoder
# separates demand occurrence states, captures positive-demand magnitude, and
# produces a stable latent representation. These functions are diagnostic only;
# they are not required by the rolling inference path.
# =============================================================================


def diagnose_training_batch(b, preds, epoch, bi, n_diag_batches=3):
    """Print diagnostics for the first few batches."""
    if bi >= n_diag_batches:
        return
    y = b["y"]
    active_cnt = (y > 0).sum().item()
    total_cnt  = y.numel()
    mu_mean    = torch.stack([mu for mu, _ in preds], dim=0).mean().item()
    y_active_mean = y[y > 0].mean().item() if active_cnt > 0 else 0.0
    print(
        f"  [batch {bi}] active={active_cnt}/{total_cnt} "
        f"({100*active_cnt/total_cnt:.1f}%) "
        f"mu_mean={mu_mean:.2f} "
        f"y_active_mean={y_active_mean:.2f}"
    )


# =============================================================================
# Model Training
#
# Optimize the model, apply early stopping, and retain the best validation state.
# =============================================================================



# =============================================================================
# Prediction and Diagnostic DataFrames
#
# Run Monte Carlo inference and convert model outputs into demand quantiles,
# exposure forecasts, and optional diagnostic tables.
# =============================================================================



# =============================================================================
# Non-Rolling Execution Helpers
#
# Entry points retained for single-run experiments and targeted sparse cohorts.
# =============================================================================

def filter_extreme_asins(data_high, demand_col="fbi_demand", asin_col="asin", q=0.99):
    """Memory-safe extreme-ASIN filtering.

    Important: ``data_high`` is already a dedicated dataframe created by the caller,
    so this function intentionally consumes/mutates it in place to avoid making a
    second ~10-12 GB copy.
    """
    import gc
    _all_t0 = time.time()
    df = data_high  # NO full dataframe copy

    print(
        f"[EXTREME-SAFE 01] ENTER | rows={len(df):,} | "
        f"asins={df[asin_col].nunique():,} | q={q} | in_place=True",
        flush=True,
    )

    _t0 = time.time()
    print(f"[EXTREME-SAFE 02] clean demand column START", flush=True)
    demand_numeric = pd.to_numeric(df[demand_col], errors="coerce")
    demand_numeric = demand_numeric.fillna(0).clip(lower=0)
    df[demand_col] = demand_numeric
    del demand_numeric
    gc.collect()
    print(f"[EXTREME-SAFE 03] clean demand column DONE | elapsed={time.time()-_t0:.2f}s", flush=True)

    _t0 = time.time()
    print(f"[EXTREME-SAFE 04] positive quantile START", flush=True)
    positive_mask = df[demand_col].to_numpy(copy=False) > 0
    positive_values = df.loc[positive_mask, demand_col]
    if len(positive_values) == 0:
        del positive_mask, positive_values
        gc.collect()
        print(f"[EXTREME-SAFE 05] NO POSITIVE DEMAND | elapsed={time.time()-_all_t0:.2f}s", flush=True)
        return df, pd.DataFrame(), np.nan
    cap = float(positive_values.quantile(q))
    del positive_mask, positive_values
    gc.collect()
    print(f"[EXTREME-SAFE 05] quantile DONE | cap={cap:.6f} | elapsed={time.time()-_t0:.2f}s", flush=True)

    _t0 = time.time()
    print("[EXTREME-SAFE 06] groupby peak START", flush=True)
    asin_peak = (
        df.groupby(asin_col, sort=False, observed=True)[demand_col]
        .max()
        .reset_index(name="asin_max")
    )
    bad_mask = asin_peak["asin_max"] > cap
    bad_asins = asin_peak.loc[bad_mask, asin_col]
    removed = asin_peak.loc[bad_mask].copy()
    print(
        f"[EXTREME-SAFE 07] groupby peak DONE | groups={len(asin_peak):,} | "
        f"bad_asins={len(removed):,} | elapsed={time.time()-_t0:.2f}s",
        flush=True,
    )

    _t0 = time.time()
    print("[EXTREME-SAFE 08] locate rows to drop START", flush=True)
    row_bad_mask = df[asin_col].isin(bad_asins).to_numpy(dtype=bool, copy=False)
    bad_positions = np.flatnonzero(row_bad_mask)
    bad_index = df.index.take(bad_positions)
    n_bad_rows = len(bad_index)
    del row_bad_mask, bad_positions, bad_asins, bad_mask, asin_peak
    gc.collect()
    print(
        f"[EXTREME-SAFE 09] rows to drop READY | bad_rows={n_bad_rows:,} | "
        f"elapsed={time.time()-_t0:.2f}s",
        flush=True,
    )

    _t0 = time.time()
    print(
        f"[EXTREME-SAFE 10] in-place drop START | rows_before={len(df):,}",
        flush=True,
    )
    df.drop(index=bad_index, inplace=True)
    del bad_index
    gc.collect()
    print(
        f"[EXTREME-SAFE 11] in-place drop DONE | rows_after={len(df):,} | "
        f"elapsed={time.time()-_t0:.2f}s",
        flush=True,
    )

    _t0 = time.time()
    print("[EXTREME-SAFE 12] reset index START", flush=True)
    df.reset_index(drop=True, inplace=True)
    gc.collect()
    print(
        f"[EXTREME-SAFE 13] reset index DONE | asins={df[asin_col].nunique():,} | "
        f"elapsed={time.time()-_t0:.2f}s",
        flush=True,
    )

    print(
        f"[EXTREME-SAFE 14] RETURN | cap={cap:.1f} | removed_asins={len(removed):,} | "
        f"rows={len(df):,} | total_elapsed={time.time()-_all_t0:.2f}s",
        flush=True,
    )
    return df, removed, cap



# =============================================================================
# Sparse-Cohort Evaluation
#
# Attach sparsity labels and summarize WAPE behavior by demand sparsity group.
# =============================================================================


# =============================================================================
# Horizon Diagnostics by Sparsity Group
#
# Compare forecast behavior across horizons for low-, medium-, and high-sparsity
# ASIN cohorts.
# =============================================================================


# =============================================================================
# SCOT Alignment and Standardized Evaluation
#
# Normalize join keys, align model forecasts to SCOT rows, and call the external
# standardized WAPE functions.
# =============================================================================


# =====================================================

# =============================================================================
# External Exposure Covariates
#
# Load predicted total, buy-box, and in-stock DPH and append them to demand
# future_context. Realized future DPH is never used as a demand input.
# =============================================================================

_ORIGINAL_LOAD_REAL_DATA_BEFORE_EXTERNAL_EXP3 = load_real_data


def load_real_data(data_raw, dph_cap_q=0.995, dph_cap_end_week=None):
    """
    Override original load_real_data to inject external exposure-3 hats into future_context.

    Added future context columns:
      external_total_dph_hat_log
      external_buy_box_dph_hat_log
      external_instock_dph_hat_log

    These are predicted future covariates, not true future DPH.
    """
    _ext_all_t0 = time.perf_counter()
    print("[EXT-HAT 01] external-hat wrapper ENTER", flush=True)
    print("[EXT-HAT 02] calling base load_real_data START", flush=True)

    data, context_dim, context_cols = _ORIGINAL_LOAD_REAL_DATA_BEFORE_EXTERNAL_EXP3(
        data_raw=data_raw,
        dph_cap_q=dph_cap_q,
        dph_cap_end_week=dph_cap_end_week,
    )

    print(
        f"[EXT-HAT 03] base load_real_data RETURNED | "
        f"asins={len(data):,} | elapsed={(time.perf_counter()-_ext_all_t0)/60:.2f}m",
        flush=True,
    )

    required = [
        "asin",
        "order_week",
        "attn_pred_total_log",
        "attn_pred_buy_box_log",
        "attn_pred_instock_log",
    ]

    if not all(c in data_raw.columns for c in required):
        print("\nExternal exposure-3 columns not found. Using original future_context.", flush=True)
        print("[EXT-HAT 99] external-hat wrapper RETURN without hats", flush=True)
        return data, context_dim, context_cols

    _t = time.perf_counter()
    print(
        f"[EXT-HAT 04] select/copy external columns START | rows={len(data_raw):,}",
        flush=True,
    )
    ext = data_raw[required].copy()
    print(
        f"[EXT-HAT 05] select/copy external columns DONE | "
        f"elapsed={time.perf_counter()-_t:.2f}s",
        flush=True,
    )

    _t = time.perf_counter()
    print("[EXT-HAT 06] type conversion START", flush=True)
    ext["asin"] = ext["asin"].astype(str)
    ext["order_week"] = pd.to_datetime(ext["order_week"])

    for c in ["attn_pred_total_log", "attn_pred_buy_box_log", "attn_pred_instock_log"]:
        ext[c] = pd.to_numeric(ext[c], errors="coerce").fillna(0.0).clip(lower=0.0)

    print(
        f"[EXT-HAT 07] type conversion DONE | elapsed={time.perf_counter()-_t:.2f}s",
        flush=True,
    )

    _t = time.perf_counter()
    print("[EXT-HAT 08] sort/groupby START", flush=True)
    ext = (
        ext.sort_values(["asin", "order_week"])
        .groupby(["asin", "order_week"], as_index=False)
        .agg(
            attn_pred_total_log=("attn_pred_total_log", "mean"),
            attn_pred_buy_box_log=("attn_pred_buy_box_log", "mean"),
            attn_pred_instock_log=("attn_pred_instock_log", "mean"),
        )
    )
    print(
        f"[EXT-HAT 09] sort/groupby DONE | rows={len(ext):,} | "
        f"asins={ext['asin'].nunique():,} | elapsed={(time.perf_counter()-_t)/60:.2f}m",
        flush=True,
    )

    _t = time.perf_counter()
    print("[EXT-HAT 10] build ASIN lookup START", flush=True)
    ext_by_asin = {
        asin_key: group.reset_index(drop=True)
        for asin_key, group in ext.groupby("asin", sort=False)
    }
    print(
        f"[EXT-HAT 11] build ASIN lookup DONE | groups={len(ext_by_asin):,} | "
        f"elapsed={time.perf_counter()-_t:.2f}s",
        flush=True,
    )

    new_cols = [
        "external_total_dph_hat_log",
        "external_buy_box_dph_hat_log",
        "external_instock_dph_hat_log",
    ]

    added_any = False
    n_asins = len(data)
    _attach_t0 = time.perf_counter()
    aligned_count = 0
    missing_count = 0
    print(
        f"[EXT-HAT 12] attach hats to per-ASIN future_context START | asins={n_asins:,}",
        flush=True,
    )

    for i, (asin, d) in enumerate(data.items(), start=1):
        sub = ext_by_asin.get(str(asin))

        if sub is None:
            missing_count += 1
            sub = pd.DataFrame({
                "order_week": pd.to_datetime(d["week"]),
                "attn_pred_total_log": 0.0,
                "attn_pred_buy_box_log": 0.0,
                "attn_pred_instock_log": 0.0,
            })
        elif len(sub) != len(d["week"]):
            aligned_count += 1
            week_df = pd.DataFrame({"order_week": pd.to_datetime(d["week"])})
            sub = week_df.merge(
                sub.drop(columns=["asin"], errors="ignore"),
                on="order_week",
                how="left",
            )

        arr = sub[[
            "attn_pred_total_log",
            "attn_pred_buy_box_log",
            "attn_pred_instock_log",
        ]].fillna(0.0).values.astype(np.float32)

        old_fc = d["future_context"]
        d["future_context"] = np.concatenate([old_fc, arr], axis=1)
        added_any = True

        if i == 1 or i % 1000 == 0 or i == n_asins:
            elapsed = time.perf_counter() - _attach_t0
            rate = i / max(elapsed, 1e-9)
            eta = (n_asins - i) / max(rate, 1e-9)
            print(
                f"[EXT-HAT 13] attach {i:,}/{n_asins:,} "
                f"({100*i/max(n_asins,1):.1f}%) | "
                f"elapsed={elapsed/60:.1f}m | ETA={eta/60:.1f}m | "
                f"realigned={aligned_count:,} | missing={missing_count:,}",
                flush=True,
            )

    print(
        f"[EXT-HAT 14] attach hats DONE | elapsed={(time.perf_counter()-_attach_t0)/60:.2f}m",
        flush=True,
    )

    if added_any:
        context_cols = context_cols + new_cols
        context_dim = len(context_cols)

        print("\n" + "=" * 100)
        print("EXTERNAL EXPOSURE-3 HATS ADDED TO FUTURE_CONTEXT")
        print("=" * 100)
        print("Added context cols:", new_cols)
        print("New context dim:", context_dim)

    print(
        f"[EXT-HAT 15] external-hat wrapper RETURN | "
        f"total_elapsed={(time.perf_counter()-_ext_all_t0)/60:.2f}m",
        flush=True,
    )
    return data, context_dim, context_cols



# =============================================================================
# Package-Aware ASIN Graph Context
#
# Build peer relationships using category and relaxed package comparability.
# Graph statistics are calculated at the forecast origin from historical values
# only and are appended to future_context.
# =============================================================================

GRAPH_CONTEXT_COLS = [
    "graph_peer_total_mean13_log",
    "graph_peer_buybox_mean13_log",
    "graph_peer_instock_mean13_log",
    "graph_peer_demand_mean13_log",
    "graph_peer_active_rate13",
    "graph_peer_zero_rate13",
    "graph_peer_count_log",
    "graph_peer_rank_prior",
    "graph_same_hbt_peer_rate",
    "graph_top10_peer_rate",
]


def _graph_mode_str(x, default="MISSING"):
    try:
        s = pd.Series(x).astype(str).replace({"nan": default, "None": default, "": default})
        if len(s) == 0:
            return default
        return str(s.mode().iloc[0]) if len(s.mode()) else default
    except Exception:
        return default


def _graph_build_meta_from_raw(data_raw):
    """Static product metadata used for graph neighbor construction.

    Safe optimization: operate only on columns required by the graph instead
    of copying the full rolling dataframe. The grouping, sorting, mode, and
    last-observation rules are unchanged, so graph metadata is identical.
    """
    t0 = time.perf_counter()
    print(f"[GRAPH-META] START | rows={len(data_raw):,}", flush=True)

    asin_col = "asin" if "asin" in data_raw.columns else ("ASIN" if "ASIN" in data_raw.columns else None)
    if asin_col is None:
        print("[GRAPH-META] no ASIN column; returning empty metadata", flush=True)
        return {}

    required = [
        asin_col, "order_week", "pkg_height", "pkg_length", "pkg_width",
        "pkg_weight", "our_price", "ind_top10_brand", "category_code", "hbt",
    ]
    present = [c for c in required if c in data_raw.columns]
    print(f"[GRAPH-META 01] select/copy columns START | cols={present}", flush=True)
    _step_t = time.perf_counter()
    # This is the main memory optimization: copy only graph-relevant columns.
    df = data_raw.loc[:, present].copy()
    print(
        f"[GRAPH-META 02] select/copy columns DONE | rows={len(df):,} | "
        f"elapsed={(time.perf_counter()-_step_t)/60:.2f}m | "
        f"memory_gb={df.memory_usage(deep=True).sum()/1024**3:.2f}",
        flush=True,
    )

    _step_t = time.perf_counter()
    print("[GRAPH-META 03] ASIN astype(str) START", flush=True)
    df[asin_col] = df[asin_col].astype(str)
    print(f"[GRAPH-META 04] ASIN astype(str) DONE | elapsed={time.perf_counter()-_step_t:.2f}s", flush=True)

    if "order_week" in df.columns:
        _step_t = time.perf_counter()
        print("[GRAPH-META 05] order_week datetime conversion START", flush=True)
        df["order_week"] = pd.to_datetime(df["order_week"], errors="coerce")
        print(f"[GRAPH-META 06] order_week datetime conversion DONE | elapsed={time.perf_counter()-_step_t:.2f}s", flush=True)

        _step_t = time.perf_counter()
        print(f"[GRAPH-META 07] sort START | keys={[asin_col, 'order_week']}", flush=True)
        # Keep the same pandas sorting behavior as the original implementation.
        df = df.sort_values([asin_col, "order_week"])
        print(f"[GRAPH-META 08] sort DONE | elapsed={(time.perf_counter()-_step_t)/60:.2f}m", flush=True)

    _step_t = time.perf_counter()
    print("[GRAPH-META 09] numeric conversion START", flush=True)
    for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight", "our_price", "ind_top10_brand"]:
        if c not in df.columns:
            df[c] = np.nan if c.startswith("pkg_") else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"[GRAPH-META 10] numeric conversion DONE | elapsed={time.perf_counter()-_step_t:.2f}s", flush=True)

    if "category_code" not in df.columns:
        df["category_code"] = "UNKNOWN"
    if "hbt" not in df.columns:
        df["hbt"] = "MISSING"

    meta = {}
    _step_t = time.perf_counter()
    print("[GRAPH-META 11] groupby object/ngroups START", flush=True)
    groups = df.groupby(asin_col, sort=True)
    n_groups = groups.ngroups
    print(f"[GRAPH-META 12] groupby object/ngroups DONE | groups={n_groups:,} | elapsed={time.perf_counter()-_step_t:.2f}s", flush=True)
    print("[GRAPH-META 13] per-ASIN metadata loop START", flush=True)
    _loop_t = time.perf_counter()
    for i, (asin, g) in enumerate(groups, 1):
        dims = {}
        for c in ["pkg_height", "pkg_length", "pkg_width", "pkg_weight"]:
            vals = pd.to_numeric(g[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            v = float(vals.iloc[-1]) if len(vals) else np.nan
            dims[c] = v
            dims[f"log_{c}"] = float(np.log1p(max(v, 0.0))) if np.isfinite(v) else np.nan
        vol = dims.get("pkg_height", np.nan) * dims.get("pkg_length", np.nan) * dims.get("pkg_width", np.nan)
        dims["pkg_volume"] = vol
        dims["log_pkg_volume"] = float(np.log1p(max(vol, 0.0))) if np.isfinite(vol) else np.nan
        price_vals = pd.to_numeric(g["our_price"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        price = float(price_vals.iloc[-1]) if len(price_vals) else 0.0
        meta[str(asin)] = {
            "category_code": _graph_mode_str(g["category_code"], "UNKNOWN"),
            "hbt": _graph_mode_str(g["hbt"], "MISSING").lower(),
            "ind_top10_brand": float(pd.to_numeric(g["ind_top10_brand"], errors="coerce").fillna(0.0).iloc[-1]),
            "log_our_price": float(np.log1p(max(price, 0.0))),
            **dims,
        }
        if i % 1000 == 0 or i == n_groups:
            elapsed = time.perf_counter() - _loop_t
            rate = i / max(elapsed, 1e-9)
            eta = (n_groups - i) / max(rate, 1e-9)
            print(
                f"[GRAPH-META 14] loop {i:,}/{n_groups:,} ({100*i/max(n_groups,1):.1f}%) | "
                f"elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                flush=True,
            )

    print(f"[GRAPH-META 15] per-ASIN metadata loop DONE | elapsed={(time.perf_counter()-_loop_t)/60:.2f}m", flush=True)
    print(f"[GRAPH-META] DONE | ASINs={len(meta):,} | {(time.perf_counter()-t0)/60:.2f} min", flush=True)
    return meta


def _graph_pkg_relaxed_similar(mi, mj, max_mean_log_gap=0.40, max_volume_log_gap=0.75, max_weight_log_gap=0.55):
    """Relaxed package comparability: same physical scale, not exact duplicate."""
    dim_cols = ["log_pkg_height", "log_pkg_length", "log_pkg_width", "log_pkg_weight"]
    gaps = []
    for c in dim_cols:
        a, b = mi.get(c, np.nan), mj.get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            gaps.append(abs(float(a) - float(b)))
    if len(gaps) == 0:
        return False
    mean_gap = float(np.mean(gaps))
    vol_gap = abs(float(mi.get("log_pkg_volume", np.nan)) - float(mj.get("log_pkg_volume", np.nan))) \
        if np.isfinite(mi.get("log_pkg_volume", np.nan)) and np.isfinite(mj.get("log_pkg_volume", np.nan)) else mean_gap
    wt_gap = abs(float(mi.get("log_pkg_weight", np.nan)) - float(mj.get("log_pkg_weight", np.nan))) \
        if np.isfinite(mi.get("log_pkg_weight", np.nan)) and np.isfinite(mj.get("log_pkg_weight", np.nan)) else mean_gap
    return (mean_gap <= max_mean_log_gap) and (vol_gap <= max_volume_log_gap) and (wt_gap <= max_weight_log_gap)


def _graph_static_similarity(mi, mj):
    """Deterministic candidate-edge score based only on static product metadata."""
    same_cat = float(mi.get("category_code", "UNKNOWN") == mj.get("category_code", "UNKNOWN"))
    same_hbt = float(mi.get("hbt", "missing") == mj.get("hbt", "missing"))
    same_top10 = float(mi.get("ind_top10_brand", 0.0) == mj.get("ind_top10_brand", 0.0))

    gaps = []
    for c in ["log_pkg_height", "log_pkg_length", "log_pkg_width", "log_pkg_weight"]:
        a, b = mi.get(c, np.nan), mj.get(c, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            gaps.append(abs(float(a) - float(b)))
    pkg_sim = float(np.exp(-np.mean(gaps))) if gaps else 0.0
    price_gap = abs(float(mi.get("log_our_price", 0.0)) - float(mj.get("log_our_price", 0.0)))
    price_sim = float(np.exp(-price_gap))
    return 0.40 * same_cat + 0.15 * same_hbt + 0.25 * pkg_sim + 0.15 * price_sim + 0.05 * same_top10


def _graph_recent_mean(arr, end, window=13):
    x = np.asarray(arr[max(0, end-window):end], dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.mean(np.clip(x, 0, None)))


def _graph_strength_for_asin(d, end):
    return (
        0.30 * np.log1p(_graph_recent_mean(d.get("total_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("buy_box_dph", []), end, 13)) +
        0.25 * np.log1p(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13)) +
        0.20 * np.log1p(_graph_recent_mean(d.get("demand", []), end, 13))
    )


def _graph_percentile_rank(value, values):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(vals) <= 1:
        return 0.5
    return float((np.sum(vals < value) + 0.5 * np.sum(vals == value)) / len(vals))


def _graph_add_context_cols_to_data(data, context_cols, data_raw=None):
    """Build the final future-context layout in one allocation and attach metadata.

    This combines the previous append-zero-columns and later column-reorder passes.
    Values, graph columns, and final column order are unchanged; only redundant
    NumPy allocations/copies are removed.
    """
    t0 = time.perf_counter()
    old_cols = list(context_cols)
    add_cols = [c for c in GRAPH_CONTEXT_COLS if c not in old_cols]
    appended_cols = old_cols + add_cols

    if all(c in appended_cols for c in EXTERNAL_HAT_COLS):
        non_hat_cols = [c for c in appended_cols if c not in EXTERNAL_HAT_COLS]
        final_cols = non_hat_cols + EXTERNAL_HAT_COLS
    else:
        final_cols = appended_cols

    old_pos = {c: i for i, c in enumerate(old_cols)}
    final_pos = {c: i for i, c in enumerate(final_cols)}
    n_asins = len(data)
    print(
        f"[GRAPH-CONTEXT] layout START | ASINs={n_asins:,} | "
        f"old_dim={len(old_cols)} | new_dim={len(final_cols)} | added={len(add_cols)}",
        flush=True,
    )

    print("[GRAPH-CONTEXT 01] future_context reallocation loop START", flush=True)
    _array_t = time.perf_counter()
    for i, d in enumerate(data.values(), 1):
        old_fc = np.asarray(d["future_context"], dtype=np.float32)
        # Allocate the exact final shape once. New graph columns remain zero.
        new_fc = np.zeros((old_fc.shape[0], len(final_cols)), dtype=np.float32)
        for col, src_idx in old_pos.items():
            dst_idx = final_pos.get(col)
            if dst_idx is not None:
                new_fc[:, dst_idx] = old_fc[:, src_idx]
        d["future_context"] = new_fc

        if i % 1000 == 0 or i == n_asins:
            elapsed = time.perf_counter() - _array_t
            rate = i / max(elapsed, 1e-9)
            eta = (n_asins-i) / max(rate, 1e-9)
            print(
                f"[GRAPH-CONTEXT 02] arrays {i:,}/{n_asins:,} ({100*i/max(n_asins,1):.1f}%) | "
                f"elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                flush=True,
            )

    print(f"[GRAPH-CONTEXT 03] future_context reallocation loop DONE | elapsed={(time.perf_counter()-_array_t)/60:.2f}m", flush=True)
    print(f"[GRAPH-CONTEXT 04] metadata build CALL", flush=True)
    meta = _graph_build_meta_from_raw(data_raw) if data_raw is not None else {}
    default_meta = {
        "category_code": "UNKNOWN", "hbt": "missing", "ind_top10_brand": 0.0,
        "log_our_price": 0.0, "log_pkg_height": np.nan, "log_pkg_length": np.nan,
        "log_pkg_width": np.nan, "log_pkg_weight": np.nan, "log_pkg_volume": np.nan,
    }
    graph_idx = {c: final_pos[c] for c in GRAPH_CONTEXT_COLS if c in final_pos}

    print(f"[GRAPH-CONTEXT 05] metadata build RETURNED | meta_asins={len(meta):,}", flush=True)
    print(f"[GRAPH-CONTEXT 06] attach metadata START", flush=True)
    _attach_t = time.perf_counter()
    for i, (asin, d) in enumerate(data.items(), 1):
        d["context_cols"] = final_cols
        d["graph_context_idx"] = graph_idx.copy()
        d["graph_meta"] = meta.get(str(asin), default_meta.copy())
        d["dph_proxy_context_idx"] = {
            c: final_pos[c]
            for c in d.get("dph_proxy_context_idx", {})
            if c in final_pos
        }
        if i % 1000 == 0 or i == n_asins:
            elapsed = time.perf_counter() - _attach_t
            rate = i / max(elapsed, 1e-9)
            eta = (n_asins-i) / max(rate, 1e-9)
            print(
                f"[GRAPH-CONTEXT 07] attach {i:,}/{n_asins:,} ({100*i/max(n_asins,1):.1f}%) | "
                f"elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                flush=True,
            )

    print(f"[GRAPH-CONTEXT 08] attach metadata DONE | elapsed={(time.perf_counter()-_attach_t)/60:.2f}m", flush=True)
    print(f"[GRAPH-CONTEXT] DONE | {(time.perf_counter()-t0)/60:.2f} min", flush=True)
    return data, len(final_cols), final_cols


_GRAPH_NEIGHBOR_MAP_CACHE = {}


def _graph_build_neighbor_map(data, min_neighbors=3):
    """Build the exact original neighbor map, with reuse for train/validation datasets."""
    cache_key = (id(data), len(data), int(min_neighbors))
    cached = _GRAPH_NEIGHBOR_MAP_CACHE.get(cache_key)
    if cached is not None:
        print("[GRAPH-NEIGHBORS] cache HIT", flush=True)
        return cached

    t0 = time.perf_counter()
    print(f"[GRAPH-NEIGHBORS] START | ASINs={len(data):,}", flush=True)
    asins = list(data.keys())
    by_cat = {}
    for a in asins:
        cat = data[a].get("graph_meta", {}).get("category_code", "UNKNOWN")
        by_cat.setdefault(cat, []).append(a)

    nbrs = {}
    for i, a in enumerate(asins, 1):
        mi = data[a].get("graph_meta", {})
        same_cat = by_cat.get(mi.get("category_code", "UNKNOWN"), [])
        cand = []
        for b in same_cat:
            if b == a:
                continue
            mj = data[b].get("graph_meta", {})
            if _graph_pkg_relaxed_similar(mi, mj):
                cand.append(b)
        if len(cand) < min_neighbors:
            cand = [b for b in same_cat if b != a]
        nbrs[a] = cand
        if i % 5000 == 0 or i == len(asins):
            print(
                f"[GRAPH-NEIGHBORS] progress {i:,}/{len(asins):,} | "
                f"{(time.perf_counter()-t0)/60:.2f} min",
                flush=True,
            )

    _GRAPH_NEIGHBOR_MAP_CACHE[cache_key] = nbrs
    print(f"[GRAPH-NEIGHBORS] DONE | {(time.perf_counter()-t0)/60:.2f} min", flush=True)
    return nbrs


class _GraphContextMixin:
    def _init_graph_context(self, min_graph_neighbors=3):
        self.graph_neighbor_map = _graph_build_neighbor_map(self.data, min_neighbors=min_graph_neighbors)
        self._graph_context_cache = {}
        counts = [len(v) for v in self.graph_neighbor_map.values()]
        if len(counts):
            print("Package-aware graph context enabled | ASINs:", len(counts),
                  "| median neighbors:", int(np.median(counts)),
                  "| mean neighbors:", round(float(np.mean(counts)), 2),
                  "| min/max:", int(np.min(counts)), int(np.max(counts)))

    def _compute_graph_context_vec(self, asin, end):
        key = (str(asin), int(end))
        if key in self._graph_context_cache:
            return self._graph_context_cache[key]
        d_i = self.data[asin]
        idx = d_i.get("graph_context_idx", {})
        if len(idx) == 0:
            vec = np.zeros(len(GRAPH_CONTEXT_COLS), dtype=np.float32)
            self._graph_context_cache[key] = vec
            return vec
        nbrs = self.graph_neighbor_map.get(asin, [])
        if len(nbrs) == 0:
            nbrs = [asin]

        vals_total, vals_buy, vals_inst, vals_dem, active_rates = [], [], [], [], []
        strengths = []
        same_hbt, top10 = [], []
        hbt_i = d_i.get("graph_meta", {}).get("hbt", "missing")
        for b in nbrs:
            d = self.data[b]
            vals_total.append(_graph_recent_mean(d.get("total_dph", []), end, 13))
            vals_buy.append(_graph_recent_mean(d.get("buy_box_dph", []), end, 13))
            vals_inst.append(_graph_recent_mean(d.get("in_stock_dph", d.get("instock_raw", [])), end, 13))
            vals_dem.append(_graph_recent_mean(d.get("demand", []), end, 13))
            x = np.asarray(d.get("in_stock_dph", d.get("instock_raw", []))[max(0, end-13):end], dtype=float)
            active_rates.append(float(np.mean(x > 0)) if len(x) else 0.0)
            strengths.append(_graph_strength_for_asin(d, end))
            same_hbt.append(1.0 if d.get("graph_meta", {}).get("hbt", "missing") == hbt_i else 0.0)
            top10.append(float(d.get("graph_meta", {}).get("ind_top10_brand", 0.0)))

        own_strength = _graph_strength_for_asin(d_i, end)
        all_strengths = strengths + [own_strength]
        rank_prior = _graph_percentile_rank(own_strength, all_strengths)
        peer_count = max(len(nbrs), 1)
        vec_map = {
            "graph_peer_total_mean13_log": np.log1p(np.mean(vals_total) if len(vals_total) else 0.0),
            "graph_peer_buybox_mean13_log": np.log1p(np.mean(vals_buy) if len(vals_buy) else 0.0),
            "graph_peer_instock_mean13_log": np.log1p(np.mean(vals_inst) if len(vals_inst) else 0.0),
            "graph_peer_demand_mean13_log": np.log1p(np.mean(vals_dem) if len(vals_dem) else 0.0),
            "graph_peer_active_rate13": float(np.mean(active_rates) if len(active_rates) else 0.0),
            "graph_peer_zero_rate13": float(1.0 - np.mean(active_rates) if len(active_rates) else 1.0),
            "graph_peer_count_log": np.log1p(peer_count),
            "graph_peer_rank_prior": rank_prior,
            "graph_same_hbt_peer_rate": float(np.mean(same_hbt) if len(same_hbt) else 0.0),
            "graph_top10_peer_rate": float(np.mean(top10) if len(top10) else 0.0),
        }
        vec = np.array([vec_map[c] for c in GRAPH_CONTEXT_COLS], dtype=np.float32)
        self._graph_context_cache[key] = vec
        return vec

    def _inject_graph_context(self, fc, d, asin, end):
        idx = d.get("graph_context_idx", {})
        if len(idx) == 0 or fc is None or len(fc) == 0:
            return fc
        base_vec = self._compute_graph_context_vec(asin, end)
        H = fc.shape[0]
        for step_h in range(H):
            # keep graph prior strongest near origin but still available at long horizon
            h_decay = 0.65 + 0.35 * np.exp(-0.06 * step_h)
            for k, col in enumerate(GRAPH_CONTEXT_COLS):
                if col in idx:
                    fc[step_h, idx[col]] = float(base_vec[k]) * h_decay
        return fc

    def _dynamic_topk_peers(self, asin, top_k, min_similarity=0.55):
        """Return at most Top-K candidates that pass a static similarity floor."""
        mi = self.data[asin].get("graph_meta", {})
        candidates = self.graph_neighbor_map.get(asin, [])
        scored = [
            (_graph_static_similarity(mi, self.data[b].get("graph_meta", {})), str(b), b)
            for b in candidates if b in self.data and b != asin
        ]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [
            (b, score) for score, _, b in scored
            if score >= float(min_similarity)
        ][:int(top_k)]


# Demand dataset integration
_ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT = load_real_data
_ORIGINAL_DEMAND_DATASET_BEFORE_GRAPH_CONTEXT = DemandDataset


# =============================================================================
# Parallel Eager Sample Construction
#
# Build materialized Dataset samples across forked worker processes when
# available. The serial path is used automatically if multiprocessing fails.
# =============================================================================
_MP_BUILD_DATASET = None


EXTERNAL_HAT_COLS = [
    "external_total_dph_hat_log",
    "external_buy_box_dph_hat_log",
    "external_instock_dph_hat_log",
]


def load_real_data(data_raw, dph_cap_q=0.995, dph_cap_end_week=None):
    _wrapper_t0 = time.perf_counter()
    print("[PROFILE-WRAPPER 01] graph-context wrapper ENTERED", flush=True)
    print(
        f"[PROFILE-WRAPPER 02] active base loader="
        f"{getattr(_ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT, '__name__', type(_ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT).__name__)}",
        flush=True,
    )
    print("[PROFILE-WRAPPER 03] CALL pre-graph loader START", flush=True)

    _pregraph_result = _ORIGINAL_LOAD_REAL_DATA_BEFORE_GRAPH_CONTEXT(
        data_raw,
        dph_cap_q=dph_cap_q,
        dph_cap_end_week=dph_cap_end_week,
    )

    print(
        f"[PROFILE-WRAPPER 04] pre-graph loader RETURNED | "
        f"elapsed={(time.perf_counter()-_wrapper_t0)/60:.2f}m",
        flush=True,
    )
    _unpack_t0 = time.perf_counter()
    print("[PROFILE-WRAPPER 05] unpack pre-graph tuple START", flush=True)
    data, context_dim, context_cols = _pregraph_result
    print(
        f"[PROFILE-WRAPPER 06] unpack pre-graph tuple DONE | "
        f"asins={len(data):,} | context_dim={context_dim} | "
        f"elapsed={time.perf_counter()-_unpack_t0:.6f}s",
        flush=True,
    )
    del _pregraph_result
    print("[PROFILE-WRAPPER 07] GRAPH PIPELINE ENTER", flush=True)

    _graph_t0 = time.perf_counter()
    print("[PROFILE-GRAPH 01] graph preprocessing START", flush=True)
    data, context_dim, context_cols = _graph_add_context_cols_to_data(
        data,
        context_cols,
        data_raw=data_raw,
    )
    print(
        f"[PROFILE-GRAPH 02] graph preprocessing DONE | "
        f"elapsed={(time.perf_counter()-_graph_t0)/60:.2f}m",
        flush=True,
    )

    print("\n" + "=" * 100)
    print("PACKAGE-AWARE RELATION GRAPH FEATURES ADDED TO DEMAND FUTURE_CONTEXT")
    print("Graph cols:", GRAPH_CONTEXT_COLS)
    print("External exposure hats kept as last 3 columns:", all(c in context_cols[-3:] for c in EXTERNAL_HAT_COLS))
    print("New context dim:", context_dim)
    print("=" * 100)
    print(
        f"[PROFILE-WRAPPER 08] graph-context wrapper RETURN | "
        f"total_elapsed={(time.perf_counter()-_wrapper_t0)/60:.2f}m",
        flush=True,
    )
    return data, context_dim, context_cols


class DemandDataset(_GraphContextMixin, _ORIGINAL_DEMAND_DATASET_BEFORE_GRAPH_CONTEXT):
    """Eager graph-aware dataset.

    Materializes every sliding-window sample once during ``__init__``. This
    restores the pre-lazy behavior while preserving the same window boundaries,
    DPH proxy formulas, graph-context injection, targets, and float32 dtypes.
    """

    def __init__(self, data, history=52, horizon=3, mode="train", val_weeks=20,
                 min_graph_neighbors=3, num_build_workers=None,
                 graph_variant="legacy", dynamic_graph_top_k=10,
                 dynamic_graph_min_similarity=0.55):
        _ds_t0 = time.perf_counter()
        self.data = data
        self.history = int(history)
        self.horizon = int(horizon)
        self.val_weeks = int(val_weeks)
        self.mode = str(mode)
        self.graph_variant = str(graph_variant).lower()
        if self.graph_variant not in {"legacy", "dynamic_signed"}:
            raise ValueError("graph_variant must be 'legacy' or 'dynamic_signed'.")
        self.dynamic_graph_top_k = int(dynamic_graph_top_k)
        self.dynamic_graph_min_similarity = float(dynamic_graph_min_similarity)
        self.samples = []

        print(
            f"[STAGE] DemandDataset EAGER START | mode={self.mode} | "
            f"ASINs={len(data):,} | history={self.history} | horizon={self.horizon}",
            flush=True,
        )

        graph_t0 = time.perf_counter()
        print(f"[DATASET-EAGER] graph neighbor initialization START | mode={self.mode}", flush=True)
        self._init_graph_context(min_graph_neighbors=min_graph_neighbors)
        self.dynamic_peer_map = (
            {
                a: self._dynamic_topk_peers(
                    a, self.dynamic_graph_top_k, self.dynamic_graph_min_similarity
                )
                for a in self.data
            }
            if self.graph_variant == "dynamic_signed" else {}
        )
        print(
            f"[DATASET-EAGER] graph neighbor initialization DONE | mode={self.mode} | "
            f"elapsed={time.perf_counter()-graph_t0:.2f}s",
            flush=True,
        )

        total_asins = len(data)
        eligible_asins = 0
        report_every = max(100, min(500, total_asins // 20 if total_asins else 100))
        build_t0 = time.perf_counter()

        def as_float_tensor(arr):
            a = np.asarray(arr, dtype=np.float32)
            if not a.flags.c_contiguous:
                a = np.ascontiguousarray(a)
            return torch.from_numpy(a)

        for asin_i, (asin, d) in enumerate(data.items(), start=1):
            T = len(d["demand"])
            if self.mode == "train":
                starts = range(max(0, T - self.val_weeks - self.horizon - self.history + 1))
            else:
                s = T - self.history - self.horizon
                starts = [s] if s >= 0 else []

            added = 0
            for start in starts:
                history_end = start + self.history
                target_end = history_end + self.horizon
                fc = self._make_future_context_with_dph_proxies(
                    d=d,
                    start=start,
                    history=self.history,
                    horizon=self.horizon,
                )
                if self.graph_variant == "legacy":
                    fc = self._inject_graph_context(fc, d, asin, history_end)
                else:
                    # Keep the exact same context layout for a fair ablation, but
                    # disable legacy peer means. Dynamic messages enter in-model.
                    for col_idx in d.get("graph_context_idx", {}).values():
                        fc[:, col_idx] = 0.0

                sample = {
                    "x": as_float_tensor(d["features"][start:history_end]),
                    "future_context": as_float_tensor(fc),
                    "y": as_float_tensor(d["demand"][history_end:target_end]),
                    "asin": asin,
                    "target_week": [str(w)[:10] for w in d["week"][history_end:target_end]],
                    "oos": as_float_tensor(d["oos"][history_end:target_end]),
                    "our_price": as_float_tensor(d["price_raw"][history_end:target_end]),
                    "pkg_volume": as_float_tensor(d["pkg_volume_raw"][history_end:target_end]),
                    "future_instock": as_float_tensor(d["instock_raw"][history_end:target_end]),
                    "future_total_dph": as_float_tensor(d["total_dph_raw"][history_end:target_end]),
                    "future_buy_box_dph": as_float_tensor(d["buy_box_dph_raw"][history_end:target_end]),
                }
                if self.graph_variant == "dynamic_signed":
                    sample["_graph_asin"] = asin
                    sample["_graph_origin_week"] = str(d["week"][history_end - 1])[:10]
                self.samples.append(sample)
                added += 1

            if added:
                eligible_asins += 1

            if asin_i % report_every == 0 or asin_i == total_asins:
                elapsed = time.perf_counter() - build_t0
                rate = asin_i / max(elapsed, 1e-9)
                eta = (total_asins - asin_i) / max(rate, 1e-9)
                print(
                    f"[DATASET-EAGER] build progress | mode={self.mode} | "
                    f"ASINs={asin_i:,}/{total_asins:,} | eligible={eligible_asins:,} | "
                    f"samples={len(self.samples):,} | elapsed={elapsed/60:.2f}m | ETA={eta/60:.2f}m",
                    flush=True,
                )

        print(
            f"[STAGE] DemandDataset EAGER DONE | mode={self.mode} | "
            f"eligible_asins={eligible_asins:,} | samples={len(self.samples):,} | "
            f"total_elapsed={(time.perf_counter()-_ds_t0)/60:.2f}m",
            flush=True,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]
        if self.graph_variant != "dynamic_signed":
            return sample

        out = {k: v for k, v in sample.items() if not k.startswith("_graph_")}
        asin = sample["_graph_asin"]
        origin_week = np.datetime64(sample["_graph_origin_week"])
        K, T = self.dynamic_graph_top_k, self.history
        input_dim = int(out["x"].shape[-1])
        peer_x = np.zeros((K, T, input_dim), dtype=np.float32)
        peer_y = np.zeros((K, self.horizon), dtype=np.float32)
        peer_mask = np.zeros(K, dtype=np.float32)
        peer_static = np.zeros(K, dtype=np.float32)

        for k, (peer, static_score) in enumerate(self.dynamic_peer_map.get(asin, [])):
            d = self.data[peer]
            weeks = np.asarray(d["week"], dtype="datetime64[D]")
            end = int(np.searchsorted(weeks, origin_week, side="right"))
            hist_start = max(0, end - T)
            hist = np.asarray(d["features"][hist_start:end], dtype=np.float32)
            if len(hist):
                peer_x[k, -len(hist):] = hist
            future = np.asarray(d["demand"][end:end + self.horizon], dtype=np.float32)
            peer_y[k, :len(future)] = future
            peer_mask[k] = float(len(hist) > 0)
            peer_static[k] = float(static_score)

        out["peer_x"] = torch.from_numpy(peer_x)
        out["peer_y"] = torch.from_numpy(peer_y)
        out["peer_mask"] = torch.from_numpy(peer_mask)
        out["peer_static"] = torch.from_numpy(peer_static)
        return out


def _dataloader_worker_init_fn(worker_id):
    # Seeds are deterministic for a fixed DataLoader generator seed.  The
    # Dataset itself has no stochastic transforms, but this keeps future
    # worker-side additions reproducible.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    print(
        f"[DATALOADER] worker READY | worker={worker_id} | pid={os.getpid()} | seed={worker_seed}",
        flush=True,
    )


def make_demand_dataloader(dataset, batch_size, shuffle, seed=42, num_workers=None, name="loader"):
    """GPU-friendly DataLoader with conservative worker/prefetch defaults."""
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        # Four is a safe starting point for the large shared per-ASIN data dict.
        num_workers = min(4, max(0, cpu_count - 1))
    num_workers = max(0, int(num_workers))

    generator = torch.Generator()
    generator.manual_seed(int(seed))

    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        generator=generator,
    )
    if num_workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=2,
            worker_init_fn=_dataloader_worker_init_fn,
        )

    print(
        f"[DATALOADER] create | name={name} | samples={len(dataset):,} | "
        f"batch_size={batch_size} | shuffle={shuffle} | workers={num_workers} | "
        f"pin_memory={kwargs['pin_memory']} | "
        f"persistent={num_workers > 0} | prefetch={2 if num_workers > 0 else 0}",
        flush=True,
    )
    loader = DataLoader(**kwargs)
    print(
        f"[DATALOADER] ready | name={name} | batches={len(loader):,}",
        flush=True,
    )
    return loader


# =============================================================================
# Joint Exposure and Demand Model
#
# A shared historical encoder feeds two future-facing branches:
#
#   Exposure covariate decoder
#     Predicts total, buy-box, and in-stock DPH for each future week.
#
#   Demand decoder
#     Consumes future-known covariates together with predicted exposure and
#     produces the demand distribution used for Monte Carlo quantiles.
#
# Predicted exposure connects the two branches; realized future exposure is
# never passed to the demand decoder.
# =============================================================================


# -----------------------------------------------------------------------------
# Joint Model Composition
#
# Shared encoder -> exposure covariate decoder -> demand decoder.
# -----------------------------------------------------------------------------
class DynamicSignedGraphMessage(nn.Module):
    """Activity-gated, z-conditioned signed messages over fixed Top-K peers.

    Static product similarity defines only the candidate pool. Encoder states
    dynamically assign neutral/positive/competitive probabilities at each FCD.
    The sampled shared z enters message values, not candidate selection.
    """
    def __init__(self, d_model=32, d_z=16, horizon=3, hidden=64):
        super().__init__()
        self.horizon = horizon
        self.activity_head = nn.Linear(d_model, 1)
        edge_dim = 4 * d_model + 1
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(hidden, 3),
        )
        self.z_proj = nn.Linear(d_z, d_model)
        self.pos_value = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        self.neg_value = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())
        self.message_out = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.mu_head = nn.Linear(d_model, horizon)
        self.alpha_head = nn.Linear(d_model, horizon)
        nn.init.zeros_(self.mu_head.weight); nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.alpha_head.weight); nn.init.zeros_(self.alpha_head.bias)

    def relation_logits(self, target_phi, peer_phi, peer_static):
        target = target_phi[:, None, :].expand_as(peer_phi)
        pair = torch.cat([
            target, peer_phi, torch.abs(target - peer_phi), target * peer_phi,
            peer_static.unsqueeze(-1),
        ], dim=-1)
        return self.edge_net(pair)

    def forward(self, target_phi, peer_phi, peer_mask, peer_static, z,
                relation_logits=None):
        if relation_logits is None:
            relation_logits = self.relation_logits(target_phi, peer_phi, peer_static)
        relation_prob = torch.softmax(relation_logits, dim=-1)

        target_active = torch.sigmoid(self.activity_head(target_phi)).squeeze(-1)
        peer_active = torch.sigmoid(self.activity_head(peer_phi)).squeeze(-1)
        pair_gate = 1.0 - (1.0 - target_active[:, None]) * (1.0 - peer_active)
        valid = peer_mask.float()

        pos_w = relation_prob[..., 1] * pair_gate * valid
        neg_w = relation_prob[..., 2] * pair_gate * valid
        pos_w = pos_w / pos_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        neg_w = neg_w / neg_w.sum(dim=1, keepdim=True).clamp_min(1e-6)

        z_peer = self.z_proj(z)[:, None, :].expand_as(peer_phi)
        pos_value = self.pos_value(torch.cat([peer_phi, z_peer], dim=-1))
        neg_value = self.neg_value(torch.cat([peer_phi, z_peer], dim=-1))
        pos_msg = torch.sum(pos_w.unsqueeze(-1) * pos_value, dim=1)
        neg_msg = torch.sum(neg_w.unsqueeze(-1) * neg_value, dim=1)
        message = self.message_out(torch.cat([target_phi, pos_msg, neg_msg], dim=-1))
        return {
            "mu_residual": self.mu_head(message),
            "alpha_residual": self.alpha_head(message),
            "target_activity_logit": self.activity_head(target_phi).squeeze(-1),
            "peer_activity_logits": self.activity_head(peer_phi).squeeze(-1),
            "relation_logits": relation_logits,
            "positive_weight": pos_w,
            "competitive_weight": neg_w,
        }


class JointExposureDemandH3(nn.Module):
    """Experimental joint H3 model.

    Structure
    ---------
    shared demand-history encoder
        -> exposure decoder -> total/buy-box/in-stock H1-H3
        -> demand decoder, conditioned on predicted exposure -> NB demand H1-H3

    By default ``detach_exposure_for_demand=False`` for true end-to-end training:
    demand loss can update the exposure decoder through predicted exposure.
    Set it to True for the safer semi-joint ablation.
    """
    def __init__(self, input_dim, context_dim, d_model=32, d_z=16,
                 horizon=3, prior_scale=0.3,
                 detach_exposure_for_demand=False,
                 graph_variant="legacy"):
        super().__init__()
        if horizon != 3:
            raise ValueError("This experimental file is intentionally H3 only.")
        if context_dim < 3:
            raise ValueError("future_context must reserve its final 3 columns for exposure hats.")

        self.horizon = horizon
        self.context_dim = context_dim
        self.base_context_dim = context_dim - 3
        self.d_z = d_z
        self.detach_exposure_for_demand = bool(detach_exposure_for_demand)
        self.graph_variant = str(graph_variant).lower()
        if self.graph_variant not in {"legacy", "dynamic_signed"}:
            raise ValueError("graph_variant must be 'legacy' or 'dynamic_signed'.")

        # One shared history encoder for both tasks.
        self.encoder = TCNSparseAttnEncoder(input_dim, d_model, horizon)

        # Exposure decoder: independent future decoder/head.
        self.exposure_decoder = DecoderPeakCrossAttention(
            d_model=d_model,
            context_dim=self.base_context_dim,
            horizon=horizon,
            n_heads=4,
            dropout=0.1,
            future_tcn_dilations=(1, 2),
        )
        exp_in = d_model + self.base_context_dim
        self.exp_total_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.exp_ratio_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )
        self.exp_active_head = nn.Sequential(
            nn.Linear(exp_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 3)
        )

        # Demand latent-index and uncertainty branches.
        self.z_generator = ContextZGenerator(d_model, context_dim, d_z, horizon)
        self.epinet = Epinet(d_model, d_z, horizon, prior_scale)

        # Separate demand decoder.
        self.demand_decoder = DecoderPeakCrossAttention(
            d_model=d_model,
            context_dim=context_dim,
            horizon=horizon,
            n_heads=4,
            dropout=0.1,
            future_tcn_dilations=(1, 2),
        )
        demand_dec_in = d_model + context_dim
        self.peak_mu_head = nn.Sequential(
            nn.Linear(demand_dec_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.peak_gate_head = nn.Sequential(
            nn.Linear(demand_dec_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.decoder_epinet = DecoderEpinet(
            d_decoder=d_model,
            context_dim=context_dim,
            d_z=d_z,
            hidden=64,
            prior_scale=0.20,
        )
        self.decoder_epinet_scale = 0.35
        self.dynamic_graph = (
            DynamicSignedGraphMessage(d_model=d_model, d_z=d_z, horizon=horizon)
            if self.graph_variant == "dynamic_signed" else None
        )

        nn.init.constant_(self.exp_total_head[-1].bias, 5.5)
        nn.init.zeros_(self.exp_ratio_head[-1].weight)
        nn.init.constant_(self.exp_ratio_head[-1].bias, 0.7)
        nn.init.zeros_(self.peak_mu_head[-1].weight)
        nn.init.constant_(self.peak_mu_head[-1].bias, -4.0)
        nn.init.zeros_(self.peak_gate_head[-1].weight)
        nn.init.constant_(self.peak_gate_head[-1].bias, -1.5)

    def _base_future_context(self, future_context):
        # Final 3 external-hat slots are intentionally excluded from exposure prediction.
        return future_context[:, :, :self.base_context_dim]

    def predict_exposure(self, H_enc, b_t, peak_score, future_context):
        fc_base = self._base_future_context(future_context)
        dec_h = self.exposure_decoder(
            H_enc, fc_base, b_t=b_t, peak_score=peak_score
        )
        z = torch.cat([dec_h, fc_base], dim=-1)

        total = F.softplus(self.exp_total_head(z).squeeze(-1))
        ratios = torch.sigmoid(self.exp_ratio_head(z))
        buy = total * ratios[..., 0]
        instock = total * ratios[..., 1]
        exposure = torch.stack([total, buy, instock], dim=-1)
        active_logits = self.exp_active_head(z)
        return exposure, active_logits, dec_h

    def _demand_context(self, future_context, exposure_hat):
        exp_for_demand = (
            exposure_hat.detach() if self.detach_exposure_for_demand else exposure_hat
        )
        fc = future_context.clone()
        fc[:, :, -3:] = torch.log1p(exp_for_demand.clamp_min(0.0))
        return fc

    def _combine_one_z(self, mu_base, alpha_base, phi, dec_h,
                       demand_context, peak_mu, peak_gate, z,
                       graph_output=None):
        mu_e, al_e = self.epinet(phi, z)
        dec_mu_e, dec_al_e = self.decoder_epinet(dec_h, demand_context, z)
        dec_mu_e = self.decoder_epinet_scale * dec_mu_e
        dec_al_e = self.decoder_epinet_scale * dec_al_e

        graph_mu = 0.0 if graph_output is None else graph_output["mu_residual"]
        graph_alpha = 0.0 if graph_output is None else graph_output["alpha_residual"]
        mu_normal = F.softplus(mu_base + mu_e + dec_mu_e + graph_mu)
        mu = mu_normal + peak_gate * peak_mu
        alpha = F.softplus(alpha_base + al_e + dec_al_e + graph_alpha) + 1e-4
        return mu, alpha

    def forward(self, x, future_context, nZ=8,
                peer_x=None, peer_mask=None, peer_static=None):
        H_enc, h_t, b_t, peak_score = self.encoder.encode(x)

        peer_phi = None
        relation_logits = None
        if self.graph_variant == "dynamic_signed":
            if peer_x is None or peer_mask is None or peer_static is None:
                raise ValueError("dynamic_signed graph requires peer_x, peer_mask, and peer_static.")
            B, K, T, Din = peer_x.shape
            _, peer_phi_flat, _, _ = self.encoder.encode(peer_x.reshape(B * K, T, Din))
            peer_phi = peer_phi_flat.reshape(B, K, -1)
            relation_logits = self.dynamic_graph.relation_logits(h_t, peer_phi, peer_static)

        exposure_hat, exposure_active_logits, exposure_dec_h = self.predict_exposure(
            H_enc, b_t, peak_score, future_context
        )
        demand_context = self._demand_context(future_context, exposure_hat)

        mu_base = F.softplus(self.encoder.base_head(h_t))
        alpha_base = F.softplus(self.encoder.alpha_head(h_t)) + 1e-4
        phi = h_t.detach()
        z_mean, z_std = self.z_generator(phi, demand_context)
        z_reg = 0.001 * (z_mean.square() + z_std.square()).mean()

        demand_dec_h = self.demand_decoder(
            H_enc, demand_context, b_t=b_t, peak_score=peak_score
        )
        demand_dec_in = torch.cat([demand_dec_h, demand_context], dim=-1)
        peak_mu = F.softplus(self.peak_mu_head(demand_dec_in).squeeze(-1))
        peak_gate = torch.sigmoid(self.peak_gate_head(demand_dec_in).squeeze(-1))

        preds = []
        graph_outputs = []
        for _ in range(nZ):
            z = z_mean + z_std * torch.randn_like(z_mean)
            graph_output = None
            if self.dynamic_graph is not None:
                graph_output = self.dynamic_graph(
                    h_t, peer_phi, peer_mask, peer_static, z,
                    relation_logits=relation_logits,
                )
                graph_outputs.append(graph_output)
            preds.append(self._combine_one_z(
                mu_base, alpha_base, phi, demand_dec_h,
                demand_context, peak_mu, peak_gate, z,
                graph_output=graph_output,
            ))

        return {
            "demand_preds": preds,
            "z_reg": z_reg,
            "exposure_hat": exposure_hat,
            "exposure_active_logits": exposure_active_logits,
            "demand_context": demand_context,
            "graph_outputs": graph_outputs,
            "relation_logits": relation_logits,
        }

    def predict(self, x, future_context, M=100,
                peer_x=None, peer_mask=None, peer_static=None):
        self.eval()
        with torch.no_grad():
            out = self.forward(
                x, future_context, nZ=M,
                peer_x=peer_x, peer_mask=peer_mask, peer_static=peer_static,
            )
            samples = []
            for mu, alpha in out["demand_preds"]:
                dist = torch.distributions.NegativeBinomial(
                    total_count=(1.0 / alpha).clamp_min(1e-4),
                    probs=(mu * alpha / (1.0 + mu * alpha)).clamp(1e-6, 1 - 1e-6),
                )
                samples.append(dist.sample().float())
            samples = torch.stack(samples, dim=1)
            p50 = samples.quantile(0.5, dim=1)
            p70 = torch.maximum(samples.quantile(0.7, dim=1), p50)
            p90 = torch.maximum(samples.quantile(0.9, dim=1), p70)
            return p50, p70, p90, out["exposure_hat"]


def joint_exposure_loss(exposure_hat, active_logits, true_exposure):
    """Gentle H3 exposure loss on log magnitude + occurrence + hierarchy."""
    true_exposure = true_exposure.clamp_min(0.0)
    pred_log = torch.log1p(exposure_hat.clamp_min(0.0))
    true_log = torch.log1p(true_exposure)

    mag = F.smooth_l1_loss(pred_log, true_log)
    active_target = (true_exposure > 0).float()
    active = F.binary_cross_entropy_with_logits(active_logits, active_target)

    # Mild mean calibration per channel/horizon.
    pred_mean = exposure_hat.mean(dim=0)
    true_mean = true_exposure.mean(dim=0)
    mean_calib = torch.abs(
        torch.log1p(pred_mean) - torch.log1p(true_mean)
    ).mean()

    # Hierarchical constraints are already architectural; keep only a tiny numerical penalty.
    hierarchy = (
        F.relu(exposure_hat[..., 1] - exposure_hat[..., 0]).mean()
        + F.relu(exposure_hat[..., 2] - exposure_hat[..., 0]).mean()
    )
    return mag + 0.20 * active + 0.10 * mean_calib + 0.01 * hierarchy


def dynamic_signed_graph_aux_loss(batch, out):
    """Leakage-safe inputs with supervised dynamic relation/activity targets.

    Future demand is used only as a training label. Zero-zero pairs are excluded
    from relation learning because they contain no useful ranking information.
    """
    if out.get("relation_logits") is None or "peer_y" not in batch:
        zero = batch["y"].new_tensor(0.0)
        return zero, {"activity": zero, "relation": zero}

    relation_logits = out["relation_logits"]
    graph0 = out["graph_outputs"][0]
    y = batch["y"].clamp_min(0.0)
    peer_y = batch["peer_y"].clamp_min(0.0)
    peer_mask = batch["peer_mask"] > 0
    target_active = (y.sum(dim=-1) > 0).float()
    peer_active = (peer_y.sum(dim=-1) > 0).float()

    activity = F.binary_cross_entropy_with_logits(
        graph0["target_activity_logit"], target_active
    )
    valid_peer = peer_mask.float().sum().clamp_min(1.0)
    activity = activity + (
        F.binary_cross_entropy_with_logits(
            graph0["peer_activity_logits"], peer_active, reduction="none"
        ) * peer_mask.float()
    ).sum() / valid_peer

    # Dynamic signed label: compare future-vs-recent-history movement. Class 0
    # neutral, 1 positive/co-moving, 2 competitive/opposite-moving.
    target_hist = batch["x"][:, -4:, 0].mean(dim=1)
    peer_hist = batch["peer_x"][:, :, -4:, 0].mean(dim=2)
    target_delta = torch.log1p(y.mean(dim=-1)) - target_hist
    peer_delta = torch.log1p(peer_y.mean(dim=-1)) - peer_hist
    product = target_delta[:, None] * peer_delta
    relation_target = torch.zeros_like(product, dtype=torch.long)
    meaningful = (target_delta[:, None].abs() > 0.05) & (peer_delta.abs() > 0.05)
    relation_target[(product > 0) & meaningful] = 1
    relation_target[(product < 0) & meaningful] = 2

    # At least one future-active endpoint is required; 0-0 pairs do not rank.
    relation_mask = peer_mask & (
        (target_active[:, None] > 0) | (peer_active > 0)
    )
    relation_ce = F.cross_entropy(
        relation_logits.reshape(-1, 3), relation_target.reshape(-1), reduction="none"
    ).reshape_as(relation_target)
    relation = (relation_ce * relation_mask.float()).sum() / relation_mask.float().sum().clamp_min(1.0)
    return activity + relation, {"activity": activity, "relation": relation}


def train_joint_exposure_demand_h3(
    model, tr_ld, va_ld, epochs=60, lr=1e-3, nZ=8,
    beta_tail=0.5, lambda_under=0.0, lambda_over=0.0, lambda_z_reg=1.0,
    lambda_exposure=0.50, patience=6, lambda_graph_aux=0.10,
):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_val, best_sd, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        sums = {"total": 0.0, "demand": 0.0, "exposure": 0.0, "graph": 0.0}

        for b in tr_ld:
            b = _move_batch_to_device(b, DEVICE)
            true_exp = torch.stack([
                b["future_total_dph"],
                b["future_buy_box_dph"],
                b["future_instock"],
            ], dim=-1)
            out = model(
                b["x"], b["future_context"], nZ=nZ,
                peer_x=b.get("peer_x"), peer_mask=b.get("peer_mask"),
                peer_static=b.get("peer_static"),
            )

            demand_nll = sum(
                tail_weighted_negbin_nll(b["y"], mu, alpha, beta_tail=beta_tail)
                for mu, alpha in out["demand_preds"]
            ) / nZ
            demand_loss = demand_nll + lambda_z_reg * out["z_reg"]

            # Optional asymmetric bias losses. Defaults are both zero, so the
            # model is trained without an explicit under/over-forecast preference.
            if lambda_under > 0.0 or lambda_over > 0.0:
                mu_mean = torch.stack(
                    [m for m, _ in out["demand_preds"]], dim=1
                ).mean(dim=1)
                if lambda_under > 0.0:
                    demand_loss = demand_loss + lambda_under * active_underforecast_loss(
                        b["y"], mu_mean, log_scale=True
                    )
                if lambda_over > 0.0:
                    demand_loss = demand_loss + lambda_over * active_overforecast_loss(
                        b["y"], mu_mean, log_scale=True
                    )

            exp_loss = joint_exposure_loss(
                out["exposure_hat"], out["exposure_active_logits"], true_exp
            )
            graph_loss, _ = dynamic_signed_graph_aux_loss(b, out)
            loss = demand_loss + lambda_exposure * exp_loss + lambda_graph_aux * graph_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            sums["total"] += float(loss.item())
            sums["demand"] += float(demand_loss.item())
            sums["exposure"] += float(exp_loss.item())
            sums["graph"] += float(graph_loss.item())

        sch.step()

        model.eval()
        val_total = 0.0
        val_demand = 0.0
        val_exp = 0.0
        with torch.no_grad():
            for b in va_ld:
                b = _move_batch_to_device(b, DEVICE)
                true_exp = torch.stack([
                    b["future_total_dph"], b["future_buy_box_dph"], b["future_instock"]
                ], dim=-1)
                out = model(
                    b["x"], b["future_context"], nZ=min(8, nZ),
                    peer_x=b.get("peer_x"), peer_mask=b.get("peer_mask"),
                    peer_static=b.get("peer_static"),
                )
                mu_mean = torch.stack([m for m, _ in out["demand_preds"]], dim=1).mean(dim=1)
                al_mean = torch.stack([a for _, a in out["demand_preds"]], dim=1).mean(dim=1)
                dloss = tail_weighted_negbin_nll(
                    b["y"], mu_mean, al_mean, beta_tail=beta_tail
                )
                eloss = joint_exposure_loss(
                    out["exposure_hat"], out["exposure_active_logits"], true_exp
                )
                val_demand += float(dloss.item())
                val_exp += float(eloss.item())
                graph_loss, _ = dynamic_signed_graph_aux_loss(b, out)
                val_total += float((dloss + lambda_exposure * eloss + lambda_graph_aux * graph_loss).item())

        ntr = max(1, len(tr_ld)); nva = max(1, len(va_ld))
        val_total /= nva; val_demand /= nva; val_exp /= nva
        improved = val_total < best_val
        if improved:
            best_val = val_total
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"Epoch {epoch+1:3d} | total={sums['total']/ntr:.4f} | "
            f"demand={sums['demand']/ntr:.4f} | exposure={sums['exposure']/ntr:.4f} | "
            f"graph={sums['graph']/ntr:.4f} | "
            f"val_total={val_total:.4f} | val_demand={val_demand:.4f} | "
            f"val_exposure={val_exp:.4f}" + (" *" if improved else "")
        )
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    print(f"Best joint val: {best_val:.4f}")


def generate_joint_forecast_df(model, va_ld, M=100):
    rows = []
    model = model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = _move_batch_to_device(b, DEVICE)
            p50, p70, p90, exp_hat = model.predict(
                b["x"], b["future_context"], M=M,
                peer_x=b.get("peer_x"), peer_mask=b.get("peer_mask"),
                peer_static=b.get("peer_static"),
            )
            for i in range(b["y"].shape[0]):
                for h in range(b["y"].shape[1]):
                    rows.append({
                        "asin": b["asin"][i],
                        "order_week": pd.to_datetime(b["target_week"][h][i]),
                        "fcst_week_index": h + 1,
                        "fbi_demand": b["y"][i, h].item(),
                        "our_price": b["our_price"][i, h].item(),
                        "true_amt": b["y"][i, h].item() * b["our_price"][i, h].item(),
                        "pkg_volume": b["pkg_volume"][i, h].item(),
                        "true_size": b["y"][i, h].item() * b["pkg_volume"][i, h].item(),
                        "true_future_total_dph": b["future_total_dph"][i, h].item(),
                        "true_future_buy_box_dph": b["future_buy_box_dph"][i, h].item(),
                        "true_future_instock": b["future_instock"][i, h].item(),
                        "pred_total_dph_hat": exp_hat[i, h, 0].item(),
                        "pred_buy_box_dph_hat": exp_hat[i, h, 1].item(),
                        "pred_instock_dph_hat": exp_hat[i, h, 2].item(),
                        "scot_oos": b["oos"][i, h].item(),
                        "oos": b["oos"][i, h].item(),
                        "oos_status": b["oos"][i, h].item(),
                        "p50_amxl": p50[i, h].item(),
                        "p70_amxl": p70[i, h].item(),
                        "p90_amxl": p90[i, h].item(),
                    })
    return pd.DataFrame(rows)


def compute_dynamic_graph_diagnostics(model, va_ld):
    """Summarize learned edge types and the zero-zero masking behavior."""
    if getattr(model, "graph_variant", "legacy") != "dynamic_signed":
        return {}
    totals = {
        "graph_candidate_edges": 0.0, "graph_rank_eligible_edges": 0.0,
        "graph_positive_prob": 0.0, "graph_competitive_prob": 0.0,
        "graph_neutral_prob": 0.0, "graph_edge_entropy": 0.0,
    }
    model.eval()
    with torch.no_grad():
        for b in va_ld:
            b = _move_batch_to_device(b, DEVICE)
            out = model(
                b["x"], b["future_context"], nZ=1,
                peer_x=b["peer_x"], peer_mask=b["peer_mask"],
                peer_static=b["peer_static"],
            )
            prob = torch.softmax(out["relation_logits"], dim=-1)
            mask = b["peer_mask"] > 0
            target_active = b["y"].sum(dim=-1) > 0
            peer_active = b["peer_y"].sum(dim=-1) > 0
            rank_mask = mask & (target_active[:, None] | peer_active)
            n = mask.float().sum()
            totals["graph_candidate_edges"] += float(n.item())
            totals["graph_rank_eligible_edges"] += float(rank_mask.float().sum().item())
            if n.item() > 0:
                totals["graph_neutral_prob"] += float((prob[..., 0] * mask).sum().item())
                totals["graph_positive_prob"] += float((prob[..., 1] * mask).sum().item())
                totals["graph_competitive_prob"] += float((prob[..., 2] * mask).sum().item())
                entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
                totals["graph_edge_entropy"] += float((entropy * mask).sum().item())
    denom = max(totals["graph_candidate_edges"], 1.0)
    for key in ["graph_neutral_prob", "graph_positive_prob", "graph_competitive_prob", "graph_edge_entropy"]:
        totals[key] /= denom
    totals["graph_rank_eligible_rate"] = totals["graph_rank_eligible_edges"] / denom
    return totals


def build_prediction_export_df(forecast_df):
    """
    Build the single user-facing prediction.csv for one rolling cut.

    The model forecasts the demand distribution directly. Therefore, the
    exported demand predictions are P50, P70, and P90; there is intentionally
    no separate ``predicted_demand`` column.

    This function only formats the saved CSV. It does not change training,
    inference, or any WAPE calculation.
    """
    df = forecast_df.copy()

    rename_map = {
        "fbi_demand": "actual_demand",
        "p50_amxl": "demand_p50",
        "p70_amxl": "demand_p70",
        "p90_amxl": "demand_p90",
        "true_future_total_dph": "actual_total_dph",
        "true_future_buy_box_dph": "actual_buy_box_dph",
        "true_future_instock": "actual_instock_dph",
        "pred_total_dph_hat": "predicted_total_dph",
        "pred_buy_box_dph_hat": "predicted_buy_box_dph",
        "pred_instock_dph_hat": "predicted_instock_dph",
    }
    df = df.rename(columns=rename_map)

    preferred_columns = [
        "asin",
        "order_week",
        "fcst_week_index",
        "data_cut",
        "scot_fcd",
        "eval_cut",
        "actual_demand",
        "demand_p50",
        "demand_p70",
        "demand_p90",
        "actual_total_dph",
        "predicted_total_dph",
        "actual_buy_box_dph",
        "predicted_buy_box_dph",
        "actual_instock_dph",
        "predicted_instock_dph",
        "oos_status",
        "our_price",
        "pkg_volume",
    ]
    first = [c for c in preferred_columns if c in df.columns]
    rest = [c for c in df.columns if c not in first]
    return df[first + rest]


def restore_internal_prediction_columns(prediction_df):
    """
    Restore internal column names when resume_existing loads prediction.csv.

    The existing WAPE code continues to receive the same internal columns it
    used before this export-only cleanup.
    """
    reverse_map = {
        "actual_demand": "fbi_demand",
        "demand_p50": "p50_amxl",
        "demand_p70": "p70_amxl",
        "demand_p90": "p90_amxl",
        "actual_total_dph": "true_future_total_dph",
        "actual_buy_box_dph": "true_future_buy_box_dph",
        "actual_instock_dph": "true_future_instock",
        "predicted_total_dph": "pred_total_dph_hat",
        "predicted_buy_box_dph": "pred_buy_box_dph_hat",
        "predicted_instock_dph": "pred_instock_dph_hat",
    }
    return prediction_df.rename(columns=reverse_map)


def save_prediction_csv(forecast_df, prediction_path):
    """Save exactly one prediction.csv for a completed rolling cut."""
    prediction_path = Path(prediction_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    export_df = build_prediction_export_df(forecast_df)
    export_df.to_csv(prediction_path, index=False)
    print(f"Saved prediction CSV: {prediction_path}", flush=True)
    return export_df


def _joint_exposure_metrics(df):
    out = []
    for name, true_col, pred_col in [
        ("total", "true_future_total_dph", "pred_total_dph_hat"),
        ("buy_box", "true_future_buy_box_dph", "pred_buy_box_dph_hat"),
        ("in_stock", "true_future_instock", "pred_instock_dph_hat"),
    ]:
        y = df[true_col].to_numpy(float)
        p = df[pred_col].to_numpy(float)
        denom = np.abs(y).sum() + 1e-8
        out.append({
            "channel": name,
            "wape": np.abs(p-y).sum()/denom,
            "ratio": p.sum()/(y.sum()+1e-8),
            "corr": np.corrcoef(y, p)[0,1] if np.std(y)>1e-8 and np.std(p)>1e-8 else np.nan,
            "true_mean": y.mean(),
            "pred_mean": p.mean(),
        })
    return pd.DataFrame(out)


def run_joint_exposure_demand_h3_end2end(
    data_raw1,
    scot_df=None,
    n_asins=5000,
    seed=42,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    beta_tail=0.5,
    patience=6,
    lambda_under=0.0,
    lambda_over=0.0,
    lambda_exposure=0.50,
    detach_exposure_for_demand=False,
    remove_extreme=True,
    extreme_q=0.99,
    output_csv="joint_exposure_demand_h3_forecast_two_step_wape_aligned.csv",
    remove_oos_dp=True,
    dph_cap_end_week=None,
    graph_variant="legacy",
    dynamic_graph_top_k=10,
    dynamic_graph_min_similarity=0.55,
    lambda_graph_aux=0.10,
):
    """Run non-rolling H3 joint model on one snapshot.

    V1 keeps the existing demand preprocessing and SCOT cohort logic, but it no
    longer reads an exposure CSV. Future exposure is predicted internally.
    """
    if horizon != 3:
        raise ValueError("Use horizon=3 for this file.")

    # Reset all local RNGs so legacy-vs-dynamic comparison changes only Graph.
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    print("=" * 88)
    print("JOINT EXPOSURE + DEMAND H3 END-TO-END V1.2 | TWO-STEP WAPE-ALIGNED")
    print("Shared encoder: YES | Separate decoders: YES")
    print(f"Demand gradient through exposure_hat: {not detach_exposure_for_demand}")
    print(
        f"Graph variant: {graph_variant} | dynamic max-K: {dynamic_graph_top_k} | "
        f"minimum static similarity: {dynamic_graph_min_similarity} | DEVICE: {DEVICE}"
    )
    print("=" * 88)

    if scot_df is None:
        raise ValueError(
            "scot_df is required: WAPE is reported exclusively via the "
            "boss-approved calculate_wape_using_lp_oos2/quick_error_check "
            "pipeline, which needs real SCOT p50/p70 forecasts to compare "
            "against. Pass scot_df instead of running without it."
        )

    raw = data_raw1.copy()
    # Reserve the last three future-context columns without reading an external CSV.
    # NOTE: load_real_data's external-exposure attach step requires these exact
    # _log-suffixed names (see `required` in attach_external_exposure3_to_raw_data's
    # override) to append the 3 dedicated hat slots. Using _dph here silently skips
    # that step, and JointExposureDemandH3 then treats whatever real feature ends up
    # last in context_cols (e.g. a graph-peer column) as the reserved hat slot instead.
    for c in ["attn_pred_total_log", "attn_pred_buy_box_log", "attn_pred_instock_log"]:
        raw[c] = 0.0

    data_small_raw, sample_asin_df, intersect_asin_df = prepare_data_from_sample_scot_intersection(
        data_raw1=raw, scot_df=scot_df, n_asins=n_asins, seed=seed
    )

    data_small, asin_stats = add_zero_rate_group(data_small_raw, (0.4, 0.7))
    # Match the two-step H3 demand model exactly:
    # zero_group is diagnostic only; do not filter to high_sparse.
    data_use = data_small.copy()
    print("\n" + "=" * 80)
    print("JOINT COHORT MATCHED TO TWO-STEP H3")
    print("=" * 80)
    print("Sampled ASINs:", len(sample_asin_df))
    print("ASINs after SCOT intersection:", len(intersect_asin_df))
    print("ASINs before extreme filtering:", data_use["asin"].nunique())
    print("Sparse groups are diagnostics only; no high_sparse-only filtering.", flush=True)

    _stage_t0 = time.time()
    print(
        f"[STAGE] extreme filtering START | rows={len(data_use):,} | "
        f"asins={data_use['asin'].nunique():,} | remove_extreme={remove_extreme} | q={extreme_q}",
        flush=True,
    )
    if remove_extreme:
        data_use, removed_extreme, extreme_cap = filter_extreme_asins(data_use, q=extreme_q)
    else:
        removed_extreme, extreme_cap = pd.DataFrame(), np.nan
    print(
        f"[STAGE] extreme filtering DONE | rows={len(data_use):,} | "
        f"asins={data_use['asin'].nunique():,} | removed={len(removed_extreme):,} | "
        f"cap={extreme_cap} | elapsed={time.time() - _stage_t0:.1f}s",
        flush=True,
    )

    _stage_t0 = time.time()
    print(
        f"[STAGE] load_real_data CALL | rows={len(data_use):,} | "
        f"asins={data_use['asin'].nunique():,}",
        flush=True,
    )
    data, context_dim, context_cols = load_real_data(
        data_use,
        dph_cap_q=0.995,
        dph_cap_end_week=dph_cap_end_week,
    )
    print(
        f"[STAGE] load_real_data RETURNED | built_asins={len(data):,} | "
        f"context_dim={context_dim} | elapsed={time.time() - _stage_t0:.1f}s",
        flush=True,
    )
    if context_cols[-3:] != EXTERNAL_HAT_COLS:
        raise RuntimeError(
            "load_real_data did not append the reserved exposure-hat "
            f"placeholder columns as the last 3 context columns "
            f"(got {context_cols[-3:]}, expected {EXTERNAL_HAT_COLS}). "
            "JointExposureDemandH3 assumes context_dim - 3 == base context "
            "and would silently overwrite whatever real feature ended up "
            "last instead. Check the placeholder column names set above "
            "match the _log-suffixed names load_real_data requires."
        )
    _ds_t0 = time.perf_counter()
    print("[PROFILE-DATASET] train lazy dataset START", flush=True)
    tr_ds = DemandDataset(
        data, history, horizon, "train", horizon, num_build_workers=4,
        graph_variant=graph_variant, dynamic_graph_top_k=dynamic_graph_top_k,
        dynamic_graph_min_similarity=dynamic_graph_min_similarity,
    )
    print(
        f"[PROFILE-DATASET] train lazy dataset DONE | samples={len(tr_ds):,} | elapsed={time.perf_counter()-_ds_t0:.2f}s",
        flush=True,
    )

    _ds_t0 = time.perf_counter()
    print("[PROFILE-DATASET] validation lazy dataset START", flush=True)
    va_ds = DemandDataset(
        data, history, horizon, "val", horizon, num_build_workers=4,
        graph_variant=graph_variant, dynamic_graph_top_k=dynamic_graph_top_k,
        dynamic_graph_min_similarity=dynamic_graph_min_similarity,
    )
    print(
        f"[PROFILE-DATASET] validation lazy dataset DONE | samples={len(va_ds):,} | elapsed={time.perf_counter()-_ds_t0:.2f}s",
        flush=True,
    )

    _loader_t0 = time.perf_counter()
    print("[PROFILE-LOADER] DataLoader creation START", flush=True)
    tr_ld = make_demand_dataloader(tr_ds, batch_size, True, seed=seed, num_workers=0, name="train")
    va_ld = make_demand_dataloader(va_ds, batch_size, False, seed=seed + 1, num_workers=0, name="val")
    print(
        f"[PROFILE-LOADER] DataLoader creation DONE | elapsed={time.perf_counter()-_loader_t0:.2f}s",
        flush=True,
    )
    if len(tr_ds) == 0 or len(va_ds) == 0:
        raise RuntimeError("No train/validation samples. Check history/horizon and data length.")

    input_dim = int(tr_ds[0]["x"].shape[-1])
    print(f"ASINs used: {len(data):,} | Train={len(tr_ds):,} | Val={len(va_ds):,}")
    print(f"input_dim={input_dim} | context_dim={context_dim} | H={horizon}")

    model = JointExposureDemandH3(
        input_dim=input_dim,
        context_dim=context_dim,
        d_model=d_model,
        d_z=d_z,
        horizon=horizon,
        prior_scale=0.3,
        detach_exposure_for_demand=detach_exposure_for_demand,
        graph_variant=graph_variant,
    ).to(DEVICE)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training on: {DEVICE}")

    train_joint_exposure_demand_h3(
        model, tr_ld, va_ld,
        epochs=epochs, lr=1e-3, nZ=8,
        beta_tail=beta_tail,
        lambda_under=lambda_under,
        lambda_over=lambda_over,
        lambda_z_reg=1.0,
        lambda_exposure=lambda_exposure,
        lambda_graph_aux=lambda_graph_aux,
        patience=patience,
    )

    forecast_df = generate_joint_forecast_df(model, va_ld, M=M_eval)
    forecast_df["zero_group_run"] = "all_sample_scot_intersection"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_csv, index=False)

    exposure_metrics = _joint_exposure_metrics(forecast_df)
    exposure_diag = compute_exposure_hat_diagnostics(forecast_df)
    graph_diagnostics = compute_dynamic_graph_diagnostics(model, va_ld)

    print("\nJOINT EXPOSURE METRICS (overall, legacy summary)")
    print(exposure_metrics.round(4).to_string(index=False))
    print_exposure_hat_diagnostics(
        exposure_diag,
        title="EXPOSURE HAT DIAGNOSTICS | NON-ROLLING",
    )
    print(f"Saved joint forecast: {output_csv}")

    result = {
        "model": model,
        "forecast_df": forecast_df,
        "exposure_metrics": exposure_metrics,
        "exposure_hat_diagnostics": exposure_diag,
        "tr_ld": tr_ld,
        "va_ld": va_ld,
        "context_cols": context_cols,
        "sample_asin_df": sample_asin_df,
        "intersect_asin_df": intersect_asin_df,
        "asin_stats": asin_stats,
        "removed_extreme": removed_extreme,
        "extreme_cap": extreme_cap,
        "graph_variant": graph_variant,
        "graph_diagnostics": graph_diagnostics,
    }

    # Demand WAPE is reported exclusively through the approved
    # calculate_wape_using_lp_oos2 / quick_error_check / weekly_error_check
    # pipeline below -- no locally rolled abs(pred-true)/true formula.
    # Cohort and evaluation order match the standalone two-step H3 model:
    # sample n_asins -> SCOT intersection -> all sparse groups -> same extreme
    # filter -> H1-H3 validation rows -> same OOS-DP filter -> same real-SCOT
    # join -> same calculate_wape_using_lp_oos2 / quick_error_check WAPE.
    result["real_scot_outputs"] = _evaluate_standard_wape_against_scot(
        result=result,
        scot_df=scot_df,
        data_raw1=data_raw1,
        asin_stats=asin_stats,
        remove_oos_dp=remove_oos_dp,
        source="lp",
    )
    result["final_wape"] = result["real_scot_outputs"]

    return result



# =============================================================================
# Exposure Forecast Diagnostics
#
# Summarize exposure accuracy overall, by forecast horizon, and by rolling cut.
# =============================================================================

def _safe_exposure_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) <= 1e-8 or np.std(y_pred) <= 1e-8:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_exposure_active_auc(y_true, score):
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return np.nan

    y_true = np.asarray(y_true, dtype=float)
    score = np.asarray(score, dtype=float)
    active = (y_true > 0).astype(int)

    if len(np.unique(active)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(active, score))
    except Exception:
        return np.nan


def _exposure_metric_row(
    sub,
    channel,
    true_col,
    pred_col,
):
    valid = sub[
        ["asin", true_col, pred_col]
    ].dropna()

    if valid.empty:
        return {
            "channel": channel,
            "n_rows": 0,
            "n_asins": 0,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "pred_true_ratio": np.nan,
            "wape": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "overbias": np.nan,
            "underbias": np.nan,
            "true_zero_rate": np.nan,
            "pred_zero_rate": np.nan,
            "active_auc": np.nan,
        }

    y_true = pd.to_numeric(
        valid[true_col], errors="coerce"
    ).to_numpy(dtype=float)
    y_pred = pd.to_numeric(
        valid[pred_col], errors="coerce"
    ).to_numpy(dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if len(y_true) == 0:
        return {
            "channel": channel,
            "n_rows": 0,
            "n_asins": 0,
            "true_mean": np.nan,
            "pred_mean": np.nan,
            "pred_true_ratio": np.nan,
            "wape": np.nan,
            "mae": np.nan,
            "corr": np.nan,
            "overbias": np.nan,
            "underbias": np.nan,
            "true_zero_rate": np.nan,
            "pred_zero_rate": np.nan,
            "active_auc": np.nan,
        }

    error = y_pred - y_true
    denom = np.abs(y_true).sum() + 1e-8

    return {
        "channel": channel,
        "n_rows": len(y_true),
        "n_asins": valid.loc[finite, "asin"].astype(str).nunique(),
        "true_mean": y_true.mean(),
        "pred_mean": y_pred.mean(),
        "pred_true_ratio": y_pred.sum() / (y_true.sum() + 1e-8),
        "wape": np.abs(error).sum() / denom,
        "mae": np.abs(error).mean(),
        "corr": _safe_exposure_corr(y_true, y_pred),
        "overbias": np.maximum(error, 0.0).sum() / denom,
        "underbias": np.maximum(-error, 0.0).sum() / denom,
        "true_zero_rate": np.mean(y_true <= 0),
        "pred_zero_rate": np.mean(y_pred <= 0),
        "active_auc": _safe_exposure_active_auc(y_true, y_pred),
    }


def compute_exposure_hat_diagnostics(
    forecast_df,
    data_cut=None,
):
    """
    Compute exposure_hat diagnostics for total, buy-box, and in-stock.

    Returns:
      overall_df
      by_horizon_df
      by_cut_horizon_df
    """
    df = forecast_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    channel_specs = [
        (
            "total",
            "true_future_total_dph",
            "pred_total_dph_hat",
        ),
        (
            "buy_box",
            "true_future_buy_box_dph",
            "pred_buy_box_dph_hat",
        ),
        (
            "in_stock",
            "true_future_instock",
            "pred_instock_dph_hat",
        ),
    ]

    required = ["asin", "fcst_week_index"]
    for _, true_col, pred_col in channel_specs:
        required.extend([true_col, pred_col])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing exposure diagnostic columns: {missing}"
        )

    df["asin"] = df["asin"].astype(str)
    df["fcst_week_index"] = pd.to_numeric(
        df["fcst_week_index"], errors="coerce"
    )

    if data_cut is not None:
        df["data_cut"] = pd.Timestamp(data_cut)

    overall_rows = []
    by_h_rows = []
    by_cut_h_rows = []

    for channel, true_col, pred_col in channel_specs:
        overall_rows.append(
            _exposure_metric_row(
                df,
                channel,
                true_col,
                pred_col,
            )
        )

    horizons = sorted(
        int(h)
        for h in df["fcst_week_index"].dropna().unique()
    )

    for horizon in horizons:
        sub_h = df[df["fcst_week_index"] == horizon]

        for channel, true_col, pred_col in channel_specs:
            row = _exposure_metric_row(
                sub_h,
                channel,
                true_col,
                pred_col,
            )
            row["horizon"] = horizon
            by_h_rows.append(row)

    if "data_cut" in df.columns:
        for cut_value, sub_cut in df.groupby(
            "data_cut", dropna=False
        ):
            for horizon in horizons:
                sub_h = sub_cut[
                    sub_cut["fcst_week_index"] == horizon
                ]

                for channel, true_col, pred_col in channel_specs:
                    row = _exposure_metric_row(
                        sub_h,
                        channel,
                        true_col,
                        pred_col,
                    )
                    row["data_cut"] = cut_value
                    row["horizon"] = horizon
                    by_cut_h_rows.append(row)

    return {
        "overall": pd.DataFrame(overall_rows),
        "by_horizon": pd.DataFrame(by_h_rows),
        "by_cut_horizon": pd.DataFrame(by_cut_h_rows),
    }


def print_exposure_hat_diagnostics(
    diagnostics,
    title="EXPOSURE HAT DIAGNOSTICS",
):
    overall_df = diagnostics["overall"]
    by_horizon_df = diagnostics["by_horizon"]

    print("\n" + "=" * 100)
    print(title + " — OVERALL")
    print("=" * 100)
    print(
        overall_df[
            [
                "channel",
                "n_rows",
                "n_asins",
                "true_mean",
                "pred_mean",
                "pred_true_ratio",
                "wape",
                "corr",
                "overbias",
                "underbias",
                "active_auc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 100)
    print(title + " — BY HORIZON")
    print("=" * 100)
    print(
        by_horizon_df[
            [
                "horizon",
                "channel",
                "n_rows",
                "n_asins",
                "true_mean",
                "pred_mean",
                "pred_true_ratio",
                "wape",
                "corr",
                "overbias",
                "underbias",
                "active_auc",
            ]
        ]
        .sort_values(["horizon", "channel"])
        .round(4)
        .to_string(index=False)
    )


# =============================================================================
# Rolling S3 Orchestration
#
# For each snapshot pair:
#   1. read origin, evaluation, and SCOT data;
#   2. construct the joint eligible cohort;
#   3. build features and eager datasets;
#   4. train the joint model;
#   5. generate demand and exposure predictions;
#   6. run standardized SCOT-aligned evaluation;
#   7. save one prediction.csv.
# =============================================================================

import io
import re
from pathlib import Path
import boto3


ROLLING_S3_BUCKET = "amxl-asin-forecast590184089576"
ROLLING_DATA_PREFIX = (
    "amxl-asin-forecast-intern/data_for_model/"
    "df_head_body_add_holiday_"
)
ROLLING_SCOT_PREFIX = "amxl-asin-forecast-intern/scotforecast/"


def _list_s3_keys(bucket, prefix, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key")
            if key:
                keys.append(key)

    return keys



def list_joint_rolling_snapshot_pairs(
    bucket=ROLLING_S3_BUCKET,
    data_prefix=ROLLING_DATA_PREFIX,
    scot_prefix=ROLLING_SCOT_PREFIX,
    s3_client=None,
):
    """
    Build a leakage-controlled rolling backtest pairing.

    For origin snapshot date D (Saturday):
      - SCOT FCD is D + 1 day (Sunday)
      - evaluation snapshot is D + 21 days

    Why D + 21?
      The evaluation snapshot's final observed Sunday is the SCOT H3 week.
      Therefore the existing non-rolling validation logic (last 3 weeks)
      produces exactly:
        H1 = FCD
        H2 = FCD + 7 days
        H3 = FCD + 14 days

    Example:
      origin snapshot: 2025-10-04, max observed week 2025-09-28
      SCOT FCD:        2025-10-05
      eval snapshot:   2025-10-25, max observed week 2025-10-19
      model val H1-H3: 2025-10-05, 2025-10-12, 2025-10-19
    """
    s3_client = s3_client or boto3.client("s3")

    data_keys = _list_s3_keys(bucket, data_prefix, s3_client)
    scot_keys = _list_s3_keys(bucket, scot_prefix, s3_client)

    data_pattern = re.compile(
        r"df_head_body_add_holiday_"
        r"(\d{4}-\d{2}-\d{2})_?ETLM_[vV]3\.csv$"
    )
    scot_pattern = re.compile(
        r"from(\d{4}-\d{2}-\d{2})_20weeks__"
        r"headbody_scot_fcst_(?:no_refresh|refresh)\.parquet$"
    )

    data_by_date = {}
    for key in data_keys:
        match = data_pattern.search(key)
        if match:
            data_by_date[pd.Timestamp(match.group(1)).normalize()] = key

    scot_by_date = {}
    for key in scot_keys:
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
    for origin_cut, origin_key in sorted(data_by_date.items()):
        scot_fcd = origin_cut + pd.Timedelta(days=1)
        eval_cut = origin_cut + pd.Timedelta(days=21)

        rows.append({
            "data_cut": origin_cut,
            "scot_fcd": scot_fcd,
            "eval_cut": eval_cut,
            "data_key": origin_key,
            "eval_data_key": data_by_date.get(eval_cut),
            "scot_key": scot_by_date.get(scot_fcd),
            "has_eval_snapshot": eval_cut in data_by_date,
            "has_scot": scot_fcd in scot_by_date,
        })

    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs["is_complete_pair"] = (
            pairs["has_eval_snapshot"] & pairs["has_scot"]
        )
    else:
        pairs["is_complete_pair"] = pd.Series(dtype=bool)

    print("=" * 96)
    print("ROLLING ORIGIN / EVALUATION SNAPSHOT / SCOT PAIR CHECK")
    print("=" * 96)
    print("Feature snapshots:", len(data_by_date))
    print("SCOT forecast files:", len(scot_by_date))
    print(
        "Complete rolling pairs:",
        int(pairs["is_complete_pair"].sum()) if len(pairs) else 0,
    )
    print(
        "Missing evaluation snapshot:",
        int((~pairs["has_eval_snapshot"]).sum()) if len(pairs) else 0,
    )
    print(
        "Missing SCOT file:",
        int((~pairs["has_scot"]).sum()) if len(pairs) else 0,
    )

    return pairs


def _read_s3_csv(bucket, key, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    _read_t0 = time.perf_counter()
    print(f"[STAGE] parsing CSV bytes START | bytes={len(body):,}", flush=True)
    _df_read = pd.read_csv(io.BytesIO(body))
    print(f"[STAGE] parsing CSV bytes DONE | rows={len(_df_read):,} | cols={len(_df_read.columns)} | {(time.perf_counter()-_read_t0):.1f}s", flush=True)
    return _df_read


def _read_s3_parquet(bucket, key, s3_client=None):
    s3_client = s3_client or boto3.client("s3")
    body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def _extract_wape_metrics(wape_object, prefix):
    """
    Flatten the P50/P70 object returned by the existing two-step evaluator.

    The original quick_error_check may return a Series, dict, DataFrame,
    or scalar depending on the notebook helper implementation.
    """
    output = {}

    if wape_object is None:
        return output

    if isinstance(wape_object, pd.DataFrame):
        if len(wape_object) == 1:
            for key, value in wape_object.iloc[0].items():
                if np.isscalar(value):
                    output[f"{prefix}_{key}"] = value
        else:
            output[f"{prefix}_repr"] = wape_object.to_json(orient="records")
        return output

    if isinstance(wape_object, pd.Series):
        for key, value in wape_object.items():
            if np.isscalar(value):
                output[f"{prefix}_{key}"] = value
        return output

    if isinstance(wape_object, dict):
        for key, value in wape_object.items():
            if np.isscalar(value):
                output[f"{prefix}_{key}"] = value
            else:
                output[f"{prefix}_{key}"] = str(value)
        return output

    if np.isscalar(wape_object):
        output[prefix] = wape_object
    else:
        output[f"{prefix}_repr"] = str(wape_object)

    return output


def _safe_file_token(value):
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def run_joint_h3_s3_rolling_scot_p50_p70(
    n_asins=5000,
    seed=42,
    max_snapshots=None,
    select_latest=False,
    snapshot_stride=1,
    start_date=None,
    end_date=None,
    epochs=60,
    history=52,
    horizon=3,
    d_model=32,
    d_z=16,
    batch_size=64,
    M_eval=100,
    beta_tail=0.5,
    patience=6,
    lambda_under=0.0,
    lambda_over=0.0,
    lambda_exposure=0.50,
    detach_exposure_for_demand=False,
    remove_extreme=True,
    extreme_q=0.99,
    remove_oos_dp=True,
    output_root="joint_true_rolling_h3_original_wape_exposure_diag",
    resume_existing=True,
    continue_on_error=True,
    bucket=ROLLING_S3_BUCKET,
    data_prefix=ROLLING_DATA_PREFIX,
    scot_prefix=ROLLING_SCOT_PREFIX,
    graph_variant="legacy",
    dynamic_graph_top_k=10,
    dynamic_graph_min_similarity=0.55,
    lambda_graph_aux=0.10,
):
    """
    Rolling Joint H3 evaluation.

    For every matched cut:
      1. Read one feature snapshot CSV.
      2. Read the SCOT parquet whose FCD is snapshot date + 1 day.
      3. Intersect origin, evaluation, SCOT, and Chris ASIN cohorts.
      4. Randomly sample n_asins from that fully joined cohort.
      5. Train the unchanged Joint H3 V1.2 model.
      6. Evaluate P50, P70, and P90 with the existing standardized WAPE pipeline.
      7. Save exactly one prediction.csv after each completed cut.

    CSV export does not change model training, inference, or WAPE calculations.
    Model architecture and losses are unchanged from the non-rolling V1.2 file.
    """
    if horizon != 3:
        raise ValueError("This rolling experiment is H3 only.")

    missing_wape_helpers = [
        name
        for name in [
            "calculate_wape_using_lp_oos2",
            "quick_error_check",
            "weekly_error_check",
        ]
        if name not in globals()
    ]
    if missing_wape_helpers:
        raise RuntimeError(
            "The rolling file intentionally does not implement WAPE. "
            "Load the same existing helper functions used by the two-stage "
            "model before running. Missing: "
            + ", ".join(missing_wape_helpers)
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    s3_client = boto3.client("s3")

    pairs = list_joint_rolling_snapshot_pairs(
        bucket=bucket,
        data_prefix=data_prefix,
        scot_prefix=scot_prefix,
        s3_client=s3_client,
    )

    pairs = pairs[pairs["is_complete_pair"]].copy()

    if start_date is not None:
        pairs = pairs[
            pairs["data_cut"] >= pd.Timestamp(start_date).normalize()
        ]
    if end_date is not None:
        pairs = pairs[
            pairs["data_cut"] <= pd.Timestamp(end_date).normalize()
        ]

    pairs = pairs.sort_values("data_cut").iloc[::max(1, int(snapshot_stride))]

    if max_snapshots is not None:
        max_snapshots = int(max_snapshots)
        pairs = (
            pairs.tail(max_snapshots)
            if select_latest
            else pairs.head(max_snapshots)
        )

    pairs = pairs.reset_index(drop=True)

    print("\n" + "=" * 88)
    print("JOINT H3 TRUE ROLLING + ORIGINAL WAPE + EXPOSURE DIAGNOSTICS")
    print("=" * 88)
    print("Cuts selected:", len(pairs))
    print("Random ASIN sample after full joint per cut:", "ALL" if n_asins is None else n_asins)
    print("Seed:", seed)
    print("Output root:", output_root.resolve())
    print("Model unchanged from non-rolling V1.2: YES")
    print("Metrics retained: P50, P70, and P90")
    print(
        f"Graph variant: {graph_variant} | dynamic max-K: {dynamic_graph_top_k} | "
        f"minimum static similarity: {dynamic_graph_min_similarity} | DEVICE: {DEVICE}"
    )
    print(
        "WAPE: calls original calculate_wape_using_lp_oos2 "
        "+ quick_error_check; no local WAPE formula"
    )

    summary_rows = []
    all_forecasts = []
    all_aligned = []
    all_exposure_overall = []
    all_exposure_by_horizon = []
    all_exposure_by_cut_horizon = []

    for run_index, row in pairs.iterrows():
        data_cut = pd.Timestamp(row["data_cut"])
        scot_fcd = pd.Timestamp(row["scot_fcd"])
        cut_token = _safe_file_token(data_cut)

        # SageMaker/JupyterLab-friendly layout: keep all rolling outputs in
        # one existing root directory and distinguish cuts by filename.
        prediction_path = output_root / f"prediction_cut_{cut_token}.csv"

        print("\n" + "#" * 100)
        print(
            f"ROLLING CUT {run_index + 1}/{len(pairs)} | "
            f"data_cut={data_cut.date()} | "
            f"SCOT_FCD={scot_fcd.date()}"
        )
        print("#" * 100)

        try:
            origin_raw = _read_s3_csv(
                bucket,
                row["data_key"],
                s3_client,
            )
            eval_raw = _read_s3_csv(
                bucket,
                row["eval_data_key"],
                s3_client,
            )
            scot_df = _read_s3_parquet(
                bucket,
                row["scot_key"],
                s3_client,
            )

            for frame in [origin_raw, eval_raw, scot_df]:
                frame["asin"] = frame["asin"].astype(str).str.strip()

            origin_raw["order_week"] = pd.to_datetime(
                origin_raw["order_week"], errors="coerce"
            )
            eval_raw["order_week"] = pd.to_datetime(
                eval_raw["order_week"], errors="coerce"
            )
            scot_df["order_week"] = pd.to_datetime(
                scot_df["order_week"], errors="coerce"
            )

            # Build the fully joined eligible cohort FIRST:
            # origin snapshot ∩ evaluation snapshot ∩ SCOT ∩ Chris.
            # Only after that intersection do we apply n_asins sampling.
            origin_set = set(origin_raw["asin"].dropna().unique())
            scot_set = set(scot_df["asin"].dropna().unique())
            eval_set = set(eval_raw["asin"].dropna().unique())
            eligible_joint_asins = sorted(
                origin_set & scot_set & eval_set & CHRIS_ASIN_SET
            )

            print(
                f"[CHRIS-JOINT] eligible before sampling | "
                f"origin={len(origin_set):,} | scot={len(scot_set):,} | "
                f"eval={len(eval_set):,} | chris={len(CHRIS_ASIN_SET):,} | "
                f"joint={len(eligible_joint_asins):,}",
                flush=True,
            )

            if not eligible_joint_asins:
                raise RuntimeError(
                    "No ASINs remain after origin, evaluation, SCOT, and "
                    "Chris intersection."
                )

            if n_asins is None:
                rolling_asins = eligible_joint_asins
            else:
                rng = np.random.default_rng(seed)
                rolling_asins = sorted(
                    rng.choice(
                        np.asarray(eligible_joint_asins, dtype=object),
                        size=min(int(n_asins), len(eligible_joint_asins)),
                        replace=False,
                    ).tolist()
                )

            print(
                f"[CHRIS-JOINT] after n_asins sampling | "
                f"requested={'ALL' if n_asins is None else int(n_asins):} | "
                f"selected={len(rolling_asins):,}",
                flush=True,
            )

            # Store every rolling cut directly under output_root. The cut date
            # is part of the filename, so no per-cut subdirectory is required.
            prediction_path = output_root / f"prediction_cut_{cut_token}.csv"
            print(
                f"[ROLLING-OUTPUT] prediction={prediction_path}",
                flush=True,
            )

            # Use the later evaluation snapshot to supply the actual H1-H3
            # rows, restricted to the already-joined and then sampled cohort.
            data_raw1 = eval_raw[
                eval_raw["asin"].isin(rolling_asins)
            ].copy()

            if data_raw1.empty:
                raise RuntimeError(
                    "No rows remain after joined-cohort sampling."
                )

            expected_h1 = pd.Timestamp(row["scot_fcd"])
            expected_h2 = expected_h1 + pd.Timedelta(days=7)
            expected_h3 = expected_h1 + pd.Timedelta(days=14)

            # Last actually observable week at this rolling forecast origin.
            # Estimate the global DPH cap only from these origin-snapshot rows;
            # the later evaluation snapshot is still used to supply H1-H3 targets.
            origin_cap_rows = origin_raw[
                origin_raw["asin"].isin(rolling_asins)
            ]
            dph_cap_end_week = origin_cap_rows["order_week"].max()
            if pd.isna(dph_cap_end_week):
                raise RuntimeError(
                    "Unable to determine the DPH-cap cutoff week from the "
                    "origin snapshot for the selected rolling cohort."
                )

            print("=" * 96)
            print("ROLLING COHORT + TARGET-WEEK CONSTRUCTION")
            print("=" * 96)
            print("Origin snapshot:", pd.Timestamp(row["data_cut"]).date())
            print("Evaluation snapshot:", pd.Timestamp(row["eval_cut"]).date())
            print("SCOT FCD:", expected_h1.date())
            print(
                "Origin max observed week:",
                origin_raw["order_week"].max(),
            )
            print(
                "DPH cap history cutoff week:",
                pd.Timestamp(dph_cap_end_week),
            )
            print(
                "Evaluation max observed week:",
                eval_raw["order_week"].max(),
            )
            print(
                "Expected H1/H2/H3:",
                [expected_h1, expected_h2, expected_h3],
            )
            print("Joint cohort sampled ASINs:", len(rolling_asins))
            print("SCOT ASINs:", len(scot_set))
            print("Evaluation ASINs:", len(eval_set))
            print("Final rolling ASIN intersection:", len(rolling_asins))
            print(
                f"Evaluation rows used={len(data_raw1):,} | "
                f"ASINs={data_raw1['asin'].nunique():,}"
            )

            if (
                resume_existing
                and prediction_path.exists()
            ):
                print("Existing outputs found; loading instead of retraining.")
                prediction_df = pd.read_csv(
                    prediction_path,
                    parse_dates=["order_week"],
                )
                forecast_df = restore_internal_prediction_columns(
                    prediction_df
                )
                loaded_asins = set(
                    forecast_df["asin"].astype(str).str.strip().dropna().unique()
                )
                if not loaded_asins:
                    raise RuntimeError(
                        "Resume validation failed: "
                        "the cached prediction file contains zero ASINs."
                    )

                # Existing prediction.csv is authoritative for resume. If this
                # rolling cut was completed previously, evaluate exactly the ASINs
                # stored in that file; do not compare them with a newly sampled
                # cohort, which may differ across reruns or parameter settings.
                data_raw1 = eval_raw[
                    eval_raw["asin"].astype(str).str.strip().isin(loaded_asins)
                ].copy()
                scot_df = scot_df[
                    scot_df["asin"].astype(str).str.strip().isin(loaded_asins)
                ].copy()
                rolling_asins = sorted(loaded_asins)

                eval_loaded_asins = set(
                    data_raw1["asin"].astype(str).str.strip().dropna().unique()
                )
                scot_loaded_asins = set(
                    scot_df["asin"].astype(str).str.strip().dropna().unique()
                )
                print(
                    f"[RESUME-CACHE] using existing prediction.csv | "
                    f"prediction_asins={len(loaded_asins):,} | "
                    f"available_in_eval={len(eval_loaded_asins):,} | "
                    f"available_in_scot={len(scot_loaded_asins):,} | "
                    f"missing_eval={len(loaded_asins - eval_loaded_asins):,} | "
                    f"missing_scot={len(loaded_asins - scot_loaded_asins):,}",
                    flush=True,
                )

                # Use the standardized evaluator directly, including on resume.
                result_stub = {
                    "forecast_df": forecast_df,
                }
                real_scot = _evaluate_standard_wape_against_scot(
                    result=result_stub,
                    scot_df=scot_df,
                    data_raw1=data_raw1,
                    asin_stats=None,
                    remove_oos_dp=remove_oos_dp,
                    source="lp",
                )
                aligned_df = real_scot[
                    "forecast_df_scot_real"
                ].copy()

                aligned_df["data_cut"] = data_cut
                aligned_df["scot_fcd"] = scot_fcd
                aligned_df["eval_cut"] = pd.Timestamp(row["eval_cut"])
                aligned_df["data_s3_key"] = row["data_key"]
                aligned_df["scot_s3_key"] = row["scot_key"]
                # Aligned SCOT rows remain in memory for WAPE only.

                exposure_diag = compute_exposure_hat_diagnostics(
                    forecast_df,
                    data_cut=data_cut,
                )
                print_exposure_hat_diagnostics(
                    exposure_diag,
                    title=(
                        "EXPOSURE HAT DIAGNOSTICS | "
                        f"CUT={data_cut.date()} | LOADED"
                    ),
                )



                metrics = {}
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p50_wape"),
                        "p50",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p70_wape"),
                        "p70",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p90_wape"),
                        "p90",
                    )
                )
                metrics["p50_penalty_diff"] = real_scot.get(
                    "p50_penalty_diff"
                )
                metrics["p70_penalty_diff"] = real_scot.get(
                    "p70_penalty_diff"
                )
                metrics["p90_penalty_diff"] = real_scot.get(
                    "p90_penalty_diff"
                )

                result = None
                status = "loaded_existing"

            else:
                result = run_joint_exposure_demand_h3_end2end(
                    data_raw1=data_raw1,
                    scot_df=scot_df,
                    n_asins=len(rolling_asins),
                    seed=seed,
                    epochs=epochs,
                    history=history,
                    horizon=horizon,
                    d_model=d_model,
                    d_z=d_z,
                    batch_size=batch_size,
                    M_eval=M_eval,
                    beta_tail=beta_tail,
                    patience=patience,
                    lambda_under=lambda_under,
                    lambda_over=lambda_over,
                    lambda_exposure=lambda_exposure,
                    detach_exposure_for_demand=(
                        detach_exposure_for_demand
                    ),
                    remove_extreme=remove_extreme,
                    extreme_q=extreme_q,
                    output_csv=str(prediction_path),
                    remove_oos_dp=remove_oos_dp,
                    dph_cap_end_week=dph_cap_end_week,
                    graph_variant=graph_variant,
                    dynamic_graph_top_k=dynamic_graph_top_k,
                    dynamic_graph_min_similarity=dynamic_graph_min_similarity,
                    lambda_graph_aux=lambda_graph_aux,
                )

                if "real_scot_outputs" not in result:
                    raise RuntimeError(
                        "SCOT evaluation did not produce real_scot_outputs."
                    )

                real_scot = result["real_scot_outputs"]
                forecast_df = result["forecast_df"].copy()

                joint_weeks = sorted(
                    pd.to_datetime(
                        forecast_df["order_week"], errors="coerce"
                    ).dropna().unique()
                )
                scot_h3_weeks = sorted(
                    pd.to_datetime(
                        scot_df["order_week"], errors="coerce"
                    ).dropna().unique()
                )[:3]
                expected_weeks = [
                    expected_h1.to_datetime64(),
                    expected_h2.to_datetime64(),
                    expected_h3.to_datetime64(),
                ]

                print("\n" + "=" * 96)
                print("ROLLING TARGET WEEK ASSERTION")
                print("=" * 96)
                print("Joint unique weeks:", joint_weeks)
                print("SCOT first 3 weeks:", scot_h3_weeks)
                print("Expected H1-H3:", expected_weeks)

                if {pd.Timestamp(w) for w in joint_weeks} != {pd.Timestamp(w) for w in expected_weeks}:
                    raise RuntimeError(
                        "Joint validation weeks do not equal the rolling "
                        "SCOT H1-H3 weeks. The evaluation snapshot pairing "
                        "or dataset split is incorrect."
                    )

                aligned_df = real_scot[
                    "forecast_df_scot_real"
                ].copy()

                forecast_df["data_cut"] = data_cut
                forecast_df["scot_fcd"] = scot_fcd
                forecast_df["eval_cut"] = pd.Timestamp(row["eval_cut"])
                forecast_df["data_s3_key"] = row["data_key"]
                forecast_df["scot_s3_key"] = row["scot_key"]

                aligned_df["data_cut"] = data_cut
                aligned_df["scot_fcd"] = scot_fcd
                aligned_df["data_s3_key"] = row["data_key"]
                aligned_df["scot_s3_key"] = row["scot_key"]

                save_prediction_csv(forecast_df, prediction_path)

                # Exposure hat diagnostics for this rolling cut.
                exposure_diag = compute_exposure_hat_diagnostics(
                    forecast_df,
                    data_cut=data_cut,
                )
                print_exposure_hat_diagnostics(
                    exposure_diag,
                    title=(
                        "EXPOSURE HAT DIAGNOSTICS | "
                        f"CUT={data_cut.date()}"
                    ),
                )



                # Keep the original two-step P50/P70 outputs in summary.
                metrics = {}
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p50_wape"),
                        "p50",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p70_wape"),
                        "p70",
                    )
                )
                metrics.update(
                    _extract_wape_metrics(
                        real_scot.get("p90_wape"),
                        "p90",
                    )
                )

                metrics["p50_penalty_diff"] = real_scot.get(
                    "p50_penalty_diff"
                )
                metrics["p70_penalty_diff"] = real_scot.get(
                    "p70_penalty_diff"
                )
                metrics["p90_penalty_diff"] = real_scot.get(
                    "p90_penalty_diff"
                )

                status = "trained"

            summary = {
                "data_cut": data_cut,
                "scot_fcd": scot_fcd,
                "status": status,
                "data_s3_key": row["data_key"],
                "scot_s3_key": row["scot_key"],
                "forecast_rows": len(forecast_df),
                "forecast_asins": forecast_df["asin"].nunique(),
                "aligned_rows": len(aligned_df),
                "aligned_asins": aligned_df["asin"].nunique(),
                "aligned_week_min": aligned_df["order_week"].min(),
                "aligned_week_max": aligned_df["order_week"].max(),
                "raw_asin_intersection": (
                    real_scot.get("alignment_debug", {}).get("raw_asin_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "canonical_asin_intersection": (
                    real_scot.get("alignment_debug", {}).get("canonical_asin_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "raw_week_intersection": (
                    real_scot.get("alignment_debug", {}).get("raw_week_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                "sunday_week_intersection": (
                    real_scot.get("alignment_debug", {}).get("sunday_week_intersection")
                    if "real_scot" in locals() else np.nan
                ),
                **{
                    f"exposure_{row['channel']}_{metric}": row[metric]
                    for _, row in exposure_diag["overall"].iterrows()
                    for metric in [
                        "wape",
                        "pred_true_ratio",
                        "corr",
                        "overbias",
                        "underbias",
                        "active_auc",
                    ]
                },
                **metrics,
                "error": "",
                "graph_variant": graph_variant,
                **(
                    result.get("graph_diagnostics", {})
                    if isinstance(result, dict) else {}
                ),
            }

            summary_rows.append(summary)
            all_forecasts.append(forecast_df)
            all_aligned.append(aligned_df)

            exposure_overall_cut = exposure_diag["overall"].copy()
            exposure_overall_cut["data_cut"] = data_cut
            all_exposure_overall.append(exposure_overall_cut)

            exposure_by_h_cut = exposure_diag["by_horizon"].copy()
            exposure_by_h_cut["data_cut"] = data_cut
            all_exposure_by_horizon.append(exposure_by_h_cut)

            if not exposure_diag["by_cut_horizon"].empty:
                all_exposure_by_cut_horizon.append(
                    exposure_diag["by_cut_horizon"].copy()
                )


            print(f"Completed cut output: {prediction_path}", flush=True)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print(
                f"FAILED data_cut={data_cut.date()} | "
                f"{error_message}"
            )

            summary_rows.append({
                "data_cut": data_cut,
                "scot_fcd": scot_fcd,
                "status": "failed",
                "data_s3_key": row["data_key"],
                "scot_s3_key": row["scot_key"],
                "forecast_rows": 0,
                "forecast_asins": 0,
                "aligned_rows": 0,
                "aligned_asins": 0,
                "error": error_message,
            })


            if not continue_on_error:
                raise

    summary_df = pd.DataFrame(summary_rows)

    if all_forecasts:
        full_forecast_df = pd.concat(
            all_forecasts,
            ignore_index=True,
        )
    else:
        full_forecast_df = pd.DataFrame()

    if all_aligned:
        full_aligned_df = pd.concat(
            all_aligned,
            ignore_index=True,
        )
    else:
        full_aligned_df = pd.DataFrame()


    if all_exposure_overall:
        exposure_overall_full = pd.concat(
            all_exposure_overall,
            ignore_index=True,
        )
    else:
        exposure_overall_full = pd.DataFrame()

    if all_exposure_by_horizon:
        exposure_by_horizon_full = pd.concat(
            all_exposure_by_horizon,
            ignore_index=True,
        )
    else:
        exposure_by_horizon_full = pd.DataFrame()

    if all_exposure_by_cut_horizon:
        exposure_by_cut_horizon_full = pd.concat(
            all_exposure_by_cut_horizon,
            ignore_index=True,
        )
    else:
        exposure_by_cut_horizon_full = pd.DataFrame()

    # Diagnostics across all rolling cuts combined.
    if not full_forecast_df.empty:
        combined_exposure_diag = compute_exposure_hat_diagnostics(
            full_forecast_df
        )
        print_exposure_hat_diagnostics(
            combined_exposure_diag,
            title="EXPOSURE HAT DIAGNOSTICS | ALL ROLLING CUTS",
        )
    else:
        combined_exposure_diag = {
            "overall": pd.DataFrame(),
            "by_horizon": pd.DataFrame(),
            "by_cut_horizon": pd.DataFrame(),
        }

    print("\n" + "=" * 88)
    print("ROLLING RUN COMPLETE")
    print("=" * 88)
    print("Successful cuts:", int((summary_df["status"] != "failed").sum()))
    print("Failed cuts:", int((summary_df["status"] == "failed").sum()))

    return {
        "summary": summary_df,
        "forecast_df": full_forecast_df,
        "aligned_scot_df": full_aligned_df,
        "exposure_hat_overall_by_cut": exposure_overall_full,
        "exposure_hat_by_cut_and_horizon": exposure_by_horizon_full,
        "exposure_hat_full_diagnostics": combined_exposure_diag,
        "snapshot_pairs": pairs,
        "output_root": str(output_root),
    }


# ============================================================================
# GRAPH COMPARISON RUNNER
# ============================================================================
def run_dynamic_vs_legacy_graph_comparison(
    output_root="joint_h3_usage_graph_comparison",
    **rolling_kwargs,
):
    """Matched ablation: run the new dynamic Graph first, then legacy Graph."""
    root = Path(output_root)
    common = dict(rolling_kwargs)
    common["resume_existing"] = common.get("resume_existing", True)

    print("\n" + "=" * 100)
    print("GRAPH ABLATION A/2 | NEW ACTIVITY-GATED DYNAMIC SIGNED GRAPH")
    print("=" * 100)
    dynamic = run_joint_h3_s3_rolling_scot_p50_p70(
        **common,
        graph_variant="dynamic_signed",
        output_root=str(root / "dynamic_signed_graph"),
    )

    print("\n" + "=" * 100)
    print("GRAPH ABLATION B/2 | OLD LEGACY PEER-MEAN GRAPH")
    print("=" * 100)
    legacy = run_joint_h3_s3_rolling_scot_p50_p70(
        **common,
        graph_variant="legacy",
        output_root=str(root / "legacy_graph"),
    )

    legacy_summary = legacy["summary"].copy()
    dynamic_summary = dynamic["summary"].copy()
    legacy_summary["graph_variant"] = "legacy"
    dynamic_summary["graph_variant"] = "dynamic_signed"
    comparison = pd.concat([dynamic_summary, legacy_summary], ignore_index=True)
    comparison_path = root / "graph_comparison_summary.csv"
    root.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    print(f"Graph comparison summary saved: {comparison_path}")
    return {
        "comparison": comparison,
        "legacy": legacy,
        "dynamic_signed": dynamic,
        "comparison_path": str(comparison_path),
    }


# Backward-compatible alias for notebooks that already referenced the old name.
run_legacy_vs_dynamic_graph_comparison = run_dynamic_vs_legacy_graph_comparison


# ============================================================================
# USAGE GRAPH COMPARISON: SAME SETTING AS QUICK VALIDATION
# ============================================================================
# This is the requested matched experiment. Both runs use the same first two
# rolling cuts, seed, sampled cohort logic, model dimensions, training epochs,
# WAPE evaluator, and exposure loss. Only graph_variant changes.
#
# graph_comparison = run_dynamic_vs_legacy_graph_comparison(
#     n_asins=5000,
#     seed=42,
#     max_snapshots=2,
#     select_latest=False,
#     epochs=3,
#     patience=2,
#     history=52,
#     horizon=3,
#     d_model=32,
#     d_z=16,
#     batch_size=64,
#     M_eval=100,
#     lambda_exposure=0.50,
#     dynamic_graph_top_k=10,
#     dynamic_graph_min_similarity=0.55,
#     lambda_graph_aux=0.10,
#     detach_exposure_for_demand=False,
#     remove_oos_dp=True,
#     resume_existing=False,  # force all four matched trainings for a clean ablation
#     continue_on_error=False,
#     output_root="joint_h3_usage_graph_comparison_5000",
# )


# ============================================================================
# USAGE 1 OF 3: QUICK VALIDATION ON THE FIRST TWO ROLLING CUTS
# ============================================================================
# Use this small run to verify S3 loading, cohort construction, eager dataset
# creation, training, WAPE evaluation, and per-cut CSV export.
# Files are saved directly under joint_h3_usage1_quick_validation_5000/.
#
# rolling_joint_h3_test = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=5000,
#     seed=42,
#     max_snapshots=2,
#     select_latest=False,
#     epochs=3,
#     patience=2,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_h3_usage1_quick_validation_5000",
#     resume_existing=True,
#     continue_on_error=False,
# )


# ============================================================================
# USAGE 2 OF 3: FULL ROLLING RUN WITH A FIXED SAMPLE SIZE
# ============================================================================
# For each cut, the code first intersects the origin snapshot, evaluation
# snapshot, SCOT cohort, and Chris cohort. It then samples up to n_asins from
# that fully joined eligible cohort.
#
# Each completed cut writes one date-stamped CSV directly under
# joint_h3_usage2_full_rolling_15000/, for example:
#   prediction_cut_2025-10-04.csv
#
# Each file contains:
#   - ASIN, order week, and forecast horizon
#   - actual demand and model demand P50/P70/P90
#   - actual and predicted total/buy-box/in-stock DPH
#   - OOS status, price, and package volume
#
# rolling_joint_h3 = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=15000,
#     seed=42,
#     max_snapshots=None,
#     select_latest=False,
#     epochs=60,
#     patience=6,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_h3_usage2_full_rolling_15000",
#     resume_existing=True,
#     continue_on_error=True,
# )


# ============================================================================
# USAGE 3 OF 3: FULL ROLLING RUN WITH ALL ELIGIBLE ASINS
# ============================================================================
# Setting n_asins=None disables sampling. Every cut uses the complete
# origin/evaluation/SCOT/Chris intersection. Use a separate output_root so
# resume files remain isolated from sampled runs. Date-stamped CSV files
# are saved directly under joint_h3_usage3_full_rolling_all_asins/.
#
# rolling_joint_h3_all_asins = run_joint_h3_s3_rolling_scot_p50_p70(
#     n_asins=None,
#     seed=42,
#     max_snapshots=None,
#     select_latest=False,
#     epochs=60,
#     patience=6,
#     history=52,
#     horizon=3,
#     batch_size=64,
#     lambda_exposure=0.50,
#     detach_exposure_for_demand=False,
#     output_root="joint_h3_usage3_full_rolling_all_asins",
#     resume_existing=True,
#     continue_on_error=True,
# )
