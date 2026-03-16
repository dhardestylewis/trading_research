"""Branch C — Score calibration and monotone ranking quality.

Move from binary trigger to a calibrated score surface that supports
cross-sectional ranking and capital allocation.

Metrics:
  - Isotonic-calibrated reliability diagram
  - Ventile monotonicity (rank-weighted return ordering)
  - Cross-sectional spread (top-K vs bottom-K per bar)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.utils.logging import get_logger

log = get_logger("score_calibration")


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Compute calibration (reliability) curve data.

    Returns DataFrame with columns: bin_midpoint, predicted_prob,
    observed_freq, count per bin.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows: list[dict] = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_midpoint": (bins[b] + bins[b + 1]) / 2,
            "predicted_prob": y_prob[mask].mean(),
            "observed_freq": y_true[mask].mean(),
            "count": int(mask.sum()),
        })

    return pd.DataFrame(rows)


def isotonic_calibrate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_test: np.ndarray | None = None,
) -> tuple[np.ndarray, IsotonicRegression]:
    """Fit isotonic regression and return calibrated probabilities.

    If y_prob_test is provided, calibrate those instead (train/test split).
    """
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    target = y_prob_test if y_prob_test is not None else y_prob
    calibrated = iso.predict(target)
    return calibrated, iso


def monotone_rank_quality(
    preds: pd.DataFrame,
    n_bins: int = 20,
    prob_col: str = "y_pred_prob",
    ret_col: str = "fwd_ret_1h",
) -> pd.DataFrame:
    """Measure rank-weighted return by ventile.

    Sort predictions into ventile bins, compute mean realized return per
    bin.  A well-calibrated model should show monotone ordering: higher
    predicted probability → higher realized return.

    Returns DataFrame with columns: ventile, mean_prob, mean_return,
    count, cum_return.
    """
    df = preds.copy()
    df["ventile"] = pd.qcut(df[prob_col], n_bins, labels=False, duplicates="drop")

    rows: list[dict] = []
    for v, grp in df.groupby("ventile"):
        gross = grp[ret_col].values
        rows.append({
            "ventile": int(v),
            "mean_prob": grp[prob_col].mean(),
            "mean_return": gross.mean(),
            "mean_return_bps": gross.mean() * 10_000,
            "count": len(grp),
            "hit_rate": (gross > 0).mean(),
        })

    result = pd.DataFrame(rows).sort_values("ventile")

    # Compute monotonicity score (Spearman correlation between ventile and mean return)
    if len(result) >= 3:
        from scipy.stats import spearmanr
        corr, pval = spearmanr(result["ventile"], result["mean_return"])
        log.info("  Monotonicity (Spearman): %.3f (p=%.4f)", corr, pval)

    return result


def cross_sectional_ranking(
    preds: pd.DataFrame,
    top_k: int = 3,
    prob_col: str = "y_pred_prob",
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
) -> pd.DataFrame:
    """Cross-sectional ranking: go long top-K, flat otherwise, per bar.

    Groups predictions by timestamp, ranks assets, and measures the
    spread between top-K and bottom-K basket returns.

    Returns DataFrame with columns: timestamp, top_k_return, bottom_k_return,
    spread, n_assets.
    """
    cost = cost_bps / 10_000.0
    rows: list[dict] = []

    for ts, grp in preds.groupby("timestamp"):
        if len(grp) < 2 * top_k:
            continue

        ranked = grp.sort_values(prob_col, ascending=False)
        top = ranked.head(top_k)
        bottom = ranked.tail(top_k)

        top_ret = top[ret_col].mean() - 2 * cost
        bottom_ret = bottom[ret_col].mean() - 2 * cost
        spread = top_ret - bottom_ret

        rows.append({
            "timestamp": ts,
            "top_k_return": top_ret,
            "bottom_k_return": bottom_ret,
            "spread": spread,
            "spread_bps": spread * 10_000,
            "n_assets": len(grp),
        })

    return pd.DataFrame(rows)


def calibration_study(
    preds: pd.DataFrame,
    *,
    n_bins: int = 20,
    top_k: int = 3,
    prob_col: str = "y_pred_prob",
    ret_col: str = "fwd_ret_1h",
    cost_bps: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full score-calibration study.

    Returns
    -------
    calibration_curve : reliability diagram data
    rank_quality : ventile monotonicity table
    cs_ranking : cross-sectional spread per bar
    """
    log.info("═══ Branch C: Score calibration ═══")

    y_true = preds["y_true"].values if "y_true" in preds.columns else (preds[ret_col] > 0).astype(float).values
    y_prob = preds[prob_col].values

    # 1. Calibration curve (raw + isotonic)
    cal_raw = compute_calibration_curve(y_true, y_prob, n_bins=n_bins)
    cal_raw["variant"] = "raw"

    calibrated, _ = isotonic_calibrate(y_true, y_prob)
    cal_iso = compute_calibration_curve(y_true, calibrated, n_bins=n_bins)
    cal_iso["variant"] = "isotonic"

    calibration_curve = pd.concat([cal_raw, cal_iso], ignore_index=True)
    log.info("  Calibration curve: %d bins × 2 variants", n_bins)

    # ECE (Expected Calibration Error)
    for variant in ["raw", "isotonic"]:
        v_df = calibration_curve[calibration_curve["variant"] == variant]
        if not v_df.empty:
            weights = v_df["count"].values / v_df["count"].sum()
            ece = np.sum(weights * np.abs(v_df["predicted_prob"].values - v_df["observed_freq"].values))
            log.info("  ECE (%s): %.4f", variant, ece)

    # 2. Monotone rank quality
    rank_quality = monotone_rank_quality(preds, n_bins=n_bins, prob_col=prob_col, ret_col=ret_col)
    log.info("  Rank quality: %d ventiles", len(rank_quality))

    # Check top-vs-bottom spread
    if len(rank_quality) >= 2:
        top_v = rank_quality.nlargest(1, "ventile")["mean_return_bps"].values[0]
        bottom_v = rank_quality.nsmallest(1, "ventile")["mean_return_bps"].values[0]
        log.info("  Top-bottom ventile spread: %.1f bps", top_v - bottom_v)

    # 3. Cross-sectional ranking
    cs_ranking = cross_sectional_ranking(
        preds, top_k=top_k, prob_col=prob_col, ret_col=ret_col, cost_bps=cost_bps,
    )
    if not cs_ranking.empty:
        log.info("  Cross-sectional: %d bars, mean spread=%.1f bps",
                 len(cs_ranking), cs_ranking["spread_bps"].mean())

    return calibration_curve, rank_quality, cs_ranking
