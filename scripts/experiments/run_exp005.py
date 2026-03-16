"""exp005 experiment runner — Live fill validation.

Objective: Validate whether the production-candidate policy survives
queue/priority penalty stress testing before paper deployment.

Single policy: pooled-train SOL-deploy + sep_3bar_t0.55 + NOT_rebound

Entry modes:
  1. Marketable near-open entry
  2. Passive limit at open −5 bps
  3. Passive limit at open −10 bps

Usage:
    python run_exp005.py
    python run_exp005.py configs/experiments/crypto_1h_exp005.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.download_crypto import run as download_data
from src.data.build_panel import build as build_panel
from src.features.build_features import build as build_features
from src.labels.build_labels import build as build_labels
from src.validation.fold_builder import build_folds
from src.diagnostics.regime_labeller import label_regimes
from src.diagnostics.pooled_vs_solo import pooled_vs_solo_comparison
from src.diagnostics.regime_gated_policy import regime_gated_policy_study
from src.diagnostics.passive_entry import passive_entry_study
from src.diagnostics.queue_priority_penalty import queue_penalty_study
from src.diagnostics.live_fill_validation import run_fill_validation_report
from src.reporting.exp005_report import (
    build_exp005_summary,
    plot_penalty_heatmap,
    plot_entry_mode_comparison,
)
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp005")


def _load_features_with_identifiers(data_cfg_path: str) -> pd.DataFrame:
    """Load features parquet and merge back asset + timestamp from the panel."""
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    feat_path = Path("data/processed/features/features.parquet")

    panel = load_parquet(panel_path)
    features = load_parquet(feat_path)

    features["asset"] = panel["asset"].values
    features["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)
    return features


def _get_feature_cols(features_df: pd.DataFrame) -> list[str]:
    """Return numeric feature column names (exclude identifiers)."""
    exclude = {"asset", "timestamp"}
    return [c for c in features_df.columns if c not in exclude]


def _build_entry_mode_trades(
    sol_panel: pd.DataFrame,
    sol_preds: pd.DataFrame,
    entry_mode: dict,
    policy_cfg: dict,
) -> pd.DataFrame:
    """Build trade-level DataFrame for a single entry mode.

    Uses the passive entry study for limit orders and direct
    fill simulation for market orders.
    """
    threshold = policy_cfg.get("threshold", 0.55)
    cost_bps = policy_cfg.get("cost_bps", 15.0)

    entry_type = entry_mode.get("type", "market")
    offset_bps = entry_mode.get("offset_bps", 0.0)

    if entry_type == "passive_limit":
        # Use passive entry study to get filled trades
        passive_df = passive_entry_study(
            sol_panel, sol_preds,
            limit_offsets_bps=[offset_bps],
            threshold=threshold,
            cost_bps=cost_bps,
        )
        # Build per-trade returns from the passive model
        trades = _simulate_passive_trades(
            sol_panel, sol_preds, offset_bps, threshold, cost_bps,
        )
        return trades
    else:
        # Marketable near-open: use close_to_next_open fill
        trades = _simulate_market_trades(
            sol_panel, sol_preds, threshold, cost_bps,
        )
        return trades


def _simulate_passive_trades(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    offset_bps: float,
    threshold: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Simulate passive limit order fills.

    Entry: limit order at next bar's open - offset_bps.
    Fill condition: next bar's low ≤ limit price (touch model).
    Exit: next bar's close.
    """
    merged = preds.merge(
        panel[["asset", "timestamp", "open", "low", "close"]],
        on=["asset", "timestamp"],
        how="left",
    )

    # We need the NEXT bar's OHLC for entry simulation
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["next_open"] = merged["open"].shift(-1)
    merged["next_low"] = merged["low"].shift(-1)
    merged["next_close"] = merged["close"].shift(-1)

    active = merged[merged["y_pred_prob"] > threshold].copy()
    active = active.dropna(subset=["next_open", "next_low", "next_close"])
    if active.empty:
        return pd.DataFrame()

    # Limit price = next bar's open - offset
    active["limit_price"] = active["next_open"] * (1 - offset_bps / 10_000)

    # Touch model: filled if next bar's low ≤ limit_price
    active["filled"] = active["next_low"] <= active["limit_price"]

    # Forward return for filled trades
    filled = active[active["filled"]].copy()
    if filled.empty:
        return pd.DataFrame()

    cost_frac = cost_bps / 10_000
    filled["fill_price"] = filled["limit_price"]
    # Return = (next_close - fill_price) / fill_price - cost
    filled["gross_return"] = (filled["next_close"] - filled["fill_price"]) / filled["fill_price"]
    filled["net_return"] = filled["gross_return"] - 2 * cost_frac

    keep_cols = ["asset", "timestamp", "y_pred_prob", "fill_price",
                 "gross_return", "net_return"]
    if "fold_id" in filled.columns:
        keep_cols.append("fold_id")

    return filled[[c for c in keep_cols if c in filled.columns]]


def _simulate_market_trades(
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    threshold: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Simulate marketable near-open entry."""
    merged = preds.merge(
        panel[["asset", "timestamp", "open", "close"]],
        on=["asset", "timestamp"],
        how="left",
    )

    # Shift to get next bar's open as entry
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["next_open"] = merged["open"].shift(-1)
    merged["next_close"] = merged["close"].shift(-1)

    active = merged[merged["y_pred_prob"] > threshold].copy()
    active = active.dropna(subset=["next_open", "next_close"])
    if active.empty:
        return pd.DataFrame()

    cost_frac = cost_bps / 10_000
    active["fill_price"] = active["next_open"]
    active["gross_return"] = (active["next_close"] - active["next_open"]) / active["next_open"]
    active["net_return"] = active["gross_return"] - 2 * cost_frac

    keep_cols = ["asset", "timestamp", "y_pred_prob", "fill_price",
                 "gross_return", "net_return"]
    if "fold_id" in active.columns:
        keep_cols.append("fold_id")

    return active[[c for c in keep_cols if c in active.columns]]


def main(config_path: str | None = None):
    if config_path is None:
        config_path = (
            sys.argv[1]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
            else "configs/experiments/crypto_1h_exp005.yaml"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    target_asset = cfg.get("target_asset", "SOL-USD")
    target_model = cfg.get("target_model", "lightgbm")
    policy_cfg = cfg.get("policy", {})
    go_no_go_cfg = cfg.get("go_no_go", {})
    entry_modes = cfg.get("entry_modes", [])
    qp_cfg = cfg.get("queue_penalty", {})

    log.info("═══ Starting experiment: %s ═══", exp_id)
    log.info("Policy: %s", policy_cfg)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load data ────────────────────────────────────────────────
    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)

    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    if not panel_path.exists():
        log.info("Panel not found — downloading data and building panel…")
        download_data(cfg["data_config"])
        build_panel(cfg["data_config"])
    panel = load_parquet(panel_path)
    if hasattr(panel["timestamp"].dtype, "tz") and panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Load predictions ─────────────────────────────────────────
    pred_path = cfg.get("exp004_predictions", "data/artifacts/predictions/exp004_sol_predictions.parquet")
    log.info("Loading predictions from %s", pred_path)
    preds = load_parquet(pred_path)
    log.info("Loaded %d prediction rows", len(preds))
    if hasattr(preds["timestamp"].dtype, "tz") and preds["timestamp"].dt.tz is not None:
        preds["timestamp"] = preds["timestamp"].dt.tz_localize(None)

    # ── Load features for regime labelling ────────────────────────
    features = _load_features_with_identifiers(cfg["data_config"])

    regime_feat_cols = [
        "realized_vol_24h", "ret_24h", "ret_1h", "drawdown_168h",
        "drawdown_24h", "dollar_volume_24h", "is_weekend", "hour_of_day",
    ]
    available_cols = [c for c in regime_feat_cols if c in features.columns]
    feat_for_merge = features[["asset", "timestamp"] + available_cols].copy()
    preds_with_feats = preds.merge(
        feat_for_merge, on=["asset", "timestamp"], how="left", suffixes=("", "_feat")
    )
    preds_labelled = label_regimes(preds_with_feats)

    # Filter to target asset + model
    sol_preds = preds_labelled[
        (preds_labelled["asset"] == target_asset) &
        (preds_labelled["model_name"] == target_model)
    ].copy()
    log.info("SOL LightGBM predictions: %d rows", len(sol_preds))

    # Apply regime gate
    gate = policy_cfg.get("regime_gate", "NOT_rebound")
    if gate == "NOT_rebound" and "regime" in sol_preds.columns:
        sol_preds_gated = sol_preds[sol_preds["regime"] != "rebound"].copy()
    elif gate.startswith("AND_") and "regime" in sol_preds.columns:
        gate_regime = gate.replace("AND_", "")
        sol_preds_gated = sol_preds[sol_preds["regime"] == gate_regime].copy()
    else:
        sol_preds_gated = sol_preds.copy()
    log.info("After %s gate: %d prediction rows", gate, len(sol_preds_gated))

    sol_panel = panel[panel["asset"] == target_asset].copy()

    # ═══════════════════════════════════════════════════════════════
    #  Per entry mode: build trades + run penalty study
    # ═══════════════════════════════════════════════════════════════
    all_penalty_rows: list[pd.DataFrame] = []
    entry_summary_rows: list[dict] = []

    for em_cfg in entry_modes:
        em_name = em_cfg["name"]
        log.info("═══ Entry mode: %s ═══", em_name)

        trades = _build_entry_mode_trades(sol_panel, sol_preds_gated, em_cfg, policy_cfg)
        if trades.empty:
            log.warning("No trades for entry mode %s — skipping", em_name)
            continue

        # Apply sep_gap (minimum bars between trades)
        sep_gap = policy_cfg.get("sep_gap", 3)
        if sep_gap > 0 and "timestamp" in trades.columns:
            trades = trades.sort_values("timestamp").reset_index(drop=True)
            kept_indices = [0]
            for i in range(1, len(trades)):
                last_kept = kept_indices[-1]
                gap = trades.loc[i, "timestamp"] - trades.loc[last_kept, "timestamp"]
                if hasattr(gap, "total_seconds"):
                    gap_bars = gap.total_seconds() / 3600
                else:
                    gap_bars = sep_gap + 1
                if gap_bars >= sep_gap:
                    kept_indices.append(i)
            trades = trades.loc[kept_indices].reset_index(drop=True)

        log.info("Trades after sep_gap filter: %d", len(trades))

        # Baseline metrics
        baseline = run_fill_validation_report(trades, em_name)
        fill_rate = len(trades) / max(len(sol_preds_gated[sol_preds_gated["y_pred_prob"] > policy_cfg.get("threshold", 0.55)]), 1)
        baseline["fill_rate"] = fill_rate
        entry_summary_rows.append(baseline)

        # Queue penalty study
        penalty_df = queue_penalty_study(
            trades,
            fill_probability_haircuts=qp_cfg.get("fill_probability_haircuts", [1.0, 0.8, 0.6, 0.4]),
            fill_price_penalties_bps=qp_cfg.get("fill_price_penalties_bps", [0.0, 2.0, 5.0]),
        )
        penalty_df["entry_mode"] = em_name
        all_penalty_rows.append(penalty_df)

        # Save per-mode tables
        trades.to_csv(tbl_dir / f"trades_{em_name}.csv", index=False)
        penalty_df.to_csv(tbl_dir / f"penalty_{em_name}.csv", index=False)

    # ── Combine results ──────────────────────────────────────────
    entry_summary_df = pd.DataFrame(entry_summary_rows)
    if all_penalty_rows:
        full_penalty_df = pd.concat(all_penalty_rows, ignore_index=True)
    else:
        full_penalty_df = pd.DataFrame()

    entry_summary_df.to_csv(tbl_dir / "entry_summary.csv", index=False)
    full_penalty_df.to_csv(tbl_dir / "penalty_grid.csv", index=False)

    # ═══════════════════════════════════════════════════════════════
    #  Plots
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Generating plots ═══")

    if not entry_summary_df.empty:
        plot_entry_mode_comparison(entry_summary_df, fig_dir)

    for em_cfg in entry_modes:
        em_name = em_cfg["name"]
        if not full_penalty_df.empty:
            plot_penalty_heatmap(full_penalty_df, em_name, fig_dir)

    # ═══════════════════════════════════════════════════════════════
    #  Summary Report
    # ═══════════════════════════════════════════════════════════════
    log.info("═══ Building summary report ═══")

    summary_path = build_exp005_summary(
        report_dir,
        penalty_df=full_penalty_df,
        entry_summary_df=entry_summary_df,
        go_no_go_cfg=go_no_go_cfg,
        policy_cfg=policy_cfg,
        cfg=cfg,
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    # Save predictions
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        save_parquet(sol_preds_gated, pred_dir / "exp005_sol_gated_predictions.parquet")
        log.info("Saved gated predictions")


if __name__ == "__main__":
    main()
