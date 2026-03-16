"""Experiment reconciliation — exp008/exp009 policy object comparison.

Builds a structured table comparing the exact policy object, data pipeline,
and results across experiment branches to diagnose contradictions
(e.g. exp008 Branch B SOL = −20.6 bps vs exp009 Branch B SOL = +37.4 bps).

Works from saved CSVs and config files — no model re-training required.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logging import get_logger

log = get_logger("experiment_reconciliation")


# ── Config extraction helpers ──────────────────────────────────


def _load_config(path: str | Path) -> dict:
    """Load a YAML config, return empty dict on failure."""
    p = Path(path)
    if not p.exists():
        log.warning("Config not found: %s", p)
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def _extract_policy_params(cfg: dict) -> dict:
    """Extract the policy parameters that matter for reconciliation."""
    policy = cfg.get("policy", {})
    return {
        "threshold": policy.get("threshold"),
        "sep_gap": policy.get("sep_gap"),
        "regime_gate": policy.get("regime_gate"),
        "cost_bps": policy.get("cost_bps"),
        "training_mode": policy.get("training_mode"),
    }


# ── Reconciliation row builders ────────────────────────────────


def _build_row_from_csv(
    *,
    experiment_id: str,
    branch: str,
    description: str,
    cfg: dict,
    csv_path: Path | None,
    asset_filter: str = "SOL-USD",
    pool_name_filter: str | None = None,
    training_pool: str,
    deploy_asset: str,
    data_config_key: str = "data_config",
    known_bugs: str = "",
) -> dict:
    """Build one reconciliation row from a saved results CSV."""
    policy = _extract_policy_params(cfg)

    row = {
        "experiment_id": experiment_id,
        "branch": branch,
        "description": description,
        "training_pool": training_pool,
        "deploy_asset": deploy_asset,
        "threshold": policy.get("threshold"),
        "sep_gap": policy.get("sep_gap"),
        "regime_gate": policy.get("regime_gate"),
        "cost_bps": policy.get("cost_bps"),
        "data_pipeline": cfg.get(data_config_key, "unknown"),
        "known_bugs_fixed": known_bugs,
        "trade_count": np.nan,
        "fold_count": np.nan,
        "mean_net_bps": np.nan,
        "sharpe": np.nan,
        "fold_profitability": np.nan,
    }

    if csv_path is not None and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)

            # Filter to asset
            if "asset" in df.columns:
                df = df[df["asset"] == asset_filter]

            # Filter to pool name if specified
            if pool_name_filter and "pool_name" in df.columns:
                df = df[df["pool_name"] == pool_name_filter]

            if not df.empty:
                row["trade_count"] = df["trade_count"].sum() if "trade_count" in df.columns else np.nan
                row["fold_count"] = df["fold_count"].values[0] if "fold_count" in df.columns else np.nan
                row["mean_net_bps"] = df["mean_net_bps"].values[0] if "mean_net_bps" in df.columns else np.nan
                row["sharpe"] = df["sharpe"].values[0] if "sharpe" in df.columns else np.nan
                row["fold_profitability"] = df["fold_profitability"].values[0] if "fold_profitability" in df.columns else np.nan
        except Exception as e:
            log.warning("Failed to read %s: %s", csv_path, e)

    return row


# ── Main reconciliation function ───────────────────────────────


def build_reconciliation_table(
    *,
    exp008_config_path: str = "configs/experiments/crypto_1h_exp008.yaml",
    exp009_config_path: str = "configs/experiments/crypto_1h_exp009.yaml",
    exp008_report_dir: str = "reports/exp008",
    exp009_report_dir: str = "reports/exp009",
) -> pd.DataFrame:
    """Build a side-by-side reconciliation table for SOL across exp008 and exp009.

    Compares:
      - exp008 Branch B (universe expansion) SOL
      - exp009 Branch A (original 3-asset pool) SOL
      - exp009 Branch B (full 8-asset pool) SOL
      - exp009 Branch C (family-pooled) SOL

    Returns
    -------
    DataFrame with one row per branch, all policy parameters aligned.
    """
    log.info("Building exp008/exp009 reconciliation table")

    cfg008 = _load_config(exp008_config_path)
    cfg009 = _load_config(exp009_config_path)

    exp008_tbl = Path(exp008_report_dir) / "tables"
    exp009_tbl = Path(exp009_report_dir) / "tables"

    rows = []

    # ── exp008 Branch B: Universe expansion SOL ────────────────
    ue_csv = exp008_tbl / "universe_expansion.csv"
    ue_cfg = cfg008.get("universe_expansion", {})
    ue_pool = ["SOL-USD", "BTC-USD", "ETH-USD"] + ue_cfg.get("expansion_assets", [])

    rows.append(_build_row_from_csv(
        experiment_id="exp008",
        branch="B_universe_expansion",
        description="Pooled all 8 assets → SOL deploy",
        cfg=cfg008,
        csv_path=ue_csv if ue_csv.exists() else None,
        asset_filter="SOL-USD",
        training_pool=", ".join(ue_pool),
        deploy_asset="SOL-USD",
        data_config_key="data_config_expanded",
        known_bugs="Pre-exp009 data pipeline (mixed history lengths may cause NoneType.strip)",
    ))

    # ── exp009 Branch A: Original 3-asset pool SOL ─────────────
    dilution_csv = exp009_tbl / "dilution_comparison.csv"

    rows.append(_build_row_from_csv(
        experiment_id="exp009",
        branch="A_original_3_pool",
        description="Pooled SOL/BTC/ETH → SOL deploy",
        cfg=cfg009,
        csv_path=dilution_csv if dilution_csv.exists() else None,
        asset_filter="SOL-USD",
        pool_name_filter="original_3",
        training_pool="SOL-USD, BTC-USD, ETH-USD",
        deploy_asset="SOL-USD",
        data_config_key="data_config_expanded",
        known_bugs="exp009 data pipeline fixes applied",
    ))

    # ── exp009 Branch B: Full 8-asset pool SOL ─────────────────
    rows.append(_build_row_from_csv(
        experiment_id="exp009",
        branch="B_full_8_pool",
        description="Pooled all 8 assets → SOL deploy",
        cfg=cfg009,
        csv_path=dilution_csv if dilution_csv.exists() else None,
        asset_filter="SOL-USD",
        pool_name_filter="full_8",
        training_pool="SOL-USD, BTC-USD, ETH-USD, APT-USD, SUI-USD, NEAR-USD, AVAX-USD, DOT-USD",
        deploy_asset="SOL-USD",
        data_config_key="data_config_expanded",
        known_bugs="exp009 data pipeline fixes applied",
    ))

    # ── exp009 Branch C: Family-pooled SOL ─────────────────────
    family_csv = exp009_tbl / "family_pooled.csv"

    rows.append(_build_row_from_csv(
        experiment_id="exp009",
        branch="C_family_pooled",
        description="Family pool (SOL/APT/SUI/NEAR) → SOL deploy",
        cfg=cfg009,
        csv_path=family_csv if family_csv.exists() else None,
        asset_filter="SOL-USD",
        training_pool="SOL-USD, APT-USD, SUI-USD, NEAR-USD",
        deploy_asset="SOL-USD",
        data_config_key="data_config_expanded",
        known_bugs="exp009 data pipeline fixes applied",
    ))

    df = pd.DataFrame(rows)

    # Reorder columns for readability
    col_order = [
        "experiment_id", "branch", "description",
        "training_pool", "deploy_asset",
        "threshold", "sep_gap", "regime_gate", "cost_bps",
        "trade_count", "fold_count", "mean_net_bps", "sharpe", "fold_profitability",
        "data_pipeline", "known_bugs_fixed",
    ]
    cols = [c for c in col_order if c in df.columns]
    df = df[cols]

    log.info("Reconciliation table:\n%s", df.to_string(index=False))
    return df
