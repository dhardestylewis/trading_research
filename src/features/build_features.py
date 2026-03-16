"""Orchestrate feature generation across all families."""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from src.features.price_features import compute_price_features
from src.features.volume_features import compute_volume_features
from src.features.regime_features import compute_regime_features
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("build_features")


def build(panel_path: str | Path, feature_cfg_path: str | None = None) -> pd.DataFrame:
    """Build all features from a panel parquet. Returns the feature DataFrame."""
    panel = load_parquet(panel_path)
    log.info("Panel loaded: %d rows, assets: %s", len(panel), panel["asset"].unique().tolist())

    # ── Per-asset features ───────────────────────────────────────────
    price_parts: list[pd.DataFrame] = []
    vol_parts: list[pd.DataFrame] = []

    for asset, g in panel.groupby("asset", sort=False):
        g_sorted = g.sort_values("timestamp")
        price_parts.append(compute_price_features(g_sorted))
        vol_parts.append(compute_volume_features(g_sorted))

    price_feats = pd.concat(price_parts).loc[panel.index]
    vol_feats = pd.concat(vol_parts).loc[panel.index]

    # ── Panel-wide regime features ───────────────────────────────────
    regime_feats = compute_regime_features(panel)

    # ── Combine ──────────────────────────────────────────────────────
    features = pd.concat([
        panel[["asset", "timestamp"]],
        price_feats,
        vol_feats,
        regime_feats,
    ], axis=1)

    # ── Winsorize ────────────────────────────────────────────────────
    if feature_cfg_path:
        with open(feature_cfg_path, encoding="utf-8") as f:
            fcfg = yaml.safe_load(f)
        if fcfg.get("winsorize", False):
            lo, hi = fcfg["winsorize_limits"]
            num_cols = features.select_dtypes(include=[np.number]).columns.difference(["hour_of_day", "day_of_week", "is_weekend"])
            for col in num_cols:
                lb = features[col].quantile(lo)
                ub = features[col].quantile(hi)
                features[col] = features[col].clip(lb, ub)

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = Path("data/processed/features")
    if feature_cfg_path:
        with open(feature_cfg_path, encoding="utf-8") as f:
            fcfg = yaml.safe_load(f)
        out_dir = Path(fcfg.get("output_dir", out_dir))
    ensure_dir(out_dir)
    out_path = out_dir / "features.parquet"
    save_parquet(features, out_path)
    log.info("Features saved: %d rows × %d cols → %s", len(features), len(features.columns), out_path)
    return features


if __name__ == "__main__":
    import sys
    panel_p = sys.argv[1] if len(sys.argv) > 1 else "data/processed/panel/panel.parquet"
    feat_cfg = sys.argv[2] if len(sys.argv) > 2 else "configs/features/baseline_features_v1.yaml"
    build(panel_p, feat_cfg)
