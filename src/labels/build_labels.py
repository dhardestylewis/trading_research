"""Orchestrate label generation."""
from __future__ import annotations
from pathlib import Path

import yaml

from src.labels.forward_returns import compute_forward_labels
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("build_labels")


def build(
    panel_path: str | Path,
    label_cfg_path: str,
    backtest_cfg_path: str,
) -> "pd.DataFrame":
    import pandas as pd

    panel = load_parquet(panel_path)

    with open(label_cfg_path, encoding="utf-8") as f:
        lcfg = yaml.safe_load(f)
    with open(backtest_cfg_path, encoding="utf-8") as f:
        bcfg = yaml.safe_load(f)

    horizons = lcfg["horizons"]
    cost_regime = lcfg.get("label_cost_regime", "base")
    one_way_bps = bcfg["costs"][cost_regime]["one_way_bps"]

    labels = compute_forward_labels(panel, horizons, one_way_cost_bps=one_way_bps)

    # attach identifiers
    labels = pd.concat([panel[["asset", "timestamp"]], labels], axis=1)

    out_dir = Path(lcfg.get("output_dir", "data/processed/labels"))
    ensure_dir(out_dir)
    out_path = out_dir / "labels.parquet"
    save_parquet(labels, out_path)
    log.info("Labels saved: %d rows × %d cols → %s", len(labels), len(labels.columns), out_path)
    return labels


if __name__ == "__main__":
    import sys
    build(
        sys.argv[1] if len(sys.argv) > 1 else "data/processed/panel/panel.parquet",
        sys.argv[2] if len(sys.argv) > 2 else "configs/labels/horizon_labels_v1.yaml",
        sys.argv[3] if len(sys.argv) > 3 else "configs/backtests/long_flat_v1.yaml",
    )
