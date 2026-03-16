"""Build a stacked (asset, timestamp) panel from normalized per-asset CSVs."""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import yaml

from src.data.normalize_bars import normalize
from src.utils.io import save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("build_panel")


def build(cfg_path: str) -> pd.DataFrame:
    """Stack all assets into one panel parquet."""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = Path(cfg["raw_dir"])
    labels = cfg.get("asset_labels", {})
    frames: list[pd.DataFrame] = []

    for symbol in cfg["asset_universe"]:
        label = labels.get(symbol, symbol.replace("/", "-"))
        csv_path = raw_dir / f"{label}.csv"
        if not csv_path.exists():
            log.warning("Raw CSV not found for %s at %s — skipping", label, csv_path)
            continue
        df = normalize(csv_path, label)
        log.info("  %s: %d bars", label, len(df))
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True).sort_values(["asset", "timestamp"]).reset_index(drop=True)

    panel_dir = Path(cfg["panel_dir"])
    ensure_dir(panel_dir)
    out_path = panel_dir / "panel.parquet"
    save_parquet(panel, out_path)
    log.info("Panel saved: %d rows → %s", len(panel), out_path)
    return panel


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "configs/data/crypto_1h.yaml")
