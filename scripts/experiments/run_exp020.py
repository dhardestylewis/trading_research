"""Run exp020: Execution Cost Modeling."""
import sys
import yaml
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp019.feature_engineer import build_extended_features
from src.research.exp020.cost_modeling import train_cost_model, build_cost_surface
from src.reporting.exp020_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp020")

def _checkpoint_exists(out_dir: Path, name: str) -> bool:
    return (out_dir / f"{name}.parquet").exists() or (out_dir / f"{name}.pkl").exists()

def _load_checkpoint(out_dir: Path, name: str):
    pkl_path = out_dir / f"{name}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    pq_path = out_dir / f"{name}.parquet"
    if pq_path.exists():
        return pd.read_parquet(pq_path)
    return None

def _save_checkpoint(out_dir: Path, name: str, obj):
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(out_dir / f"{name}.parquet", index=False)
    else:
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    feature_cfg = cfg["features"]
    model_cfg = cfg["model"]
    report_cfg = cfg.get("reporting", {})

    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load panel ──────────────────────────────────────────
    panel_path = Path(data_cfg["panel_path"])
    log.info("Loading panel from %s ...", panel_path)
    panel = pd.read_parquet(panel_path)

    universe = data_cfg.get("asset_universe")
    if universe:
        panel = panel[panel["asset"].isin(universe)].reset_index(drop=True)

    log.info("Panel: %d rows, %d assets", len(panel), panel["asset"].nunique())

    if "dollar_volume" not in panel.columns:
        panel["dollar_volume"] = panel["volume"] * panel["close"]

    # In exp019, exec_loss_bps was calculated dynamically. We need to ensure we have a proxy for exp020 training.
    # We will compute a simple rolling absolute return as a baseline exec_loss_bps proxy for now, 
    # to maintain compatibility if the underlying data generator hasn't baked it in.
    if "exec_loss_bps" not in panel.columns:
        panel["exec_loss_bps"] = panel.groupby("asset")["close"].transform(lambda x: (x.pct_change().abs() * 10000).shift(-1))
        # Drop where target is nan
        panel = panel.dropna(subset=["exec_loss_bps"]).reset_index(drop=True)

    # ── Step 2: Build extended features ─────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2: Feature Engineering (Reusing Exp019 base)")
    log.info("=" * 60)

    if _checkpoint_exists(out_dir, "features"):
        log.info("  ↳ Loading cached features")
        features = _load_checkpoint(out_dir, "features")
    else:
        features = build_extended_features(panel, feature_cfg)
        _save_checkpoint(out_dir, "features", features)
    log.info("Features: %d rows × %d cols", len(features), len(features.columns))

    # ── Step 3: Train Execution Cost Model ──────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3: Training Cost Model")
    log.info("=" * 60)

    feature_cols = [c for c in features.columns if c not in ["asset", "timestamp", "open", "high", "low", "close", "volume", "dollar_volume", "exec_loss_bps"]]

    if _checkpoint_exists(out_dir, "model_result"):
        log.info("  ↳ Loading cached model result")
        model_result = _load_checkpoint(out_dir, "model_result")
    else:
        model_result = train_cost_model(features, panel["exec_loss_bps"], feature_cols, model_cfg)
        _save_checkpoint(out_dir, "model_result", model_result)

    oof_preds = model_result["oof_predictions"]
    
    # ── Step 4: Build Cost Surface ──────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 4: Building Cost Surface")
    log.info("=" * 60)

    surface_df = build_cost_surface(features, oof_preds)
    _save_checkpoint(out_dir, "cost_surface", surface_df)

    # ── Step 5: Generate Report ─────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 5: Report Generation")
    log.info("=" * 60)

    report_path = generate_report(
        features=features,
        surface_df=surface_df,
        model_result=model_result,
        output_dir=out_dir,
        report_dir=Path(report_cfg.get("output_dir", "reports/exp020"))
    )
    log.info("\nReport saved to: %s", report_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/crypto_1h_exp020.yaml")
