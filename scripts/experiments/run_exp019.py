"""Run exp019: Latent Cell Discovery under the Money Equation.

Pipeline (with checkpoint/resume — each step checks for cached output):
1. Load expanded panel data
2. Build extended features (spread, vol, compression, microstructure, momentum)
3. Discover latent market states via clustering
4. Train multi-head LightGBM (gross move, exec loss, net move)
5. Extract interpretable cells from top latent states
6. Apply hard economic kill gates
7. Generate report
"""
import sys
import yaml
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from src.research.exp019.feature_engineer import build_extended_features
from src.research.exp019.state_discovery import discover_states
from src.research.exp019.net_expectancy_model import train_multi_head
from src.research.exp019.cell_extractor import extract_cells
from src.research.exp019.economic_gates import apply_gates
from src.reporting.exp019_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp019")


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
    state_cfg = cfg["state_discovery"]
    model_cfg = cfg["model"]
    exec_cfg = cfg["execution"]
    gates_cfg = cfg["gates"]
    extract_cfg = cfg["cell_extraction"]
    report_cfg = cfg.get("reporting", {})

    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load panel ──────────────────────────────────────────
    panel_path = Path(data_cfg["panel_path"])
    log.info("Loading panel from %s ...", panel_path)
    panel = pd.read_parquet(panel_path)

    # Filter to configured universe if specified
    universe = data_cfg.get("asset_universe")
    if universe:
        panel = panel[panel["asset"].isin(universe)].reset_index(drop=True)

    log.info("Panel: %d rows, %d assets", len(panel), panel["asset"].nunique())
    log.info("Assets: %s", sorted(panel["asset"].unique().tolist()))

    # Ensure dollar_volume exists
    if "dollar_volume" not in panel.columns:
        panel["dollar_volume"] = panel["volume"] * panel["close"]

    # ── Step 2: Build extended features ─────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 2: Extended Feature Engineering")
    log.info("=" * 60)

    if _checkpoint_exists(out_dir, "features"):
        log.info("  ↳ Loading cached features")
        features = _load_checkpoint(out_dir, "features")
    else:
        features = build_extended_features(panel, feature_cfg)
        _save_checkpoint(out_dir, "features", features)
    log.info("Features: %d rows × %d cols", len(features), len(features.columns))

    # ── Step 3: Discover latent states ──────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 3: Latent State Discovery")
    log.info("=" * 60)

    if _checkpoint_exists(out_dir, "state_result"):
        log.info("  ↳ Loading cached state result")
        state_result = _load_checkpoint(out_dir, "state_result")
    else:
        state_result = discover_states(features, state_cfg)
        _save_checkpoint(out_dir, "state_result", state_result)

    labels = state_result["best_labels"]
    feature_cols = state_result["feature_cols"]

    # Save cluster assignments (always, for easy inspection)
    cluster_df = features[["asset", "timestamp"]].copy()
    cluster_df["cluster"] = labels
    _save_checkpoint(out_dir, "cluster_assignments", cluster_df)

    # ── Step 4: Train multi-head model (per horizon) ─────────────
    horizons = cfg.get("horizons", [{"bars": 1, "label": "1h"}])
    all_model_results = {}

    for hz in horizons:
        hz_bars = hz["bars"]
        hz_label = hz["label"]

        log.info("\n" + "=" * 60)
        log.info("STEP 4: Multi-Head Model — horizon %s (%d bars)", hz_label, hz_bars)
        log.info("=" * 60)

        oof_name = f"oof_predictions_{hz_label}"
        model_name = f"model_result_{hz_label}"

        if _checkpoint_exists(out_dir, model_name):
            log.info("  ↳ Loading cached model result for %s", hz_label)
            model_result = _load_checkpoint(out_dir, model_name)
        else:
            model_result = train_multi_head(
                features, panel, feature_cols,
                hz_bars, model_cfg, exec_cfg,
            )
            # Save OOF predictions as parquet (fast to reload)
            _save_checkpoint(out_dir, oof_name, model_result["combined_predictions"])
            # Save full model result as pickle (includes trained models)
            _save_checkpoint(out_dir, model_name, model_result)

        all_model_results[hz_label] = model_result

    # ── Step 5: Extract interpretable cells ──────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 5: Cell Extraction")
    log.info("=" * 60)

    # Use the first horizon's predictions for cell extraction
    primary_hz = horizons[0]["label"]
    primary_combined = all_model_results[primary_hz]["combined_predictions"]

    cell_result = extract_cells(
        features, primary_combined, labels,
        feature_cols, exec_cfg, extract_cfg,
    )

    # Save cell economics
    _save_checkpoint(out_dir, "cell_economics", cell_result["cell_economics"])

    # ── Step 6: Apply economic kill gates ────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 6: Economic Kill Gates")
    log.info("=" * 60)

    gate_result = apply_gates(
        cell_result["cell_economics"],
        cell_result["cell_cards"],
        primary_combined,
        labels,
        gates_cfg,
    )

    # Save gate results
    _save_checkpoint(out_dir, "gate_results", gate_result["gate_results"])

    # ── Step 7: Generate report ──────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("STEP 7: Report Generation")
    log.info("=" * 60)

    report_path = generate_report(
        feature_summary={
            "n_rows": len(features),
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "nan_rate": features[feature_cols].isnull().mean().mean(),
        },
        state_result=state_result,
        model_results=all_model_results,
        cell_result=cell_result,
        gate_result=gate_result,
        horizons=[h["label"] for h in horizons],
        output_dir=out_dir,
        report_dir=Path(report_cfg.get("output_dir", "reports/exp019")),
    )
    log.info("\nReport saved to: %s", report_path)

    # ── Final verdict ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FINAL VERDICT")
    log.info("=" * 60)
    log.info("  %s", gate_result["verdict"])
    log.info("  Advancing cells: %s", gate_result["advancing_cells"])
    log.info("  Killed cells: %s", gate_result["killed_cells"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "configs/experiments/crypto_1h_exp019.yaml")
