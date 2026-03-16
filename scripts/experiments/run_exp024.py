"""Run exp024: Execution Cost Surface and Alpha-Cost Join.

Orchestrates all branches A-F:
  A — Historical cost surface construction
  B — Cost prediction model (LightGBM/CatBoost/quantile walk-forward)
  C — Alpha-cost join with frozen exp022/023 signal
  D — Low-cost execution state discovery
  E — Live paper cost logger (scaffold)
  F — Cost model freshness / cadence study

Branch verdict: If no joint high-alpha/low-cost bucket has positive
median AND trimmed-mean net markout, the directional microstructure
alpha program is closed.
"""
import os
import sys
import yaml
import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("run_exp024")


# ── Utilities ────────────────────────────────────────────────────

def load_config(path="configs/experiments/crypto_microstructure_exp024.yaml"):
    logger.info("Loading config from %s", path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _checkpoint_path(out_dir: Path, name: str, ext: str = "parquet") -> Path:
    return out_dir / f"{name}.{ext}"


def _save_checkpoint(out_dir: Path, name: str, obj):
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        obj.to_parquet(_checkpoint_path(out_dir, name))
        logger.info("  ✓ Saved checkpoint: %s.parquet", name)
    else:
        with open(_checkpoint_path(out_dir, name, "pkl"), "wb") as f:
            pickle.dump(obj, f)
        logger.info("  ✓ Saved checkpoint: %s.pkl", name)


def _load_checkpoint(out_dir: Path, name: str):
    pq = _checkpoint_path(out_dir, name)
    if pq.exists():
        return pd.read_parquet(pq)
    pkl = _checkpoint_path(out_dir, name, "pkl")
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    return None


# ── Branch A ─────────────────────────────────────────────────────

def run_branch_a(config, out_dir: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH A: Historical Cost Surface Construction")
    logger.info("=" * 60)

    cached = _load_checkpoint(out_dir, "cost_surface_base")
    if cached is not None:
        logger.info("  ↳ Loaded cached cost surface: %d rows", len(cached))
        return cached

    from src.data.processing.build_execution_surface import build_execution_surface

    data_cfg = config.get("data", {})
    exec_cfg = config.get("execution", {})

    assets = [a.replace("-", "").upper()
              for a in config.get("universe", {}).get("assets", ["BTC-USDT", "ETH-USDT", "SOL-USDT"])]

    surface = build_execution_surface(
        flow_bar_dir=data_cfg.get("flow_bar_dir", "data/processed/flow_bars"),
        assets=assets,
        fee_bps=exec_cfg.get("fee_bps", 4.0),
        latency_buckets_ms=exec_cfg.get("latency_buckets_ms", [0, 100, 250, 500, 1000]),
    )

    if surface.empty:
        logger.warning("Branch A produced empty surface — running in scaffold mode")
        return pd.DataFrame()

    _save_checkpoint(out_dir, "cost_surface_base", surface)
    return surface


# ── Branch B ─────────────────────────────────────────────────────

def run_branch_b(config, surface: pd.DataFrame, out_dir: Path) -> dict:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH B: Cost Prediction Model")
    logger.info("=" * 60)

    if surface.empty:
        logger.warning("  ↳ Empty surface; skipping Branch B")
        return {}

    cached = _load_checkpoint(out_dir, "cost_model_results")
    if cached is not None:
        logger.info("  ↳ Loaded cached model results")
        return cached

    from src.models.cost_surface_model import CostSurfaceModelTrainer

    training_cfg = config.get("training", {})
    model_classes = config.get("models", {}).get("classes", ["LightGBM"])
    targets = config.get("models", {}).get("cost_targets", [
        "shortfall_1s_bps", "shortfall_5s_bps",
        "adverse_markout_1s_bps", "adverse_markout_5s_bps",
    ])

    trainer = CostSurfaceModelTrainer(
        targets=targets,
        model_classes=model_classes,
    )

    # Use 6h forward slices for Branch B (Branch F tests finer cadences)
    train_window = training_cfg.get("min_train_seconds", 259200)  # 3d default
    forward_slice = 21600  # 6h slices for faster walk-forward

    results = trainer.train_walk_forward(
        surface,
        train_window_seconds=train_window,
        forward_slice_seconds=forward_slice,
    )

    _save_checkpoint(out_dir, "cost_model_results", results)
    return results


# ── Branch C ─────────────────────────────────────────────────────

def run_branch_c(config, surface: pd.DataFrame, model_results: dict, out_dir: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH C: Alpha-Cost Join")
    logger.info("=" * 60)

    if surface.empty:
        logger.warning("  ↳ Empty surface; skipping Branch C")
        return pd.DataFrame()

    from src.models.alpha_cost_join import run_alpha_cost_join

    # Subsample for speed — bucket-level medians don't need 5.7M rows
    if len(surface) > 500_000:
        logger.info("  Subsampling surface from %d to 500K rows for alpha-cost join", len(surface))
        surface_sample = surface.sample(n=500_000, random_state=42)
    else:
        surface_sample = surface

    # Collect cost prediction DataFrames from model results
    cost_preds = {}
    for key, res in model_results.items():
        if "oof_predictions" in res:
            cost_preds[key] = res["oof_predictions"]

    join_path = str(out_dir / "alpha_cost_join.csv")
    join_df = run_alpha_cost_join(surface_sample, cost_preds, output_path=join_path)

    return join_df


# ── Branch D ─────────────────────────────────────────────────────

def run_branch_d(config, surface: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH D: Low-Cost State Discovery")
    logger.info("=" * 60)

    if surface.empty:
        logger.warning("  ↳ Empty surface; skipping Branch D")
        return pd.DataFrame()

    from src.research.exp024.execution_state_discovery import discover_execution_states

    # Subsample for speed — state discovery only needs representative sample
    if len(surface) > 500_000:
        logger.info("  Subsampling surface from %d to 500K rows for state discovery", len(surface))
        surface = surface.sample(n=500_000, random_state=42)

    cells_path = str(out_dir / "execution_cells.csv")
    result = discover_execution_states(surface, output_path=cells_path)

    return result.get("state_cards", pd.DataFrame())


# ── Branch E ─────────────────────────────────────────────────────

def run_branch_e(config, surface: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH E: Live Paper Cost Logger (scaffold)")
    logger.info("=" * 60)

    from src.eval.live_paper_cost_eval import LivePaperCostLogger, compute_replay_vs_live_gap

    paper_logger = LivePaperCostLogger(
        log_path=str(out_dir / "live_paper_cost_log.csv")
    )

    # Simulate with tail of surface as "live" data
    if not surface.empty:
        sample = surface.tail(min(500, len(surface)))
        for idx, row in sample.iterrows():
            book = {
                "bid": row.get("best_bid", 100),
                "ask": row.get("best_ask", 101),
            }
            features = row.to_dict()
            paper_logger.score_and_log(idx, row.get("asset", "UNK"), features, book)

        paper_logger.save()

    # Compute replay-vs-live gap (simulated: compare full surface vs paper log)
    live_df = paper_logger.to_dataframe()
    replay_vs_live = compute_replay_vs_live_gap(
        surface, live_df, cost_col="shortfall_1s_bps"
    )
    logger.info("Replay vs Live:\n%s", replay_vs_live.to_string())

    return replay_vs_live


# ── Branch F ─────────────────────────────────────────────────────

def run_branch_f(config, surface: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    logger.info("\n" + "=" * 60)
    logger.info("BRANCH F: Cost Model Freshness / Cadence Study")
    logger.info("=" * 60)

    if surface.empty:
        logger.warning("  ↳ Empty surface; skipping Branch F")
        return pd.DataFrame()

    from src.eval.shortfall_replay import evaluate_cost_freshness

    training_cfg = config.get("training", {})
    train_windows = training_cfg.get("train_windows", ["3d", "7d", "14d", "30d"])
    refresh_cadences = training_cfg.get("forward_slices", ["1h", "6h", "12h", "24h"])

    freshness = evaluate_cost_freshness(
        surface,
        train_windows=train_windows,
        refresh_cadences=refresh_cadences,
    )

    _save_checkpoint(out_dir, "freshness_study", freshness)
    return freshness


# ── Reporting ────────────────────────────────────────────────────

def run_reporting(
    config, surface, model_results, alpha_cost_join,
    state_cards, freshness_df, replay_vs_live,
):
    logger.info("\n" + "=" * 60)
    logger.info("REPORTING")
    logger.info("=" * 60)

    from src.reporting.exp024_report import generate_all_reports

    report_dir = config.get("reporting", {}).get("output_dir", "reports/exp024")
    report_path = generate_all_reports(
        surface=surface,
        model_results=model_results,
        alpha_cost_join=alpha_cost_join,
        state_cards=state_cards,
        freshness_df=freshness_df,
        replay_vs_live=replay_vs_live,
        report_dir=report_dir,
    )
    logger.info("Report: %s", report_path)
    return report_path


# ── Main Pipeline ────────────────────────────────────────────────

def run_full_pipeline(config_path=None):
    logger.info("Starting exp024 pipeline: Execution Cost Surface and Alpha-Cost Join")

    try:
        config = load_config(config_path or "configs/experiments/crypto_microstructure_exp024.yaml")
        out_dir = Path(config.get("data", {}).get("output_dir", "data/processed/exp024_cost_surface"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # Branch A
        surface = run_branch_a(config, out_dir)

        # Branch B
        model_results = run_branch_b(config, surface, out_dir)

        # Branch C
        alpha_cost_join = run_branch_c(config, surface, model_results, out_dir)

        # Branch D
        state_cards = run_branch_d(config, surface, out_dir)

        # Branch E
        replay_vs_live = run_branch_e(config, surface, out_dir)

        # Branch F
        freshness_df = run_branch_f(config, surface, out_dir)

        # Report
        run_reporting(
            config, surface, model_results, alpha_cost_join,
            state_cards, freshness_df, replay_vs_live,
        )

        logger.info("\n" + "=" * 60)
        logger.info("exp024 pipeline completed successfully.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_full_pipeline(config_arg)
