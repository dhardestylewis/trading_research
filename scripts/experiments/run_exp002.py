"""exp002 experiment runner.

Usage:
    python run_exp002.py                                          # full run
    python run_exp002.py --diagnostics-only                       # diagnostics on exp001 predictions
    python run_exp002.py configs/experiments/crypto_1h_exp002.yaml
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.diagnostics.regime_labeller import label_regimes
from src.diagnostics.decile_analysis import decile_metrics, tail_quantile_metrics, check_monotonicity
from src.diagnostics.regime_performance import regime_conditional_metrics
from src.diagnostics.asset_isolation import pooled_per_asset_metrics
from src.diagnostics.robustness_grid import robustness_grid
from src.diagnostics.fold_attribution import compute_fold_descriptors, regress_fold_pnl
from src.backtest.conditional_policy import evaluate_policies
from src.reporting.diagnostic_tables import save_all_diagnostic_tables
from src.reporting.diagnostic_plots import save_all_diagnostic_plots
from src.utils.io import load_parquet, save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("run_exp002")


def _load_features_with_identifiers(data_cfg_path: str) -> pd.DataFrame:
    """Load features parquet and merge back asset + timestamp from the panel."""
    with open(data_cfg_path) as f:
        data_cfg = yaml.safe_load(f)
    panel_path = Path(data_cfg["panel_dir"]) / "panel.parquet"
    feat_path = Path("data/processed/features/features.parquet")

    panel = load_parquet(panel_path)
    features = load_parquet(feat_path)

    # Attach identifiers
    features["asset"] = panel["asset"].values
    features["timestamp"] = pd.DatetimeIndex(panel["timestamp"].values).tz_localize(None)
    return features


def main(config_path: str | None = None, diagnostics_only: bool = False):
    if config_path is None:
        config_path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "configs/experiments/crypto_1h_exp002.yaml"
    if "--diagnostics-only" in sys.argv:
        diagnostics_only = True

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp_id = cfg["experiment_id"]
    log.info("═══ Starting experiment: %s ═══", exp_id)

    report_dir = Path(cfg["reporting"]["output_dir"])
    fig_dir = ensure_dir(report_dir / "figures")
    tbl_dir = ensure_dir(report_dir / "tables")

    # ── Load exp001 predictions ──────────────────────────────────
    pred_path = cfg.get("exp001_predictions", "data/artifacts/predictions/crypto_1h_exp001_predictions.parquet")
    log.info("Loading predictions from %s", pred_path)
    preds = load_parquet(pred_path)
    log.info("Loaded %d prediction rows", len(preds))

    # Ensure tz-naive timestamps
    if hasattr(preds["timestamp"].dtype, "tz") and preds["timestamp"].dt.tz is not None:
        preds["timestamp"] = preds["timestamp"].dt.tz_localize(None)

    # ── Load features for regime labelling ───────────────────────
    log.info("Loading features for regime labelling…")
    features = _load_features_with_identifiers(cfg["data_config"])

    # ── Load fold definitions ────────────────────────────────────
    fold_path = Path("data/artifacts/folds/fold_definitions.parquet")
    fold_df = None
    if fold_path.exists():
        fold_df = load_parquet(fold_path)
        for col in ("start", "end"):
            if fold_df[col].dt.tz is not None:
                fold_df[col] = fold_df[col].dt.tz_localize(None)
        log.info("Loaded %d fold definitions", fold_df["fold_id"].nunique())

    # ── Step 1: Label regimes on predictions ─────────────────────
    log.info("── Step 1: Labelling regimes ──")
    # Merge feature columns needed for regime labelling into preds
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
    log.info("  Regime columns added: %d", sum(c.startswith("regime_") for c in preds_labelled.columns))

    # ── Step 2: Diagnostic tables ────────────────────────────────
    log.info("── Step 2: Generating diagnostic tables ──")

    # Branch B: Score deciles
    log.info("  Branch B: Score-decile analysis…")
    decile_df = decile_metrics(preds_labelled, cost_bps=15.0)
    decile_df.to_csv(tbl_dir / "score_deciles.csv", index=False)

    tail_df = tail_quantile_metrics(preds_labelled, cost_bps=15.0)
    tail_df.to_csv(tbl_dir / "tail_quantile_metrics.csv", index=False)

    mono_df = check_monotonicity(decile_df)
    mono_df.to_csv(tbl_dir / "monotonicity.csv", index=False)

    # Branch A: Asset-mode metrics
    log.info("  Branch A: Asset-mode metrics…")
    asset_df = pooled_per_asset_metrics(preds_labelled, cost_bps=15.0)
    asset_df.to_csv(tbl_dir / "asset_mode_metrics.csv", index=False)

    # Branch C: Regime metrics
    log.info("  Branch C: Regime conditional metrics…")
    regime_df = regime_conditional_metrics(preds_labelled, cost_bps=15.0)
    regime_df.to_csv(tbl_dir / "regime_metrics.csv", index=False)

    # Branch C: Fold attribution
    fold_desc = None
    fold_reg = None
    if fold_df is not None:
        log.info("  Branch C: Fold regime attribution…")
        features_with_ids = features.copy()
        fold_desc = compute_fold_descriptors(features_with_ids, fold_df, preds_labelled, cost_bps=15.0)
        fold_desc.to_csv(tbl_dir / "fold_regime_attribution.csv", index=False)

        fold_reg = regress_fold_pnl(fold_desc)
        if not fold_reg.empty:
            fold_reg.to_csv(tbl_dir / "fold_regime_regression.csv", index=False)

    # Branch D: Policy comparison
    log.info("  Branch D: Conditional policy evaluation…")
    policy_df = evaluate_policies(preds_labelled, cost_bps=15.0)
    policy_df.to_csv(tbl_dir / "policy_comparison.csv", index=False)

    # Branch E: Execution robustness
    log.info("  Branch E: Execution robustness grid…")
    robust_df = robustness_grid(preds_labelled)
    robust_df.to_csv(tbl_dir / "delay_cost_robustness.csv", index=False)

    # ── Step 3: Diagnostic plots ─────────────────────────────────
    log.info("── Step 3: Generating diagnostic plots ──")
    save_all_diagnostic_plots(
        preds=preds_labelled,
        decile_df=decile_df,
        regime_df=regime_df,
        fold_desc=fold_desc,
        robustness_df=robust_df,
        policy_df=policy_df,
        fig_dir=fig_dir,
    )

    # ── Step 4: Build summary report ─────────────────────────────
    log.info("── Step 4: Building summary report ──")
    summary_path = _build_summary(
        report_dir, decile_df, tail_df, mono_df, asset_df,
        regime_df, fold_desc, fold_reg, policy_df, robust_df, cfg
    )
    log.info("═══ Report generated: %s ═══", summary_path)

    # ── Save enriched predictions ────────────────────────────────
    if cfg["reporting"].get("save_predictions", True):
        pred_dir = ensure_dir("data/artifacts/predictions")
        save_parquet(preds_labelled, pred_dir / "exp002_predictions.parquet")
        log.info("Saved enriched predictions to %s", pred_dir / "exp002_predictions.parquet")


def _build_summary(
    report_dir: Path,
    decile_df, tail_df, mono_df, asset_df,
    regime_df, fold_desc, fold_reg, policy_df, robust_df,
    cfg: dict,
) -> Path:
    """Build exp002 summary.md with all diagnostic sections and recommendation."""
    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp002\n")
    lines.append("## Primary Question\n")
    lines.append("**Is the LightGBM edge a stable conditional inefficiency, or just a thin SOL-heavy artifact?**\n")

    # ── Section A: Signal Existence ──────────────────────────────
    lines.append("---\n")
    lines.append("## Section A — Signal Existence\n")
    lines.append("### Score-Decile Metrics\n")
    if not decile_df.empty:
        lines.append(decile_df.drop(columns=["asset_composition"], errors="ignore").to_markdown(index=False))
        lines.append("\n")

    lines.append("### Tail-Quantile Metrics\n")
    if not tail_df.empty:
        lines.append(tail_df.to_markdown(index=False))
        lines.append("\n")

    # ── Section B: Score Usability ───────────────────────────────
    lines.append("---\n")
    lines.append("## Section B — Score Usability\n")
    lines.append("### Monotonicity Check\n")
    if not mono_df.empty:
        lines.append(mono_df.to_markdown(index=False))
        lines.append("\n")

    # ── Section C: Regime Dependence ─────────────────────────────
    lines.append("---\n")
    lines.append("## Section C — Regime Dependence\n")
    lines.append("### Conditional Sharpe by Regime\n")
    if not regime_df.empty:
        lines.append(regime_df.to_markdown(index=False))
        lines.append("\n")

    if fold_reg is not None and not fold_reg.empty:
        lines.append("### Fold PnL Regression on Regime Descriptors\n")
        lines.append(fold_reg.to_markdown(index=False))
        lines.append("\n")

    # ── Section D: Execution Robustness ──────────────────────────
    lines.append("---\n")
    lines.append("## Section D — Execution Robustness\n")
    if not robust_df.empty:
        # Show taker-only summary for clarity
        taker = robust_df[robust_df["fill_mode"] == "taker"]
        if not taker.empty:
            lines.append("### Delay × Cost (Taker Fill)\n")
            lines.append(taker[["model_name", "delay_bars", "cost_regime", "sharpe",
                                "cumulative_return", "max_drawdown", "trade_count"]].to_markdown(index=False))
            lines.append("\n")

    # ── Section E: Portability ───────────────────────────────────
    lines.append("---\n")
    lines.append("## Section E — Portability\n")
    lines.append("### Per-Asset-Mode Metrics\n")
    if not asset_df.empty:
        lines.append(asset_df.to_markdown(index=False))
        lines.append("\n")

    # ── Policy Comparison ────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Policy Comparison\n")
    if not policy_df.empty:
        # Show top 20 policies by Sharpe
        top = policy_df.sort_values("sharpe", ascending=False).head(20)
        lines.append(top.to_markdown(index=False))
        lines.append("\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Figures\n")
    fig_dir = report_dir / "figures"
    if fig_dir.exists():
        for fig in sorted(fig_dir.glob("*.png")):
            lines.append(f"![{fig.stem}]({fig.name})\n")

    # ── Go / No-Go ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Go / No-Go Assessment\n")
    gates = _evaluate_exp002_gates(
        decile_df, mono_df, policy_df, robust_df, asset_df, cfg
    )
    for gate_name, (passed, detail) in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"- **{gate_name}**: {status} — {detail}")
    lines.append("\n")

    # ── Recommendation ───────────────────────────────────────────
    rec = _determine_recommendation(gates, asset_df)
    lines.append(f"\n### Recommendation: **{rec}**\n")

    out_path = report_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _evaluate_exp002_gates(decile_df, mono_df, policy_df, robust_df, asset_df, cfg) -> dict:
    """Evaluate 6 go/no-go gates."""
    gates: dict = {}
    criteria = cfg.get("success_criteria", {})

    # Gate 1: Fold profitability > 60%
    if not policy_df.empty:
        best_fold_prof = policy_df["fold_profitability"].max()
        gates["Fold Stability"] = (
            best_fold_prof > criteria.get("fold_profitability_min", 0.60),
            f"best={best_fold_prof:.0%}"
        )

    # Gate 2: Asset concentration < 75%
    if not policy_df.empty and "max_asset_share" in policy_df.columns:
        # Use the best policy by Sharpe
        best_row = policy_df.loc[policy_df["sharpe"].idxmax()]
        max_share = best_row.get("max_asset_share", 1.0)
        gates["Asset Concentration"] = (
            max_share < criteria.get("max_single_asset_share", 0.75),
            f"max share={max_share:.0%} in best policy"
        )

    # Gate 3: Tail monotonicity
    if not mono_df.empty:
        lgb_mono = mono_df[mono_df["model_name"] == "lightgbm"]
        if not lgb_mono.empty:
            rho = lgb_mono.iloc[0]["spearman_rho"]
            is_mono = lgb_mono.iloc[0]["is_monotonic"]
            gates["Tail Monotonicity"] = (
                rho > 0.5,
                f"Spearman ρ={rho:.3f}, strict_mono={is_mono}"
            )

    # Gate 4: Regime improvement ≥ 10pp over baseline
    if not policy_df.empty:
        baseline = policy_df[policy_df["policy"] == "baseline"]
        gated = policy_df[policy_df["policy"].isin(["regime_gated", "hybrid"])]
        if not baseline.empty and not gated.empty:
            base_prof = baseline["fold_profitability"].max()
            best_gated_prof = gated["fold_profitability"].max()
            improvement = best_gated_prof - base_prof
            gates["Regime Improvement"] = (
                improvement >= 0.10,
                f"baseline={base_prof:.0%}, best gated={best_gated_prof:.0%}, Δ={improvement:+.0%}"
            )

    # Gate 5: Execution robustness — 1-bar delay stays positive
    if not robust_df.empty:
        taker_1bar = robust_df[
            (robust_df["delay_bars"] == 1) &
            (robust_df["fill_mode"] == "taker") &
            (robust_df["cost_regime"] == "base")
        ]
        if not taker_1bar.empty:
            lgb = taker_1bar[taker_1bar["model_name"] == "lightgbm"]
            if not lgb.empty:
                delay_sharpe = lgb.iloc[0]["sharpe"]
                gates["Execution Robustness"] = (
                    delay_sharpe > 0,
                    f"1-bar delay + base cost Sharpe={delay_sharpe:.3f}"
                )

    # Gate 6: Trade sufficiency ≥ 100
    if not policy_df.empty:
        best_row = policy_df.loc[policy_df["sharpe"].idxmax()]
        tc = best_row.get("trade_count", 0)
        gates["Trade Sufficiency"] = (
            tc >= criteria.get("min_trade_count", 100),
            f"trade count={tc} in best policy"
        )

    return gates


def _determine_recommendation(gates: dict, asset_df: pd.DataFrame) -> str:
    """Determine one of four recommendation outcomes."""
    all_pass = all(p for p, _ in gates.values())
    delay_pass = gates.get("Execution Robustness", (False, ""))[0]
    fold_pass = gates.get("Fold Stability", (False, ""))[0]
    concentration_pass = gates.get("Asset Concentration", (False, ""))[0]

    if not delay_pass and not fold_pass:
        return "A — No-go, redesign. Edge does not survive diagnostics."

    if not concentration_pass and fold_pass:
        # Check if SOL-only looks good
        return "B — No-go for pooled, but SOL-only research is warranted."

    if all_pass or (delay_pass and fold_pass and concentration_pass):
        return "C — Proceed to selective paper trading."

    if fold_pass and delay_pass and not concentration_pass:
        return "D — Expand to structurally similar assets before paper trading."

    # Fallback
    passed = sum(1 for p, _ in gates.values() if p)
    return f"A — No-go, redesign ({passed}/{len(gates)} gates passed)."


if __name__ == "__main__":
    main()
