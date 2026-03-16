"""Build a markdown summary report from experiment results."""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from src.utils.io import load_csv, ensure_dir


def build(report_dir: str | Path) -> Path:
    """Generate summary.md from the tables and figures in report_dir."""
    rd = Path(report_dir)
    ensure_dir(rd)

    # Load key tables
    mc = load_csv(rd / "tables" / "model_comparison.csv")
    cs = load_csv(rd / "tables" / "cost_sensitivity.csv")
    ts = load_csv(rd / "tables" / "threshold_sensitivity.csv")
    am = load_csv(rd / "tables" / "asset_metrics.csv")
    fm = load_csv(rd / "tables" / "fold_metrics.csv")

    lines: list[str] = []
    lines.append("# Experiment Report: crypto_1h_exp001\n")

    # ── Key results ──────────────────────────────────────────────
    lines.append("## Key Results\n")
    base_55 = mc[(mc["cost_regime"] == "base") & (mc["threshold"] == 0.55)]
    if not base_55.empty:
        lines.append("### Model Comparison (base cost, τ=0.55)\n")
        lines.append(base_55[["model_name", "sharpe", "cumulative_return", "max_drawdown",
                               "hit_rate_trades", "profit_factor", "num_trades", "exposure_fraction"]].to_markdown(index=False))
        lines.append("\n")

    # ── Cost sensitivity ─────────────────────────────────────────
    lines.append("## Cost Sensitivity (τ=0.55)\n")
    lines.append(cs.to_markdown(index=False))
    lines.append("\n")

    # ── Threshold sensitivity ────────────────────────────────────
    lines.append("## Threshold Sensitivity (base cost)\n")
    lines.append(ts.to_markdown(index=False))
    lines.append("\n")

    # ── Per-asset breakdown ──────────────────────────────────────
    lines.append("## Per-Asset Breakdown (base cost, τ=0.55)\n")
    lines.append(am[["model_name", "asset", "sharpe", "cumulative_return", "max_drawdown"]].to_markdown(index=False))
    lines.append("\n")

    # ── Fold stability ───────────────────────────────────────────
    lines.append("## Fold Stability (base cost, τ=0.55)\n")
    base_fold = fm[(fm["cost_regime"] == "base") & (fm["threshold"] == 0.55)]
    if not base_fold.empty:
        lines.append(base_fold[["model_name", "fold_id", "sharpe", "cumulative_return", "max_drawdown"]].to_markdown(index=False))
    lines.append("\n")

    # ── Figures ──────────────────────────────────────────────────
    lines.append("## Figures\n")
    fig_dir = rd / "figures"
    if fig_dir.exists():
        for fig in sorted(fig_dir.glob("*.png")):
            lines.append(f"![{fig.stem}]({fig.name})\n")

    # ── Go / No-Go ───────────────────────────────────────────────
    lines.append("## Go / No-Go Assessment\n")
    go = _evaluate_gates(mc, am, fm)
    for gate_name, (passed, detail) in go.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        lines.append(f"- **{gate_name}**: {status} — {detail}")
    lines.append("\n")

    overall = all(passed for passed, _ in go.values())
    lines.append(f"\n### Recommendation: **{'GO — proceed to paper trading' if overall else 'NO-GO — refine before paper trading'}**\n")

    out_path = rd / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _evaluate_gates(mc: pd.DataFrame, am: pd.DataFrame, fm: pd.DataFrame) -> dict:
    """Evaluate go/no-go gates. Returns {gate_name: (passed, detail)}."""
    gates: dict = {}

    # Gate 1: Net Edge — at least one model beats naive on net Sharpe (base cost, τ=0.55)
    base_55 = mc[(mc["cost_regime"] == "base") & (mc["threshold"] == 0.55)]
    naive_sharpe = base_55.loc[base_55["model_name"] == "naive_momentum", "sharpe"].values
    naive_s = naive_sharpe[0] if len(naive_sharpe) > 0 else 0
    ml_sharpes = base_55.loc[base_55["model_name"] != "naive_momentum", "sharpe"]
    gate1 = bool((ml_sharpes > naive_s).any())
    gates["Net Edge (beat naive Sharpe)"] = (gate1, f"naive={naive_s:.3f}, best ML={ml_sharpes.max():.3f}" if len(ml_sharpes) > 0 else "no ML models")

    # Gate 2: Stable Profitability — positive net return across majority of folds
    base_fold = fm[(fm["cost_regime"] == "base") & (fm["threshold"] == 0.55)]
    for model in base_fold["model_name"].unique():
        mf = base_fold[base_fold["model_name"] == model]
        frac_pos = (mf["cumulative_return"] > 0).mean()
        if model != "naive_momentum":
            gates[f"Fold Stability ({model})"] = (frac_pos > 0.5, f"{frac_pos:.0%} of folds profitable")

    # Gate 3: No Asset Concentration — no single asset > 70% of PnL
    for model in am["model_name"].unique():
        ma = am[am["model_name"] == model]
        if len(ma) > 0 and ma["cumulative_return"].sum() != 0:
            max_share = ma["cumulative_return"].abs().max() / ma["cumulative_return"].abs().sum()
            gates[f"Asset Diversification ({model})"] = (max_share < 0.70, f"max share={max_share:.0%}")

    # Gate 4: Threshold Robustness
    base_mc = mc[mc["cost_regime"] == "base"]
    for model in base_mc["model_name"].unique():
        if model == "naive_momentum":
            continue
        mm = base_mc[base_mc["model_name"] == model]
        positive_thresholds = (mm["cumulative_return"] > 0).sum()
        gates[f"Threshold Robustness ({model})"] = (positive_thresholds >= 2, f"{positive_thresholds}/{len(mm)} thresholds positive")

    return gates
