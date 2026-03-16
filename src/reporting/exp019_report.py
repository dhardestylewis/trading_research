"""Report generator for exp019: Latent Cell Discovery.

Generates a markdown report with:
  - Feature summary
  - Cluster profiles
  - Model OOF performance per head/horizon
  - Cell economics ranking
  - Cell cards with rule descriptions
  - Gate results
  - Final verdict
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.utils.logging import get_logger

log = get_logger("exp019_report")


def _fmt_bps(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:+.1f}"


def _feature_section(summary: dict) -> str:
    lines = [
        "## Feature Summary\n",
        f"- **Rows**: {summary['n_rows']:,}",
        f"- **Feature columns**: {summary['n_features']}",
        f"- **Average NaN rate**: {summary['nan_rate']:.2%}",
        "",
    ]
    return "\n".join(lines)


def _cluster_section(state_result: dict) -> str:
    profiles = state_result["profiles"]
    sweep = state_result["sweep_results"]
    best_n = state_result["best_n_clusters"]
    best_score = state_result["best_score"]

    lines = [
        "## Latent State Discovery\n",
        f"**Best model**: n_clusters={best_n}, score={best_score:.1f}\n",
        "### Cluster Sweep\n",
        "| n_clusters | Score | Viable | Total |",
        "|-----------|-------|--------|-------|",
    ]
    for s in sweep:
        lines.append(
            f"| {s['n_clusters']} | {s['score']:.1f} | "
            f"{s['viable_clusters']} | {s['total_clusters']} |"
        )

    lines.append("\n### Cluster Profiles\n")
    lines.append("| Cluster | Samples | Frac | Assets | Top Asset | Top Frac | Viable |")
    lines.append("|---------|---------|------|--------|-----------|----------|--------|")

    for _, row in profiles.iterrows():
        lines.append(
            f"| {int(row['cluster_id'])} | {int(row['n_samples']):,} | "
            f"{row['frac_of_total']:.2%} | {row.get('n_assets', '?')} | "
            f"{row.get('top_asset', '?')} | {row.get('top_asset_frac', '?'):.0%} | "
            f"{'✓' if row['viable'] else '✗'} |"
        )

    return "\n".join(lines)


def _model_section(model_results: dict, horizons: list[str]) -> str:
    lines = ["## Multi-Head Model Performance\n"]

    for hz in horizons:
        if hz not in model_results:
            continue
        result = model_results[hz]
        heads = result["heads"]

        lines.append(f"### Horizon: {hz}\n")
        lines.append("| Head | OOF Corr | OOF MAE (bps) | Folds | Samples |")
        lines.append("|------|----------|---------------|-------|---------|")

        for name, head in heads.items():
            m = head.metrics
            lines.append(
                f"| {name} | {m.get('oof_correlation', 'N/A')} | "
                f"{m.get('oof_mae_bps', 'N/A')} | "
                f"{m.get('n_folds', 0)} | {m.get('n_oof_samples', 0):,} |"
            )

        # Top features for net_move head
        if "net_move_bps" in heads:
            imp = heads["net_move_bps"].feature_importance
            if len(imp) > 0:
                lines.append(f"\n**Top 15 features (net_move head, {hz})**:\n")
                for _, frow in imp.head(15).iterrows():
                    lines.append(f"- `{frow['feature']}`: {frow['importance']:.0f}")

        lines.append("")

    return "\n".join(lines)


def _cell_section(cell_result: dict) -> str:
    econ = cell_result["cell_economics"]
    cards = cell_result["cell_cards"]

    lines = [
        "## Cell Economics Ranking\n",
        "| Rank | Cluster | Samples | Median Net | Mean Net | Trim Mean | "
        "Pct Pos | Assets | Stressed Med |",
        "|------|---------|---------|------------|----------|-----------|"
        "---------|--------|-------------|",
    ]

    for rank, (_, row) in enumerate(econ.iterrows(), 1):
        lines.append(
            f"| {rank} | {int(row['cluster_id'])} | {int(row['n_samples']):,} | "
            f"{_fmt_bps(row['median_net_bps'])} | {_fmt_bps(row['mean_net_bps'])} | "
            f"{_fmt_bps(row['trimmed_mean_net_bps'])} | "
            f"{row['pct_positive']:.1f}% | {int(row['n_assets'])} | "
            f"{_fmt_bps(row['stressed_median_bps'])} |"
        )

    # Cell cards
    if cards:
        lines.append("\n## Cell Cards (Top Cells)\n")
        for card in cards:
            lines.append(f"### Cell {card['cluster_id']}\n")
            lines.append(f"- **Samples**: {card['n_samples']:,}")
            lines.append(f"- **Median net**: {_fmt_bps(card['median_net_bps'])} bps")
            lines.append(f"- **Trimmed mean net**: {_fmt_bps(card['trimmed_mean_net_bps'])} bps")
            lines.append(f"- **Pct positive**: {card['pct_positive']:.1f}%")
            lines.append(f"- **Assets**: {card['n_assets']}")
            lines.append(f"- **Stressed median**: {_fmt_bps(card['stressed_median_bps'])} bps")
            lines.append(f"\n**Approximate rule description**:\n```\n{card['rule_description']}\n```\n")

            if card.get("prototypes"):
                lines.append("**Prototype examples**:\n")
                lines.append("| Asset | Timestamp |")
                lines.append("|-------|-----------|")
                for p in card["prototypes"][:5]:
                    lines.append(f"| {p.get('asset', '?')} | {p.get('timestamp', '?')} |")
            lines.append("")

    return "\n".join(lines)


def _gate_section(gate_result: dict) -> str:
    gdf = gate_result["gate_results"]
    adv = gate_result["advancing_cells"]
    killed = gate_result["killed_cells"]
    verdict = gate_result["verdict"]

    lines = [
        "## Economic Kill Gates\n",
        "| Cluster | Dedup N | Trades | Median | Trim Mean | Assets | "
        "Conc | Stress | ALL |",
        "|---------|---------|--------|--------|-----------|--------|"
        "-----|--------|-----|",
    ]

    for _, row in gdf.iterrows():
        def _check(v):
            return "✓" if v else "✗"

        lines.append(
            f"| {int(row['cluster_id'])} | {int(row['n_deduped_trades'])} | "
            f"{_check(row['gate_trade_count'])} | "
            f"{_check(row['gate_median_positive'])} | "
            f"{_check(row['gate_trimmed_mean_positive'])} | "
            f"{_check(row['gate_asset_diversity'])} | "
            f"{_check(row['gate_no_concentration'])} | "
            f"{_check(row['gate_stress_test'])} | "
            f"**{_check(row['all_gates_pass'])}** |"
        )

    lines.append(f"\n**Advancing**: {adv if adv else 'None'}")
    lines.append(f"**Killed**: {killed if killed else 'None'}")
    lines.append(f"\n> **Verdict**: {verdict}")

    return "\n".join(lines)


def generate_report(
    feature_summary: dict,
    state_result: dict,
    model_results: dict,
    cell_result: dict,
    gate_result: dict,
    horizons: list[str],
    output_dir: Path,
    report_dir: Path,
) -> Path:
    """Generate the full exp019 markdown report."""
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = [
        f"# Exp019 — Latent Cell Discovery Report\n",
        f"*Generated: {ts}*\n",
        "---\n",
        _feature_section(feature_summary),
        "---\n",
        _cluster_section(state_result),
        "---\n",
        _model_section(model_results, horizons),
        "---\n",
        _cell_section(cell_result),
        "---\n",
        _gate_section(gate_result),
    ]

    report = "\n".join(sections)
    report_path = report_dir / "exp019_report.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("Report written: %s (%d chars)", report_path, len(report))
    return report_path
