"""Exp024 Report Generator — Execution Cost Surface and Alpha-Cost Join.

Generates 6 tables as required by the experiment specification:
  Table 1: Alpha-cost gap (asset × horizon × bucket)
  Table 2: Cost model performance (OOS corr, MAE, bucket sep)
  Table 3: Low-cost state cards
  Table 4: Alpha-cost join comparison (gross vs cost vs joint ranking)
  Table 5: Freshness (window × cadence)
  Table 6: Replay vs live paper
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger("exp024_report")


def _save_table(df: pd.DataFrame, report_dir: Path, name: str):
    """Save a table as both CSV and formatted text."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    txt_path = report_dir / f"{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"  {name.replace('_', ' ').upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    logger.info("Saved table: %s", csv_path)


def _build_table1_alpha_cost_gap(
    alpha_cost_join: pd.DataFrame,
    surface: pd.DataFrame,
) -> pd.DataFrame:
    """Table 1 — Alpha-cost gap: the most important table.

    Asset × horizon × bucket: predicted gross, realized gross,
    predicted cost, realized cost, net realized.
    """
    if alpha_cost_join.empty:
        logger.warning("Empty alpha-cost join; Table 1 will be empty")
        return pd.DataFrame()

    # The join table already contains the key columns
    # Rename for presentation clarity
    cols_to_show = [
        c for c in ["ranking", "bucket", "n_obs", "pred_gross_1s",
                     "realized_gross_1s", "pred_cost", "realized_cost",
                     "net_markout_median", "net_markout_trimmed_mean"]
        if c in alpha_cost_join.columns
    ]
    return alpha_cost_join[cols_to_show].copy() if cols_to_show else alpha_cost_join


def _build_table2_cost_model_performance(
    model_results: Dict[str, Any],
) -> pd.DataFrame:
    """Table 2 — Cost model performance."""
    rows = []
    for key, res in model_results.items():
        avg = res.get("aggregate_metrics", {})
        rows.append({
            "target_model": key,
            "mean_spearman": avg.get("mean_spearman", None),
            "mean_mae": avg.get("mean_mae", None),
            "mean_decile_sep": avg.get("mean_decile_sep", None),
            "n_folds": avg.get("n_folds", 0),
        })
    return pd.DataFrame(rows)


def _build_table3_state_cards(
    state_cards: pd.DataFrame,
) -> pd.DataFrame:
    """Table 3 — Low-cost execution state cards."""
    if state_cards.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "cluster", "sample_size", "median_shortfall_bps", "mean_shortfall_bps",
        "asset_mix", "median_spread_bps", "median_hour", "weekend_frac",
        "prototype_timestamps",
    ] if c in state_cards.columns]
    return state_cards[cols].copy() if cols else state_cards


def _build_table4_join_comparison(
    alpha_cost_join: pd.DataFrame,
) -> pd.DataFrame:
    """Table 4 — Compare gross-only vs cost-only vs joint ranking."""
    if alpha_cost_join.empty:
        return pd.DataFrame()

    top_bucket_rows = []
    for ranking in alpha_cost_join["ranking"].unique() if "ranking" in alpha_cost_join.columns else []:
        sub = alpha_cost_join[alpha_cost_join["ranking"] == ranking]
        if "bucket" in sub.columns:
            top = sub[sub["bucket"] == sub["bucket"].max()]
            if not top.empty:
                row = top.iloc[0].to_dict()
                row["ranking_strategy"] = ranking
                top_bucket_rows.append(row)

    return pd.DataFrame(top_bucket_rows)


def _build_table5_freshness(
    freshness_df: pd.DataFrame,
) -> pd.DataFrame:
    """Table 5 — Freshness study: window × cadence."""
    if freshness_df.empty:
        return pd.DataFrame()
    cols = [c for c in [
        "train_window", "refresh_cadence", "mean_spearman", "std_spearman",
        "mean_mae", "n_folds", "stability",
    ] if c in freshness_df.columns]
    return freshness_df[cols].copy() if cols else freshness_df


def _build_table6_replay_vs_live(
    replay_vs_live: pd.DataFrame,
) -> pd.DataFrame:
    """Table 6 — Replay vs live paper."""
    return replay_vs_live.copy() if not replay_vs_live.empty else pd.DataFrame()


def _evaluate_gates(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
    table5: pd.DataFrame,
    table6: pd.DataFrame,
    state_cards: pd.DataFrame,
) -> Dict[str, Dict]:
    """Evaluate the 5 hard gates and produce verdicts."""
    gates = {}

    # Gate 1: Cost model usefulness
    gate1_pass = False
    if not table2.empty and "mean_spearman" in table2.columns:
        # Check if any model beats quantile baseline
        baselines = table2[table2["target_model"].str.contains("quantile", case=False, na=False)]
        models = table2[~table2["target_model"].str.contains("quantile", case=False, na=False)]
        if not baselines.empty and not models.empty:
            best_model_sp = models["mean_spearman"].max()
            best_baseline_sp = baselines["mean_spearman"].max()
            gate1_pass = best_model_sp > best_baseline_sp + 0.02
        elif not models.empty:
            gate1_pass = models["mean_spearman"].max() > 0.1
    gates["gate_1_cost_model_usefulness"] = {
        "pass": gate1_pass,
        "detail": "ML model Spearman vs baseline",
    }

    # Gate 2: Cost compression
    gate2_pass = False
    if not state_cards.empty and "median_shortfall_bps" in state_cards.columns:
        unconditional_median = state_cards["median_shortfall_bps"].median()
        cheapest_median = state_cards["median_shortfall_bps"].min()
        if unconditional_median > 0:
            compression = 1 - (cheapest_median / unconditional_median)
            gate2_pass = compression >= 0.50
            gates["gate_2_cost_compression"] = {
                "pass": gate2_pass,
                "compression_pct": float(compression),
                "cheapest_bps": float(cheapest_median),
                "unconditional_bps": float(unconditional_median),
            }
        else:
            gates["gate_2_cost_compression"] = {"pass": False, "detail": "Non-positive unconditional median"}
    else:
        gates["gate_2_cost_compression"] = {"pass": False, "detail": "No state cards"}

    # Gate 3: Alpha-cost salvage
    gate3_pass = False
    if not table1.empty:
        joint_rows = table1[table1["ranking"] == "joint"] if "ranking" in table1.columns else table1
        if not joint_rows.empty and "net_markout_median" in joint_rows.columns:
            best_median = joint_rows["net_markout_median"].max()
            best_trimmed = joint_rows.get("net_markout_trimmed_mean", pd.Series([0])).max()
            gate3_pass = best_median > 0 and best_trimmed > 0
    gates["gate_3_alpha_cost_salvage"] = {
        "pass": gate3_pass,
        "detail": "Joint high-alpha/low-cost bucket positive median+trimmed mean",
    }

    # Gate 4: Live paper consistency
    gate4_pass = False
    if not table6.empty and "gap_bps" in table6.columns:
        max_gap = table6["gap_bps"].abs().max()
        gate4_pass = max_gap < 5.0  # within 5 bps tolerance
    gates["gate_4_live_paper_consistency"] = {
        "pass": gate4_pass,
        "detail": "Replay-vs-live gap within tolerance",
    }

    # Gate 5: Refresh stability
    gate5_pass = False
    if not table5.empty and "stability" in table5.columns:
        gate5_pass = table5["stability"].max() > 0.5
    gates["gate_5_refresh_stability"] = {
        "pass": gate5_pass,
        "detail": "Stable training-window/refresh-cadence exists",
    }

    return gates


def generate_all_reports(
    surface: pd.DataFrame,
    model_results: Dict[str, Any],
    alpha_cost_join: pd.DataFrame,
    state_cards: pd.DataFrame,
    freshness_df: pd.DataFrame,
    replay_vs_live: pd.DataFrame,
    report_dir: str = "reports/exp024",
) -> str:
    """Generate all 6 tables and the summary report.

    Returns the path to the summary report.
    """
    rd = Path(report_dir)
    rd.mkdir(parents=True, exist_ok=True)

    # Build tables
    t1 = _build_table1_alpha_cost_gap(alpha_cost_join, surface)
    t2 = _build_table2_cost_model_performance(model_results)
    t3 = _build_table3_state_cards(state_cards)
    t4 = _build_table4_join_comparison(alpha_cost_join)
    t5 = _build_table5_freshness(freshness_df)
    t6 = _build_table6_replay_vs_live(replay_vs_live)

    # Save all tables
    for name, df in [
        ("table1_alpha_cost_gap", t1),
        ("table2_cost_model_performance", t2),
        ("table3_low_cost_state_cards", t3),
        ("table4_alpha_cost_join_comparison", t4),
        ("table5_freshness", t5),
        ("table6_replay_vs_live", t6),
    ]:
        if not df.empty:
            _save_table(df, rd, name)

    # Evaluate gates
    gates = _evaluate_gates(t1, t2, t5, t6, state_cards)

    # Write summary report
    report_path = rd / "exp024_summary.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  EXP024 — EXECUTION COST SURFACE AND ALPHA-COST JOIN\n")
        f.write("=" * 80 + "\n\n")

        f.write("BRANCH VERDICT RULE:\n")
        f.write("  If no joint high-alpha/low-cost bucket has positive median\n")
        f.write("  AND trimmed-mean net markout, the directional microstructure\n")
        f.write("  alpha program is CLOSED.\n\n")

        f.write("-" * 80 + "\n")
        f.write("  GATE VERDICTS\n")
        f.write("-" * 80 + "\n\n")
        all_pass = True
        for gate_name, gate_result in gates.items():
            status = "PASS" if gate_result["pass"] else "FAIL"
            f.write(f"  {gate_name}: {status}\n")
            for k, v in gate_result.items():
                if k != "pass":
                    f.write(f"    {k}: {v}\n")
            f.write("\n")
            if not gate_result["pass"]:
                all_pass = False

        f.write("-" * 80 + "\n")
        f.write("  OVERALL VERDICT\n")
        f.write("-" * 80 + "\n\n")

        gate3 = gates.get("gate_3_alpha_cost_salvage", {})
        if gate3.get("pass"):
            f.write("  SUCCESS A — ALPHA SALVAGE\n")
            f.write("  Joint high-alpha/low-cost bucket has positive net expectancy.\n")
            f.write("  exp025 → live paper execution-gated short-horizon alpha.\n")
        else:
            gate1 = gates.get("gate_1_cost_model_usefulness", {})
            if gate1.get("pass"):
                f.write("  SUCCESS B — EXECUTION INTELLIGENCE PRODUCT\n")
                f.write("  Cost model is learnable but alpha salvage fails.\n")
                f.write("  Directional microstructure alpha program CLOSED.\n")
                f.write("  Execution cost model survives as standalone asset.\n")
            else:
                f.write("  FAIL — DIRECTIONAL BRANCH CLOSED\n")
                f.write("  No viable alpha-cost cell found. Cost model not useful enough.\n")
                f.write("  Short-horizon directional alpha program PERMANENTLY CLOSED.\n")

        # Append tables inline
        for name, df in [
            ("TABLE 1: ALPHA-COST GAP", t1),
            ("TABLE 2: COST MODEL PERFORMANCE", t2),
            ("TABLE 3: LOW-COST STATE CARDS", t3),
            ("TABLE 4: ALPHA-COST JOIN COMPARISON", t4),
            ("TABLE 5: FRESHNESS", t5),
            ("TABLE 6: REPLAY VS LIVE PAPER", t6),
        ]:
            f.write(f"\n{'='*80}\n  {name}\n{'='*80}\n\n")
            if not df.empty:
                f.write(df.to_string(index=False))
            else:
                f.write("  (no data)")
            f.write("\n")

    logger.info("Report saved to %s", report_path)
    return str(report_path)
