"""Run exp018: RV Rule Audit (Diagnostic Salvage).

Pipeline:
1. Load exp017 trade data and spread panel
2. Run trade deduplicator (cross-horizon + temporal)
3. Run horizon audit on original trades
4. Run central tendency audit on deduplicated trades
5. Generate regression-based direction signals → simulate new trades
6. Run corrected pair-level gates on both rule families
7. Generate report with full audit and final verdict
"""
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp017.spread_definitions import compute_spreads
from src.research.exp017.trade_simulator import simulate_trades
from src.research.exp018.trade_deduplicator import full_dedup_pipeline
from src.research.exp018.horizon_audit import run_horizon_audit
from src.research.exp018.central_tendency_audit import run_central_tendency_audit
from src.research.exp018.regression_direction import generate_regression_signals
from src.research.exp018.diagnostic_gates import compare_rule_families
from src.reporting.exp018_report import generate_report
from src.utils.logging import get_logger

# Re-use event mask builder from exp017
from run_exp017 import _build_event_mask

log = get_logger("run_exp018")


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    spread_cfg = cfg["spread"]
    events_cfg = cfg["events"]
    regression_cfg = cfg["regression_rule"]
    exec_cfg = cfg["execution"]
    audit_cfg = cfg["audit"]
    gates_cfg = cfg["gates"]

    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load exp017 trades ──────────────────────────────
    trades_path = Path(data_cfg["exp017_trades_path"])
    log.info(f"Loading exp017 trades from {trades_path}...")
    trades_df = pd.read_parquet(trades_path)
    log.info(f"Loaded {len(trades_df)} trades, "
             f"{trades_df['pair'].nunique()} pairs")

    # ── Step 2: Trade deduplication ─────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("BRANCH A: Trade Deduplication")
    log.info("=" * 60)

    dedup_result = full_dedup_pipeline(
        trades_df, min_gap_bars=audit_cfg["min_gap_bars"])

    deduped_trades = dedup_result["deduplicated_trades"]
    log.info(f"Dedup pipeline: {dedup_result['original_count']} → "
             f"{dedup_result['after_horizon_collapse']} → "
             f"{dedup_result['after_temporal_dedup']}")

    # ── Step 3: Horizon audit ───────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("BRANCH B: Horizon Sensitivity Audit")
    log.info("=" * 60)

    horizon_result = run_horizon_audit(trades_df)
    hs = horizon_result["summary"]
    if hs:
        log.info(f"Horizon audit: {hs.get('pct_identical', '?')}% of "
                 f"horizon pairs are statistically identical")
        log.info(f"Exit breakdown: {hs.get('overall_pct_stop', '?')}% stop, "
                 f"{hs.get('overall_pct_tp', '?')}% TP, "
                 f"{hs.get('overall_pct_horizon', '?')}% horizon")

    # ── Step 4: Central tendency audit ──────────────────────────
    log.info("\n" + "=" * 60)
    log.info("BRANCH C: Central Tendency Audit")
    log.info("=" * 60)

    tendency_result = run_central_tendency_audit(deduped_trades)
    flags = tendency_result["flags"]
    log.info(f"Central tendency flags: {flags}")

    # ── Step 5: Regression direction signals ────────────────────
    log.info("\n" + "=" * 60)
    log.info("BRANCH D: Regression Direction Rule")
    log.info("=" * 60)

    # Reload panel and compute spreads for regression
    panel_path = Path(data_cfg["panel_path"])
    log.info(f"Loading panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)

    spread_df = compute_spreads(
        panel,
        data_cfg["rv_pairs"],
        spread_cfg["definitions"],
        beta_lookback=spread_cfg["beta_lookback"],
        zscore_lookback=spread_cfg["zscore_lookback"],
    )

    regression_trades_list = []
    pairs = data_cfg["rv_pairs"]

    for pair in pairs:
        pair_label = f"{pair[0]}/{pair[1]}"
        pair_spread = spread_df[spread_df["pair"] == pair_label].copy()
        pair_spread = pair_spread.sort_values("timestamp").reset_index(drop=True)

        if pair_spread.empty:
            log.warning(f"No spread data for {pair_label}")
            continue

        # Build event mask
        event_mask = _build_event_mask(pair_spread, panel, events_cfg, pair)

        # Generate regression signals
        signals = generate_regression_signals(
            pair_spread, spread_cfg["definitions"],
            regression_cfg, event_mask
        )

        if signals.empty:
            log.warning(f"No regression signals for {pair_label}")
            continue

        log.info(f"Generated {len(signals)} regression signals for {pair_label}")

        # Simulate trades using same execution model
        reg_trades = simulate_trades(pair_spread, signals, exec_cfg)

        if not reg_trades.empty:
            regression_trades_list.append(reg_trades)
            log.info(f"Simulated {len(reg_trades)} regression trades for {pair_label}")

    if regression_trades_list:
        regression_trades_df = pd.concat(regression_trades_list, ignore_index=True)
        log.info(f"\nTotal regression trades: {len(regression_trades_df)}")
    else:
        regression_trades_df = pd.DataFrame()
        log.warning("No regression trades generated")

    # Dedup the regression trades too
    if not regression_trades_df.empty:
        reg_dedup = full_dedup_pipeline(
            regression_trades_df, min_gap_bars=audit_cfg["min_gap_bars"])
        regression_deduped = reg_dedup["deduplicated_trades"]
        log.info(f"Regression dedup: {reg_dedup['original_count']} → "
                 f"{reg_dedup['after_temporal_dedup']}")
    else:
        regression_deduped = pd.DataFrame()

    # ── Step 6: Corrected gates on both families ────────────────
    log.info("\n" + "=" * 60)
    log.info("CORRECTED PAIR-LEVEL GATE EVALUATION")
    log.info("=" * 60)

    gate_comparison = compare_rule_families(
        deduped_trades, regression_deduped, gates_cfg)

    log.info(f"Heuristic verdict: {gate_comparison['heuristic_gates']['verdict']}")
    log.info(f"Regression verdict: {gate_comparison['regression_gates']['verdict']}")

    # Print pair-level detail
    for label, gates in [("HEURISTIC", gate_comparison["heuristic_gates"]),
                          ("REGRESSION", gate_comparison["regression_gates"])]:
        log.info(f"\n  {label} pair details:")
        for pair, detail in gates["pair_details"].items():
            log.info(f"    {pair}: n={detail['n_trades']}, "
                     f"median={detail['median_net_bps']}, "
                     f"mean={detail['mean_net_bps']}, "
                     f"frac_pos={detail['frac_positive']}")

    # ── Step 7: Final verdict ───────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FINAL VERDICT")
    log.info("=" * 60)

    heuristic_pass = gate_comparison["heuristic_gates"]["overall_pass"]
    regression_pass = gate_comparison["regression_gates"]["overall_pass"]

    if regression_pass:
        final_verdict = ("SALVAGEABLE — regression rule produces positive pair-level "
                         "economics. Direction rules were the problem, not the opportunity.")
    elif heuristic_pass:
        final_verdict = ("PARTIAL — heuristic rules pass after dedup. Original failure "
                         "was primarily a measurement artifact.")
    else:
        any_regression_positive = any(
            d.get("passes_median", False)
            for d in gate_comparison["regression_gates"]["pair_details"].values()
        )
        if any_regression_positive:
            final_verdict = ("MARGINAL — some pairs show improvement with regression rule "
                             "but not enough to pass. Further work may be warranted on "
                             "direction identification.")
        else:
            final_verdict = ("KILL — neither heuristic nor regression rules produce "
                             "positive pair-level economics. The RV excursion opportunity "
                             "is fundamentally not capturable with ex-ante direction locking. "
                             "Close RV directional branch.")

    log.info(f"  {final_verdict}")

    # ── Step 8: Generate report ─────────────────────────────────
    report_path = generate_report(
        dedup_result=dedup_result,
        horizon_result=horizon_result,
        tendency_result=tendency_result,
        gate_comparison=gate_comparison,
        regression_deduped=regression_deduped,
        final_verdict=final_verdict,
        output_dir=out_dir,
    )
    log.info(f"\nReport saved to: {report_path}")

    # Save key outputs
    if not deduped_trades.empty:
        deduped_trades.to_parquet(out_dir / "heuristic_deduped.parquet", index=False)
    if not regression_deduped.empty:
        regression_deduped.to_parquet(out_dir / "regression_deduped.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "configs/experiments/crypto_1h_exp018.yaml")
