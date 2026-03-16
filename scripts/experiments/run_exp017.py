"""Run exp017: Direction-Locked RV Execution Study.

Pipeline:
1. Load panel parquet
2. Build ratio panel (reuse exp016)
3. Compute spread definitions (4 types)
4. Extract event states (reuse exp015)
5. Generate direction-locked signals from each rule
6. Simulate trades with two-leg execution model
7. Aggregate PnL and build summary tables
8. Generate report with side-by-side comparison
9. Evaluate kill gates and write final verdict
"""
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.research.exp016.relative_value_scanner import build_ratio_panel
from src.research.exp015.event_extraction import extract_events, add_atr
from src.research.exp017.spread_definitions import compute_spreads
from src.research.exp017.direction_rules import generate_direction_signals
from src.research.exp017.trade_simulator import simulate_trades
from src.research.exp017.spread_pnl import (
    build_trade_summary,
    build_excursion_comparison,
    evaluate_kill_gates,
)
from src.reporting.exp017_report import generate_report
from src.utils.logging import get_logger

log = get_logger("run_exp017")


def _build_event_mask(spread_pair_df: pd.DataFrame, panel: pd.DataFrame,
                      events_cfg: dict, pair: list[str]) -> np.ndarray:
    """Build event mask for a pair's spread panel rows.

    Apply vol_expansion and range_breakout events on the BASE asset,
    then align to the spread panel timestamps.
    """
    col_key = "asset" if "asset" in panel.columns else "symbol"
    base = pair[0]
    base_label = (f"{base}-USD" if f"{base}-USD" in panel[col_key].values
                  else f"{base}-USDT")

    base_panel = panel[panel[col_key] == base_label].copy()
    if base_panel.empty:
        log.warning(f"No base panel data for {base_label}")
        return np.ones(len(spread_pair_df), dtype=bool)

    base_panel = base_panel.sort_values("timestamp").reset_index(drop=True)

    # Vol expansion event
    mask_vol = np.zeros(len(base_panel), dtype=bool)
    if "vol_expansion" in events_cfg:
        cfg = events_cfg["vol_expansion"]
        atr = add_atr(base_panel, cfg["atr_window"])
        baseline = atr.rolling(cfg["baseline_window"]).mean()
        mask_vol = (atr > cfg["multiplier"] * baseline).values

    # Range breakout event
    mask_brk = np.zeros(len(base_panel), dtype=bool)
    if "range_breakout" in events_cfg:
        cfg = events_cfg["range_breakout"]
        rolling_high = base_panel["high"].shift(1).rolling(cfg["window"]).max()
        rolling_low = base_panel["low"].shift(1).rolling(cfg["window"]).min()
        mask_brk = ((base_panel["close"] > rolling_high) |
                    (base_panel["close"] < rolling_low)).values

    # Combined: either event active
    combined = mask_vol | mask_brk

    # Create a timestamp->event lookup
    event_lookup = dict(zip(base_panel["timestamp"].values, combined))

    # Align to spread panel timestamps
    aligned_mask = np.array([
        event_lookup.get(ts, False)
        for ts in spread_pair_df["timestamp"].values
    ], dtype=bool)

    active_count = aligned_mask.sum()
    log.info(f"Event mask for {base}: {active_count}/{len(aligned_mask)} "
             f"bars active ({active_count/len(aligned_mask)*100:.1f}%)")
    return aligned_mask


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    spread_cfg = cfg["spread"]
    events_cfg = cfg["events"]
    rules_cfg = cfg["direction_rules"]
    exec_cfg = cfg["execution"]
    gates_cfg = cfg["gates"]

    panel_path = Path(data_cfg["panel_path"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load panel
    log.info(f"Loading panel from {panel_path}...")
    panel = pd.read_parquet(panel_path)
    log.info(f"Panel loaded: {len(panel)} rows, "
             f"{panel['asset' if 'asset' in panel.columns else 'symbol'].nunique()} assets")

    # 2. Compute spread definitions
    log.info("Computing spread definitions...")
    spread_df = compute_spreads(
        panel,
        data_cfg["rv_pairs"],
        spread_cfg["definitions"],
        beta_lookback=spread_cfg["beta_lookback"],
        zscore_lookback=spread_cfg["zscore_lookback"],
    )

    if spread_df.empty:
        log.error("No spread data computed. Exiting.")
        return

    log.info(f"Spread panel: {len(spread_df)} rows, "
             f"{spread_df['pair'].nunique()} pairs")

    # 3-6. For each pair: extract events, generate signals, simulate trades
    all_trades = []
    pairs = data_cfg["rv_pairs"]

    for pair in pairs:
        pair_label = f"{pair[0]}/{pair[1]}"
        log.info(f"\n{'='*60}")
        log.info(f"Processing pair: {pair_label}")
        log.info(f"{'='*60}")

        pair_spread = spread_df[spread_df["pair"] == pair_label].copy()
        pair_spread = pair_spread.sort_values("timestamp").reset_index(drop=True)

        if pair_spread.empty:
            log.warning(f"No spread data for {pair_label}, skipping")
            continue

        # Build event mask from underlying asset events
        event_mask = _build_event_mask(pair_spread, panel, events_cfg, pair)

        # Generate direction-locked signals
        signals = generate_direction_signals(
            pair_spread, rules_cfg, spread_cfg["definitions"], event_mask
        )

        if signals.empty:
            log.warning(f"No signals generated for {pair_label}")
            continue

        log.info(f"Generated {len(signals)} signals for {pair_label}")

        # Simulate trades
        trades = simulate_trades(pair_spread, signals, exec_cfg)

        if not trades.empty:
            all_trades.append(trades)
            log.info(f"Simulated {len(trades)} trades for {pair_label}")
        else:
            log.warning(f"No trades simulated for {pair_label}")

    # 7. Aggregate
    if not all_trades:
        log.error("No trades generated across any pair. Report will be empty.")
        all_trades_df = pd.DataFrame()
    else:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
        log.info(f"\nTotal trades across all pairs: {len(all_trades_df)}")

    # Build summary
    summary_df = build_trade_summary(all_trades_df)

    # Build side-by-side comparison
    comparison_df = build_excursion_comparison(
        all_trades_df, spread_df, exec_cfg["holding_horizons_hours"]
    )

    # 8. Evaluate kill gates
    kill_gates = evaluate_kill_gates(summary_df, gates_cfg)

    log.info("\n" + "="*60)
    log.info("KILL GATE RESULTS:")
    for key, val in kill_gates.items():
        if isinstance(val, dict) and "pass" in val:
            status = "PASS" if val["pass"] else "FAIL"
            log.info(f"  {key}: {val['value']} vs {val['threshold']} → {status}")
        elif key == "overall_pass":
            log.info(f"  OVERALL: {'PASS' if val else 'FAIL'}")
    log.info("="*60)

    # 9. Generate report
    report_path = generate_report(
        all_trades_df, summary_df, comparison_df, kill_gates, out_dir
    )
    log.info(f"\nReport saved to: {report_path}")

    # Also save raw data
    if not all_trades_df.empty:
        all_trades_df.to_parquet(out_dir / "all_trades.parquet", index=False)
    if not summary_df.empty:
        summary_df.to_parquet(out_dir / "summary_stats.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "configs/experiments/crypto_1h_exp017.yaml")
