"""
spike_tsfm_chronos.py — Chronos Trajectory Forecasting PoC on SOL-USD

Tests whether Chronos can produce useful multi-step price trajectory 
forecasts that could be used for entry/exit timing.

Run with: .venv_chronos\Scripts\python spike_tsfm_chronos.py
"""
import logging
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("spike_chronos")

FORECAST_HORIZON = 8   # 8 bars = 8 hours
CONTEXT_LENGTH = 256   # how much history to feed
COST_BPS = 14


def load_sol_data() -> pd.DataFrame:
    """Load SOL-USD hourly data from the panel."""
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy()
    sol = sol.sort_values('timestamp').reset_index(drop=True)
    log.info(f"Loaded {len(sol)} SOL-USD bars")
    return sol


def run_chronos_backtest(sol: pd.DataFrame, n_windows: int = 100) -> pd.DataFrame:
    """
    Walk-forward backtest of Chronos trajectory forecasts.
    """
    from chronos import ChronosPipeline
    
    log.info("Loading Chronos model...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    log.info("Chronos model loaded")
    
    closes = sol['close'].values
    timestamps = sol['timestamp'].values
    
    start_idx = CONTEXT_LENGTH
    end_idx = len(closes) - FORECAST_HORIZON
    
    if end_idx - start_idx < n_windows:
        n_windows = end_idx - start_idx
    
    window_indices = np.linspace(start_idx, end_idx - 1, n_windows, dtype=int)
    results = []
    
    for i, idx in enumerate(window_indices):
        context = closes[idx - CONTEXT_LENGTH:idx]
        realized = closes[idx:idx + FORECAST_HORIZON]
        
        if len(realized) < FORECAST_HORIZON:
            continue
        
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        entry_price = context[-1]
        
        t0 = time.time()
        forecast = pipeline.predict(
            context_tensor,
            prediction_length=FORECAST_HORIZON,
            num_samples=20,
        )
        inference_ms = (time.time() - t0) * 1000
        
        # forecast shape: (1, num_samples, prediction_length)
        samples = forecast[0].numpy()
        median_forecast = np.median(samples, axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)
        
        # Metrics
        predicted_return = (median_forecast[-1] - entry_price) / entry_price
        realized_return = (realized[-1] - entry_price) / entry_price
        direction_correct = (predicted_return > 0) == (realized_return > 0)
        
        # Peak timing: when does the predicted trajectory peak vs realized?
        pred_peak_bar = np.argmax(np.abs(median_forecast - entry_price))
        real_peak_bar = np.argmax(np.abs(realized - entry_price))
        peak_timing_error = abs(pred_peak_bar - real_peak_bar)
        
        # Trajectory-optimal PnL: enter at predicted low, exit at predicted high
        if predicted_return > 0:
            pred_entry_bar = np.argmin(median_forecast)
            pred_exit_bar = np.argmax(median_forecast[pred_entry_bar:]) + pred_entry_bar
            if pred_exit_bar > pred_entry_bar:
                # What would realized PnL be if you entered/exited at those bars?
                traj_pnl_bps = (realized[pred_exit_bar] - realized[pred_entry_bar]) / entry_price * 10000
            else:
                traj_pnl_bps = realized_return * 10000
        else:
            traj_pnl_bps = realized_return * 10000
        
        # Simple fixed-horizon PnL
        simple_gross_bps = realized_return * 10000
        
        # Confidence width
        ci_width_bps = np.mean((p90 - p10) / entry_price * 10000)
        
        results.append({
            'window': i,
            'timestamp': timestamps[idx],
            'entry_price': entry_price,
            'predicted_return_bps': predicted_return * 10000,
            'realized_return_bps': realized_return * 10000,
            'direction_correct': direction_correct,
            'peak_timing_error_bars': peak_timing_error,
            'traj_optimal_pnl_bps': traj_pnl_bps,
            'simple_fixed_pnl_bps': simple_gross_bps,
            'ci_width_bps': ci_width_bps,
            'inference_ms': inference_ms,
        })
        
        if (i + 1) % 10 == 0:
            dir_so_far = np.mean([r['direction_correct'] for r in results])
            log.info(f"Window {i+1}/{n_windows}: dir_acc={dir_so_far:.1%}, "
                     f"pred={predicted_return*10000:.1f}bps, "
                     f"real={realized_return*10000:.1f}bps, "
                     f"timing_err={peak_timing_error}bars, "
                     f"{inference_ms:.0f}ms")
    
    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame, output_dir: str = "reports/spike_tsfm") -> str:
    """Generate the Chronos spike report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "chronos_backtest_results.csv", index=False)
    
    dir_acc = results['direction_correct'].mean()
    mean_timing_err = results['peak_timing_error_bars'].mean()
    mean_traj_pnl = results['traj_optimal_pnl_bps'].mean()
    mean_fixed_pnl = results['simple_fixed_pnl_bps'].mean()
    mean_ci = results['ci_width_bps'].mean()
    mean_inference = results['inference_ms'].mean()
    
    # High-confidence subset (tightest CI decile)
    results['ci_decile'] = pd.qcut(results['ci_width_bps'], 10, labels=False, duplicates='drop')
    tight = results[results['ci_decile'] == 0]
    
    lines = [
        "# Chronos Trajectory Forecasting Spike — SOL-USD",
        "",
        "## 1. Overall Metrics",
        f"- Windows tested: **{len(results)}**",
        f"- Direction accuracy: **{dir_acc:.1%}** (random = 50%)",
        f"- Mean peak timing error: **{mean_timing_err:.1f} bars** (out of 8)",
        f"- Mean trajectory-optimal PnL: **{mean_traj_pnl:.1f} bps**",
        f"- Mean fixed-horizon PnL: **{mean_fixed_pnl:.1f} bps**",
        f"- Trajectory vs Fixed advantage: **{mean_traj_pnl - mean_fixed_pnl:.1f} bps**",
        f"- Mean CI width: **{mean_ci:.1f} bps**",
        f"- Mean inference time: **{mean_inference:.0f} ms**",
        "",
        "## 2. High-Confidence Subset (Tightest 10% CI)",
    ]
    if len(tight) > 0:
        lines.extend([
            f"- Trades: **{len(tight)}**",
            f"- Direction accuracy: **{tight['direction_correct'].mean():.1%}**",
            f"- Mean traj-optimal PnL: **{tight['traj_optimal_pnl_bps'].mean():.1f} bps**",
        ])
    
    lines.extend(["", "## 3. Verdict"])
    
    if dir_acc > 0.55 and mean_timing_err < 3:
        lines.append("**PROMISING** — Chronos shows directional and timing edge.")
    elif dir_acc > 0.52:
        lines.append("**MARGINAL** — Slight directional edge. Needs enrichment.")
    else:
        lines.append("**NO EDGE** — Raw Chronos does not outperform random on SOL-USD.")
    
    report_path = out / "chronos_spike_report.md"
    report_path.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    sol = load_sol_data()
    results = run_chronos_backtest(sol, n_windows=100)
    report = generate_report(results)
    log.info(f"Done: {report}")
