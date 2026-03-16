"""
spike_chronos_large.py — Chronos-T5-Large on SOL-USD

Same backtest as the small model, but using chronos-t5-large (710M params)
to see if a bigger model improves directional accuracy beyond 54%.

Run with: .venv_chronos\Scripts\python spike_chronos_large.py
"""
import logging
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("spike_chronos_large")

FORECAST_HORIZON = 8
CONTEXT_LENGTH = 256
COST_BPS = 14


def load_sol_data() -> pd.DataFrame:
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy()
    sol = sol.sort_values('timestamp').reset_index(drop=True)
    log.info(f"Loaded {len(sol)} SOL-USD bars")
    return sol


def run_backtest(sol: pd.DataFrame, n_windows: int = 100) -> pd.DataFrame:
    from chronos import ChronosPipeline

    log.info("Loading Chronos-T5-Large (710M params)...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large",
        device_map="cpu",
        torch_dtype=torch.float32,
    )
    log.info("Model loaded")

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

        entry_price = context[-1]
        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        t0 = time.time()
        forecast = pipeline.predict(ctx_tensor, prediction_length=FORECAST_HORIZON, num_samples=20)
        inference_ms = (time.time() - t0) * 1000

        samples = forecast[0].numpy()
        median_fc = np.median(samples, axis=0)
        p10 = np.percentile(samples, 10, axis=0)
        p90 = np.percentile(samples, 90, axis=0)

        pred_ret = (median_fc[-1] - entry_price) / entry_price
        real_ret = (realized[-1] - entry_price) / entry_price
        direction_correct = (pred_ret > 0) == (real_ret > 0)

        pred_peak = np.argmax(np.abs(median_fc - entry_price))
        real_peak = np.argmax(np.abs(realized - entry_price))
        timing_err = abs(pred_peak - real_peak)

        # Trajectory-optimal PnL
        if pred_ret > 0:
            ent = np.argmin(median_fc)
            ext = np.argmax(median_fc[ent:]) + ent
            traj_pnl = (realized[ext] - realized[ent]) / entry_price * 10000 if ext > ent else real_ret * 10000
        else:
            traj_pnl = real_ret * 10000

        ci_width = np.mean((p90 - p10) / entry_price * 10000)

        results.append({
            'window': i, 'timestamp': timestamps[idx], 'entry_price': entry_price,
            'predicted_return_bps': pred_ret * 10000, 'realized_return_bps': real_ret * 10000,
            'direction_correct': direction_correct, 'peak_timing_error_bars': timing_err,
            'traj_optimal_pnl_bps': traj_pnl, 'simple_fixed_pnl_bps': real_ret * 10000,
            'ci_width_bps': ci_width, 'inference_ms': inference_ms,
        })

        if (i + 1) % 10 == 0:
            dacc = np.mean([r['direction_correct'] for r in results])
            log.info(f"Window {i+1}/{n_windows}: dir_acc={dacc:.1%}, "
                     f"pred={pred_ret*10000:.1f}bps, real={real_ret*10000:.1f}bps, "
                     f"timing={timing_err}bars, {inference_ms:.0f}ms")

    return pd.DataFrame(results)


def generate_report(results: pd.DataFrame) -> str:
    out = Path("reports/spike_tsfm")
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / "chronos_large_backtest_results.csv", index=False)

    dir_acc = results['direction_correct'].mean()
    lines = [
        "# Chronos-T5-Large Trajectory Spike — SOL-USD", "",
        "## Metrics",
        f"- Direction accuracy: **{dir_acc:.1%}** (Small was 54.0%)",
        f"- Mean peak timing error: **{results['peak_timing_error_bars'].mean():.1f} bars**",
        f"- Mean traj-optimal PnL: **{results['traj_optimal_pnl_bps'].mean():.1f} bps**",
        f"- Mean fixed PnL: **{results['simple_fixed_pnl_bps'].mean():.1f} bps**",
        f"- Trajectory advantage: **{results['traj_optimal_pnl_bps'].mean() - results['simple_fixed_pnl_bps'].mean():.1f} bps**",
        f"- Mean CI width: **{results['ci_width_bps'].mean():.1f} bps**",
        f"- Mean inference: **{results['inference_ms'].mean():.0f} ms**",
        "", "## Verdict",
    ]
    if dir_acc > 0.55:
        lines.append("**PROMISING** — Large model shows meaningful directional improvement.")
    elif dir_acc > 0.52:
        lines.append("**MARGINAL** — Modest improvement over small model.")
    else:
        lines.append("**NO IMPROVEMENT** — Larger model does not help on raw price.")

    path = out / "chronos_large_spike_report.md"
    path.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {path}")
    return str(path)


if __name__ == "__main__":
    sol = load_sol_data()
    results = run_backtest(sol, n_windows=100)
    report = generate_report(results)
    log.info(f"Done: {report}")
