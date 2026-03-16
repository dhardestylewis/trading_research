"""
Phase 1: Bulk Chronos trajectory inference for all SOL-USD bars.
Saves trajectory features to CSV for the walk-forward stack to consume.

Run with: .venv_chronos\Scripts\python spike_chronos_bulk.py
"""
import logging, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from chronos import ChronosPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("chronos_bulk")

FORECAST_HORIZON = 8
CONTEXT_LENGTH = 256
BATCH_SIZE = 8  # Process multiple windows at once


def main():
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy().sort_values('timestamp').reset_index(drop=True)
    log.info(f"Loaded {len(sol)} bars")

    closes = sol['close'].values
    timestamps = sol['timestamp'].values

    log.info("Loading Chronos-T5-Small...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32,
    )
    log.info("Model loaded")

    # Generate trajectories for every eligible bar
    results = []
    eligible = range(CONTEXT_LENGTH, len(closes) - FORECAST_HORIZON)
    total = len(list(eligible))
    
    # Sample 500 evenly spaced windows (~5 min instead of 45)
    all_eligible = list(range(CONTEXT_LENGTH, len(closes) - FORECAST_HORIZON))
    n_sample = 500
    sample_indices = [all_eligible[int(i)] for i in np.linspace(0, len(all_eligible)-1, n_sample)]
    log.info(f"Processing {len(sample_indices)} sampled bars (of {len(all_eligible)} eligible)")
    
    t0_total = time.time()
    
    for batch_start in range(0, len(sample_indices), BATCH_SIZE):
        batch_indices = sample_indices[batch_start:batch_start + BATCH_SIZE]
        
        # Build context tensors for the batch
        contexts = []
        for idx in batch_indices:
            ctx = closes[idx - CONTEXT_LENGTH:idx]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
        
        t0 = time.time()
        # Chronos supports batched prediction
        forecasts = pipeline.predict(
            contexts,
            prediction_length=FORECAST_HORIZON,
            num_samples=20,
        )
        batch_ms = (time.time() - t0) * 1000
        
        for j, idx in enumerate(batch_indices):
            samples = forecasts[j].numpy()  # (num_samples, horizon)
            median = np.median(samples, axis=0)
            p10 = np.percentile(samples, 10, axis=0)
            p90 = np.percentile(samples, 90, axis=0)
            
            entry_price = closes[idx]
            
            # Derived trajectory features
            pred_return = (median[-1] - entry_price) / entry_price
            pred_min_bar = int(np.argmin(median))
            pred_max_bar = int(np.argmax(median))
            pred_range_bps = (max(median) - min(median)) / entry_price * 10000
            ci_width_bps = float(np.mean((p90 - p10) / entry_price * 10000))
            
            # Trajectory shape classification
            # "bullish": predicted low before predicted high (dip then rip)
            # "bearish": predicted high before predicted low
            shape = "bullish" if pred_min_bar < pred_max_bar else "bearish"
            
            # Trajectory strength: how much of the forecast is directional vs noise
            monotonicity = np.corrcoef(np.arange(FORECAST_HORIZON), median)[0, 1]
            
            results.append({
                'bar_idx': idx,
                'timestamp': timestamps[idx],
                'entry_price': entry_price,
                'pred_return_bps': pred_return * 10000,
                'pred_min_bar': pred_min_bar,
                'pred_max_bar': pred_max_bar,
                'pred_range_bps': pred_range_bps,
                'ci_width_bps': ci_width_bps,
                'traj_shape': shape,
                'monotonicity': monotonicity,
                # Store full trajectory for detailed analysis
                'median_0': median[0], 'median_1': median[1],
                'median_2': median[2], 'median_3': median[3],
                'median_4': median[4], 'median_5': median[5],
                'median_6': median[6], 'median_7': median[7],
            })
        
        done = batch_start + len(batch_indices)
        if done % 80 == 0 or done >= len(sample_indices):
            elapsed = time.time() - t0_total
            rate = done / elapsed
            eta = (len(sample_indices) - done) / rate if rate > 0 else 0
            log.info(f"Progress: {done}/{len(sample_indices)} "
                     f"({done/len(sample_indices)*100:.0f}%) "
                     f"ETA: {eta/60:.1f}min "
                     f"batch={batch_ms:.0f}ms")
    
    df = pd.DataFrame(results)
    out = Path("reports/spike_tsfm")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "chronos_bulk_trajectories.csv", index=False)
    log.info(f"Saved {len(df)} trajectories to {out / 'chronos_bulk_trajectories.csv'}")
    
    # Quick stats
    log.info(f"Bullish trajectories: {(df['traj_shape']=='bullish').mean():.1%}")
    log.info(f"Mean pred range: {df['pred_range_bps'].mean():.0f} bps")
    log.info(f"Mean CI width: {df['ci_width_bps'].mean():.0f} bps")
    log.info(f"Total time: {(time.time()-t0_total)/60:.1f} min")


if __name__ == "__main__":
    main()
