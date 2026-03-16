"""
tabpfn_forward_sim.py — Self-consistent forward simulation.

Uses TabPFN's own predicted return distribution to autoregressively generate
synthetic future XLE price paths, while simultaneously tracking the strategy's
fire/no-fire decisions and P&L on each generated path.

50 simulations, 1 quarter horizon (~441 market hours).
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("fwd_sim")

import sklearn.utils.validation
_o1 = sklearn.utils.validation.check_X_y
def _p1(*a, **k):
    if 'force_all_finite' in k: k['ensure_all_finite'] = k.pop('force_all_finite')
    return _o1(*a, **k)
sklearn.utils.validation.check_X_y = _p1
_o2 = sklearn.utils.validation.check_array
def _p2(*a, **k):
    if 'force_all_finite' in k: k['ensure_all_finite'] = k.pop('force_all_finite')
    return _o2(*a, **k)
sklearn.utils.validation.check_array = _p2

import torch, torch.nn.modules.transformer as _t, typing
_t.Optional = typing.Optional; _t.Tensor = torch.Tensor
_t.Module = torch.nn.Module; _t.Linear = torch.nn.Linear
_t.Dropout = torch.nn.Dropout; _t.LayerNorm = torch.nn.LayerNorm
_t.MultiheadAttention = torch.nn.MultiheadAttention
if not hasattr(_t, '_get_activation_fn'):
    def _g(a):
        if a == "relu": return torch.nn.functional.relu
        elif a == "gelu": return torch.nn.functional.gelu
        raise RuntimeError(a)
    _t._get_activation_fn = _g

from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt

COST_BPS = 5
PCA_DIM = 47
WINDOW = 500
RETRAIN_EVERY = 200
N_SIMS = 50
HORIZON = 441  # ~1 quarter of market hours

def main():
    out = Path("data/processed/lane_discovery")
    rpt = Path("reports/spike_tsfm")
    rpt.mkdir(parents=True, exist_ok=True)

    # Load real XLE data with features
    rich = pd.read_parquet(out / "panel_rich.parquet")
    xle = rich[rich['asset'] == 'XLE'].copy()
    xle['timestamp'] = pd.to_datetime(xle['timestamp'], utc=True)
    xle = xle.sort_values('timestamp').reset_index(drop=True)

    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in xle.columns if c not in exclude and pd.api.types.is_numeric_dtype(xle[c])]

    # Use all available data as the seed
    seed_data = xle[feat_cols + ['fwd_ret_8']].copy()
    seed_data = seed_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Get the last WINDOW rows as the starting context
    seed_end = len(seed_data)
    
    # For bin edges: build from the last WINDOW of real data
    last_window = seed_data.iloc[max(0, seed_end - WINDOW):seed_end]
    y_seed = last_window['fwd_ret_8'].values * 10000
    _, bin_edges = pd.qcut(y_seed, 10, labels=False, duplicates='drop', retbins=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # center of each bin in bps
    
    log.info(f"Seed data: {len(seed_data)} rows, {len(feat_cols)} features")
    log.info(f"Bin centers (bps): {[f'{b:.0f}' for b in bin_centers]}")
    log.info(f"Starting {N_SIMS} simulations, {HORIZON} steps each")

    all_balances = []  # (n_sims, horizon+1) 
    all_fired_counts = []
    
    for sim in range(N_SIMS):
        t0 = time.time()
        balance = 1000.0
        balance_path = [balance]
        fired_count = 0
        
        # Copy the feature window — we'll evolve it autoregressively
        # For simplicity: keep the real feature matrix fixed and just predict
        # from the current window. After each step, shift the window by 1 row
        # using the sampled return to update the fwd_ret column.
        
        # Build initial model
        window_X = seed_data.iloc[max(0, seed_end - WINDOW):seed_end][feat_cols].values.copy()
        window_y = y_seed.copy()
        
        model = None
        scaler = None
        pca = None
        last_train = -1
        
        for step in range(HORIZON):
            # Retrain if needed
            if model is None or step - last_train > RETRAIN_EVERY:
                nc = min(PCA_DIM, window_X.shape[1], window_X.shape[0])
                scaler = StandardScaler()
                pca = PCA(n_components=nc, random_state=42)
                Xp = pca.fit_transform(scaler.fit_transform(window_X))
                bins = pd.qcut(window_y, 10, labels=False, duplicates='drop')
                model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
                model.fit(Xp, bins)
                last_train = step
            
            # Predict on the latest row
            x_test = window_X[-1:].copy()
            xtp = pca.transform(scaler.transform(x_test))
            proba = model.predict_proba(xtp)[0]
            
            # Fire decision
            top_conv = float(sum(proba[-2:])) if len(proba) >= 2 else float(proba[-1])
            fires = top_conv > 0.25
            
            # Sample a return from the predicted distribution
            n_bins = len(proba)
            centers = bin_centers[:n_bins] if n_bins <= len(bin_centers) else bin_centers
            sampled_ret_bps = np.random.choice(centers, p=proba)
            
            # Update balance if fired
            if fires:
                dollar_gain = (sampled_ret_bps - COST_BPS) / 10000 * (balance / 4)
                balance += dollar_gain
                fired_count += 1
            
            balance_path.append(balance)
            
            # Shift window: drop first row, add a new row
            # New row = last row with slight noise (features drift slowly)
            new_row = window_X[-1:].copy() + np.random.normal(0, 0.01, window_X[-1:].shape)
            window_X = np.vstack([window_X[1:], new_row])
            window_y = np.append(window_y[1:], sampled_ret_bps)
        
        elapsed = time.time() - t0
        pct = (balance / 1000 - 1) * 100
        all_balances.append(balance_path)
        all_fired_counts.append(fired_count)
        log.info(f"  Sim {sim+1}/{N_SIMS}: ${balance:.0f} ({pct:+.1f}%), {fired_count} fires, {elapsed:.0f}s")
    
    all_balances = np.array(all_balances)
    
    # Stats
    finals = all_balances[:, -1]
    log.info(f"\n=== Results ({N_SIMS} simulations) ===")
    log.info(f"  Median final: ${np.median(finals):.0f} ({(np.median(finals)/1000-1)*100:+.1f}%)")
    log.info(f"  5th pct: ${np.percentile(finals,5):.0f}")
    log.info(f"  95th pct: ${np.percentile(finals,95):.0f}")
    log.info(f"  P(loss): {(finals < 1000).mean():.0%}")
    log.info(f"  Mean fires/sim: {np.mean(all_fired_counts):.0f}")
    
    # Save results
    pd.DataFrame({'final_balance': finals, 'fires': all_fired_counts}).to_csv(
        rpt / 'forward_sim_results.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    hours = np.arange(all_balances.shape[1])
    p5 = np.percentile(all_balances, 5, axis=0)
    p25 = np.percentile(all_balances, 25, axis=0)
    p50 = np.percentile(all_balances, 50, axis=0)
    p75 = np.percentile(all_balances, 75, axis=0)
    p95 = np.percentile(all_balances, 95, axis=0)
    
    ax = axes[0]
    ax.fill_between(hours, p5, p95, alpha=0.15, color='darkorange', label='5-95th %ile')
    ax.fill_between(hours, p25, p75, alpha=0.3, color='darkorange', label='25-75th %ile')
    ax.plot(hours, p50, color='darkorange', linewidth=2, label='Median')
    ax.axhline(y=1000, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Market Hours Forward', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_title('Self-Consistent TabPFN Forward Sim\nXLE, $1k, 1 Quarter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.hist(finals, bins=25, color='darkorange', edgecolor='white', alpha=0.7)
    ax2.axvline(x=1000, color='red', linestyle='--', label='Breakeven', linewidth=2)
    ax2.axvline(x=np.median(finals), color='green', linestyle='--', label=f'Median ${np.median(finals):.0f}', linewidth=2)
    ax2.set_xlabel('Final Portfolio Value ($)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Outcome Distribution\nP(loss)={(finals<1000).mean()*100:.0f}%', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = Path('reports/visualizations/tabpfn_forward_sim.png')
    plt.savefig(fig_path, dpi=300)
    log.info(f"Saved to {fig_path}")

if __name__ == "__main__":
    main()
