"""
tabpfn_forward_sim_quick.py — Quick 10-simulation forward projection.
~100 steps (2 weeks), 10 sims. Should finish in ~30 min.
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("fwd_sim_quick")

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
N_SIMS = 10
HORIZON = 100  # ~2 weeks of market hours

def main():
    out = Path("data/processed/lane_discovery")
    rpt = Path("reports/spike_tsfm"); rpt.mkdir(parents=True, exist_ok=True)

    rich = pd.read_parquet(out / "panel_rich.parquet")
    xle = rich[rich['asset'] == 'XLE'].copy()
    xle['timestamp'] = pd.to_datetime(xle['timestamp'], utc=True)
    xle = xle.sort_values('timestamp').reset_index(drop=True)

    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in xle.columns if c not in exclude and pd.api.types.is_numeric_dtype(xle[c])]

    seed_data = xle[feat_cols + ['fwd_ret_8']].copy()
    seed_data = seed_data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    seed_end = len(seed_data)

    last_window = seed_data.iloc[max(0, seed_end - WINDOW):seed_end]
    y_seed = last_window['fwd_ret_8'].values * 10000
    _, bin_edges = pd.qcut(y_seed, 10, labels=False, duplicates='drop', retbins=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    log.info(f"Seed: {len(seed_data)} rows, {len(feat_cols)} feats, {N_SIMS} sims × {HORIZON} steps")

    all_balances = []
    all_fired = []

    for sim in range(N_SIMS):
        t0 = time.time()
        balance = 1000.0
        path = [balance]
        fired_count = 0

        window_X = seed_data.iloc[max(0, seed_end - WINDOW):seed_end][feat_cols].values.copy()
        window_y = y_seed.copy()
        model = None; scaler = None; pca = None; last_train = -1

        for step in range(HORIZON):
            if model is None or step - last_train > RETRAIN_EVERY:
                nc = min(PCA_DIM, window_X.shape[1], window_X.shape[0])
                scaler = StandardScaler()
                pca = PCA(n_components=nc, random_state=42)
                Xp = pca.fit_transform(scaler.fit_transform(window_X))
                bins = pd.qcut(window_y, 10, labels=False, duplicates='drop')
                model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
                model.fit(Xp, bins)
                last_train = step

            xtp = pca.transform(scaler.transform(window_X[-1:]))
            proba = model.predict_proba(xtp)[0]
            top_conv = float(sum(proba[-2:])) if len(proba) >= 2 else float(proba[-1])
            fires = top_conv > 0.25

            n_bins = len(proba)
            centers = bin_centers[:n_bins]
            sampled_ret = np.random.choice(centers, p=proba)

            if fires:
                balance += (sampled_ret - COST_BPS) / 10000 * (balance / 4)
                fired_count += 1

            path.append(balance)
            new_row = window_X[-1:] + np.random.normal(0, 0.01, window_X[-1:].shape)
            window_X = np.vstack([window_X[1:], new_row])
            window_y = np.append(window_y[1:], sampled_ret)

        elapsed = time.time() - t0
        pct = (balance / 1000 - 1) * 100
        all_balances.append(path)
        all_fired.append(fired_count)
        log.info(f"  Sim {sim+1}/{N_SIMS}: ${balance:.0f} ({pct:+.1f}%), {fired_count} fires, {elapsed:.0f}s")

    all_balances = np.array(all_balances)
    finals = all_balances[:, -1]

    # Write results to file user can read
    results_path = rpt / 'forward_sim_quick_results.md'
    with open(results_path, 'w') as f:
        f.write("# Self-Consistent TabPFN Forward Sim (Quick)\n\n")
        f.write(f"**{N_SIMS} simulations, {HORIZON} steps (~2 weeks forward)**\n\n")
        f.write("| Sim | Final $ | Return | Fires |\n|---|---|---|---|\n")
        for i in range(N_SIMS):
            f.write(f"| {i+1} | ${finals[i]:.0f} | {(finals[i]/1000-1)*100:+.1f}% | {all_fired[i]} |\n")
        f.write(f"\n**Median**: ${np.median(finals):.0f} ({(np.median(finals)/1000-1)*100:+.1f}%)\n")
        f.write(f"**P(loss)**: {(finals<1000).mean()*100:.0f}%\n")
        f.write(f"**Range**: ${finals.min():.0f} to ${finals.max():.0f}\n")

    log.info(f"Results saved to {results_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.arange(all_balances.shape[1])
    for i in range(N_SIMS):
        ax.plot(hours, all_balances[i], alpha=0.4, linewidth=1)
    ax.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Breakeven')
    ax.set_xlabel('Market Hours Forward')
    ax.set_ylabel('Portfolio Value ($)')
    ax.set_title(f'TabPFN Self-Consistent Forward Sim — XLE, $1k\n{N_SIMS} sims, {HORIZON} steps (~2 weeks)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = Path('reports/visualizations/tabpfn_forward_sim_quick.png')
    plt.savefig(fig_path, dpi=300)
    log.info(f"Plot saved to {fig_path}")

if __name__ == "__main__":
    main()
