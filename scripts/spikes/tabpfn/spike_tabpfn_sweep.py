"""
spike_tabpfn_sweep.py — Hyperparameter sweep for TabPFN on SOL-USD

Tests different configurations to find the optimal TabPFN setup:
- N_ensemble_configurations: 4, 8, 16, 32
- PCA dimensions: 20, 35, 47, all
- Conviction threshold: top-decile (0.25), top-quintile (0.40)
- Training window: 300, 500, 1000

Uses the same walk-forward framework as the stack backtest.
Run with: python spike_tabpfn_sweep.py
"""
import logging, time, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("tabpfn_sweep")

# TabPFN monkey patches
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

COST_BPS = 14
HORIZON = 8
RETRAIN_EVERY = 200

# Sweep grid — keep it small to finish in ~30 min
CONFIGS = [
    {"n_ensemble": 4,  "pca_dim": 20,   "threshold": 0.25, "train_window": 500},
    {"n_ensemble": 4,  "pca_dim": 47,   "threshold": 0.25, "train_window": 500},  # baseline
    {"n_ensemble": 16, "pca_dim": 47,   "threshold": 0.25, "train_window": 500},
    {"n_ensemble": 32, "pca_dim": 47,   "threshold": 0.25, "train_window": 500},
    {"n_ensemble": 4,  "pca_dim": 47,   "threshold": 0.40, "train_window": 500},  # top quintile
    {"n_ensemble": 4,  "pca_dim": 47,   "threshold": 0.25, "train_window": 300},
    {"n_ensemble": 4,  "pca_dim": 47,   "threshold": 0.25, "train_window": 1000},
    {"n_ensemble": 16, "pca_dim": 35,   "threshold": 0.30, "train_window": 500},  # middle ground
]


def run_config(valid, feat_cols, cfg):
    """Run walk-forward for one configuration, testing only every 55th bar for speed."""
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    target = 'fwd_ret_8'
    
    # Test every 55th bar to keep each config under 3 min
    test_indices = list(range(300, len(valid), 55))
    
    for row_i in test_indices:
        # Retrain if needed
        if model is None or row_i - last_train > RETRAIN_EVERY:
            tw = cfg['train_window']
            train = valid.iloc[max(0, row_i-tw):row_i].copy()
            y = train[target].values * 10000
            if len(train) < 100: continue
            
            X = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pd_dim = cfg['pca_dim'] if cfg['pca_dim'] != 'all' else X.shape[1]
            nc = min(pd_dim, X.shape[1], X.shape[0])
            scaler = StandardScaler()
            pca = PCA(n_components=nc, random_state=42)
            Xp = pca.fit_transform(scaler.fit_transform(X))
            bins = pd.qcut(y, 10, labels=False, duplicates='drop')
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=cfg['n_ensemble'])
            model.fit(Xp, bins)
            last_train = row_i
        
        # Score
        Xt = valid.iloc[row_i:row_i+1][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xtp = pca.transform(scaler.transform(Xt))
        proba = model.predict_proba(Xtp)[0]
        n_bins = len(proba)
        top_conv = float(sum(proba[-2:])) if n_bins >= 2 else float(proba[-1])
        fires = top_conv > cfg['threshold']
        
        real_ret = valid.iloc[row_i][target]
        pnl = (real_ret * 10000 - COST_BPS) if fires else 0
        
        results.append({
            'bar_idx': row_i, 'fires': fires,
            'top_conviction': top_conv, 'pnl': pnl,
            'realized_bps': real_ret * 10000,
        })
    
    return pd.DataFrame(results)


def main():
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy().sort_values('timestamp').reset_index(drop=True)
    rich = build_rich_perp_state_features(sol, horizons=[8])
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    
    target = 'fwd_ret_8'
    valid = rich[rich[target].notna()].copy().reset_index(drop=True)
    log.info(f"Panel: {len(valid)} valid rows, {len(feat_cols)} features")
    
    sweep_results = []
    
    for i, cfg in enumerate(CONFIGS):
        log.info(f"=== Config {i+1}/{len(CONFIGS)}: {cfg} ===")
        t0 = time.time()
        df = run_config(valid, feat_cols, cfg)
        elapsed = time.time() - t0
        
        trades = df[df['fires']]
        if len(trades) > 0:
            mean_pnl = trades['pnl'].mean()
            win_rate = (trades['pnl'] > 0).mean()
            sharpe = mean_pnl / (trades['pnl'].std() + 1e-8) * np.sqrt(252)
            total_pnl = trades['pnl'].sum()
        else:
            mean_pnl = win_rate = sharpe = total_pnl = 0
        
        result = {
            **cfg,
            'n_trades': len(trades),
            'n_tested': len(df),
            'fire_rate': len(trades) / max(len(df), 1),
            'mean_pnl': mean_pnl,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'sharpe': sharpe,
            'elapsed_s': elapsed,
        }
        sweep_results.append(result)
        log.info(f"  → {len(trades)} trades, {mean_pnl:.1f} bps/trade, "
                 f"win={win_rate:.1%}, sharpe={sharpe:.2f}, {elapsed:.0f}s")
    
    # Report
    sdf = pd.DataFrame(sweep_results).sort_values('sharpe', ascending=False)
    out = Path("reports/spike_tsfm"); out.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(out / "tabpfn_sweep_results.csv", index=False)
    
    lines = [
        "# TabPFN Hyperparameter Sweep — SOL-USD 8h", "",
        "| Ensemble | PCA | Threshold | Window | Trades | Mean PnL | Win Rate | Sharpe |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in sdf.iterrows():
        lines.append(
            f"| {int(r['n_ensemble'])} | {int(r['pca_dim'])} | {r['threshold']:.2f} | "
            f"{int(r['train_window'])} | {int(r['n_trades'])} | {r['mean_pnl']:.1f} | "
            f"{r['win_rate']:.1%} | {r['sharpe']:.2f} |"
        )
    
    best = sdf.iloc[0]
    lines.extend(["", 
        f"**Best config**: ensemble={int(best['n_ensemble'])}, "
        f"pca={int(best['pca_dim'])}, thresh={best['threshold']:.2f}, "
        f"window={int(best['train_window'])} → "
        f"**{best['mean_pnl']:.1f} bps/trade, Sharpe {best['sharpe']:.2f}**"
    ])
    
    p = out / "tabpfn_sweep_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {p}")


if __name__ == "__main__":
    main()
