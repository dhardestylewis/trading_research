"""
spike_tsfm_stack.py — Two-Model Stack: Chronos Trajectory × TabPFN Conviction

Run with: .venv_chronos\Scripts\python spike_tsfm_stack.py
"""
import logging, time, json
import numpy as np
import pandas as pd
import torch
# Monkey-patch torch.load for TabPFN 0.1.9 compatibility with newer torch
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("spike_stack")

FORECAST_HORIZON = 8
CONTEXT_LENGTH = 256
COST_BPS = 14
MAX_TRAIN_SAMPLES = 1000
PCA_COMPONENTS = 47

# TabPFN monkey patches
import sklearn.utils.validation
_orig_check_X_y = sklearn.utils.validation.check_X_y
def _p1(*a, **kw):
    if 'force_all_finite' in kw: kw['ensure_all_finite'] = kw.pop('force_all_finite')
    return _orig_check_X_y(*a, **kw)
sklearn.utils.validation.check_X_y = _p1

_orig_ca = sklearn.utils.validation.check_array
def _p2(*a, **kw):
    if 'force_all_finite' in kw: kw['ensure_all_finite'] = kw.pop('force_all_finite')
    return _orig_ca(*a, **kw)
sklearn.utils.validation.check_array = _p2

import torch.nn.modules.transformer as _tmr
import typing
_tmr.Optional = typing.Optional
_tmr.Tensor = torch.Tensor
_tmr.Module = torch.nn.Module
_tmr.Linear = torch.nn.Linear
_tmr.Dropout = torch.nn.Dropout
_tmr.LayerNorm = torch.nn.LayerNorm
_tmr.MultiheadAttention = torch.nn.MultiheadAttention
if not hasattr(_tmr, '_get_activation_fn'):
    def _gaf(act):
        if act == "relu": return torch.nn.functional.relu
        elif act == "gelu": return torch.nn.functional.gelu
        raise RuntimeError(act)
    _tmr._get_activation_fn = _gaf

from tabpfn import TabPFNClassifier
from chronos import ChronosPipeline


def load_data():
    """Load panel and build features."""
    # Import from project src
    import sys
    sys.path.insert(0, '.')
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy().sort_values('timestamp').reset_index(drop=True)
    
    rich_df = build_rich_perp_state_features(sol, horizons=[8])
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich_df[c])]
    
    log.info(f"Data: {len(rich_df)} rows, {len(feat_cols)} features")
    return sol, rich_df, feat_cols


def run_stack(sol, rich_df, feat_cols, n_windows=50):
    """Two-model stack backtest."""
    # Load Chronos
    log.info("Loading Chronos model...")
    chronos = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map="cpu", torch_dtype=torch.float32
    )
    log.info("Chronos loaded")
    
    closes = rich_df['close'].values if 'close' in rich_df.columns else sol['close'].values
    timestamps = rich_df['timestamp'].values if 'timestamp' in rich_df.columns else sol['timestamp'].values
    target_col = 'fwd_ret_8'
    
    valid = rich_df[rich_df[target_col].notna()].copy()
    start_idx = CONTEXT_LENGTH + 100
    end_idx = len(valid) - FORECAST_HORIZON
    window_indices = np.linspace(start_idx, end_idx - 1, min(n_windows, end_idx - start_idx), dtype=int)
    
    results = []
    tabpfn_model = None
    pca_model = None
    scaler = None
    last_train = -1
    
    for i, idx in enumerate(window_indices):
        # --- TabPFN: retrain every 50 windows ---
        if tabpfn_model is None or idx - last_train > 50:
            train = valid.iloc[max(0, idx-500):idx].copy()
            y = train[target_col].values * 10000
            if len(train) < 100: continue
            if len(train) > MAX_TRAIN_SAMPLES:
                train = train.iloc[-MAX_TRAIN_SAMPLES:]
                y = train[target_col].values * 10000
            
            X = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            nc = min(PCA_COMPONENTS, X.shape[1], X.shape[0])
            scaler = StandardScaler()
            pca_model = PCA(n_components=nc, random_state=42)
            Xp = pca_model.fit_transform(scaler.fit_transform(X))
            
            bins = pd.qcut(y, 10, labels=False, duplicates='drop')
            tabpfn_model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            tabpfn_model.fit(Xp, bins)
            last_train = idx
            log.info(f"TabPFN retrained at {i}, n={len(train)}")
        
        # Score with TabPFN
        Xt = valid.iloc[idx:idx+1][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xtp = pca_model.transform(scaler.transform(Xt))
        proba = tabpfn_model.predict_proba(Xtp)[0]
        top_conv = sum(proba[-2:]) if len(proba) >= 2 else proba[-1]
        tabpfn_fires = top_conv > 0.3
        
        # --- Chronos trajectory (only if TabPFN fires) ---
        traj_ok = False
        if tabpfn_fires and idx >= CONTEXT_LENGTH:
            ctx = torch.tensor(closes[idx-CONTEXT_LENGTH:idx], dtype=torch.float32).unsqueeze(0)
            fc = chronos.predict(ctx, prediction_length=FORECAST_HORIZON, num_samples=20)
            med = fc[0].numpy()
            median_fc = np.median(med, axis=0)
            entry = closes[idx]
            lo_bar = np.argmin(median_fc)
            hi_bar = np.argmax(median_fc[lo_bar:]) + lo_bar
            pred_range = (max(median_fc) - min(median_fc)) / entry * 10000
            traj_ok = lo_bar < hi_bar and pred_range > 20
        
        stack_fires = tabpfn_fires and traj_ok
        
        # Realized
        entry = closes[idx]
        realized = closes[idx:idx+FORECAST_HORIZON]
        if len(realized) < FORECAST_HORIZON: continue
        real_ret = (realized[-1] - entry) / entry
        
        # PnL
        if stack_fires:
            lo_bar = np.argmin(median_fc)
            hi_bar = np.argmax(median_fc[lo_bar:]) + lo_bar
            if hi_bar > lo_bar and hi_bar < FORECAST_HORIZON:
                stack_pnl = (realized[hi_bar] - realized[lo_bar]) / entry * 10000 - COST_BPS
            else:
                stack_pnl = real_ret * 10000 - COST_BPS
        else:
            stack_pnl = 0
        
        simple_pnl = (real_ret * 10000 - COST_BPS) if tabpfn_fires else 0
        
        results.append({
            'window': i, 'tabpfn_fires': tabpfn_fires, 'top_conviction': top_conv,
            'traj_favorable': traj_ok, 'stack_fires': stack_fires,
            'realized_bps': real_ret * 10000,
            'stack_pnl': stack_pnl, 'tabpfn_only_pnl': simple_pnl,
        })
        
        if (i+1) % 10 == 0:
            st = [r for r in results if r['stack_fires']]
            log.info(f"W{i+1}/{len(window_indices)}: stack_trades={len(st)}, "
                     f"stack_pnl={np.mean([r['stack_pnl'] for r in st]):.1f}" if st else 
                     f"W{i+1}/{len(window_indices)}: no stack trades yet")
    
    return pd.DataFrame(results)


def report(df):
    out = Path("reports/spike_tsfm"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "stack_results.csv", index=False)
    
    st = df[df['stack_fires']]; tp = df[df['tabpfn_fires']]
    lines = [
        "# Two-Model Stack: Chronos × TabPFN", "",
        f"- Total windows: **{len(df)}**",
        f"- TabPFN fires: **{len(tp)}** ({len(tp)/len(df)*100:.0f}%)",
        f"- Stack fires: **{len(st)}** ({len(st)/len(df)*100:.0f}%)", "",
    ]
    if len(st) > 0:
        lines += [
            "## PnL (net 14bps)",
            f"- Stack: **{st['stack_pnl'].mean():.1f} bps/trade**, total **{st['stack_pnl'].sum():.0f} bps**, WR **{(st['stack_pnl']>0).mean():.0%}**",
            f"- TabPFN-only: **{tp['tabpfn_only_pnl'].mean():.1f} bps/trade**, total **{tp['tabpfn_only_pnl'].sum():.0f} bps**, WR **{(tp['tabpfn_only_pnl']>0).mean():.0%}**",
        ]
    lines += ["", "## Verdict"]
    if len(st) > 0 and st['stack_pnl'].mean() > 5:
        lines.append("**PROMISING**")
    elif len(st) > 0 and st['stack_pnl'].mean() > 0:
        lines.append("**MARGINAL**")
    else:
        lines.append("**NO EDGE**")
    
    p = out / "stack_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {p}")
    return str(p)


if __name__ == "__main__":
    sol, rich_df, feat_cols = load_data()
    df = run_stack(sol, rich_df, feat_cols, n_windows=50)
    report(df)
