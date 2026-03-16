"""
spike_walkforward_stack.py — Phase 2: Proper Walk-Forward with Chronos + TabPFN

Loads pre-computed Chronos trajectory features from CSV, builds TabPFN 
features, and runs a walk-forward backtest where we only trade when
BOTH models agree.

Run with: python spike_walkforward_stack.py
"""
import logging, time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("walkforward_stack")

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

import torch.nn.modules.transformer as _t
import typing
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
MAX_TRAIN = 1000
PCA_DIM = 47
HORIZON = 8
RETRAIN_EVERY = 200  # bars between TabPFN retraining


def main():
    # Load data + features
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    
    panel = pd.read_parquet("data/processed/panel_expanded/panel.parquet")
    col = 'symbol' if 'symbol' in panel.columns else 'asset'
    sol = panel[panel[col] == 'SOL-USD'].copy().sort_values('timestamp').reset_index(drop=True)
    rich = build_rich_perp_state_features(sol, horizons=[8])
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    log.info(f"Panel: {len(rich)} rows, {len(feat_cols)} features")
    
    # Load Chronos trajectories
    traj = pd.read_csv("reports/spike_tsfm/chronos_bulk_trajectories.csv")
    traj_set = set(traj['bar_idx'].values)
    log.info(f"Chronos trajectories: {len(traj)} bars")
    
    # Walk-forward: only test on bars where we have Chronos predictions
    target = 'fwd_ret_8'
    valid = rich[rich[target].notna()].copy().reset_index(drop=True)
    
    results = []
    model = None; scaler = None; pca = None
    last_train = -1
    
    for row_i in range(len(valid)):
        idx = valid.index[row_i]
        
        # Skip if no Chronos prediction for this bar
        if idx not in traj_set:
            continue
        
        # Need enough history for training
        if row_i < 300:
            continue
        
        # Retrain TabPFN periodically
        if model is None or row_i - last_train > RETRAIN_EVERY:
            train = valid.iloc[max(0, row_i-500):row_i].copy()
            y = train[target].values * 10000
            if len(train) < 100: continue
            if len(train) > MAX_TRAIN:
                train = train.iloc[-MAX_TRAIN:]
                y = train[target].values * 10000
            
            X = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            nc = min(PCA_DIM, X.shape[1], X.shape[0])
            scaler = StandardScaler()
            pca = PCA(n_components=nc, random_state=42)
            Xp = pca.fit_transform(scaler.fit_transform(X))
            
            bins = pd.qcut(y, 10, labels=False, duplicates='drop')
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            model.fit(Xp, bins)
            last_train = row_i
            log.info(f"TabPFN retrained at bar {idx}, n_train={len(train)}")
        
        # Score with TabPFN
        Xt = valid.iloc[row_i:row_i+1][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xtp = pca.transform(scaler.transform(Xt))
        proba = model.predict_proba(Xtp)[0]
        n_bins = len(proba)
        top_conv = float(sum(proba[-2:])) if n_bins >= 2 else float(proba[-1])
        tabpfn_fires = top_conv > 0.25  # fires on top quartile of conviction
        
        # Get Chronos trajectory features
        traj_row = traj[traj['bar_idx'] == idx].iloc[0]
        traj_shape = traj_row['traj_shape']
        traj_range = traj_row['pred_range_bps']
        ci_width = traj_row['ci_width_bps']
        pred_min_bar = int(traj_row['pred_min_bar'])
        pred_max_bar = int(traj_row['pred_max_bar'])
        
        # Chronos fires if trajectory is bullish AND range is meaningful
        chronos_fires = traj_shape == 'bullish' and traj_range > 20
        
        # Combined stack
        stack_fires = tabpfn_fires and chronos_fires
        
        # Realized
        closes = valid['close'].values if 'close' in valid.columns else sol['close'].values
        entry = closes[idx]
        realized = closes[idx:idx+HORIZON]
        if len(realized) < HORIZON: continue
        real_ret = (realized[-1] - entry) / entry
        
        # PnL calculations
        # TabPFN only: trade every time TabPFN fires, fixed horizon
        tabpfn_pnl = (real_ret * 10000 - COST_BPS) if tabpfn_fires else 0
        
        # Stack: use Chronos entry/exit timing when both fire
        if stack_fires and pred_max_bar > pred_min_bar and pred_max_bar < HORIZON:
            # Enter at predicted dip bar, exit at predicted peak bar
            stack_pnl = (realized[pred_max_bar] - realized[pred_min_bar]) / entry * 10000 - COST_BPS
        elif stack_fires:
            stack_pnl = real_ret * 10000 - COST_BPS
        else:
            stack_pnl = 0
        
        # Chronos-only: trade every bullish trajectory
        chronos_only_pnl = (real_ret * 10000 - COST_BPS) if chronos_fires else 0
        
        results.append({
            'bar_idx': idx, 'tabpfn_fires': tabpfn_fires, 'chronos_fires': chronos_fires,
            'stack_fires': stack_fires, 'top_conviction': top_conv,
            'traj_shape': traj_shape, 'traj_range_bps': traj_range,
            'realized_bps': real_ret * 10000,
            'tabpfn_pnl': tabpfn_pnl, 'chronos_pnl': chronos_only_pnl,
            'stack_pnl': stack_pnl,
        })
        
        if len(results) % 50 == 0:
            st = [r for r in results if r['stack_fires']]
            tp = [r for r in results if r['tabpfn_fires']]
            log.info(f"Bars processed: {len(results)}, "
                     f"stack_trades={len(st)} ({np.mean([r['stack_pnl'] for r in st]):.1f}bps), "
                     f"tabpfn_trades={len(tp)} ({np.mean([r['tabpfn_pnl'] for r in tp]):.1f}bps)"
                     if st and tp else f"Bars: {len(results)}")
    
    df = pd.DataFrame(results)
    report(df)


def report(df):
    out = Path("reports/spike_tsfm"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "walkforward_stack_results.csv", index=False)
    
    st = df[df['stack_fires']]; tp = df[df['tabpfn_fires']]; co = df[df['chronos_fires']]
    
    lines = [
        "# Walk-Forward Stack Backtest: Chronos × TabPFN — SOL-USD", "",
        "## Setup",
        "- Walk-forward with periodic TabPFN retraining (every 200 bars)",
        "- Chronos trajectory pre-computed (500 evenly spaced windows)",
        "- Entry/exit timing from Chronos predicted dip→peak bars",
        f"- Total test bars: **{len(df)}**", "",
        "## Signal Counts",
        f"- TabPFN fires: **{len(tp)}** ({len(tp)/len(df)*100:.0f}%)",
        f"- Chronos fires: **{len(co)}** ({len(co)/len(df)*100:.0f}%)",
        f"- Stack fires (both): **{len(st)}** ({len(st)/len(df)*100:.0f}%)", "",
        "## PnL Comparison (net of 14 bps)",
    ]
    
    for name, subset, col in [("TabPFN Only", tp, 'tabpfn_pnl'),
                               ("Chronos Only", co, 'chronos_pnl'),
                               ("Stack (Both)", st, 'stack_pnl')]:
        if len(subset) > 0:
            pnl = subset[col]
            lines.append(f"### {name}")
            lines.append(f"- Trades: **{len(subset)}**")
            lines.append(f"- Mean PnL: **{pnl.mean():.1f} bps/trade**")
            lines.append(f"- Total PnL: **{pnl.sum():.0f} bps**")
            lines.append(f"- Win rate: **{(pnl > 0).mean():.1%}**")
            lines.append(f"- Sharpe (approx): **{pnl.mean() / (pnl.std()+1e-8) * np.sqrt(252):.2f}**")
            lines.append("")
    
    lines.append("## Verdict")
    if len(st) > 0 and st['stack_pnl'].mean() > tp['tabpfn_pnl'].mean():
        lines.append("**STACK IMPROVES** — Combined model beats TabPFN alone.")
    elif len(st) > 0 and st['stack_pnl'].mean() > 0:
        lines.append("**STACK POSITIVE BUT NO IMPROVEMENT** — Both models are positive but stack doesn't beat TabPFN alone.")
    else:
        lines.append("**NO EDGE** — Stack does not produce a tradeable edge.")
    
    p = out / "walkforward_stack_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {p}")


if __name__ == "__main__":
    main()
