"""
spike_equity_tabpfn.py — TabPFN walk-forward on uncorrelated equity/FX ETFs

Tests whether the TabPFN top-decile edge discovered on SOL-USD transfers
to fundamentally different asset classes:
  SPY, QQQ, GLD, TLT, UUP, USO, FXI

Run with: python spike_equity_tabpfn.py
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("equity_tabpfn")

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

COST_BPS = 5  # ETF spreads are tighter than crypto
PCA_DIM = 47
HORIZON = 8
MAX_TRAIN = 1000
RETRAIN_EVERY = 200


def run_asset(asset_df, asset_name, feat_cols, target_col='fwd_ret_8'):
    valid = asset_df[asset_df[target_col].notna()].copy().reset_index(drop=True)
    if len(valid) < 400:
        return pd.DataFrame()
    
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    
    # Sample every 55th bar for speed (~3 min per asset instead of 15)
    test_indices = list(range(300, len(valid), 55))
    
    for row_i in test_indices:
        if model is None or row_i - last_train > RETRAIN_EVERY:
            train = valid.iloc[max(0, row_i-500):row_i].copy()
            y = train[target_col].values * 10000
            if len(train) < 100: continue
            if len(train) > MAX_TRAIN:
                train = train.iloc[-MAX_TRAIN:]
                y = train[target_col].values * 10000
            X = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            nc = min(PCA_DIM, X.shape[1], X.shape[0])
            scaler = StandardScaler()
            pca = PCA(n_components=nc, random_state=42)
            Xp = pca.fit_transform(scaler.fit_transform(X))
            bins = pd.qcut(y, 10, labels=False, duplicates='drop')
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            model.fit(Xp, bins)
            last_train = row_i
        
        # Score
        Xt = valid.iloc[row_i:row_i+1][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xtp = pca.transform(scaler.transform(Xt))
        proba = model.predict_proba(Xtp)[0]
        n_bins = len(proba)
        top_conv = float(sum(proba[-2:])) if n_bins >= 2 else float(proba[-1])
        fires = top_conv > 0.25
        
        real_ret = valid.iloc[row_i][target_col]
        pnl = (real_ret * 10000 - COST_BPS) if fires else 0
        
        results.append({
            'asset': asset_name, 'bar_idx': row_i,
            'fires': fires, 'top_conviction': top_conv,
            'realized_bps': real_ret * 10000, 'pnl': pnl,
        })
    
    return pd.DataFrame(results)


def main():
    rich = pd.read_parquet("data/processed/equity_panel/panel_rich.parquet")
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_4','fwd_ret_8',
               'gross_move_bps_4','gross_move_bps_8',
               'prob_tail_25_4','prob_tail_50_4','prob_tail_100_4',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    log.info(f"Features: {len(feat_cols)}")
    
    all_results = []
    for asset in sorted(rich['asset'].unique()):
        log.info(f">>> {asset}")
        t0 = time.time()
        asset_df = rich[rich['asset'] == asset].copy().sort_values('timestamp').reset_index(drop=True)
        df = run_asset(asset_df, asset, feat_cols)
        elapsed = time.time() - t0
        if len(df) > 0:
            trades = df[df['fires']]
            log.info(f"  {asset}: {len(trades)} trades, "
                     f"mean_pnl={trades['pnl'].mean():.1f} bps, "
                     f"win_rate={((trades['pnl']>0).mean()):.1%}, "
                     f"elapsed={elapsed:.0f}s")
            all_results.append(df)
        else:
            log.info(f"  {asset}: insufficient data")
    
    if not all_results:
        log.warning("No results!")
        return
    
    full = pd.concat(all_results, ignore_index=True)
    out = Path("reports/spike_tsfm"); out.mkdir(parents=True, exist_ok=True)
    full.to_csv(out / "equity_tabpfn_results.csv", index=False)
    
    # Report
    lines = [
        "# TabPFN Walk-Forward: Uncorrelated ETF Universe", "",
        "## Setup",
        "- Walk-forward with periodic retraining (every 200 bars)",
        f"- Cost assumption: **{COST_BPS} bps** (ETF spreads)",
        f"- Features: **{len(feat_cols)}** rich OHLCV-derived",
        f"- Horizon: **{HORIZON}h**", "",
        "## Results by Asset", "",
        "| Asset | Trades | Mean PnL (bps) | Total PnL | Win Rate | Sharpe |",
        "|---|---|---|---|---|---|",
    ]
    
    for asset in sorted(full['asset'].unique()):
        t = full[(full['asset'] == asset) & (full['fires'])]
        if len(t) > 0:
            pnl = t['pnl']
            lines.append(
                f"| {asset} | {len(t)} | {pnl.mean():.1f} | {pnl.sum():.0f} | "
                f"{(pnl>0).mean():.1%} | {pnl.mean()/(pnl.std()+1e-8)*np.sqrt(252):.2f} |"
            )
    
    all_trades = full[full['fires']]
    lines.extend(["", "## Aggregate",
        f"- Total trades: **{len(all_trades)}**",
        f"- Mean PnL: **{all_trades['pnl'].mean():.1f} bps**",
        f"- Win rate: **{(all_trades['pnl']>0).mean():.1%}**",
        f"- Total PnL: **{all_trades['pnl'].sum():.0f} bps**", "",
        "## Verdict",
    ])
    
    if all_trades['pnl'].mean() > 5:
        lines.append("**EDGE TRANSFERS** — TabPFN signal generalizes beyond crypto.")
    elif all_trades['pnl'].mean() > 0:
        lines.append("**MARGINAL** — Some signal but not strong enough to trade on.")
    else:
        lines.append("**NO TRANSFER** — TabPFN edge is crypto-specific, does not generalize.")
    
    p = out / "equity_tabpfn_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {p}")


if __name__ == "__main__":
    main()
