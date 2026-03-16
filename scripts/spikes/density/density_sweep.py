"""
density_sweep.py — Test rolling_500 TabPFN at multiple evaluation densities
Step sizes: 55 (baseline), 25, 10, 5, 1
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("density_sweep")

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

COST_BPS = 5
PCA_DIM = 47
RETRAIN_EVERY = 200
WINDOW = 500

STEPS = [55, 25, 10, 5, 1]
ASSETS = ['FXA', 'XLE']

def run_density(valid, feat_cols, asset, step, target='fwd_ret_8'):
    if len(valid) < 400:
        return pd.DataFrame()
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    test_indices = list(range(300, len(valid), step))
    
    for row_i in test_indices:
        if model is None or row_i - last_train > RETRAIN_EVERY:
            train_idx = np.arange(max(0, row_i - WINDOW), row_i)
            if len(train_idx) < 100: continue
            train = valid.iloc[train_idx].copy()
            y = train[target].values * 10000
            X = train[feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
            nc = min(PCA_DIM, X.shape[1], X.shape[0])
            scaler = StandardScaler()
            pca = PCA(n_components=nc, random_state=42)
            Xp = pca.fit_transform(scaler.fit_transform(X))
            bins = pd.qcut(y, 10, labels=False, duplicates='drop')
            model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
            model.fit(Xp, bins)
            last_train = row_i

        Xt = valid.iloc[row_i:row_i+1][feat_cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
        Xtp = pca.transform(scaler.transform(Xt))
        proba = model.predict_proba(Xtp)[0]
        top_conv = float(sum(proba[-2:])) if len(proba) >= 2 else float(proba[-1])
        fires = top_conv > 0.25
        real_ret = valid.iloc[row_i][target]
        net_bps = (real_ret * 10000 - COST_BPS) if fires else 0
        results.append({
            'timestamp': valid.iloc[row_i]['timestamp'],
            'asset': asset, 'step': step,
            'fires': fires, 'net_bps': net_bps
        })
    return pd.DataFrame(results)

def main():
    out = Path("data/processed/lane_discovery")
    rpt = Path("reports/spike_tsfm")
    partial_csv = rpt / "density_sweep_results.csv"
    
    rich = pd.read_parquet(out / "panel_rich.parquet")
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    
    done_set = set()
    all_results = []
    if partial_csv.exists():
        prev = pd.read_csv(partial_csv)
        for _, row in prev[['asset', 'step']].drop_duplicates().iterrows():
            done_set.add((row['asset'], int(row['step'])))
        all_results.append(prev)
        log.info(f"Resuming — {len(done_set)} combos done")
    
    for asset in ASSETS:
        adf = rich[rich['asset']==asset].copy()
        if adf.empty:
            log.warning(f"No data for {asset}")
            continue
        adf['timestamp'] = pd.to_datetime(adf['timestamp'], utc=True)
        adf = adf.sort_values('timestamp').reset_index(drop=True)
        valid = adf[adf['fwd_ret_8'].notna()].reset_index(drop=True)
        
        for step in STEPS:
            if (asset, step) in done_set:
                log.info(f">>> {asset} step={step} — SKIPPED")
                continue
            log.info(f">>> {asset} step={step}")
            t0 = time.time()
            df = run_density(valid, feat_cols, asset, step)
            elapsed = time.time() - t0
            if len(df) > 0:
                trades = df[df['fires']]
                log.info(f"  {asset} step={step}: {len(trades)} trades, "
                         f"mean={trades['net_bps'].mean():.1f}bps, "
                         f"total={trades['net_bps'].sum():.0f}bps, "
                         f"win={((trades['net_bps']>0).mean()):.0%}, {elapsed:.0f}s")
                all_results.append(df)
                full = pd.concat(all_results, ignore_index=True)
                full.to_csv(partial_csv, index=False)
    
    if not all_results:
        log.error("No results!"); return
    
    full = pd.concat(all_results, ignore_index=True)
    
    # Summary report
    lines = ["# Density Sweep: rolling_500 at Multiple Step Sizes", "",
             "| Asset | Step | Eval Points | Fires | Fire Rate | Mean PnL | Total PnL | Win Rate |",
             "|---|---|---|---|---|---|---|---|"]
    for asset in ASSETS:
        for step in STEPS:
            adf = full[(full['asset']==asset) & (full['step']==step)]
            trades = adf[adf['fires']]
            n_eval = len(adf)
            n_fire = len(trades)
            fire_rate = n_fire / n_eval if n_eval > 0 else 0
            if n_fire > 0:
                lines.append(f"| {asset} | {step} | {n_eval} | {n_fire} | {fire_rate:.0%} | "
                             f"{trades['net_bps'].mean():.1f} | {trades['net_bps'].sum():.0f} | "
                             f"{(trades['net_bps']>0).mean():.0%} |")
            else:
                lines.append(f"| {asset} | {step} | {n_eval} | 0 | 0% | — | — | — |")
    
    p = rpt / "density_sweep_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Saved report to {p}")

if __name__ == "__main__":
    main()
