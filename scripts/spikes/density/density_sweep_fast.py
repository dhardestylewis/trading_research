"""
density_sweep_fast.py — Batched TabPFN inference for dramatic speedup.

Instead of calling predict_proba() once per bar, we batch all bars between
retrains into a single call. This collapses ~200 sequential inferences into 1.
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("density_fast")

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

def run_density_batched(valid: pd.DataFrame, feat_cols: list, asset: str, step: int, target='fwd_ret_8'):
    if len(valid) < 400:
        return pd.DataFrame()

    test_indices = list(range(300, len(valid), step))
    
    # Group test indices into chunks that share the same trained model
    # Model retrains happen at: first index, and whenever gap > RETRAIN_EVERY
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    
    # Collect batches: each batch = (train_at_row, [test_rows sharing that model])
    batches = []
    current_batch_indices = []
    current_train_row = None
    
    for row_i in test_indices:
        needs_retrain = (model is None and current_train_row is None) or (row_i - (current_train_row or 0) > RETRAIN_EVERY)
        if needs_retrain:
            if current_batch_indices and current_train_row is not None:
                batches.append((current_train_row, current_batch_indices))
            current_train_row = row_i
            current_batch_indices = [row_i]
        else:
            current_batch_indices.append(row_i)
    if current_batch_indices and current_train_row is not None:
        batches.append((current_train_row, current_batch_indices))
    
    log.info(f"  {len(test_indices)} eval points grouped into {len(batches)} batches")
    
    for train_row, batch_rows in batches:
        # Train
        train_idx = np.arange(max(0, train_row - WINDOW), train_row)
        if len(train_idx) < 100:
            continue
        train = valid.iloc[train_idx].copy()
        y = train[target].values * 10000
        X = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        nc = min(PCA_DIM, X.shape[1], X.shape[0])
        scaler = StandardScaler()
        pca = PCA(n_components=nc, random_state=42)
        Xp = pca.fit_transform(scaler.fit_transform(X))
        bins = pd.qcut(y, 10, labels=False, duplicates='drop')
        model = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
        model.fit(Xp, bins)
        
        # Batch inference — transform ALL test rows at once, predict_proba in one call
        test_X = valid.iloc[batch_rows][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_Xp = pca.transform(scaler.transform(test_X))
        probas = model.predict_proba(test_Xp)  # shape: (n_batch, n_classes)
        
        for i, row_i in enumerate(batch_rows):
            proba = probas[i]
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
    rpt.mkdir(parents=True, exist_ok=True)
    partial_csv = rpt / "density_sweep_fast_results.csv"
    
    rich = pd.read_parquet(out / "panel_rich.parquet")
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    log.info(f"Features: {len(feat_cols)}")
    
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
            log.warning(f"No data for {asset}"); continue
        adf['timestamp'] = pd.to_datetime(adf['timestamp'], utc=True)
        adf = adf.sort_values('timestamp').reset_index(drop=True)
        valid = adf[adf['fwd_ret_8'].notna()].reset_index(drop=True)
        
        for step in STEPS:
            if (asset, step) in done_set:
                log.info(f">>> {asset} step={step} — SKIPPED"); continue
            log.info(f">>> {asset} step={step}")
            t0 = time.time()
            df = run_density_batched(valid, feat_cols, asset, step)
            elapsed = time.time() - t0
            if len(df) > 0:
                trades = df[df['fires']]
                log.info(f"  {asset} step={step}: {len(trades)} trades, "
                         f"mean={trades['net_bps'].mean():.1f}bps, "
                         f"total={trades['net_bps'].sum():.0f}bps, "
                         f"win={((trades['net_bps']>0).mean()):.0%}, {elapsed:.1f}s")
                all_results.append(df)
                full = pd.concat(all_results, ignore_index=True)
                full.to_csv(partial_csv, index=False)
    
    if not all_results:
        log.error("No results!"); return
    
    full = pd.concat(all_results, ignore_index=True)
    
    # Summary report
    lines = ["# Density Sweep (Batched): rolling_500 at Multiple Step Sizes", "",
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
    
    p = rpt / "density_sweep_fast_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Saved report to {p}")

if __name__ == "__main__":
    main()
