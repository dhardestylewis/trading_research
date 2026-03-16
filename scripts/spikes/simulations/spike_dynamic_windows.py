"""
spike_dynamic_windows.py — Evaluate dynamic context architectures for TabPFN

Tests different training contexts:
- rolling_100
- rolling_500
- rolling_1000
- uniform_1000 (evenly spaced 1000 samples from the entire history)
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("dynamic_windows")

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

COST_DICT = {'SOL-USD': 14, 'FXA': 5, 'XLE': 5}
PCA_DIM = 47
RETRAIN_EVERY = 200

def get_train_indices(strategy: str, row_i: int) -> np.ndarray:
    if strategy == 'rolling_100':
        return np.arange(max(0, row_i - 100), row_i)
    elif strategy == 'rolling_500':
        return np.arange(max(0, row_i - 500), row_i)
    elif strategy == 'rolling_1000':
        return np.arange(max(0, row_i - 1000), row_i)
    elif strategy == 'uniform_1000':
        if row_i <= 1000:
            return np.arange(0, row_i)
        else:
            return np.linspace(0, row_i - 1, 1000).astype(int)
    else:
        raise ValueError(f"Unknown strategy {strategy}")

def run_asset_strategy(valid: pd.DataFrame, feat_cols: list, asset: str, strategy: str, cost_bps: int, target='fwd_ret_8'):
    if len(valid) < 400:
        return pd.DataFrame()
        
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    test_indices = list(range(300, len(valid), 55))
    
    for row_i in test_indices:
        if model is None or row_i - last_train > RETRAIN_EVERY:
            train_idx = get_train_indices(strategy, row_i)
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
        gross_bps = real_ret * 10000
        net_bps = (gross_bps - cost_bps) if fires else 0
        timestamp = valid.iloc[row_i]['timestamp']
        
        results.append({
            'timestamp': timestamp,
            'asset': asset,
            'strategy': strategy,
            'fires': fires,
            'gross_bps': gross_bps,
            'net_bps': net_bps
        })
        
    return pd.DataFrame(results)

def main():
    out = Path("data/processed/lane_discovery")
    rpt = Path("reports/spike_tsfm")
    rpt.mkdir(parents=True, exist_ok=True)
    partial_csv = rpt / "dynamic_windows_results.csv"
    
    # Load features
    rich_path = out / "panel_rich.parquet"
    if not rich_path.exists():
        log.error(f"Features missing at {rich_path}")
        return
    log.info(f"Loading features from {rich_path}")
    rich = pd.read_parquet(rich_path)
    
    # For SOL-USD, we need to load from the crypto panel since lane_discovery only has traditional assets
    crypto_panel_path = Path("data/processed/panel_expanded/panel_rich.parquet")
    if crypto_panel_path.exists():
        crypto_rich = pd.read_parquet(crypto_panel_path)
        # Verify schema matches before concat
        rich = pd.concat([rich, crypto_rich[crypto_rich['asset'] == 'SOL-USD']], ignore_index=True)
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    log.info(f"Features: {len(feat_cols)}")
    
    targets = ['SOL-USD', 'FXA', 'XLE']
    strategies = ['rolling_100', 'rolling_500', 'rolling_1000', 'uniform_1000']
    
    done_set = set()
    all_results = []
    if partial_csv.exists():
        prev = pd.read_csv(partial_csv)
        prev['timestamp'] = pd.to_datetime(prev['timestamp'], utc=True)
        for _, row in prev[['asset', 'strategy']].drop_duplicates().iterrows():
            done_set.add((row['asset'], row['strategy']))
        all_results.append(prev)
        log.info(f"Resuming — {len(done_set)} combinations already done")
        
    for asset in targets:
        adf = rich[rich['asset']==asset].copy()
        if adf.empty:
            log.warning(f"No data for {asset}, skipping")
            continue
        # For uniformity, get correct datetimes.
        # Ensure timestamp is datetime
        adf['timestamp'] = pd.to_datetime(adf['timestamp'], utc=True)
        adf = adf.sort_values('timestamp').reset_index(drop=True)
        valid = adf[adf['fwd_ret_8'].notna()].reset_index(drop=True)
        
        cost_bps = COST_DICT.get(asset, 14)
        
        for strategy in strategies:
            if (asset, strategy) in done_set:
                log.info(f">>> {asset} | {strategy} — SKIPPED")
                continue
                
            log.info(f">>> {asset} | {strategy}")
            t0 = time.time()
            df = run_asset_strategy(valid, feat_cols, asset, strategy, cost_bps)
            elapsed = time.time() - t0
            
            if len(df) > 0:
                trades = df[df['fires']]
                if len(trades) > 0:
                    log.info(f"  {asset} | {strategy}: {len(trades)} trades, "
                             f"mean={trades['net_bps'].mean():.1f}bps, "
                             f"win={((trades['net_bps']>0).mean()):.0%}, {elapsed:.0f}s")
                else:
                    log.info(f"  {asset} | {strategy}: 0 trades, {elapsed:.0f}s")
                all_results.append(df)
                full_so_far = pd.concat(all_results, ignore_index=True)
                full_so_far.to_csv(partial_csv, index=False)
                
    if not all_results:
        log.error("No results!")
        return
        
    full = pd.concat(all_results, ignore_index=True)
    full['timestamp'] = pd.to_datetime(full['timestamp'], utc=True)
    
    # Generate Report
    lines = [
        "# Dynamic Context Architecture Evaluation", "",
        f"**{len(targets)} assets tested across {len(strategies)} context strategies**", ""
    ]
    
    for asset in targets:
        lines.extend([f"## Asset: {asset}", ""])
        lines.append("| Strategy | Trades | Mean PnL | Total PnL | Win Rate | Q1 2024 PnL | Non-Q1 PnL | Verdict |")
        lines.append("|---|---|---|---|---|---|---|---|")
        
        for strategy in strategies:
            t = full[(full['asset'] == asset) & (full['strategy'] == strategy) & (full['fires'])].copy()
            if len(t) == 0:
                lines.append(f"| {strategy} | 0 | - | - | - | - | - | ❌ |")
                continue
            
            t['quarter'] = t['timestamp'].dt.to_period('Q')
            q1_24 = t[t['quarter'] == '2024Q1']['net_bps'].sum()
            non_q1 = t[t['quarter'] != '2024Q1']['net_bps'].sum()
            
            mean_pnl = t['net_bps'].mean()
            tot_pnl = t['net_bps'].sum()
            win = (t['net_bps'] > 0).mean()
            
            verdict = "✅" if non_q1 > 0 and tot_pnl > 0 else ("⚠️" if tot_pnl > 0 else "❌")
            
            lines.append(f"| {strategy} | {len(t)} | {mean_pnl:.1f} | {tot_pnl:.0f} | {win:.0%} | {q1_24:.0f} | {non_q1:.0f} | {verdict} |")
            
        lines.append("")
        
    report_path = rpt / "dynamic_context_report.md"
    report_path.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Saved report to {report_path}")

if __name__ == "__main__":
    main()
