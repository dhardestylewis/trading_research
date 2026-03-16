"""
spike_lane_discovery.py — Run TabPFN walk-forward on a large universe of uncorrelated assets

Downloads hourly data via yfinance, builds features, and runs a sampled
walk-forward on each asset. Reports per-asset PnL and identifies positive lanes.

Run with: python spike_lane_discovery.py
"""
import logging, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("lane_discovery")

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

COST_BPS = 5  # ETF spreads
PCA_DIM = 47
HORIZON = 8
MAX_TRAIN = 1000
RETRAIN_EVERY = 200

# Full universe — country, sector, alternatives, vol, FX
UNIVERSE = {
    # Country / Region ETFs
    "EWJ": "Japan",
    "EWZ": "Brazil", 
    "INDA": "India",
    "EWG": "Germany",
    # Sector ETFs
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLF": "Financials",
    "XLRE": "Real Estate",
    # Alternative assets
    "SLV": "Silver",
    "DBA": "Agriculture",
    "UNG": "Natural Gas",
    "HYG": "High Yield Bonds",
    # Volatility
    "UVXY": "Volatility",
    # FX proxies
    "FXE": "Euro/USD",
    "FXY": "Yen/USD",
    "FXA": "AUD/USD",
}


def download_all():
    """Download hourly data for all assets."""
    import yfinance as yf
    
    all_dfs = []
    for ticker, desc in UNIVERSE.items():
        log.info(f"Downloading {ticker} ({desc})...")
        try:
            data = yf.download(ticker, period="2y", interval="1h", progress=False)
            if data.empty or len(data) < 200:
                log.warning(f"  {ticker}: insufficient data ({len(data)} bars)")
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            df = data.reset_index()
            ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            df = df.rename(columns={ts_col:'timestamp','Open':'open','High':'high',
                                     'Low':'low','Close':'close','Volume':'volume'})
            df['asset'] = ticker
            df['dollar_volume'] = df['close'] * df['volume']
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            all_dfs.append(df[['timestamp','asset','open','high','low','close','volume','dollar_volume']])
            log.info(f"  {ticker}: {len(df)} bars")
        except Exception as e:
            log.error(f"  {ticker}: failed — {e}")
    
    panel = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Downloaded {len(panel)} bars across {panel['asset'].nunique()} assets")
    return panel


def build_features(panel):
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    all_rich = []
    for asset in panel['asset'].unique():
        adf = panel[panel['asset']==asset].copy().sort_values('timestamp').reset_index(drop=True)
        if len(adf) < 200: continue
        try:
            rich = build_rich_perp_state_features(adf, horizons=[8])
            all_rich.append(rich)
        except Exception as e:
            log.error(f"  Feature build failed for {asset}: {e}")
    return pd.concat(all_rich, ignore_index=True) if all_rich else pd.DataFrame()


def run_asset(valid, feat_cols, asset_name, target='fwd_ret_8'):
    if len(valid) < 400:
        return pd.DataFrame()
    
    results = []
    model = None; scaler = None; pca = None; last_train = -1
    test_indices = list(range(300, len(valid), 55))
    
    for row_i in test_indices:
        if model is None or row_i - last_train > RETRAIN_EVERY:
            train = valid.iloc[max(0,row_i-500):row_i].copy()
            y = train[target].values * 10000
            if len(train) < 100: continue
            if len(train) > MAX_TRAIN:
                train = train.iloc[-MAX_TRAIN:]
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
        pnl = (real_ret * 10000 - COST_BPS) if fires else 0
        results.append({'asset':asset_name, 'fires':fires, 'pnl':pnl, 'realized_bps':real_ret*10000})
    
    return pd.DataFrame(results)


def main():
    out = Path("data/processed/lane_discovery"); out.mkdir(parents=True, exist_ok=True)
    rpt = Path("reports/spike_tsfm"); rpt.mkdir(parents=True, exist_ok=True)
    partial_csv = rpt / "lane_discovery_results.csv"
    
    # Phase 1: Download (skip if cached)
    panel_path = out / "panel.parquet"
    if panel_path.exists():
        log.info(f"Loading cached panel from {panel_path}")
        panel = pd.read_parquet(panel_path)
    else:
        panel = download_all()
        panel.to_parquet(panel_path, index=False)
    
    # Phase 2: Features (skip if cached)
    rich_path = out / "panel_rich.parquet"
    if rich_path.exists():
        log.info(f"Loading cached features from {rich_path}")
        rich = pd.read_parquet(rich_path)
    else:
        rich = build_features(panel)
        if rich.empty:
            log.error("No features built!"); return
        rich.to_parquet(rich_path, index=False)
    
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low',
               'close','volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    log.info(f"Features: {len(feat_cols)}")
    
    # Load any previously saved partial results
    done_assets = set()
    all_results = []
    if partial_csv.exists():
        prev = pd.read_csv(partial_csv)
        done_assets = set(prev['asset'].unique())
        all_results.append(prev)
        log.info(f"Resuming — {len(done_assets)} assets already done: {done_assets}")
    
    # Phase 3: Walk-forward per asset (with incremental saves)
    for asset in sorted(rich['asset'].unique()):
        if asset in done_assets:
            log.info(f">>> {asset} — SKIPPED (already done)")
            continue
        log.info(f">>> {asset}")
        t0 = time.time()
        adf = rich[rich['asset']==asset].copy().sort_values('timestamp').reset_index(drop=True)
        valid = adf[adf['fwd_ret_8'].notna()].reset_index(drop=True)
        df = run_asset(valid, feat_cols, asset)
        elapsed = time.time() - t0
        if len(df) > 0:
            trades = df[df['fires']]
            if len(trades) > 0:
                log.info(f"  {asset}: {len(trades)} trades, "
                         f"mean={trades['pnl'].mean():.1f}bps, "
                         f"win={((trades['pnl']>0).mean()):.0%}, {elapsed:.0f}s")
            else:
                log.info(f"  {asset}: 0 trades, {elapsed:.0f}s")
            all_results.append(df)
            # Incremental save after each asset
            full_so_far = pd.concat(all_results, ignore_index=True)
            full_so_far.to_csv(partial_csv, index=False)
            log.info(f"  → Saved partial results ({full_so_far['asset'].nunique()} assets so far)")
    
    if not all_results:
        log.error("No results!"); return
    
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(partial_csv, index=False)
    
    # Report
    lines = [
        "# Lane Discovery: TabPFN Walk-Forward — Expanded Universe", "",
        f"**{full['asset'].nunique()} assets tested, {COST_BPS} bps cost, {HORIZON}h horizon**", "",
        "| Asset | Category | Trades | Mean PnL | Total PnL | Win Rate | Sharpe | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    
    for asset in sorted(full['asset'].unique()):
        t = full[(full['asset']==asset) & (full['fires'])]
        cat = UNIVERSE.get(asset, "")
        if len(t) > 0:
            pnl = t['pnl']
            sh = pnl.mean() / (pnl.std()+1e-8) * np.sqrt(252)
            verdict = "✅" if pnl.mean() > 5 else ("⚠️" if pnl.mean() > 0 else "❌")
            lines.append(
                f"| {asset} | {cat} | {len(t)} | {pnl.mean():.1f} | {pnl.sum():.0f} | "
                f"{(pnl>0).mean():.0%} | {sh:.2f} | {verdict} |"
            )
        else:
            lines.append(f"| {asset} | {cat} | 0 | — | — | — | — | ❌ |")
    
    positive = full[full['fires']].groupby('asset')['pnl'].mean()
    winners = positive[positive > 5].index.tolist()
    lines.extend(["",
        f"## Summary",
        f"- Positive lanes (>5 bps): **{', '.join(winners) if winners else 'None'}**",
        f"- Total new lanes found: **{len(winners)}**",
    ])
    
    p = rpt / "lane_discovery_report.md"
    p.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Report: {p}")


if __name__ == "__main__":
    main()
