"""
canary_multi_lane.py — Multi-Lane TabPFN Paper Trading Canary

Runs TabPFN inference on all 14 validated positive lanes every hour.
Logs signals, tracks positions, and reports daily PnL.

Lanes:
  Crypto: SOL-USD (via CCXT)
  ETFs: SPY, GLD, TLT, UUP, FXI, UVXY, XLE, FXA, FXE, XLU, EWJ, XLF, INDA

Usage:
    python canary_multi_lane.py              # score all lanes
    python canary_multi_lane.py --asset SPY  # score one lane
"""
from __future__ import annotations
import argparse, json, logging, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("canary_multi")

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

# ── Config ──────────────────────────────────────────────────────
LOOKBACK = 300
PCA_DIM = 47
MAX_TRAIN = 1000
TABPFN_BINS = 10
THRESHOLD = 0.25  # top-decile conviction
HORIZON = 8

CRYPTO_LANES = {
    "SOL-USD": {"cost_bps": 14, "source": "ccxt"},
}

ETF_LANES = {
    "SPY": {"cost_bps": 5},  "GLD": {"cost_bps": 5},
    "TLT": {"cost_bps": 5},  "UUP": {"cost_bps": 5},
    "FXI": {"cost_bps": 5},  "UVXY": {"cost_bps": 10},
    "XLE": {"cost_bps": 5},  "FXA": {"cost_bps": 5},
    "FXE": {"cost_bps": 5},  "XLU": {"cost_bps": 5},
    "EWJ": {"cost_bps": 5},  "XLF": {"cost_bps": 5},
    "INDA": {"cost_bps": 5},
}

SIGNAL_LOG = Path("reports/canary/multi_lane_signals.csv")
POSITION_LOG = Path("reports/canary/multi_lane_positions.csv")
DAILY_REPORT = Path("reports/canary/daily_report.md")
STATE_PATH = Path("data/artifacts/canary_multi_state.json")


def fetch_crypto(symbol: str, limit: int = LOOKBACK) -> pd.DataFrame:
    """Fetch crypto bars via CCXT."""
    import ccxt
    exchange = ccxt.coinbase({"enableRateLimit": True})
    exchange.load_markets()
    pair = symbol.replace("-", "/")
    bars = exchange.fetch_ohlcv(pair, timeframe="1h", limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp_ms","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["asset"] = symbol
    df["dollar_volume"] = df["close"] * df["volume"]
    return df


def fetch_etf(ticker: str, limit: int = LOOKBACK) -> pd.DataFrame:
    """Fetch ETF bars via yfinance."""
    import yfinance as yf
    data = yf.download(ticker, period="1mo", interval="1h", progress=False)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index()
    ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={ts_col:'timestamp','Open':'open','High':'high',
                             'Low':'low','Close':'close','Volume':'volume'})
    df['asset'] = ticker
    df['dollar_volume'] = df['close'] * df['volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.tail(limit)


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build rich features."""
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    rich = build_rich_perp_state_features(df, horizons=[8])
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low','close',
               'volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    return rich, feat_cols


def score_lane(rich: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Train TabPFN on history and score the latest bar."""
    target = 'fwd_ret_8'
    valid = rich[rich[target].notna()].copy().reset_index(drop=True)
    if len(valid) < 50:
        return {"status": "insufficient_data"}

    train = valid.iloc[:-1].copy()
    latest = valid.iloc[-1:]

    if len(train) > MAX_TRAIN:
        recent = train.iloc[-MAX_TRAIN//2:]
        older = train.iloc[:-MAX_TRAIN//2].sample(n=MAX_TRAIN//2, random_state=42)
        train = pd.concat([older, recent]).sort_index()

    X_train = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = latest[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[target].values * 10000

    nc = min(PCA_DIM, X_train.shape[1], X_train.shape[0])
    scaler = StandardScaler()
    pca = PCA(n_components=nc, random_state=42)
    Xp = pca.fit_transform(scaler.fit_transform(X_train))
    Xt = pca.transform(scaler.transform(X_test))

    bins = pd.qcut(y, TABPFN_BINS, labels=False, duplicates='drop')
    clf = TabPFNClassifier(device='cpu', N_ensemble_configurations=4)
    clf.fit(Xp, bins)

    proba = clf.predict_proba(Xt)[0]
    top_conv = float(sum(proba[-2:])) if len(proba) >= 2 else float(proba[-1])
    fires = top_conv > THRESHOLD

    return {
        "status": "ok",
        "timestamp": str(latest['timestamp'].iloc[0]),
        "top_conviction": top_conv,
        "fires": fires,
        "n_train": len(train),
    }


def log_signal(asset: str, result: dict, cost_bps: float):
    """Append signal to unified CSV log."""
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "tick_time": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "data_timestamp": result.get("timestamp"),
        "top_conviction": result.get("top_conviction"),
        "fires": result.get("fires", False),
        "cost_bps": cost_bps,
        "n_train": result.get("n_train"),
        "status": result.get("status"),
    }
    df = pd.DataFrame([record])
    header = not SIGNAL_LOG.exists()
    df.to_csv(SIGNAL_LOG, mode='a', header=header, index=False)


def update_positions(asset: str, result: dict, cost_bps: float):
    """Track paper positions — open on fire, close after HORIZON bars."""
    POSITION_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not result.get("fires"):
        return

    record = {
        "open_time": datetime.now(timezone.utc).isoformat(),
        "asset": asset,
        "conviction": result.get("top_conviction"),
        "cost_bps": cost_bps,
        "horizon_h": HORIZON,
        "status": "OPEN",
        "close_time": None,
        "pnl_bps": None,
    }

    if POSITION_LOG.exists():
        positions = pd.read_csv(POSITION_LOG)
    else:
        positions = pd.DataFrame()

    positions = pd.concat([positions, pd.DataFrame([record])], ignore_index=True)
    positions.to_csv(POSITION_LOG, index=False)
    log.info(f"  📊 POSITION OPENED: {asset} (conviction={result['top_conviction']:.3f})")


def generate_daily_report():
    """Generate a daily summary report."""
    if not SIGNAL_LOG.exists():
        return

    signals = pd.read_csv(SIGNAL_LOG)
    signals['tick_time'] = pd.to_datetime(signals['tick_time'])
    today = signals[signals['tick_time'] > (datetime.now(timezone.utc).replace(hour=0, minute=0))]

    lines = [
        f"# Multi-Lane Canary — Daily Report",
        f"**Generated**: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Today's Signals",
        f"- Total ticks: **{len(today)}**",
        f"- Fires: **{today['fires'].sum()}**",
        "",
    ]

    if today['fires'].sum() > 0:
        fired = today[today['fires']]
        lines.append("| Asset | Conviction | Time |")
        lines.append("|---|---|---|")
        for _, r in fired.iterrows():
            lines.append(f"| {r['asset']} | {r['top_conviction']:.3f} | {r['tick_time']} |")

    lines.extend(["", "## Cumulative Stats"])
    all_fires = signals[signals['fires']]
    by_asset = all_fires.groupby('asset').size()
    lines.append(f"- Total signals ever: **{len(all_fires)}**")
    lines.append(f"- Assets that fired: **{', '.join(by_asset.index.tolist())}**")

    DAILY_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DAILY_REPORT.write_text("\n".join(lines), encoding='utf-8')
    log.info(f"Daily report: {DAILY_REPORT}")


def tick(assets: list[str] | None = None):
    """Execute one multi-lane canary tick."""
    log.info("=== Multi-Lane Canary Tick ===")
    t0 = time.time()

    all_lanes = {}
    all_lanes.update({k: {**v, "source": "ccxt"} for k, v in CRYPTO_LANES.items()})
    all_lanes.update({k: {**v, "source": "yfinance"} for k, v in ETF_LANES.items()})

    if assets:
        all_lanes = {k: v for k, v in all_lanes.items() if k in assets}

    results = {}
    for asset, cfg in all_lanes.items():
        log.info(f"  → {asset}")
        try:
            if cfg["source"] == "ccxt":
                df = fetch_crypto(asset)
            else:
                df = fetch_etf(asset)

            rich, feat_cols = build_features(df)
            result = score_lane(rich, feat_cols)
            cost = cfg["cost_bps"]

            log_signal(asset, result, cost)
            update_positions(asset, result, cost)

            if result.get("fires"):
                log.info(f"  🟢 {asset} FIRES (conviction={result['top_conviction']:.3f})")
            else:
                log.info(f"  · {asset} no signal (conv={result.get('top_conviction', 0):.3f})")

            results[asset] = result
        except Exception as e:
            log.error(f"  ✗ {asset} failed: {e}")
            log_signal(asset, {"status": f"error: {e}"}, cfg.get("cost_bps", 0))

    # Daily report
    generate_daily_report()

    # Save state
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, 'w') as f:
        json.dump({
            "last_tick": datetime.now(timezone.utc).isoformat(),
            "n_lanes": len(all_lanes),
            "n_fired": sum(1 for r in results.values() if r.get("fires")),
            "elapsed_s": time.time() - t0,
        }, f, indent=2, default=str)

    elapsed = time.time() - t0
    n_fired = sum(1 for r in results.values() if r.get("fires"))
    log.info(f"=== Tick complete: {len(results)} lanes scored, {n_fired} fired, {elapsed:.0f}s ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", nargs="*", help="Score specific assets only")
    args = parser.parse_args()
    tick(assets=args.asset)
