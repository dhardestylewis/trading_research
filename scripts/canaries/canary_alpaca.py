"""
canary_alpaca.py — XLE TabPFN Strategy with Alpaca Paper Trading

Extends the canary_multi_lane signal pipeline to execute paper trades
via Alpaca's API. Focused on XLE with rolling_500, the proven durable lane.

Setup:
  1. Sign up at https://app.alpaca.markets (free paper account)
  2. Create API keys (Paper Trading)
  3. Set environment variables:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
     Or create a .env file in the trading_research directory.

Usage:
  python canary_alpaca.py                # single tick (score + maybe trade)
  python canary_alpaca.py --loop         # run continuously during market hours
  python canary_alpaca.py --status       # show current positions & PnL
  python canary_alpaca.py --dry-run      # signal only, no orders
"""
from __future__ import annotations
import argparse, json, logging, os, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("canary_alpaca")

# ── TabPFN patches ──────────────────────────────────────────────
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
ASSET = "XLE"
COST_BPS = 5
PCA_DIM = 47
WINDOW = 500          # rolling_500 — the proven best strategy
TABPFN_BINS = 10
THRESHOLD = 0.25      # top-2-decile conviction threshold
HORIZON_HOURS = 8     # hold duration
MAX_CONCURRENT = 4    # max overlapping positions
CAPITAL_FRACTION = 0.25  # 1/MAX_CONCURRENT per trade
TICK_INTERVAL = 3600  # 1 hour in seconds

SIGNAL_LOG = Path("reports/canary/alpaca_signals.csv")
TRADE_LOG = Path("reports/canary/alpaca_trades.csv")
STATE_PATH = Path("data/artifacts/canary_alpaca_state.json")

# ── Alpaca Connection ───────────────────────────────────────────
def get_alpaca_api(paper: bool = True):
    """Connect to Alpaca API."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        log.warning("⚠️  ALPACA_API_KEY / ALPACA_SECRET_KEY not set!")
        log.warning("   Set them as environment variables or in a .env file.")
        log.warning("   Running in DRY-RUN mode (no orders will be placed).")
        return None

    import alpaca_trade_api as tradeapi
    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    # Verify connection
    try:
        account = api.get_account()
        log.info(f"✅ Alpaca connected — Paper account, "
                 f"equity=${float(account.equity):.2f}, "
                 f"buying_power=${float(account.buying_power):.2f}")
    except Exception as e:
        log.error(f"❌ Alpaca connection failed: {e}")
        return None

    return api


# ── Data & Features ─────────────────────────────────────────────
def fetch_xle_data(lookback: int = 600) -> pd.DataFrame:
    """Fetch XLE hourly bars via yfinance."""
    import yfinance as yf
    data = yf.download(ASSET, period="2y", interval="1h", progress=False)
    if data.empty:
        raise RuntimeError(f"No data for {ASSET}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data.reset_index()
    ts_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df = df.rename(columns={ts_col:'timestamp','Open':'open','High':'high',
                             'Low':'low','Close':'close','Volume':'volume'})
    df['asset'] = ASSET
    df['dollar_volume'] = df['close'] * df['volume']
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build rich features from OHLCV data."""
    from src.data.build_rich_perp_state_features import build_rich_perp_state_features
    rich = build_rich_perp_state_features(df, horizons=[8])
    exclude = {'timestamp','asset','symbol','timestamp_ms','open','high','low','close',
               'volume','dollar_volume','fwd_ret_8','gross_move_bps_8',
               'prob_tail_25_8','prob_tail_50_8','prob_tail_100_8'}
    feat_cols = [c for c in rich.columns if c not in exclude and pd.api.types.is_numeric_dtype(rich[c])]
    return rich, feat_cols


# ── TabPFN Scoring ──────────────────────────────────────────────
def score_xle(rich: pd.DataFrame, feat_cols: list[str]) -> dict:
    """Train TabPFN on rolling_500 window and score the latest bar."""
    target = 'fwd_ret_8'
    valid = rich[rich[target].notna()].copy().reset_index(drop=True)
    if len(valid) < 100:
        return {"status": "insufficient_data", "fires": False}

    # Rolling 500 window
    train_end = len(valid) - 1
    train_start = max(0, train_end - WINDOW)
    train = valid.iloc[train_start:train_end].copy()
    latest = valid.iloc[-1:]

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
        "latest_close": float(latest['close'].iloc[0]),
        "top_conviction": round(top_conv, 4),
        "full_proba": [round(p, 4) for p in proba],
        "fires": fires,
        "n_train": len(train),
    }


# ── Order Execution ─────────────────────────────────────────────
def execute_trade(api, signal: dict, dry_run: bool = False) -> dict:
    """Place a paper trade on Alpaca based on the signal."""
    if not signal.get("fires"):
        return {"action": "no_signal"}

    if api is None or dry_run:
        log.info(f"🔸 DRY-RUN: Would BUY {ASSET} (conviction={signal['top_conviction']:.3f})")
        return {"action": "dry_run", "would_fire": True}

    try:
        account = api.get_account()
        buying_power = float(account.buying_power)
        trade_amount = buying_power * CAPITAL_FRACTION

        if trade_amount < 1.0:
            log.warning("Insufficient buying power")
            return {"action": "insufficient_funds"}

        # Check current positions — don't exceed MAX_CONCURRENT
        positions = api.list_positions()
        xle_positions = [p for p in positions if p.symbol == ASSET]
        if len(xle_positions) >= MAX_CONCURRENT:
            log.info(f"Max concurrent positions ({MAX_CONCURRENT}) reached, skipping")
            return {"action": "max_positions_reached"}

        # Place market buy order
        qty = int(trade_amount / signal['latest_close'])
        if qty < 1:
            # Use fractional shares
            order = api.submit_order(
                symbol=ASSET,
                notional=round(trade_amount, 2),
                side='buy',
                type='market',
                time_in_force='day'
            )
            log.info(f"🟢 BUY {ASSET} ${trade_amount:.2f} notional "
                     f"(conviction={signal['top_conviction']:.3f})")
        else:
            order = api.submit_order(
                symbol=ASSET,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='day'
            )
            log.info(f"🟢 BUY {qty} {ASSET} @ ~${signal['latest_close']:.2f} "
                     f"(conviction={signal['top_conviction']:.3f})")

        return {
            "action": "buy",
            "order_id": order.id,
            "qty": qty,
            "notional": round(trade_amount, 2),
            "conviction": signal['top_conviction'],
        }

    except Exception as e:
        log.error(f"❌ Order failed: {e}")
        return {"action": "error", "error": str(e)}


def close_expired_positions(api):
    """Close positions that have been held for >= HORIZON_HOURS."""
    if api is None:
        return

    try:
        orders = api.list_orders(status='all', limit=100, direction='asc')
        positions = api.list_positions()
        xle_positions = [p for p in positions if p.symbol == ASSET]

        if not xle_positions:
            return

        for pos in xle_positions:
            # Check when the position was opened via order history
            # For simplicity, use position's change_today as a proxy
            # In production you'd track exact open times in STATE
            entry_time = _get_position_open_time(api, pos)
            if entry_time and (datetime.now(timezone.utc) - entry_time) > timedelta(hours=HORIZON_HOURS):
                api.close_position(ASSET)
                pnl = float(pos.unrealized_pl)
                log.info(f"🔴 CLOSED {ASSET}: PnL=${pnl:.2f} "
                         f"(held {HORIZON_HOURS}h+)")
                _log_trade(pos, pnl)

    except Exception as e:
        log.error(f"Error closing positions: {e}")


def _get_position_open_time(api, position) -> datetime | None:
    """Get position open time from state file."""
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        open_time = state.get("positions", {}).get(ASSET, {}).get("open_time")
        if open_time:
            return datetime.fromisoformat(open_time)
    return None


# ── Logging ─────────────────────────────────────────────────────
def log_signal(signal: dict):
    """Append signal to CSV log."""
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "tick_time": datetime.now(timezone.utc).isoformat(),
        "asset": ASSET,
        "data_timestamp": signal.get("timestamp"),
        "close": signal.get("latest_close"),
        "top_conviction": signal.get("top_conviction"),
        "fires": signal.get("fires", False),
        "n_train": signal.get("n_train"),
        "status": signal.get("status"),
    }
    pd.DataFrame([record]).to_csv(SIGNAL_LOG, mode='a',
                                   header=not SIGNAL_LOG.exists(), index=False)


def _log_trade(position, pnl: float):
    """Log completed trade."""
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "close_time": datetime.now(timezone.utc).isoformat(),
        "asset": ASSET,
        "qty": position.qty,
        "entry_price": position.avg_entry_price,
        "exit_price": position.current_price,
        "pnl_dollar": pnl,
        "pnl_pct": float(position.unrealized_plpc) * 100,
    }
    pd.DataFrame([record]).to_csv(TRADE_LOG, mode='a',
                                   header=not TRADE_LOG.exists(), index=False)


def save_state(signal: dict, trade_result: dict):
    """Save canary state to disk."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())

    state.update({
        "last_tick": datetime.now(timezone.utc).isoformat(),
        "last_signal": signal,
        "last_trade": trade_result,
    })

    if trade_result.get("action") == "buy":
        positions = state.get("positions", {})
        positions[ASSET] = {
            "open_time": datetime.now(timezone.utc).isoformat(),
            "order_id": trade_result.get("order_id"),
            "conviction": trade_result.get("conviction"),
        }
        state["positions"] = positions

    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ── Status ──────────────────────────────────────────────────────
def show_status(api):
    """Print current account and position status."""
    if api is None:
        log.info("No Alpaca connection. Showing local logs only.")
        if SIGNAL_LOG.exists():
            signals = pd.read_csv(SIGNAL_LOG)
            log.info(f"Total signals logged: {len(signals)}")
            log.info(f"Total fires: {signals['fires'].sum()}")
            recent = signals.tail(5)
            print("\nLast 5 signals:")
            print(recent.to_string(index=False))
        return

    account = api.get_account()
    print(f"\n{'='*50}")
    print(f"Alpaca Paper Trading Status")
    print(f"{'='*50}")
    print(f"Equity:        ${float(account.equity):,.2f}")
    print(f"Buying Power:  ${float(account.buying_power):,.2f}")
    print(f"Day P&L:       ${float(account.equity) - float(account.last_equity):+,.2f}")

    positions = api.list_positions()
    if positions:
        print(f"\nOpen Positions:")
        for p in positions:
            print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f}, "
                  f"PnL=${float(p.unrealized_pl):+.2f} ({float(p.unrealized_plpc)*100:+.1f}%)")
    else:
        print("\nNo open positions")

    if SIGNAL_LOG.exists():
        signals = pd.read_csv(SIGNAL_LOG)
        print(f"\nSignal History: {len(signals)} ticks, {signals['fires'].sum()} fires")


# ── Main Loop ───────────────────────────────────────────────────
def tick(api, dry_run: bool = False):
    """Execute one canary tick: fetch → score → trade."""
    log.info(f"{'='*50}")
    log.info(f"XLE Canary Tick — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # 1. Fetch data
    log.info("Fetching XLE hourly data...")
    df = fetch_xle_data()
    log.info(f"  Got {len(df)} bars, latest: {df['timestamp'].iloc[-1]}")

    # 2. Build features
    log.info("Building features...")
    rich, feat_cols = build_features(df)
    log.info(f"  {len(feat_cols)} features, {len(rich)} rows")

    # 3. Score
    log.info("Scoring with TabPFN (rolling_500)...")
    signal = score_xle(rich, feat_cols)
    log_signal(signal)

    if signal['status'] != 'ok':
        log.warning(f"  Signal status: {signal['status']}")
        return

    if signal['fires']:
        log.info(f"  🟢 FIRE! Conviction: {signal['top_conviction']:.3f}")
    else:
        log.info(f"  · No signal (conviction: {signal['top_conviction']:.3f})")

    # 4. Close expired positions
    close_expired_positions(api)

    # 5. Execute trade if fired
    trade_result = execute_trade(api, signal, dry_run=dry_run)

    # 6. Save state
    save_state(signal, trade_result)

    log.info(f"  Tick complete. Action: {trade_result.get('action')}")
    return signal, trade_result


def run_loop(api, dry_run: bool = False):
    """Continuously tick every hour during market hours."""
    log.info("Starting continuous canary loop (Ctrl+C to stop)")
    while True:
        try:
            tick(api, dry_run=dry_run)
        except Exception as e:
            log.error(f"Tick failed: {e}")

        # Wait until next hour
        now = datetime.now(timezone.utc)
        next_tick = now.replace(minute=5, second=0, microsecond=0) + timedelta(hours=1)
        wait = (next_tick - now).total_seconds()
        log.info(f"Next tick at {next_tick.strftime('%H:%M UTC')} ({wait/60:.0f} min)")
        time.sleep(wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLE TabPFN Alpaca Paper Trading Canary")
    parser.add_argument("--loop", action="store_true", help="Run continuously every hour")
    parser.add_argument("--status", action="store_true", help="Show account status")
    parser.add_argument("--dry-run", action="store_true", help="Signal only, no orders")
    args = parser.parse_args()

    api = get_alpaca_api(paper=True)

    if args.status:
        show_status(api)
    elif args.loop:
        run_loop(api, dry_run=args.dry_run)
    else:
        tick(api, dry_run=args.dry_run)
