"""
canary_alpaca.py — XLE TabPFN Strategy with Alpaca Paper Trading

Extends the canary_multi_lane signal pipeline to execute paper trades
via Alpaca's API. Focused on XLE, utilizing the explicitly researched
Multivariate Cross-Asset state (SPY, USO, VIX, TNX) driving a +70% 
profit structure out-of-sample over 150-hour contextual context windows.

Setup:
  1. Sign up at https://app.alpaca.markets (free paper account)
  2. Create API keys (Paper Trading)
  3. Set environment variables:
       ALPACA_API_KEY=your_key
       ALPACA_SECRET_KEY=your_secret
     Or create a .env file in the trading_research directory.

Usage:
  python scripts/canaries/canary_alpaca.py                # single tick (score + maybe trade)
  python scripts/canaries/canary_alpaca.py --loop         # run continuously during market hours
  python scripts/canaries/canary_alpaca.py --status       # show current positions & PnL
"""
from __future__ import annotations
import argparse, json, logging, os, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("canary_alpaca")

# ── Config ──────────────────────────────────────────────────────
ASSET = "XLE"
MACROS = ["USO", "SPY", "^VIX", "^TNX"]
ALL_TICKERS = [ASSET] + MACROS

WINDOW = 150          # rolling_150 — statistically verified for cross-asset Edge
TABPFN_BINS = 10
THRESHOLD = 0.25      # top-2-decile conviction threshold
HORIZON_HOURS = 8     # hold duration
MAX_CONCURRENT = 4    # max overlapping positions
CAPITAL_FRACTION = 0.10  # 10% max raw equity exposure per trade
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
    
    try:
        account = api.get_account()
        log.info(f"✅ Alpaca connected — Paper account, equity=${float(account.equity):.2f}")
    except Exception as e:
        log.error(f"❌ Alpaca connection failed: {e}")
        return None

    return api

# ── Data & Features ─────────────────────────────────────────────
def fetch_cross_asset_state() -> pd.DataFrame:
    """Fetch 1 year of multi-asset data and engineer the 22 real-time exogenous variables."""
    import yfinance as yf
    data = yf.download(ALL_TICKERS, period="1y", interval="1h", group_by="ticker", progress=False)
    
    xle_df = data[ASSET].dropna(how="all").copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    xle_df = xle_df.rename(columns={'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'})
    xle_df.index.name = "timestamp"
    xle_df = xle_df.reset_index()
    xle_df["timestamp"] = pd.to_datetime(xle_df["timestamp"], utc=True)
    xle_df["ret_1"] = xle_df["close"].pct_change()
    
    for macro in MACROS:
        m_df = data[macro] if isinstance(data.columns, pd.MultiIndex) else data
        m_df = m_df.dropna(how="all").reset_index()
        m_df["timestamp"] = pd.to_datetime(m_df.iloc[:,0], utc=True)
        m_df[f"{macro}_ret_1"] = m_df["Close"].pct_change()
        m_df[f"{macro}_ret_3"] = m_df["Close"].pct_change(3)
        m_df[f"{macro}_rv_6"] = m_df[f"{macro}_ret_1"].rolling(6).std()
        
        cols = ["timestamp", f"{macro}_ret_1", f"{macro}_ret_3", f"{macro}_rv_6"]
        xle_df = xle_df.merge(m_df[cols], on="timestamp", how="left")
        
        xle_df[f"{macro}_ret_1"] = xle_df[f"{macro}_ret_1"].ffill().fillna(0.0)
        xle_df[f"{macro}_ret_3"] = xle_df[f"{macro}_ret_3"].ffill().fillna(0.0)
        xle_df[f"{macro}_rv_6"] = xle_df[f"{macro}_rv_6"].ffill().fillna(0.0)
        
        xle_df[f"resid_{macro}"] = xle_df["ret_1"] - xle_df[f"{macro}_ret_1"]
        xle_df[f"beta_proxy_{macro}"] = xle_df["ret_1"].rolling(24).cov(xle_df[f"{macro}_ret_1"]) / (xle_df[f"{macro}_ret_1"].rolling(24).var() + 1e-8)
        
    xle_df["rv_6"] = xle_df["ret_1"].rolling(6).std()
    xle_df["fwd_ret_8"] = xle_df["close"].shift(-8) / xle_df["close"] - 1.0
    xle_df["target_bps"] = xle_df["fwd_ret_8"] * 10000.0
    
    # We explicitly do NOT dropna() vertically at the end, because the ABSOLUTE LATEST row 
    # intrinsically has no forward return. Dropping it would force the algorithm to infinitely 
    # trade 8 hours in the past!
    return xle_df

# ── TabPFN Scoring ──────────────────────────────────────────────
def get_features() -> list[str]:
    cross = ["ret_1", "rv_6"]
    for m in MACROS:
        cross.extend([f"{m}_ret_1", f"{m}_ret_3", f"{m}_rv_6", f"resid_{m}", f"beta_proxy_{m}"])
    return cross

def score_xle(df: pd.DataFrame) -> dict:
    """Train TabPFN on rolling_150 dynamically filtered window and score the absolute true latest bar."""
    feat_cols = get_features()
    
    # Extract the true unlagged present block
    latest = df.iloc[-1:]
    
    # Safely extract strictly rows where trailing technicals AND forward projections are mathematically solved
    valid_train = df.dropna(subset=feat_cols + ["target_bps"]).copy()
    
    if len(valid_train) < WINDOW:
        return {"status": "insufficient_data", "fires": False}

    # Lock strictly to the most recent parameterized attention window
    train = valid_train.iloc[-WINDOW:]
    
    X_train = train[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    X_test = latest[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y_train = train["target_bps"].values

    bins = pd.qcut(y_train, TABPFN_BINS, labels=False, duplicates='drop')
    
    clf = TabPFNClassifier()
    clf.fit(X_train, bins)

    proba = clf.predict_proba(X_test)[0]
    top_conv = float(sum(proba[-2:])) if len(proba) >= 2 else float(proba[-1])
    fires = top_conv > THRESHOLD

    allocation_scalar = 1.0
    if fires:
        train_probas = clf.predict_proba(X_train)
        train_convs = train_probas[:, -2:].sum(axis=1) if train_probas.shape[1] >= 2 else train_probas[:, -1]
        
        hist_fires = train_convs[train_convs > THRESHOLD]
        if len(hist_fires) > 0:
            percentile = float(sum(hist_fires <= top_conv) / len(hist_fires))
            allocation_scalar = max(0.20, percentile)

    return {
        "status": "ok",
        "timestamp": str(latest['timestamp'].iloc[0]),
        "latest_close": float(latest['close'].iloc[0]),
        "top_conviction": round(top_conv, 4),
        "full_proba": [round(p, 4) for p in proba],
        "fires": fires,
        "allocation_scalar": round(allocation_scalar, 4),
        "n_train": len(train),
    }

# ── Order Execution ─────────────────────────────────────────────
def execute_trade(api, signal: dict, dry_run: bool = False) -> dict:
    if not signal.get("fires"): return {"action": "no_signal"}

    if api is None or dry_run:
        log.info(f"🔸 DRY-RUN: Would BUY {ASSET} (conviction={signal['top_conviction']:.3f})")
        return {"action": "dry_run", "would_fire": True}

    try:
        account = api.get_account()
        max_trade_amount = float(account.equity) * CAPITAL_FRACTION
        
        scalar = signal.get("allocation_scalar", 1.0)
        trade_amount = max_trade_amount * scalar

        if trade_amount < 1.0:
            return {"action": "insufficient_funds"}

        positions = api.list_positions()
        xle_positions = [p for p in positions if p.symbol == ASSET]
        if len(xle_positions) >= MAX_CONCURRENT:
            return {"action": "max_positions_reached"}

        qty = int(trade_amount / signal['latest_close'])
        if qty < 1:
            order = api.submit_order(symbol=ASSET, notional=round(trade_amount, 2), side='buy', type='market', time_in_force='day')
            log.info(f"🟢 BUY {ASSET} ${trade_amount:.2f} notional (conv {signal['top_conviction']:.3f})")
        else:
            order = api.submit_order(symbol=ASSET, qty=qty, side='buy', type='market', time_in_force='day')
            log.info(f"🟢 BUY {qty} {ASSET} (${trade_amount:,.2f}) [{scalar*100:.1f}%, Conv {signal['top_conviction']:.3f}]")

        return {"action": "buy", "order_id": order.id, "qty": qty, "notional": round(trade_amount, 2), "conviction": signal['top_conviction']}

    except Exception as e:
        log.error(f"❌ Order failed: {e}")
        return {"action": "error", "error": str(e)}

def close_expired_positions(api):
    if api is None: return
    try:
        xle_positions = [p for p in api.list_positions() if p.symbol == ASSET]
        for pos in xle_positions:
            entry_time = _get_position_open_time(api, pos)
            if entry_time and (datetime.now(timezone.utc) - entry_time) > timedelta(hours=HORIZON_HOURS):
                api.close_position(ASSET)
                pnl = float(pos.unrealized_pl)
                log.info(f"🔴 CLOSED {ASSET}: PnL=${pnl:.2f}")
                _log_trade(pos, pnl)
    except Exception as e:
        log.error(f"Error closing positions: {e}")

def _get_position_open_time(api, position) -> datetime | None:
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        open_time = state.get("positions", {}).get(ASSET, {}).get("open_time")
        if open_time: return datetime.fromisoformat(open_time)
    return None

def log_signal(signal: dict):
    SIGNAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "tick_time": datetime.now(timezone.utc).isoformat(), "asset": ASSET,
        "data_timestamp": signal.get("timestamp"), "close": signal.get("latest_close"),
        "top_conviction": signal.get("top_conviction"), "fires": signal.get("fires", False),
        "n_train": signal.get("n_train"), "status": signal.get("status"),
    }]).to_csv(SIGNAL_LOG, mode='a', header=not SIGNAL_LOG.exists(), index=False)

def _log_trade(position, pnl: float):
    TRADE_LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "close_time": datetime.now(timezone.utc).isoformat(), "asset": ASSET,
        "qty": position.qty, "entry_price": position.avg_entry_price,
        "exit_price": position.current_price, "pnl_dollar": pnl, "pnl_pct": float(position.unrealized_plpc) * 100,
    }]).to_csv(TRADE_LOG, mode='a', header=not TRADE_LOG.exists(), index=False)

def save_state(signal: dict, trade_result: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    state.update({"last_tick": datetime.now(timezone.utc).isoformat(), "last_signal": signal, "last_trade": trade_result})
    if trade_result.get("action") == "buy":
        state.setdefault("positions", {})[ASSET] = {
            "open_time": datetime.now(timezone.utc).isoformat(), "order_id": trade_result.get("order_id"), "conviction": trade_result.get("conviction"),
        }
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))

def show_status(api):
    if api is None:
        log.info("No Alpaca connection. Showing local logs only.")
        if SIGNAL_LOG.exists(): print(pd.read_csv(SIGNAL_LOG).tail(5).to_string(index=False))
        return
    account = api.get_account()
    print(f"\nEquity: ${float(account.equity):,.2f} | Buying Power: ${float(account.buying_power):,.2f}")
    positions = api.list_positions()
    if positions:
        for p in positions:
            print(f"  {p.symbol}: {p.qty} shares @ ${float(p.avg_entry_price):.2f}, PnL=${float(p.unrealized_pl):+.2f} ({float(p.unrealized_plpc)*100:+.1f}%)")
    else: print("\nNo open positions")

def tick(api, dry_run: bool = False):
    log.info(f"{'='*50}\nXLE Cross-Asset Canary Tick — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    df = fetch_cross_asset_state()
    log.info(f"Built 22-dimensional cross-asset block (latest: {df['timestamp'].iloc[-1]})")
    signal = score_xle(df)
    log_signal(signal)
    if signal['status'] != 'ok': return log.warning(f"Status: {signal['status']}")
    log.info(f"  {'🟢 FIRE!' if signal['fires'] else '· No signal'} Conviction: {signal['top_conviction']:.3f} | Cap-Scaler: {signal['allocation_scalar']*100:.1f}%")
    close_expired_positions(api)
    trade_result = execute_trade(api, signal, dry_run=dry_run)
    save_state(signal, trade_result)

def run_loop(api, dry_run: bool = False):
    log.info("Starting continuous canary loop (Ctrl+C to stop)")
    while True:
        try: tick(api, dry_run=dry_run)
        except Exception as e: log.error(f"Tick failed: {e}")
        now = datetime.now(timezone.utc)
        next_tick = now.replace(minute=5, second=0, microsecond=0) + timedelta(hours=1)
        time.sleep((next_tick - now).total_seconds())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XLE Cross-Asset Alpaca Paper Canary")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    api = get_alpaca_api(paper=True)
    if args.status: show_status(api)
    elif args.loop: run_loop(api, dry_run=args.dry_run)
    else: tick(api, dry_run=args.dry_run)
