import os
import sys
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.api_clients.polymarket import PolymarketClient
from src.api_clients.manifold import ManifoldClient
from src.api_clients.kalshi import KalshiSandboxClient

PROFIT_TAKE_THRESHOLD = 0.15  # +15 cents absolute swing
STOP_LOSS_THRESHOLD = -0.15   # -15 cents absolute swing

def run():
    print("--- Starting Dynamic Exit Manager ---")
    db = DB()
    open_trades = db.get_open_trades()
    open_markets = db.get_open_markets()
    
    if not open_trades:
        print("No open trades actively swinging. Exiting.")
        return

    # Initialize all API clients
    poly_client = PolymarketClient()
    mani_client = ManifoldClient()
    kalshi_client = KalshiSandboxClient()

    print("Pre-fetching live orderbooks to hunt for exit thresholds...")
    price_map = {}
    
    # Polymarket Pull
    try:
        p_markets = poly_client.get_open_markets(limit=200)
        for m in p_markets:
            price_map[m['condition_id']] = {'yes': m.get('yes_price', 0.5), 'no': m.get('no_price', 0.5)}
    except Exception as e:
        print(f"Polymarket API Error: {e}")

    # Manifold Pull
    try:
        m_markets = mani_client.get_open_markets(limit=200)
        for m in m_markets:
            price_map[str(m['id'])] = {'yes': m.get('probability', 0.5), 'no': 1 - m.get('probability', 0.5)}
    except Exception as e:
        print(f"Manifold API Error: {e}")

    # Kalshi Pull
    try:
        k_markets = kalshi_client.get_open_markets(limit=200)
        for m in k_markets:
            price_map[str(m['ticker'])] = {'yes': m.get('yes_ask', 0.5), 'no': m.get('no_ask', 0.5)}
    except Exception as e:
        print(f"Kalshi API Error: {e}")

    print(f"Scanning {len(open_trades)} open positions against live liquidity...\n")

    closed_count = 0
    for idx, trade in enumerate(open_trades):
        market_id = trade['market_id']
        side = trade['side']
        fill_price = trade['fill_price']
        trade_id = trade['id']

        if market_id not in price_map:
             print(f"Trade {trade_id}: Market {market_id} not found in live pool (possibly resolved). Manual review needed.")
             continue
             
        # Extract live M2M
        live_price = price_map[market_id].get(side.lower(), 0.5)
        m2m_pnl = live_price - fill_price

        # Check Triggers
        if m2m_pnl >= PROFIT_TAKE_THRESHOLD:
            print(f">>> PROFIT TAKE TRIGGERED: Trade {trade_id} ({side} @ ${fill_price:.2f}) swung to ${live_price:.2f} -> Locked in +${m2m_pnl:.2f} PnL.")
            db.update_trade_pnl(trade_id, 'CLOSED', m2m_pnl)
            closed_count += 1
            
        elif m2m_pnl <= STOP_LOSS_THRESHOLD:
            print(f">>> STOP LOSS TRIGGERED: Trade {trade_id} ({side} @ ${fill_price:.2f}) bled to ${live_price:.2f} -> Truncated loss at ${m2m_pnl:.2f} PnL.")
            db.update_trade_pnl(trade_id, 'CLOSED', m2m_pnl)
            closed_count += 1
            
        else:
             print(f"Trade {trade_id}: {side} @ ${fill_price:.2f} -> Live: ${live_price:.2f} (M2M: ${m2m_pnl:.2f}) [HOLD]")

    # Update trailing M2M for all trades (including holds)
    db.get_conn = lambda: db._get_conn() # dummy
    # Actually wait, `03_grade_pnl.py` does exactly this for all OPEN trades.
    # The exit manager strictly closes trades.

    print(f"\n======================")
    print(f"Exit Manager cycle complete. Successfully closed {closed_count} highly volatile positions.")
    print(f"======================")

if __name__ == "__main__":
    run()
