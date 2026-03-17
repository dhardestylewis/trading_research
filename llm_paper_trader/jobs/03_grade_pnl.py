import sys
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to Python Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.config_loader import config

def fetch_live_price(trade, db):
    trade_id = trade['id']
    market_id = trade['market_id']
    side = trade['side']
    fill_price = trade['fill_price']
    size = trade['size_shares']
    
    # We need to map `market_id` to `exchange` by querying `markets`
    conn = db._get_conn()
    cursor = conn.execute("SELECT exchange FROM markets WHERE market_id = ?", (market_id,))
    row = cursor.fetchone()
    exchange = dict(row)['exchange'] if row else 'POLYMARKET'
    
    current_mid = fill_price  # Default safety fallback
    
    try:
        if exchange == 'POLYMARKET':
            # CLOB API perfectly accepts the string `condition_id`
            res = requests.get(f"https://clob.polymarket.com/markets/{market_id}", timeout=5)
            if res.status_code == 200:
                data = res.json()
                for t in data.get('tokens', []):
                    if t.get('outcome', '').lower() == side.lower():
                        current_mid = float(t.get('price', current_mid))
                        break
                    
        elif exchange == 'MANIFOLD':
            res = requests.get(f"https://api.manifold.markets/v0/market/{market_id}", timeout=5)
            if res.status_code == 200:
                prob = res.json().get('probability', 0.5)
                current_mid = prob if side.lower() == 'yes' else (1.0 - prob)
                
        elif exchange == 'KALSHI':
            res = requests.get(f"https://trading-api.kalshi.com/trade-api/v2/markets/{market_id}", timeout=5)
            if res.status_code == 200:
                m_data = res.json().get('market', {})
                current_mid = (m_data.get('yes_ask', 0) / 100.0) if side.lower() == 'yes' else (m_data.get('no_ask', 0) / 100.0)
    except Exception as e:
        pass # Fallback to fill price if API times out or throws

    m2m = (current_mid - fill_price) * size
    return trade_id, market_id, exchange, side, fill_price, current_mid, m2m, size


def run():
    print("--- Starting PnL Grader ---")
    db = DB(config['database']['db_path'])
    
    trades = db.get_open_trades()
    print(f"Tracking {len(trades)} open paper trades.")
    
    if not trades:
        print("No open trades. Exiting.")
        return

    print("Executing heavily concurrent live orderbook lookups...")
    total_m2m_pnl = 0.0
    
    # Use ThreadPool to aggressively fetch exactly the 324 trades with 0 loop latency
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_trade = {executor.submit(fetch_live_price, t, db): t for t in trades}
        
        for future in as_completed(future_to_trade):
            try:
                trade_id, market_id, exchange, side, fill_price, current_mid, m2m, size = future.result()
                total_m2m_pnl += m2m
                
                # Update DB asynchronously safe via sync write block
                db.update_trade_pnl(trade_id, "OPEN", m2m)
                print(f"Trade {trade_id} [{exchange}]: {side} @ {fill_price:.2f}, Live: {current_mid:.2f} -> PnL: ${m2m:+.2f}")
            except Exception as e:
                pass
        
    print(f"\n======================")
    print(f"Total Portfolio M2M: ${total_m2m_pnl:+.2f}")
    print(f"======================")

if __name__ == "__main__":
    run()
