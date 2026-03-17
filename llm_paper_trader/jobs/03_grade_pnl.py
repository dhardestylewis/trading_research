import sys
from pathlib import Path

# Add src to Python Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.config_loader import config
from src.api_clients.polymarket import PolymarketClient

def get_market_status(market_id: str, client: PolymarketClient):
    # Fetch specific market via Gamma events api (mocked here by fetching all active)
    markets = client.get_open_markets(limit=200)
    for m in markets:
        if m['market_id'] == market_id:
            return m
    return None

def run():
    print("--- Starting PnL Grader ---")
    db = DB(config['database']['db_path'])
    client = PolymarketClient()
    
    trades = db.get_open_trades()
    print(f"Tracking {len(trades)} open paper trades.")
    
    total_m2m_pnl = 0.0
    
    for t in trades:
        trade_id = t['id']
        market_id = t['market_id']
        side = t['side']
        fill_price = t['fill_price']
        size = t['size_shares']
        
        market_status = get_market_status(market_id, client)
        
        if not market_status:
           # Assuming resolved if missing from active list for simplicity in this demo
           # print(f"Trade {trade_id} (Market {market_id}) not found. Marked as assumed resolved.")
           continue
           
        current_prices = market_status.get('prices', {'Yes': 0.5, 'No': 0.5})
        current_mid = current_prices.get(side, 0.5)
        
        # Mark-to-Market PnL:
        # If I bought YES at $0.40, and it's now $0.60: edge = +$0.20 per share
        m2m = (current_mid - fill_price) * size
        total_m2m_pnl += m2m
        
        db.update_trade_pnl(trade_id, "OPEN", m2m)
        print(f"Trade {trade_id}: {side} bought @ {fill_price:.2f}, M2M Price: {current_mid:.2f} -> PnL: ${m2m:+.2f}")
        
    print(f"\n======================")
    print(f"Total Portfolio M2M: ${total_m2m_pnl:+.2f}")
    print(f"======================")

if __name__ == "__main__":
    run()
