import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to Python Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.config_loader import config
from src.api_clients.polymarket import PolymarketClient
from src.api_clients.manifold import ManifoldClient
from src.api_clients.kalshi import KalshiSandboxClient
import os
from dotenv import load_dotenv

load_dotenv()

def run():
    print("--- Starting Cross-Exchange Market Scanner ---")
    db = DB(config['database']['db_path'])
    
    # Initialize available clients natively
    clients = {
        "POLYMARKET": PolymarketClient(),
        "MANIFOLD": ManifoldClient(api_key=os.getenv("MANIFOLD_API_KEY")),
        "KALSHI": KalshiSandboxClient(
            key_id=os.getenv("KALSHI_API_KEY"), 
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY")
        )
    }
    
    thresholds = config['strategy']
    min_volume = thresholds.get('min_volume', 50000.0)
    
    total_saved = 0
    now = datetime.now(timezone.utc).isoformat()
    
    for exchange_name, client in clients.items():
        print(f"Fetching open markets from {exchange_name}...")
        try:
            markets = client.get_open_markets(limit=200)
            
            saved_count = 0
            for m in markets:
                vol = float(m.get('volume', 0))
                
                # Filter strictly by liquidity constraint for Polymarket only
                if exchange_name == "POLYMARKET" and vol < min_volume:
                    continue
                # Manifold and Kalshi Demo liquidity is lower; ingest all for Paper Trading Edge
                
                # Save to DB structure natively
                db.save_market(
                    market_id=str(m.get('id', m.get('market_id'))),
                    title=m.get('question', m.get('event_title', 'Unknown')),
                    resolution_date=m.get('resolution_date', now),
                    volume=vol,
                    question=m.get('question', 'Unknown'),
                    exchange=m.get('exchange', exchange_name)
                )
                saved_count += 1
                
            print(f" -> Saved {saved_count} liquid markets from {exchange_name}.")
            total_saved += saved_count
            
        except Exception as e:
            print(f" -> FATAL ALIGNMENT ERROR on {exchange_name}: {e}")
            
    print(f"==========================================")
    print(f"Cross-Exchange Pipeline: {total_saved} total events synced.")

if __name__ == "__main__":
    run()
