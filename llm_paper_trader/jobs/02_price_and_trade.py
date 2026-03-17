import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to Python Path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
load_dotenv(base_dir / ".env")

from src.db import DB
from src.config_loader import config
from src.api_clients.polymarket import PolymarketClient
from src.api_clients.manifold import ManifoldClient
from src.api_clients.kalshi import KalshiSandboxClient
from src.api_clients.search import SearchClient
from src.llm import LLMPricer
from src.strategy import TradingStrategy

def build_live_price_map(clients):
    print("Pre-fetching live orderbooks for structural mapping...")
    price_map = {"POLYMARKET": {}, "MANIFOLD": {}, "KALSHI": {}}
    
    try:
        poly_markets = clients["POLYMARKET"].get_open_markets(limit=200)
        for m in poly_markets:
            price_map["POLYMARKET"][m['market_id']] = m.get('prices', {'Yes': 0.5, 'No': 0.5})
            
        man_markets = clients["MANIFOLD"].get_open_markets(limit=200)
        for m in man_markets:
            p = m.get('probability', 0.5)
            price_map["MANIFOLD"][str(m['id'])] = {'Yes': p, 'No': 1.0 - p}
            
        kal_markets = clients["KALSHI"].get_open_markets(limit=200)
        for m in kal_markets:
            p = m.get('probability', 0.5)
            price_map["KALSHI"][str(m['id'])] = {'Yes': p, 'No': 1.0 - p}
            
        print("  -> Orderbooks synced.")
    except Exception as e:
        print(f"Failed to sync orderbooks: {e}")
        
    return price_map

def run():
    print("--- Starting Pricing & API Execution Engine ---")
    db = DB(config['database']['db_path'])
    
    # Initialize authentications natively
    clients = {
        "POLYMARKET": PolymarketClient(),
        "MANIFOLD": ManifoldClient(api_key=os.getenv("MANIFOLD_API_KEY")),
        "KALSHI": KalshiSandboxClient(
            key_id=os.getenv("KALSHI_API_KEY"), 
            private_key_path=os.getenv("KALSHI_PRIVATE_KEY")
        )
    }
    
    # The new fully-localized Open Weights Engine
    pricer = LLMPricer()
    search = SearchClient()
    strategy = TradingStrategy(
        edge_threshold=config['strategy']['edge_threshold'],
        trade_size_dollars=config['strategy']['trade_size_dollars']
    )
    
    # Pre-calculate fast cross-exchange live mapping
    price_map = build_live_price_map(clients)
    
    # Get tracking markets from our DB queue
    open_markets = db.get_open_markets()
    print(f"Evaluating {len(open_markets)} total markets from localized tracking DB.")
    
    # Extract structural Neocortex Memory to inject into the LLM
    top_rules = db.get_top_trading_rules(limit=10)
    active_rules_text = "\n".join([f"- {r['rule_text']} (Weight: {r['weight']:.2f})" for r in top_rules])
    if top_rules:
        print(f"Loaded {len(top_rules)} structural memory rules to override heuristic evaluation.")

    # Evaluate ALL 989 tracked markets concurrently
    for m in open_markets:
        market_id = str(m['market_id'])
        question = m['question']
        exchange = m.get('exchange', 'POLYMARKET')
        
        print(f"\nEvaluating on {exchange}: {question}")
        
        live_prices = price_map.get(exchange, {}).get(market_id)
        if not live_prices:
            print(f"  Skipping: Orderbook lookup failed on '{exchange}' for {market_id}.")
            continue
            
        print(f"  Live Market: {live_prices}")
            
        query = f"Latest news on: {question}"
        context = search.search_news(query=query, max_results=3)
        
        result = pricer.get_probability(question, context, active_rules=active_rules_text)
        p_llm = result.get('probability', 0.5)
        reasoning = result.get('reasoning', '')
        
        print(f"  Llama 3.2 3B Inference: Yes@{p_llm:.2f}")
        print(f"  Reasoning: {reasoning}")
        db.save_prediction(market_id, p_llm, context, reasoning)
        
        trade = strategy.evaluate_trade(market_id, p_llm, live_prices)
        
        if trade:
            print(f"  >>> TRADE SIGNAL: Buy {trade['side']} at {trade['fill_price']:.2f}")
            
            # Dynamic External Order Broadcasting
            if exchange == 'MANIFOLD':
                if clients["MANIFOLD"].api_key:
                    res = clients["MANIFOLD"].place_bet(market_id, trade['side'], amount=10)
                    print(f"  [MANIFOLD API] {res.get('status')} - Order Broadcasted.")
                else:
                    print(f"  [MANIFOLD API] Skipped - Keys missing.")
                    
            elif exchange == 'KALSHI':
                if clients["KALSHI"].is_authenticated:
                    res = clients["KALSHI"].place_order(market_id, trade['side'], count=1, price_cents=int(trade['fill_price']*100))
                    print(f"  [KALSHI API] {res.get('status')} - Order Broadcasted.")
                else:
                    print(f"  [KALSHI API] Skipped - Keys missing.")
            
            else:
                # Polymarket remains organically mocked in SQLite context
                pass
                
            db.save_trade(market_id, trade['side'], trade['fill_price'], trade['size_shares'])
            
        else:
            print("  No trade: insufficient edge calculated.")
        
        time.sleep(1)

if __name__ == "__main__":
    run()
