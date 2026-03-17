import os
import sys
import json
import requests
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.config_loader import config

LOSS_THRESHOLD = -0.10  # Evaluate any trade where per-share M2M PnL bled more than 10 cents

def reflect_on_trade(trade, prediction_data):
    """Hits the local Ollama daemon to extract a structural heuristic from a bleeding trade."""
    print(f"Triggering Llama 3.2 Post-Mortem on bleeding trade {trade['id']}... (PnL: {trade['pnl']:.2f})")
    
    prompt = f"""You are an elite quantitative algorithm architect. You must evaluate a failed prediction market trade and generate a strict new trading law to prevent the exact same reasoning flaw.
    
    THE FAILED TRADE:
    Question: {trade['question']}
    Your Position: You explicitly bought {trade['side']} at {trade['fill_price']:.2f}.
    The Result: The market structurally swung against you, causing a mathematical loss of {trade['pnl']:.2f}.
    
    THE CONTEXT YOU USED:
    {prediction_data['context_used']}
    
    YOUR EXACT REASONING THAT CAUSED THIS LOSS:
    {prediction_data['reasoning']}
    
    TASK:
    Analyze exactly why your logic was objectively wrong. Did you overweight stale news? Did you misinterpret geopolitical panic? Did you ignore liquidity?
    Extract the core vulnerability and output ONE SINGLE SENTENCE that defines a new strict trading rule to override this heuristic vulnerability globally.
    Be ruthless and structurally specific. Do not use generic answers like 'be careful'.
    """

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "stream": False
    }

    try:
         resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
         resp.raise_for_status()
         data = resp.json()
         return data.get('response', 'Always adhere strictly to live orderbook momentum dynamically.').strip()
    except Exception as e:
         print(f"Reflexion LLM Inference Failed: {e}")
         return "Never blindly overweight stale fundamental sentiment against contrarian momentum."

def run():
    print("--- Starting Agentic Reflexion Pipeline (Hippocampus) ---")
    db = DB()
    
    # We poll SQLite for exactly 5 of the worst unreflected trades that breached the LOSS threshold
    bleeding_trades = db.get_unreflected_losing_trades(loss_threshold=LOSS_THRESHOLD, limit=5)
    
    if not bleeding_trades:
        print("No severe unreflected bleeding trades detected. The system holds structural integrity.")
        return

    print(f"Analyzing {len(bleeding_trades)} structurally bleeding errors...")
    
    for trade in bleeding_trades:
        trade_id = trade['id']
        market_id = trade['market_id']
        domain = trade['exchange'] # Polymarket, Manifold, etc.
        raw_pnl = trade['pnl']
        
        # Link the trade state to the original inference logs
        preds = db.get_predictions_for_market(market_id)
        if not preds:
            print(f"Could not locate original reasoning logs for trade {trade_id}. Skipping.")
            db.log_reflection(trade_id, raw_pnl)
            continue
            
        pred_data = preds[0] # Most recent prediction generated the trade
        new_trading_law = reflect_on_trade(trade, pred_data)
        
        print(f"\n[Trade {trade_id} Autopsy]")
        print(f"Loss: ${raw_pnl:.2f}")
        print(f"Generated Rule: {new_trading_law}")
        
        # Structurally inject the law into SQL Memory with a Weight identically anchored to the PnL magnitude
        db.save_trading_rule(rule_text=new_trading_law, domain=domain, weight=abs(raw_pnl))
        
        # Formally mark the ledger to prevent redundant evaluating
        db.log_reflection(trade_id, raw_pnl)

    print("\n--- Daily Reflexion successfully merged into Trading Rules ---")

if __name__ == "__main__":
    run()
