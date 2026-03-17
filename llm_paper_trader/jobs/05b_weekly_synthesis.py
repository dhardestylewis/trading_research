import os
import sys
import json
import requests
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from src.db import DB
from src.config_loader import config

def synthesize_weekly_laws(rules):
    print("Initiating Llama 3.2 Neocortex Synthesis over {len(rules)} daily rules...")
    
    rules_text = "\n".join([f"- [Weight: {r['weight']}] {r['rule_text']}" for r in rules])
    
    prompt = f"""You are a master quantitative strategist. Review these daily tactical rules generated over the last week.
    
    RAW MISTAKES LOG:
    {rules_text}
    
    TASK:
    Identify overlapping psychological or logical flaws. Synthesize exactly 3 overarching 'Weekly Laws' that solve these underlying patterns globally. 
    Drop contradictory or weak rules.
    
    OUTPUT FORMAT:
    Respond STRICTLY in a valid JSON list of 3 strings. No markdown, no prose.
    Example: ["Law 1", "Law 2", "Law 3"]
    """

    payload = {
        "model": "llama3.2:3b",
        "prompt": prompt,
        "format": "json",
        "stream": False
    }

    try:
         resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=45)
         resp.raise_for_status()
         data = resp.json()
         response_text = data.get('response', '[]')
         new_laws = json.loads(response_text)
         if isinstance(new_laws, list) and len(new_laws) > 0:
             return new_laws
         return ["Always enforce strict EV mathematical thresholds against live liquidity."]
    except Exception as e:
         print(f"Synthesis inference failed: {e}")
         return ["Never trust single-source semantic news over active orderbook momentum."]

def run():
    print("--- Starting Agentic Synthesis Pipeline (Neocortex) ---")
    db = DB()
    
    # Extract the top 35 active rules mathematically weighted by PnL
    active_rules = db.get_top_trading_rules(limit=35)
    
    if len(active_rules) < 5:
        print("Not enough structural rules to justify a weekly synthesis. Holding architecture.")
    else:
        new_laws = synthesize_weekly_laws(active_rules)
        rule_ids_to_delete = [r['id'] for r in active_rules]
        
        # We must formally clear the legacy granular rules out of the database context window
        with db._get_conn() as conn:
            conn.execute(f"DELETE FROM trading_rules WHERE id IN ({','.join(['?']*len(rule_ids_to_delete))})", rule_ids_to_delete)
            conn.commit()
            
        # We insert the 3 super-rules inherently initialized with a 2.0 structural weight
        print("\n[Synthesized Weekly Laws]")
        for law in new_laws:
            print(f"> {law}")
            db.save_trading_rule(rule_text=law, domain="GLOBAL_SYNTHESIS", weight=2.0)

    # Regardless of synthesis, mathematically decay EVERYTHING to organically forget obsolete regimes
    print("\nApplying 10% Temporal Decay to all memory arrays...")
    db.apply_temporal_decay(decay_factor=0.9, prune_threshold=0.5)
    
    print("--- Memory array structurally optimized ---")

if __name__ == "__main__":
    run()
