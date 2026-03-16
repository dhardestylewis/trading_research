"""On-Chain and Ecosystem Context Ingestion (Context-Only)
Targets: DefiLlama (TVL/Volume), Dune (Flows/Unlocks), GeckoTerminal (DEX Liquidity).
"""
import requests
import logging
import pandas as pd
from typing import Dict, Any

log = logging.getLogger("onchain_context")
logging.basicConfig(level=logging.INFO)

DEFILLAMA_URL = "https://api.llama.fi"
GECKOTERMINAL_URL = "https://api.geckoterminal.com/api/v2"

def fetch_defillama_protocols() -> pd.DataFrame:
    """Fetch TVL and volume aggregates to define ecosystem regimes."""
    log.info("Fetching DefiLlama protocol statistics")
    try:
        resp = requests.get(f"{DEFILLAMA_URL}/protocols")
        if resp.status_code == 200:
            data = resp.json()
            # Reduce to top N for context
            df = pd.DataFrame(data)[["name", "category", "chains", "tvl"]].sort_values("tvl", ascending=False).head(50)
            return df
    except Exception as e:
        log.error(f"DefiLlama error: {e}")
    return pd.DataFrame()

def fetch_dune_query(query_id: str):
    """Execute and fetch a Dune query execution result (e.g. exchange flows)."""
    # Requires DUNE_API_KEY environment variable. Providing scaffold.
    log.info(f"Fetching Dune query {query_id} (Requires API Key)")
    return pd.DataFrame()

def fetch_geckoterminal_pools(network: str) -> pd.DataFrame:
    """Fetch DEX pool data and liquidity directly from GeckoTerminal."""
    log.info(f"Fetching GeckoTerminal pools for {network}")
    try:
        resp = requests.get(f"{GECKOTERMINAL_URL}/networks/{network}/pools")
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            rows = []
            for pool in data:
                attrs = pool.get("attributes", {})
                rows.append({
                    "pool_id": pool.get("id"),
                    "name": attrs.get("name"),
                    "base_token_price_usd": float(attrs.get("base_token_price_usd", 0) or 0),
                    "reserve_in_usd": float(attrs.get("reserve_in_usd", 0) or 0),
                    "volume_usd_h24": float(attrs.get("volume_usd", {}).get("h24", 0) or 0)
                })
            return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"GeckoTerminal error: {e}")
    return pd.DataFrame()

if __name__ == "__main__":
    defillama_df = fetch_defillama_protocols()
    if not defillama_df.empty:
        log.info(f"DefiLlama Top Protocol TVLs:\n{defillama_df.head(3)}")
        
    sui_pools = fetch_geckoterminal_pools("sui-network")
    if not sui_pools.empty:
        log.info(f"GeckoTerminal Top SUI Pools:\n{sui_pools.head(3)}")
