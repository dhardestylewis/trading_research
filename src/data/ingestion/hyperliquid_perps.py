"""Hyperliquid Perp Data Ingestion (Alpha-Bearing)
Targets: Funding rates, basis, perp-specific execution context.
"""
import requests
import asyncio
import json
import logging
import pandas as pd
from typing import Dict, Any

try:
    import websockets
except ImportError:
    pass

log = logging.getLogger("hyperliquid_perps")
logging.basicConfig(level=logging.INFO)

HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz/info"
HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"

def fetch_meta_and_funding():
    """Fetch global perpetuals state from Hyperliquid info endpoint."""
    log.info("Fetching Hyperliquid info (funding, mark prices)")
    try:
        response = requests.post(
            HYPERLIQUID_API_URL, 
            json={"type": "metaAndAssetCtxs"},
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            data = response.json()
            if len(data) == 2:
                meta, ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                
                rows = []
                for i, asset in enumerate(universe):
                    ctx = ctxs[i]
                    rows.append({
                        "asset": asset["name"],
                        "mark_px": float(ctx["markPx"]),
                        "funding": float(ctx["funding"]),
                        "open_interest": float(ctx["openInterest"]),
                        "day_volume": float(ctx["dayNtlVlm"])
                    })
                return pd.DataFrame(rows)
    except Exception as e:
        log.error(f"Error fetching Hyperliquid data: {e}")
    return pd.DataFrame()

async def connect_l2_websocket(symbols: list):
    """Connect to Hyperliquid WS for l2Book and trades."""
    log.info(f"Connecting to Hyperliquid WS for {symbols}")
    
    if 'websockets' not in globals():
        log.warning("websockets not installed. Skipping.")
        return

    try:
        async with websockets.connect(HYPERLIQUID_WS_URL) as ws:
            for symbol in symbols:
                msg = {
                    "method": "subscribe",
                    "subscription": {"type": "l2Book", "coin": symbol}
                }
                await ws.send(json.dumps(msg))
            
            for _ in range(20):
                resp = await ws.recv()
                data = json.loads(resp)
                if data.get("channel") == "l2Book":
                    coin = data.get("data", {}).get("coin")
                    levels = data.get("data", {}).get("levels", [])
                    log.debug(f"[{coin}] L2 Update Received: {len(levels[0])} bids, {len(levels[1])} asks")
    except Exception as e:
        log.error(f"Hyperliquid WS error: {e}")

if __name__ == "__main__":
    df = fetch_meta_and_funding()
    if not df.empty:
        log.info(f"Fetched {len(df)} perp contexts from Hyperliquid.")
        print(df.head())
    
    symbols_to_track = ["BTC", "SOL"]
    asyncio.run(connect_l2_websocket(symbols_to_track))
