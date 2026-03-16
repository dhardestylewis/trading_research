"""Coinbase Advanced Trade L2 Ingestion (Alpha-Bearing & Execution-Bearing)
Targets: Cross-venue lead/lag, order book depth, queue position.
"""
import asyncio
import json
import logging
from typing import List

try:
    import websockets
except ImportError:
    pass

log = logging.getLogger("coinbase_l2")
logging.basicConfig(level=logging.INFO)

COINBASE_WS_URL = "wss://advanced-trade-ws.coinbase.com"

class OrderBook:
    def __init__(self, product_id):
        self.product_id = product_id
        self.bids = {}
        self.asks = {}
    
    def process_update(self, changes):
        """Update local order book state."""
        for change in changes:
            price, size = float(change['price_level']), float(change['new_quantity'])
            side_dict = self.bids if change['side'] == 'bid' else self.asks
            if size == 0:
                side_dict.pop(price, None)
            else:
                side_dict[price] = size

    def get_bbo(self):
        """Get best bid and best ask."""
        best_bid = max(self.bids.keys()) if self.bids else None
        best_ask = min(self.asks.keys()) if self.asks else None
        return best_bid, best_ask
        
    def calculate_imbalance(self, levels=5):
        """Calculate alpha-bearing flow imbalance at top N levels."""
        bids_sorted = sorted(self.bids.items(), reverse=True)[:levels]
        asks_sorted = sorted(self.asks.items())[:levels]
        
        bid_vol = sum(v for p, v in bids_sorted)
        ask_vol = sum(v for p, v in asks_sorted)
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0

async def connect_l2_websocket(symbols: List[str]):
    """Connect to Coinbase Advanced Trade WS for L2 updates."""
    log.info(f"Connecting to Coinbase L2 WebSocket for {symbols}")
    
    subscribe_msg = {
        "type": "subscribe",
        "product_ids": symbols,
        "channel": "level2"
    }

    books = {sym: OrderBook(sym) for sym in symbols}
    
    try:
        if 'websockets' not in globals():
            log.warning("websockets package not installed. Mocking connection.")
            return

        async with websockets.connect(COINBASE_WS_URL) as ws:
            await ws.send(json.dumps(subscribe_msg))
            log.info("Subscribed to level2 channel.")
            
            # Listen for a few messages just for demonstration/testing
            for _ in range(50):
                msg = await ws.recv()
                data = json.loads(msg)
                
                if data.get('channel') == 'l2_data':
                    for event in data.get('events', []):
                        if event['type'] == 'snapshot' or event['type'] == 'update':
                            sym = event['product_id']
                            books[sym].process_update(event['updates'])
                            bbo = books[sym].get_bbo()
                            imb = books[sym].calculate_imbalance()
                            log.debug(f"[{sym}] BBO: {bbo} | Imbalance(5): {imb:.2f}")

    except Exception as e:
        log.error(f"WebSocket error: {e}")

if __name__ == "__main__":
    symbols_to_track = ["BTC-USD", "ETH-USD"]
    asyncio.run(connect_l2_websocket(symbols_to_track))
