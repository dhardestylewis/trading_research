import asyncio
import logging
import json
import websockets
from typing import List, Callable

logger = logging.getLogger("binance_live_capture")

class BinanceLiveCapture:
    """
    Connects to Binance WebSockets to stream tick-level data for target assets.
    Streams trades (@aggTrade) and top-of-book updates (@bookTicker).
    """
    def __init__(self, assets: List[str], on_message: Callable[[dict], None]):
        self.assets = [a.replace('-', '').lower() for a in assets]
        self.on_message = on_message
        self.streams = []
        for asset in self.assets:
            self.streams.append(f"{asset}@aggTrade")
            self.streams.append(f"{asset}@bookTicker")
        self.ws_url = f"wss://stream.binance.com:9443/stream?streams={'/'.join(self.streams)}"
        logger.info(f"Initialized BinanceLiveCapture for {self.assets}")

    async def connect_and_stream(self):
        """Connects to Binance streams and routes messages to the callback."""
        logger.info(f"Connecting to Binance WS: {self.ws_url}")
        while True:
            try:
                async for websocket in websockets.connect(self.ws_url):
                    try:
                        async for message in websocket:
                            data = json.loads(message)
                            if "data" in data:
                                self.on_message(data["data"])
                    except websockets.ConnectionClosed:
                        logger.warning("Websocket connection closed. Reconnecting...")
            except Exception as e:
                logger.error(f"Websocket error: {e}")
                await asyncio.sleep(5)

def run_capture(config_path: str):
    logger.info(f"Starting Binance Live Capture from {config_path}")
    # In practice called from an asyncio loop

