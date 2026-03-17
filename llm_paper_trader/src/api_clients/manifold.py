import json
import requests
from typing import List, Dict, Any

class ManifoldClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.manifold.markets/v0"
        self.headers = {"Authorization": f"Key {self.api_key}"} if self.api_key else {}

    def get_open_markets(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Fetch active, prominent binary markets from Manifold Markets.
        """
        try:
            url = f"{self.base_url}/markets"
            params = {"limit": limit} # Liquidity sorting removed to fix 400
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            markets = []
            for m in data:
                # Filter strictly for standard binary "YES/NO" markets
                if m.get('outcomeType') == "BINARY" and not m.get('isResolved'):
                    markets.append({
                        "exchange": "MANIFOLD",
                        "id": m.get("id"),
                        "question": m.get("question"),
                        # Manifold natively uses probability (0.00 to 1.00) as the "price" mechanism
                        "probability": float(m.get("probability", 0.5)),
                        "volume": float(m.get("volume", 0)),
                        "url": m.get("url")
                    })
            return markets
        except Exception as e:
            print(f"Manifold Extraction Error: {e}")
            return []

    def place_bet(self, contract_id: str, outcome: str, amount: int) -> Dict[str, Any]:
        """
        Places a mock paper-trade (bet) on Manifold using your authenticated Mana balance.
        Outcome must be 'YES' or 'NO'. Amount configures the total Mana block.
        """
        if not self.api_key:
            return {"status": "error", "message": "No Manifold API key provided. Cannot execute trade."}

        url = f"{self.base_url}/bet"
        payload = {
            "amount": amount,
            "contractId": contract_id,
            "outcome": outcome.upper()
        }
        
        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=30)
            resp.raise_for_status()
            return {"status": "success", "data": resp.json()}
        except Exception as e:
            print(f"Manifold Order Error: {e}")
            return {"status": "error", "message": str(e)}
