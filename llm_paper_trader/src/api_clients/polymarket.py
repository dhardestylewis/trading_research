import requests
from typing import List, Dict, Any
from datetime import datetime, timezone

class PolymarketClient:
    def __init__(self):
        self.base_url = "https://gamma-api.polymarket.com"

    def get_open_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetches open events and unwraps their underlying active binary markets.
        """
        try:
            response = requests.get(
                f"{self.base_url}/events",
                params={"limit": limit, "active": "true", "closed": "false"},
                timeout=10
            )
            response.raise_for_status()
            events = response.json()
            
            extracted_markets = []
            for event in events:
                # Some events have multiple markets. We only want simple binary markets for the bot.
                for market in event.get("markets", []):
                    if not market.get("active"):
                        continue
                    if market.get("closed"):
                        continue
                        
                    # Target simple Yes/No markets
                    import json
                    outcomes_raw = market.get("outcomes", "[]")
                    if isinstance(outcomes_raw, str):
                        try:
                            outcomes = json.loads(outcomes_raw)
                        except json.JSONDecodeError:
                            outcomes = []
                    else:
                        outcomes = outcomes_raw
                        
                    if set(outcomes) == {"Yes", "No"}:
                        # Parse out prices directly from outcomePrices string array
                        prices_raw = market.get("outcomePrices", "[]")
                        if isinstance(prices_raw, str):
                            try:
                                prices_arr = json.loads(prices_raw)
                            except json.JSONDecodeError:
                                prices_arr = []
                        else:
                            prices_arr = prices_raw
                            
                        prices = {"Yes": 0.5, "No": 0.5}
                        if len(outcomes) == len(prices_arr):
                            for i, outcome in enumerate(outcomes):
                                prices[outcome] = float(prices_arr[i])
                        
                        extracted_markets.append({
                            "market_id": market.get("conditionId"),
                            "event_title": event.get("title"),
                            "question": market.get("question"),
                            "volume": float(market.get("volume", 0)),
                            "resolution_date": market.get("endDate"),
                            "prices": prices
                        })

            return extracted_markets

        except Exception as e:
            print(f"Error fetching Polymarket data: {e}")
            return []

