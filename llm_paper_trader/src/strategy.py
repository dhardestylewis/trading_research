from typing import Optional, Dict

class TradingStrategy:
    def __init__(self, edge_threshold: float = 0.05, trade_size_dollars: float = 100.0):
        self.edge_threshold = edge_threshold
        self.trade_size_dollars = trade_size_dollars

    def evaluate_trade(self, market_id: str, p_llm: float, market_prices: Dict[str, float]) -> Optional[Dict]:
        """
        Evaluates the edge between the LLM and the Live Market.
        market_prices expects: {"Yes": 0.56, "No": 0.44}
        """
        p_market_yes = market_prices.get("Yes", 0.5)
        p_market_no = market_prices.get("No", 0.5)
        
        # We only look at the YES probability edge.
        # Edge = P_LLM - Market_Price
        edge_yes = p_llm - p_market_yes
        
        # To be safe, edge_no is evaluated as (1 - p_llm) - p_market_no
        edge_no = (1.0 - p_llm) - p_market_no

        trade_decision = None

        if edge_yes > self.edge_threshold:
            # We think YES is underpriced
            trade_decision = {
                "market_id": market_id,
                "side": "Yes",
                "fill_price": p_market_yes,
                "size_shares": self.trade_size_dollars / max(p_market_yes, 0.01),
                "edge": edge_yes
            }
        elif edge_no > self.edge_threshold:
            # We think NO is underpriced
            trade_decision = {
                "market_id": market_id,
                "side": "No",
                "fill_price": p_market_no,
                "size_shares": self.trade_size_dollars / max(p_market_no, 0.01),
                "edge": edge_no
            }

        return trade_decision
