"""Branch E — Live Paper Cost Logger.

Real-time paper process that logs score time, current bid/ask, model
gross/cost predictions, assumed taker fill, and realized markout/shortfall.
Validates replay-vs-live alignment for cost modeling.
"""
import numpy as np
import pandas as pd
import logging
import time
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger("live_paper_cost_eval")


class LivePaperCostLogger:
    """Logs real-time cost events for replay-vs-live comparison.

    Each event records:
      - score_time: when the model scored
      - asset: symbol
      - bid / ask / mid: current top of book
      - pred_gross_1s / pred_gross_5s: gross model predictions
      - pred_cost_1s / pred_cost_5s: cost model predictions
      - assumed_fill_price: taker execution assumption (ask for buy)
      - realized_mid_1s / realized_mid_5s: actual mid after 1s / 5s
      - realized_shortfall_1s / 5s: actual taker shortfall
    """

    def __init__(self, gross_model=None, cost_model=None, log_path: str = "data/processed/exp024_cost_surface/live_paper_cost_log.csv"):
        self.gross_model = gross_model
        self.cost_model = cost_model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: List[Dict] = []
        self.pending_fills: List[Dict] = []
        logger.info("LivePaperCostLogger initialized. Log: %s", self.log_path)

    def score_and_log(
        self,
        timestamp,
        asset: str,
        features: Dict,
        book: Dict,
    ) -> Dict:
        """Score a 1s bar and log the event.

        Parameters
        ----------
        timestamp : datetime-like
            Current bar timestamp.
        asset : str
            Symbol.
        features : dict
            Feature dict for model scoring.
        book : dict
            Must contain 'bid' and 'ask' keys.

        Returns
        -------
        dict with score results and decision.
        """
        bid = book.get("bid", np.nan)
        ask = book.get("ask", np.nan)
        mid = (bid + ask) / 2 if not np.isnan(bid) and not np.isnan(ask) else np.nan

        # Score gross model
        pred_gross_1s = 0.0
        pred_gross_5s = 0.0
        if self.gross_model is not None:
            try:
                feat_df = pd.DataFrame([features])
                preds = self.gross_model.predict(feat_df)
                pred_gross_1s = float(preds.get("pred_1s_gross", pd.Series([0.0])).iloc[0])
                pred_gross_5s = float(preds.get("pred_5s_gross", pd.Series([0.0])).iloc[0])
            except Exception as e:
                logger.debug("Gross model scoring failed: %s", e)

        # Score cost model (placeholder — interface depends on trained model)
        pred_cost_1s = features.get("shortfall_1s_bps", np.nan)
        pred_cost_5s = features.get("shortfall_5s_bps", np.nan)

        event = {
            "score_time": timestamp,
            "asset": asset,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "quoted_spread_bps": ((ask - bid) / mid * 10_000) if mid > 0 else np.nan,
            "pred_gross_1s_bps": pred_gross_1s,
            "pred_gross_5s_bps": pred_gross_5s,
            "pred_cost_1s_bps": pred_cost_1s,
            "pred_cost_5s_bps": pred_cost_5s,
            "assumed_fill_price": ask,  # taker buy at ask
            # Realized fields filled in later via backfill
            "realized_mid_1s": np.nan,
            "realized_mid_5s": np.nan,
            "realized_shortfall_1s_bps": np.nan,
            "realized_shortfall_5s_bps": np.nan,
        }

        self.events.append(event)
        self.pending_fills.append(event)

        return event

    def backfill_realized(self, realized_mids: Dict):
        """Backfill realized markout for pending fills.

        Parameters
        ----------
        realized_mids : dict
            Keyed by (score_time, asset) -> {"mid_1s": float, "mid_5s": float}
        """
        updated = 0
        for event in self.pending_fills:
            key = (event["score_time"], event["asset"])
            if key in realized_mids:
                rm = realized_mids[key]
                mid = event["mid"]
                if mid and mid > 0:
                    event["realized_mid_1s"] = rm.get("mid_1s", np.nan)
                    event["realized_mid_5s"] = rm.get("mid_5s", np.nan)
                    # Shortfall = fill cost relative to realized mid drift
                    fill_px = event["assumed_fill_price"]
                    if not np.isnan(rm.get("mid_1s", np.nan)):
                        event["realized_shortfall_1s_bps"] = (
                            (fill_px - rm["mid_1s"]) / mid * 10_000
                        )
                    if not np.isnan(rm.get("mid_5s", np.nan)):
                        event["realized_shortfall_5s_bps"] = (
                            (fill_px - rm["mid_5s"]) / mid * 10_000
                        )
                updated += 1
        self.pending_fills = [e for e in self.pending_fills if np.isnan(e.get("realized_mid_1s", np.nan))]
        logger.info("Backfilled %d events; %d still pending", updated, len(self.pending_fills))

    def to_dataframe(self) -> pd.DataFrame:
        """Return all events as DataFrame."""
        if not self.events:
            return pd.DataFrame()
        return pd.DataFrame(self.events)

    def save(self):
        """Persist events to disk."""
        df = self.to_dataframe()
        if df.empty:
            logger.warning("No events to save")
            return
        df.to_csv(self.log_path, index=False)
        logger.info("Saved %d events to %s", len(df), self.log_path)


def compute_replay_vs_live_gap(
    replay_df: pd.DataFrame,
    live_df: pd.DataFrame,
    cost_col: str = "shortfall_1s_bps",
) -> pd.DataFrame:
    """Compare replayed vs live shortfall for alignment validation.

    Returns per-asset summary of gap statistics.
    """
    if replay_df.empty or live_df.empty:
        logger.warning("Cannot compute replay-vs-live gap: empty input(s)")
        return pd.DataFrame()

    summary_rows = []

    # If both have asset column, compare per asset
    if "asset" in replay_df.columns and "asset" in live_df.columns:
        for asset in replay_df["asset"].unique():
            r = replay_df[replay_df["asset"] == asset]
            l_col = f"realized_{cost_col}" if f"realized_{cost_col}" in live_df.columns else cost_col
            l = live_df[live_df["asset"] == asset]
            if r.empty or l.empty or cost_col not in r.columns:
                continue
            summary_rows.append({
                "asset": asset,
                "replay_median_shortfall": r[cost_col].median(),
                "live_median_shortfall": l[l_col].median() if l_col in l.columns else np.nan,
                "gap_bps": (l[l_col].median() - r[cost_col].median()) if l_col in l.columns else np.nan,
                "replay_n": len(r),
                "live_n": len(l),
            })
    else:
        r_med = replay_df[cost_col].median() if cost_col in replay_df.columns else np.nan
        l_col = f"realized_{cost_col}" if f"realized_{cost_col}" in live_df.columns else cost_col
        l_med = live_df[l_col].median() if l_col in live_df.columns else np.nan
        summary_rows.append({
            "asset": "ALL",
            "replay_median_shortfall": r_med,
            "live_median_shortfall": l_med,
            "gap_bps": l_med - r_med if not np.isnan(l_med) and not np.isnan(r_med) else np.nan,
        })

    return pd.DataFrame(summary_rows)
