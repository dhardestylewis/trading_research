"""Paper-trade signal logger.

Records the full per-signal lifecycle for paper-trade validation:

    signal_timestamp → gate_state → order_submission → quote snapshot →
    fill/cancel outcome → post-fill price snapshots → realized PnL

Two operating modes:
  1. **Live**: call ``log_signal()`` per event, then ``flush_to_csv()``.
  2. **Simulated**: call ``simulate_from_backtest()`` on exp005 predictions
     + panel OHLC to synthesize plausible log rows until live data exists.
     The same downstream pipeline (execution_quality + exp006_report)
     consumes both formats identically.

Exchange integration (exp007+):
  An optional ``ExchangeConnector`` can be passed to ``PaperTradeLogger``
  to submit real orders via CCXT.  The connector is a thin abstraction
  so the pipeline can run with ``DryRunConnector`` (no exchange needed)
  or ``CoinbaseSandboxConnector`` (live paper orders on sandbox).
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

log = get_logger("paper_trade_logger")

# ═══════════════════════════════════════════════════════════════════
#  Per-signal record
# ═══════════════════════════════════════════════════════════════════

SIGNAL_LOG_COLUMNS = [
    "signal_timestamp",
    "gate_state",
    "order_submission_timestamp",
    "quoted_bid",
    "quoted_ask",
    "quoted_midpoint",
    "submitted_order_price",
    "acknowledgement_time",
    "fill_time",
    "fill_price",
    "cancel_status",          # filled | cancelled | partial_fill
    "midprice_after_1m",
    "midprice_after_5m",
    "midprice_after_15m",
    "midprice_after_1h",
    "realized_shortfall_vs_simulated",
    "realized_pnl_at_horizon",
    "simulated_fill_price",
    "entry_mode",
    "lane_type",
]


@dataclass
class SignalRecord:
    """One row of the paper-trade log."""
    signal_timestamp: pd.Timestamp | None = None
    gate_state: str = "open"
    order_submission_timestamp: pd.Timestamp | None = None
    quoted_bid: float = np.nan
    quoted_ask: float = np.nan
    quoted_midpoint: float = np.nan
    submitted_order_price: float = np.nan
    acknowledgement_time: pd.Timestamp | None = None
    fill_time: pd.Timestamp | None = None
    fill_price: float = np.nan
    cancel_status: str = "filled"   # filled | cancelled | partial_fill
    midprice_after_1m: float = np.nan
    midprice_after_5m: float = np.nan
    midprice_after_15m: float = np.nan
    midprice_after_1h: float = np.nan
    realized_shortfall_vs_simulated: float = np.nan
    realized_pnl_at_horizon: float = np.nan
    simulated_fill_price: float = np.nan
    entry_mode: str = ""
    lane_type: str = "primary"


# ═══════════════════════════════════════════════════════════════════
#  Exchange connector abstraction
# ═══════════════════════════════════════════════════════════════════

class ExchangeConnector(abc.ABC):
    """Abstract base for exchange order lifecycle integration."""

    @abc.abstractmethod
    def get_ticker(self, symbol: str) -> dict:
        """Return current bid/ask/mid for symbol.

        Returns dict with keys: bid, ask, mid, timestamp.
        """

    @abc.abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
    ) -> dict:
        """Submit order and return exchange response.

        Returns dict with keys: order_id, status, timestamp.
        """

    @abc.abstractmethod
    def get_order_status(self, order_id: str, symbol: str) -> dict:
        """Poll order status.

        Returns dict with keys: status, fill_price, fill_time,
        filled_amount, remaining_amount.
        """

    @abc.abstractmethod
    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order."""


class DryRunConnector(ExchangeConnector):
    """No-op connector that returns simulated fills.

    Used when no exchange credentials are available, or when running
    the pipeline against simulated data.  Preserves exp006 behavior.
    """

    def get_ticker(self, symbol: str) -> dict:
        return {"bid": np.nan, "ask": np.nan, "mid": np.nan, "timestamp": pd.Timestamp.now()}

    def submit_order(self, symbol, side, order_type, amount, price=None) -> dict:
        return {
            "order_id": f"dry_{pd.Timestamp.now().isoformat()}",
            "status": "simulated",
            "timestamp": pd.Timestamp.now(),
        }

    def get_order_status(self, order_id, symbol) -> dict:
        return {
            "status": "filled",
            "fill_price": np.nan,
            "fill_time": pd.Timestamp.now(),
            "filled_amount": 0.0,
            "remaining_amount": 0.0,
        }

    def cancel_order(self, order_id, symbol) -> dict:
        return {"status": "cancelled"}


class CoinbaseSandboxConnector(ExchangeConnector):
    """CCXT-based connector for Coinbase Advanced sandbox.

    Requires CCXT >= 4.0 and valid sandbox API credentials
    set via environment variables:
        COINBASE_SANDBOX_API_KEY
        COINBASE_SANDBOX_API_SECRET
        COINBASE_SANDBOX_PASSPHRASE  (if needed)

    This is a stub that initializes the exchange in sandbox mode.
    Fill in the actual order lifecycle when ready to go live.
    """

    def __init__(self) -> None:
        try:
            import ccxt
            import os
            self.exchange = ccxt.coinbase({
                "apiKey": os.environ.get("COINBASE_SANDBOX_API_KEY", ""),
                "secret": os.environ.get("COINBASE_SANDBOX_API_SECRET", ""),
                "password": os.environ.get("COINBASE_SANDBOX_PASSPHRASE", ""),
                "sandbox": True,
            })
            log.info("CoinbaseSandboxConnector initialized (sandbox mode)")
        except Exception as e:
            log.warning("Failed to initialize Coinbase sandbox: %s — falling back to DryRun", e)
            self._fallback = DryRunConnector()
            self.exchange = None

    def get_ticker(self, symbol: str) -> dict:
        if self.exchange is None:
            return self._fallback.get_ticker(symbol)
        ticker = self.exchange.fetch_ticker(symbol)
        return {
            "bid": ticker.get("bid", np.nan),
            "ask": ticker.get("ask", np.nan),
            "mid": (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2,
            "timestamp": pd.Timestamp.now(),
        }

    def submit_order(self, symbol, side, order_type, amount, price=None) -> dict:
        if self.exchange is None:
            return self._fallback.submit_order(symbol, side, order_type, amount, price)
        order = self.exchange.create_order(symbol, order_type, side, amount, price)
        return {
            "order_id": order["id"],
            "status": order.get("status", "open"),
            "timestamp": pd.Timestamp.now(),
        }

    def get_order_status(self, order_id, symbol) -> dict:
        if self.exchange is None:
            return self._fallback.get_order_status(order_id, symbol)
        order = self.exchange.fetch_order(order_id, symbol)
        return {
            "status": order.get("status", "unknown"),
            "fill_price": order.get("average", np.nan),
            "fill_time": pd.Timestamp(order["timestamp"], unit="ms") if order.get("timestamp") else None,
            "filled_amount": order.get("filled", 0.0),
            "remaining_amount": order.get("remaining", 0.0),
        }

    def cancel_order(self, order_id, symbol) -> dict:
        if self.exchange is None:
            return self._fallback.cancel_order(order_id, symbol)
        self.exchange.cancel_order(order_id, symbol)
        return {"status": "cancelled"}


# ═══════════════════════════════════════════════════════════════════
#  Logger class
# ═══════════════════════════════════════════════════════════════════

class PaperTradeLogger:
    """Accumulates signal records and flushes to CSV.

    Parameters
    ----------
    connector : ExchangeConnector or None
        If provided, log_signal() can optionally submit real orders
        through the connector.  If None (default), behaves exactly
        as in exp006 — pure local logging with no exchange interaction.
    """

    def __init__(self, connector: ExchangeConnector | None = None) -> None:
        self._records: list[dict] = []
        self.connector = connector or DryRunConnector()

    def log_signal(self, record: SignalRecord) -> None:
        self._records.append(asdict(record))

    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)
        return pd.DataFrame(self._records)

    def flush_to_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(p, index=False)
        log.info("Flushed %d signal records → %s", len(df), p)
        return p

    def clear(self) -> None:
        self._records.clear()


# ═══════════════════════════════════════════════════════════════════
#  Backtest simulation mode
# ═══════════════════════════════════════════════════════════════════

def _estimate_spread_bps(panel_row: pd.Series) -> float:
    """Heuristic spread estimate from OHLC bar."""
    high = panel_row.get("high", np.nan)
    low = panel_row.get("low", np.nan)
    mid = panel_row.get("close", np.nan)
    if mid > 0 and np.isfinite(high) and np.isfinite(low):
        # Rough spread ≈ 10% of the bar range, floored at 2 bps
        return max((high - low) / mid * 10_000 * 0.10, 2.0)
    return 5.0  # fallback


def simulate_from_backtest(
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    entry_mode_cfg: dict,
    policy_cfg: dict,
    lane_type: Literal["primary", "shadow"] = "primary",
) -> pd.DataFrame:
    """Synthesize paper-trade log rows from backtest predictions + OHLC.

    Uses the same fill logic as exp005 (market = next-bar open,
    passive = limit at open − offset with touch model) but enriches
    each row with the full signal-log schema.

    Parameters
    ----------
    panel : pd.DataFrame
        OHLC panel for target asset.
    predictions : pd.DataFrame
        Gated predictions with y_pred_prob, asset, timestamp.
    entry_mode_cfg : dict
        Entry mode config (name, type, offset_bps).
    policy_cfg : dict
        Policy params (threshold, cost_bps, sep_gap, regime_gate).

    Returns
    -------
    pd.DataFrame with one row per signal, full log schema.
    """
    threshold = policy_cfg.get("threshold", 0.55)
    cost_bps = policy_cfg.get("cost_bps", 15.0)
    sep_gap = policy_cfg.get("sep_gap", 3)

    entry_type = entry_mode_cfg.get("type", "market")
    entry_name = entry_mode_cfg.get("name", "unknown")
    offset_bps = entry_mode_cfg.get("offset_bps", 0.0)

    # Merge predictions with panel OHLC
    merged = predictions.merge(
        panel[["asset", "timestamp", "open", "high", "low", "close"]],
        on=["asset", "timestamp"],
        how="left",
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Shift to get next-bar OHLC (entry bar)
    for col in ["open", "high", "low", "close"]:
        merged[f"next_{col}"] = merged[col].shift(-1)

    # Also get bars +2, +5, +16 for post-fill snapshots
    # (approximations: 1m≈same bar, 5m≈same bar, 15m≈same bar, 1h≈next bar)
    merged["bar_plus_2_close"] = merged["close"].shift(-2)

    # Filter to active signals
    active = merged[merged["y_pred_prob"] > threshold].copy()
    active = active.dropna(subset=["next_open", "next_close"])

    if active.empty:
        return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)

    # Apply sep_gap
    if sep_gap > 0:
        active = active.reset_index(drop=True)
        kept = [0]
        for i in range(1, len(active)):
            gap = active.loc[i, "timestamp"] - active.loc[kept[-1], "timestamp"]
            gap_bars = gap.total_seconds() / 3600 if hasattr(gap, "total_seconds") else sep_gap + 1
            if gap_bars >= sep_gap:
                kept.append(i)
        active = active.loc[kept].reset_index(drop=True)

    rng = np.random.default_rng(42)
    records: list[dict] = []

    for _, row in active.iterrows():
        next_open = row["next_open"]
        next_high = row["next_high"]
        next_low = row["next_low"]
        next_close = row["next_close"]

        # Quote snapshot (simulated)
        spread_bps = _estimate_spread_bps(row)
        spread_frac = spread_bps / 10_000 / 2
        mid = next_open
        bid = mid * (1 - spread_frac)
        ask = mid * (1 + spread_frac)

        # Determine fill
        if entry_type == "passive_limit":
            limit_price = next_open * (1 - offset_bps / 10_000)
            filled = next_low <= limit_price
            fill_px = limit_price if filled else np.nan
            sim_fill_px = limit_price
        else:
            # Marketable near-open
            filled = True
            fill_px = next_open
            sim_fill_px = next_open

        cancel_status = "filled" if filled else "cancelled"

        # Post-fill midprice snapshots (approximations from hourly bars)
        # 1m, 5m, 15m ≈ interpolations within the entry bar
        if filled and np.isfinite(fill_px):
            # Linear interp within bar for sub-bar snapshots
            bar_range = next_high - next_low if (next_high - next_low) > 0 else 1e-10
            noise_1m = rng.normal(0, 0.0001)
            noise_5m = rng.normal(0, 0.0003)
            noise_15m = rng.normal(0, 0.0005)
            mid_1m = fill_px * (1 + noise_1m)
            mid_5m = fill_px + (next_close - fill_px) * 0.08 + fill_px * noise_5m
            mid_15m = fill_px + (next_close - fill_px) * 0.25 + fill_px * noise_15m
            mid_1h = next_close

            cost_frac = cost_bps / 10_000
            gross_ret = (next_close - fill_px) / fill_px
            net_ret = gross_ret - 2 * cost_frac
            shortfall = (fill_px - sim_fill_px) / sim_fill_px * 10_000  # bps
        else:
            mid_1m = mid_5m = mid_15m = mid_1h = np.nan
            net_ret = np.nan
            shortfall = np.nan

        # Simulated timing
        signal_ts = row["timestamp"]
        sub_ts = signal_ts + pd.Timedelta(hours=1, seconds=rng.integers(1, 10))
        ack_ts = sub_ts + pd.Timedelta(milliseconds=int(rng.integers(50, 500)))
        fill_ts = ack_ts + pd.Timedelta(milliseconds=int(rng.integers(10, 2000))) if filled else None

        # Gate state
        gate = "open"
        if "regime" in row.index:
            regime = row.get("regime", "")
            gate = f"NOT_rebound (regime={regime})"

        rec = SignalRecord(
            signal_timestamp=signal_ts,
            gate_state=gate,
            order_submission_timestamp=sub_ts,
            quoted_bid=bid,
            quoted_ask=ask,
            quoted_midpoint=mid,
            submitted_order_price=fill_px if entry_type == "market" else limit_price,
            acknowledgement_time=ack_ts,
            fill_time=fill_ts,
            fill_price=fill_px if filled else np.nan,
            cancel_status=cancel_status,
            midprice_after_1m=mid_1m,
            midprice_after_5m=mid_5m,
            midprice_after_15m=mid_15m,
            midprice_after_1h=mid_1h,
            realized_shortfall_vs_simulated=shortfall,
            realized_pnl_at_horizon=net_ret,
            simulated_fill_price=sim_fill_px,
            entry_mode=entry_name,
            lane_type=lane_type,
        )
        records.append(asdict(rec))

    df = pd.DataFrame(records)
    log.info(
        "Simulated %d signal records for %s (%s), %d filled",
        len(df), entry_name, lane_type,
        int((df["cancel_status"] == "filled").sum()),
    )
    return df
