"""Canary tick — hourly signal check for live paper canary.

══════════════════════════════════════════════════════════════
⚠️  STATUS: INFRASTRUCTURE VALIDATION ONLY (2026-03-15)
──────────────────────────────────────────────────────────────
  Post exp010: family deployment thesis falsified.
  All live-capital ambitions on this branch are FROZEN.
  This script continues solely as a systems canary —
  validating data fetch, feature build, model inference,
  and logging infrastructure. It carries NO deployment
  mandate and NO capital allocation.

  See: PROGRAM_DECISION.md, configs/research_status.yaml
══════════════════════════════════════════════════════════════

Designed to run once per hour via Windows Task Scheduler (or cron).
Each invocation:
  1. Fetches the latest ~300 1h bars for all assets via CCXT
  2. Builds features using the existing pipeline
  3. Loads (or trains) the LightGBM model
  4. Scores the latest bar for SOL-USD
  5. Checks gates: threshold, regime (NOT_rebound), sep_gap
  6. If signal fires → logs to PaperTradeLogger (appends to CSV)
  7. Exits

The companion script `run_exp007.py --mode live` then reads the
accumulated log CSV to run canary health checks and generate reports.

Usage:
    python canary_tick.py                      # default config
    python canary_tick.py --retrain             # force model retrain
    python canary_tick.py --dry-run             # log but don't submit orders
"""
from __future__ import annotations
import argparse
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logging import get_logger

log = get_logger("canary_tick")

# ── Constants ────────────────────────────────────────────────────
LOOKBACK_BARS = 300          # enough for 168h drawdown + feature warmup
MODEL_CACHE_PATH = Path("data/artifacts/models/canary_lightgbm.pkl")
CANARY_LOG_PATH = Path("reports/exp007/tables/live_paper_trade_log.csv")
STATE_PATH = Path("data/artifacts/canary_state.json")
CONFIG_PATH = "configs/experiments/crypto_1h_exp007.yaml"

# Shadow lanes: scored alongside SOL on each tick, log-only (no orders)
SHADOW_LANES = [
    {
        "asset": "APT-USD",
        "config": "configs/experiments/canary_apt.yaml",
        "log_path": Path("reports/canary/apt_shadow_log.csv"),
        "state_path": Path("data/artifacts/canary_state_apt.json"),
        "priority": "high",   # high-promise, low-sample
    },
    {
        "asset": "SUI-USD",
        "config": "configs/experiments/canary_sui.yaml",
        "log_path": Path("reports/canary/sui_shadow_log.csv"),
        "state_path": Path("data/artifacts/canary_state_sui.json"),
        "priority": "medium",  # less flashy, more robust
    },
]


def _fetch_recent_bars(
    exchange_id: str,
    symbols: list[str],
    labels: dict[str, str],
    timeframe: str = "1h",
    limit: int = LOOKBACK_BARS,
) -> pd.DataFrame:
    """Fetch recent OHLCV bars for all symbols, return stacked panel."""
    import ccxt

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    exchange.load_markets()

    frames: list[pd.DataFrame] = []
    for symbol in symbols:
        label = labels.get(symbol, symbol.replace("/", "-"))
        log.info("Fetching %d bars for %s", limit, symbol)

        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not bars:
            log.warning("No bars returned for %s", symbol)
            continue

        df = pd.DataFrame(bars, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df["asset"] = label
        df["dollar_volume"] = df["close"] * df["volume"]
        df = df.drop_duplicates(subset=["timestamp_ms"]).sort_values("timestamp_ms")
        frames.append(df)

        time.sleep(exchange.rateLimit / 1000)

    if not frames:
        raise RuntimeError("No data fetched for any symbol")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    log.info("Panel: %d rows, %d assets", len(panel), panel["asset"].nunique())
    return panel


def _build_features_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Build features using the existing pipeline (no disk I/O)."""
    from src.features.price_features import compute_price_features
    from src.features.volume_features import compute_volume_features
    from src.features.regime_features import compute_regime_features

    price_parts, vol_parts = [], []
    for _, g in panel.groupby("asset", sort=False):
        g_sorted = g.sort_values("timestamp")
        price_parts.append(compute_price_features(g_sorted))
        vol_parts.append(compute_volume_features(g_sorted))

    price_feats = pd.concat(price_parts).loc[panel.index]
    vol_feats = pd.concat(vol_parts).loc[panel.index]
    regime_feats = compute_regime_features(panel)

    features = pd.concat([
        panel[["asset", "timestamp"]],
        price_feats,
        vol_feats,
        regime_feats,
    ], axis=1)

    return features


def _build_labels_from_panel(panel: pd.DataFrame, cost_bps: float = 7.5) -> pd.DataFrame:
    """Build forward return labels inline (no disk I/O)."""
    from src.labels.forward_returns import compute_forward_labels

    labels = compute_forward_labels(panel, horizons=[1], one_way_cost_bps=cost_bps)
    labels = pd.concat([panel[["asset", "timestamp"]], labels], axis=1)
    return labels


def _train_model(features: pd.DataFrame, labels: pd.DataFrame) -> "TrainedModel":
    """Train LightGBM on all available data and cache to disk."""
    from src.models.train_lightgbm import train as train_lightgbm

    feat_cols = [c for c in features.columns if c not in ("asset", "timestamp")]
    target_col = "fwd_profitable_1h"

    merged = features.merge(
        labels[["asset", "timestamp", target_col]],
        on=["asset", "timestamp"],
        how="inner",
    )

    # Use all but the last 24h as training (recent data reserved for scoring)
    ts = merged["timestamp"]
    cutoff = ts.max() - pd.Timedelta(hours=24)
    train_df = merged[ts < cutoff].copy()
    val_df = merged[(ts >= cutoff)].copy()

    # Drop NaN rows
    valid_mask = train_df[feat_cols + [target_col]].notna().all(axis=1)
    train_df = train_df[valid_mask]

    if len(train_df) < 100:
        raise RuntimeError(f"Insufficient training data: {len(train_df)} rows")

    val_mask = val_df[feat_cols + [target_col]].notna().all(axis=1)
    val_df = val_df[val_mask] if val_mask.sum() > 10 else None

    model = train_lightgbm(
        train_df[feat_cols],
        train_df[target_col],
        val_df[feat_cols] if val_df is not None else None,
        val_df[target_col] if val_df is not None else None,
        config_path="configs/models/lightgbm_v1.yaml",
        feature_names=feat_cols,
    )

    # Cache
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_CACHE_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info("Model trained on %d rows, cached to %s", len(train_df), MODEL_CACHE_PATH)

    return model


def _load_or_train_model(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    force_retrain: bool = False,
    max_age_hours: int = 168,  # retrain weekly by default
) -> "TrainedModel":
    """Load cached model or retrain if stale/missing."""
    from src.models.predict import TrainedModel

    if not force_retrain and MODEL_CACHE_PATH.exists():
        # Check age
        mtime = datetime.fromtimestamp(MODEL_CACHE_PATH.stat().st_mtime, tz=timezone.utc)
        age_h = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_h < max_age_hours:
            log.info("Loading cached model (age: %.1fh)", age_h)
            with open(MODEL_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        else:
            log.info("Model cache stale (%.1fh old), retraining", age_h)

    return _train_model(features, labels)


def _load_last_signal_time() -> pd.Timestamp | None:
    """Load the timestamp of the last signal for sep_gap enforcement."""
    import json
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        last = state.get("last_signal_timestamp")
        if last:
            return pd.Timestamp(last)
    return None


def _save_state(last_signal_ts: pd.Timestamp | None, **extra):
    """Persist canary state between ticks."""
    import json
    state = {"last_tick": datetime.now(timezone.utc).isoformat()}
    if last_signal_ts is not None:
        state["last_signal_timestamp"] = str(last_signal_ts)
    state.update(extra)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _append_to_log(record_dict: dict):
    """Append a signal record to the live canary log CSV."""
    CANARY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record_dict])
    write_header = not CANARY_LOG_PATH.exists()
    df.to_csv(CANARY_LOG_PATH, mode="a", header=write_header, index=False)
    log.info("Appended signal to %s", CANARY_LOG_PATH)


def tick(force_retrain: bool = False, dry_run: bool = True):
    """Execute one canary tick cycle."""
    log.info("--- Canary tick at %s ---", datetime.now(timezone.utc).isoformat())

    # ── Load config ──────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    with open(cfg["data_config"]) as f:
        data_cfg = yaml.safe_load(f)

    policy = cfg.get("policy", {})
    threshold = policy.get("threshold", 0.55)
    sep_gap = policy.get("sep_gap", 3)
    regime_gate = policy.get("regime_gate", "NOT_rebound")
    cost_bps = policy.get("cost_bps", 15.0)
    target_asset = cfg.get("target_asset", "SOL-USD")

    # ── Fetch live data ──────────────────────────────────────────
    panel = _fetch_recent_bars(
        exchange_id=data_cfg["exchange_id"],
        symbols=data_cfg["asset_universe"],
        labels=data_cfg.get("asset_labels", {}),
        timeframe=data_cfg.get("bar_size", "1h"),
        limit=LOOKBACK_BARS,
    )

    # Normalize timestamps
    if panel["timestamp"].dt.tz is not None:
        panel["timestamp"] = panel["timestamp"].dt.tz_localize(None)

    # ── Build features ───────────────────────────────────────────
    features = _build_features_from_panel(panel)
    log.info("Features built: %d rows × %d cols", len(features), len(features.columns))

    # ── Build labels (for training only) ─────────────────────────
    labels = _build_labels_from_panel(panel, cost_bps=cost_bps / 2)

    # ── Load / train model ───────────────────────────────────────
    model = _load_or_train_model(features, labels, force_retrain=force_retrain)

    # ── Score latest bar for target asset ────────────────────────
    sol_mask = features["asset"] == target_asset
    sol_features = features[sol_mask].copy()

    if sol_features.empty:
        log.warning("No data for %s — skipping", target_asset)
        _save_state(None, status="no_data")
        return

    # Get the most recent bar with all features populated.
    # The absolute latest bar from the exchange is often the in-progress
    # (incomplete) candle, and edge bars may have NaN features due to
    # the 168h drawdown lookback warmup.  Walk backwards to find
    # a fully-populated bar.
    sol_features_sorted = sol_features.sort_values("timestamp", ascending=False)
    feat_cols = model.feature_names

    latest_row = None
    latest_idx = None
    latest_ts = None

    for idx, row in sol_features_sorted.iterrows():
        row_df = pd.DataFrame([row])
        if row_df[feat_cols].isna().any(axis=1).iloc[0]:
            continue
        latest_row = row_df
        latest_idx = idx
        latest_ts = row["timestamp"]
        break

    if latest_row is None:
        log.warning("No SOL bar with complete features found — skipping tick")
        _save_state(None, status="all_nan_features")
        return

    log.info("Using bar at %s (skipped %d incomplete bars)",
             latest_ts, sol_features_sorted.index.get_loc(latest_idx))

    prob = model.predict_proba(latest_row)[0]
    log.info("Signal score: %.4f (threshold: %.2f) at %s", prob, threshold, latest_ts)

    # ── Gate checks ──────────────────────────────────────────────
    signal_fires = prob > threshold

    # Regime gate
    regime = "unknown"
    if signal_fires:
        from src.diagnostics.regime_labeller import label_regimes
        # Label regimes using the full SOL history as reference
        regime_df = label_regimes(sol_features)
        latest_regime_row = regime_df.loc[latest_idx]

        if regime_gate == "NOT_rebound":
            is_rebound = latest_regime_row.get("regime_rebound", 0) == 1
            if is_rebound:
                log.info("Signal blocked by NOT_rebound gate")
                signal_fires = False
                regime = "rebound"
            else:
                regime = "not_rebound"

    # Sep gap check
    if signal_fires and sep_gap > 0:
        last_signal_ts = _load_last_signal_time()
        if last_signal_ts is not None:
            gap_hours = (latest_ts - last_signal_ts).total_seconds() / 3600
            if gap_hours < sep_gap:
                log.info("Signal blocked by sep_gap: %.1fh since last (need %d)", gap_hours, sep_gap)
                signal_fires = False

    # ── Log result ───────────────────────────────────────────────
    if signal_fires:
        log.info("✅ SIGNAL FIRES — logging to canary log")

        # Get price data for the signal bar
        sol_panel = panel[panel["asset"] == target_asset]
        bar = sol_panel.loc[sol_panel["timestamp"].idxmax()]
        next_open = bar["close"]  # best proxy for next bar's open

        # Build signal record
        from dataclasses import asdict
        from src.diagnostics.paper_trade_logger import SignalRecord

        # For primary marketable lane
        record = SignalRecord(
            signal_timestamp=latest_ts,
            gate_state=f"NOT_rebound (regime={regime})",
            order_submission_timestamp=pd.Timestamp.now(),
            quoted_bid=np.nan,      # will be filled by exchange connector
            quoted_ask=np.nan,
            quoted_midpoint=next_open,
            submitted_order_price=next_open,
            acknowledgement_time=pd.Timestamp.now(),
            fill_time=pd.Timestamp.now() if not dry_run else None,
            fill_price=next_open if not dry_run else np.nan,
            cancel_status="filled" if not dry_run else "pending",
            midprice_after_1m=np.nan,
            midprice_after_5m=np.nan,
            midprice_after_15m=np.nan,
            midprice_after_1h=np.nan,
            realized_shortfall_vs_simulated=np.nan,
            realized_pnl_at_horizon=np.nan,
            simulated_fill_price=next_open,
            entry_mode="marketable_near_open",
            lane_type="primary",
        )
        _append_to_log(asdict(record))

        # Also log shadow lanes
        for offset_bps, lane_name in [(5.0, "passive_open_minus_5bps"),
                                       (10.0, "passive_open_minus_10bps")]:
            limit_price = next_open * (1 - offset_bps / 10_000)
            shadow_record = SignalRecord(
                signal_timestamp=latest_ts,
                gate_state=f"NOT_rebound (regime={regime})",
                order_submission_timestamp=pd.Timestamp.now(),
                quoted_bid=np.nan,
                quoted_ask=np.nan,
                quoted_midpoint=next_open,
                submitted_order_price=limit_price,
                acknowledgement_time=pd.Timestamp.now(),
                fill_time=None,
                fill_price=np.nan,
                cancel_status="pending",   # resolved at next tick or EOD
                midprice_after_1m=np.nan,
                midprice_after_5m=np.nan,
                midprice_after_15m=np.nan,
                midprice_after_1h=np.nan,
                realized_shortfall_vs_simulated=np.nan,
                realized_pnl_at_horizon=np.nan,
                simulated_fill_price=limit_price,
                entry_mode=lane_name,
                lane_type="shadow",
            )
            _append_to_log(asdict(shadow_record))

        _save_state(latest_ts, status="signal_fired", prob=float(prob))
        log.info("Signal logged for all 3 lanes at %s (prob=%.4f)", latest_ts, prob)

    else:
        _save_state(_load_last_signal_time(), status="no_signal", prob=float(prob))
        log.info("No signal at %s (prob=%.4f)", latest_ts, prob)

    log.info("--- Canary tick complete ---")


def tick_shadow_lanes(model, features: pd.DataFrame, panel: pd.DataFrame):
    """Score shadow lanes (APT, SUI) using the same model and features.

    Shadow lanes are log-only — they accumulate signal data for validation
    but never submit orders or allocate capital.
    """
    for lane in SHADOW_LANES:
        asset = lane["asset"]
        log_path = lane["log_path"]
        state_path = lane["state_path"]

        log.info("--- Shadow lane: %s (priority=%s) ---", asset, lane["priority"])

        asset_mask = features["asset"] == asset
        asset_features = features[asset_mask].copy()

        if asset_features.empty:
            log.info("  No data for %s — skipping shadow lane", asset)
            continue

        # Find latest bar with complete features
        feat_cols = model.feature_names
        asset_sorted = asset_features.sort_values("timestamp", ascending=False)

        latest_row = None
        latest_ts = None
        for idx, row in asset_sorted.iterrows():
            row_df = pd.DataFrame([row])
            if row_df[feat_cols].isna().any(axis=1).iloc[0]:
                continue
            latest_row = row_df
            latest_ts = row["timestamp"]
            break

        if latest_row is None:
            log.info("  No complete features for %s — skipping", asset)
            continue

        prob = model.predict_proba(latest_row)[0]
        threshold = 0.55  # use same threshold as SOL
        signal_fires = prob > threshold

        # Log the shadow signal
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": latest_ts,
            "asset": asset,
            "prob": float(prob),
            "threshold": threshold,
            "signal_fires": signal_fires,
            "lane_type": "shadow",
            "priority": lane["priority"],
            "tick_time": datetime.now(timezone.utc).isoformat(),
        }
        df = pd.DataFrame([record])
        write_header = not log_path.exists()
        df.to_csv(log_path, mode="a", header=write_header, index=False)

        emoji = "✅" if signal_fires else "·"
        log.info("  %s %s shadow: prob=%.4f %s",
                 emoji, asset, prob, "SIGNAL" if signal_fires else "no signal")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canary tick: hourly signal check")
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Log signals but don't submit exchange orders (default)")
    parser.add_argument("--live-orders", action="store_true",
                        help="Submit real orders via exchange connector")
    parser.add_argument("--no-shadow", action="store_true",
                        help="Skip shadow lane scoring (APT, SUI)")
    args = parser.parse_args()

    tick(force_retrain=args.retrain, dry_run=not args.live_orders)

    # Score shadow lanes if model and features are available
    # NOTE: Shadow lanes reuse the model and panel from the SOL tick.
    # In production, this would be restructured to share state.
    if not args.no_shadow:
        log.info("Shadow lanes are configured but require shared model state.")
        log.info("Run with full pipeline integration to score APT/SUI shadows.")
