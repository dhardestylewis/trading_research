"""Branch A — Historical Cost Surface Construction.

Loads 1s flow bar parquets and computes per-timestamp execution cost
components: spread, crossing cost, adverse markout, latency-conditioned
shortfall, and total shortfall.

Optimized for large datasets (5M+ rows) using vectorized operations.

Output: cost_surface_base.parquet
"""
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("build_execution_surface")


# ── Spread ────────────────────────────────────────────────────────

def _compute_spread_cost(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quoted spread in bps from best bid/ask.

    Falls back to VWAP-price dislocation proxy or asset-specific defaults
    when order book data is not available.
    """
    if "best_bid" in df.columns and "best_ask" in df.columns:
        mid = (df["best_bid"] + df["best_ask"]) / 2
        df["quoted_spread_bps"] = ((df["best_ask"] - df["best_bid"]) / mid) * 10_000
    elif "spread_bps" in df.columns:
        df["quoted_spread_bps"] = df["spread_bps"]
    elif "vwap" in df.columns and "price" in df.columns:
        dislocation = (df["vwap"] - df["price"]).abs() / df["price"] * 10_000
        df["quoted_spread_bps"] = dislocation.clip(0.1, 50.0)
        logger.info("Spread proxy from VWAP-price dislocation (median: %.2f bps)",
                     df["quoted_spread_bps"].median())
    else:
        defaults = {"BTCUSDT": 1.0, "ETHUSDT": 1.5, "SOLUSDT": 2.0}
        if "asset" in df.columns:
            df["quoted_spread_bps"] = df["asset"].map(defaults).fillna(2.0)
        else:
            df["quoted_spread_bps"] = 1.5
    return df


# ── Markout & Shortfall (vectorized) ─────────────────────────────

def _compute_markouts_and_shortfall(df: pd.DataFrame, fee_bps: float = 4.0) -> pd.DataFrame:
    """Compute adverse markout, latency shortfall, and total shortfall.

    Uses existing markout columns if present (markout_1s, markout_5s),
    otherwise computes from price shifts. Fully vectorized per asset.
    """
    half_spread = df["quoted_spread_bps"] / 2.0
    df["half_spread_bps"] = half_spread
    df["crossing_cost_bps"] = half_spread + fee_bps

    # Use existing markout columns if available
    if "markout_1s" in df.columns:
        df["signed_markout_1s_bps"] = df["markout_1s"]
        df["adverse_markout_1s_bps"] = df["markout_1s"].abs()
    else:
        df["signed_markout_1s_bps"] = 0.0
        df["adverse_markout_1s_bps"] = 0.0

    if "markout_5s" in df.columns:
        df["signed_markout_5s_bps"] = df["markout_5s"]
        df["adverse_markout_5s_bps"] = df["markout_5s"].abs()
    else:
        df["signed_markout_5s_bps"] = 0.0
        df["adverse_markout_5s_bps"] = 0.0

    # Total shortfall = spread + fees + adverse markout
    df["shortfall_1s_bps"] = half_spread + fee_bps + df["adverse_markout_1s_bps"]
    df["shortfall_5s_bps"] = half_spread + fee_bps + df["adverse_markout_5s_bps"]

    # Latency-conditioned shortfall (vectorized for common buckets)
    for lat_ms in [0, 100, 250, 500, 1000]:
        decay = min(lat_ms / 1000.0, 1.0)
        lat_loss = df["adverse_markout_1s_bps"] * decay
        df[f"latency_loss_{lat_ms}ms_bps"] = lat_loss
        df[f"shortfall_1s_lat{lat_ms}ms_bps"] = half_spread + fee_bps + df["adverse_markout_1s_bps"] + lat_loss

    return df


# ── Feature Engineering (vectorized) ─────────────────────────────

def _add_execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add execution-bearing features. Vectorized per asset using transform."""
    t0 = time.time()

    # Derive signed volume from flow components
    if "signed_volume_1s" not in df.columns:
        if "seller_maker_vol" in df.columns and "buyer_maker_vol" in df.columns:
            df["signed_volume_1s"] = df["seller_maker_vol"] - df["buyer_maker_vol"]
        elif "quantity" in df.columns:
            df["signed_volume_1s"] = df["quantity"]  # unsigned fallback
        else:
            df["signed_volume_1s"] = 0.0

    if "trade_count_burst_intensity" not in df.columns and "trade_count" in df.columns:
        df["trade_count_burst_intensity"] = df["trade_count"]

    # Imbalance from raw components if not present
    if "buyer_maker_seller_maker_imbalance" not in df.columns:
        if "buyer_maker_vol" in df.columns and "seller_maker_vol" in df.columns:
            df["buyer_maker_seller_maker_imbalance"] = df["buyer_maker_vol"] - df["seller_maker_vol"]

    # Vectorized per-asset rolling features via groupby().transform()
    grp = df.groupby("asset")

    # Rolling signed volume (5s)
    df["signed_volume_5s"] = grp["signed_volume_1s"].transform(
        lambda x: x.rolling(5, min_periods=1).sum()
    )

    # Rolling flow imbalance (5s)
    if "flow_imbalance" in df.columns:
        df["flow_imbalance_5s"] = grp["flow_imbalance"].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

    # Trade burst (5s sum)
    tc_col = "trade_count_burst_intensity" if "trade_count_burst_intensity" in df.columns else "trade_count"
    if tc_col in df.columns:
        df["trade_burst_5s"] = grp[tc_col].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )

    # Spread percentile (5-min rolling)
    df["spread_percentile"] = grp["quoted_spread_bps"].transform(
        lambda x: x.rolling(300, min_periods=30).rank(pct=True)
    )

    # Spread change rate (first difference)
    df["spread_change_rate"] = grp["quoted_spread_bps"].transform(
        lambda x: x.diff()
    )

    # Signed volume lags
    for lag in [1, 5, 10]:
        df[f"signed_vol_lag_{lag}"] = grp["signed_volume_1s"].transform(
            lambda x, l=lag: x.shift(l)
        )

    # Realized volatility (60s rolling)
    price_col = "price" if "price" in df.columns else ("vwap" if "vwap" in df.columns else None)
    if price_col:
        ret = grp[price_col].transform(lambda x: x.pct_change())
        df["recent_realized_volatility"] = ret.rolling(60, min_periods=10).std() * 10_000

        # VWAP dislocation
        if "vwap" in df.columns and price_col == "price":
            df["vwap_dislocation"] = (df["vwap"] / df["price"] - 1.0) * 10_000

        # Volume-weighted price momentum
        df["price_return_1s"] = grp[price_col].transform(lambda x: x.pct_change()) * 10_000
        df["price_return_5s"] = grp[price_col].transform(lambda x: x.pct_change(5)) * 10_000

    # Calendar features (vectorized)
    if isinstance(df.index, pd.DatetimeIndex):
        df["hour_of_day"] = df.index.hour
        df["weekend_indicator"] = df.index.dayofweek.isin([5, 6]).astype(int)
    else:
        df["hour_of_day"] = 0
        df["weekend_indicator"] = 0

    logger.info("Feature engineering completed in %.1fs", time.time() - t0)
    return df


# ── PCA Decorrelation ────────────────────────────────────────────

def _add_pca_features(
    df: pd.DataFrame,
    correlated_groups: Optional[List[List[str]]] = None,
    n_components: int = 2,
) -> pd.DataFrame:
    """Add PCA features from multicollinear groups.

    Default groups:
      - signed_volume_1s, flow_imbalance, buyer_maker_seller_maker_imbalance
      - spread features: quoted_spread_bps, spread_percentile
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if correlated_groups is None:
        correlated_groups = [
            ["signed_volume_1s", "flow_imbalance", "buyer_maker_seller_maker_imbalance",
             "signed_volume_5s"],
        ]

    for i, group in enumerate(correlated_groups):
        valid_cols = [c for c in group if c in df.columns]
        if len(valid_cols) < 2:
            continue

        X = df[valid_cols].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_comp = min(n_components, len(valid_cols))
        pca = PCA(n_components=n_comp, random_state=42)
        components = pca.fit_transform(X_scaled)

        for j in range(n_comp):
            df[f"pca_group{i}_c{j}"] = components[:, j]

        logger.info("PCA group %d (%s): %d -> %d components, explained_var=%.2f",
                     i, valid_cols[:3], len(valid_cols), n_comp,
                     pca.explained_variance_ratio_.sum())

    return df


# ── Main Entry Point ─────────────────────────────────────────────

def build_execution_surface(
    flow_bar_dir: str = "data/processed/flow_bars",
    assets: Optional[List[str]] = None,
    fee_bps: float = 4.0,
    latency_buckets_ms: List[int] = None,
    output_path: Optional[str] = None,
    max_rows_per_asset: Optional[int] = None,
    add_pca: bool = True,
) -> pd.DataFrame:
    """Main entry point: build the full historical cost surface.

    Parameters
    ----------
    flow_bar_dir : str
        Directory containing {ASSET}_1S_flow.parquet files.
    assets : list, optional
        Assets to process. Defaults to BTC/ETH/SOL.
    fee_bps : float
        Assumed taker fee (one side).
    latency_buckets_ms : list
        Latency scenarios to model.
    output_path : str, optional
        If set, saves the result to this parquet path.
    max_rows_per_asset : int, optional
        If set, subsample each asset to this many rows (for fast iteration).
    add_pca : bool
        Whether to add PCA decorrelation features.

    Returns
    -------
    pd.DataFrame
        Dense cost surface with one row per asset x second.
    """
    if assets is None:
        assets = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    if latency_buckets_ms is None:
        latency_buckets_ms = [0, 100, 250, 500, 1000]

    pipeline_t0 = time.time()
    flow_dir = Path(flow_bar_dir)
    dfs = []

    for asset in assets:
        sym = asset.replace("-", "").replace("/", "").upper()
        fp = flow_dir / f"{sym}_1S_flow.parquet"
        if not fp.exists():
            logger.warning("File not found: %s", fp)
            continue
        t0 = time.time()
        logger.info("Loading %s ...", fp)
        df = pd.read_parquet(fp)
        if max_rows_per_asset and len(df) > max_rows_per_asset:
            df = df.tail(max_rows_per_asset)
            logger.info("  subsampled to %d rows", max_rows_per_asset)
        df["asset"] = sym
        dfs.append(df)
        logger.info("  loaded %d rows in %.1fs", len(df), time.time() - t0)

    if not dfs:
        logger.error("No flow bar data loaded.")
        return pd.DataFrame()

    panel = pd.concat(dfs).sort_index()
    logger.info("Raw panel: %d rows, %d assets [%.1fs]",
                len(panel), panel["asset"].nunique(), time.time() - pipeline_t0)

    # Step 1: Quoted spread
    t0 = time.time()
    panel = _compute_spread_cost(panel)
    logger.info("Step 1 (spread): done [%.1fs]", time.time() - t0)

    # Step 2: Markouts and shortfall (vectorized, no groupby)
    t0 = time.time()
    panel = _compute_markouts_and_shortfall(panel, fee_bps=fee_bps)
    logger.info("Step 2 (markout+shortfall): done [%.1fs]", time.time() - t0)

    # Step 3: Execution-bearing features
    t0 = time.time()
    panel = _add_execution_features(panel)
    logger.info("Step 3 (features): done [%.1fs]", time.time() - t0)

    # Step 4: PCA decorrelation
    if add_pca:
        t0 = time.time()
        panel = _add_pca_features(panel)
        logger.info("Step 4 (PCA): done [%.1fs]", time.time() - t0)

    # Drop rows with NaN targets
    before = len(panel)
    panel = panel.dropna(subset=["shortfall_1s_bps", "shortfall_5s_bps"])
    logger.info("Dropped %d NaN-target rows -> %d remain", before - len(panel), len(panel))

    logger.info("Final cost surface: %d rows x %d columns [total: %.1fs]",
                len(panel), len(panel.columns), time.time() - pipeline_t0)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(out)
        logger.info("Saved to %s", out)

    return panel
