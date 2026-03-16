"""Rolling walk-forward fold builder with embargo."""
from __future__ import annotations
from datetime import timedelta

import pandas as pd

from src.utils.io import save_parquet, ensure_dir
from src.utils.logging import get_logger

log = get_logger("fold_builder")


def build_folds(
    timestamps: pd.Series,
    train_days: int = 180,
    val_days: int = 30,
    test_days: int = 30,
    step_days: int = 30,
    embargo_bars: int = 4,
) -> pd.DataFrame:
    """Generate rolling walk-forward fold definitions.

    Returns a DataFrame with columns:
        fold_id, split (train/val/test), start, end
    """
    unique_ts = timestamps.drop_duplicates().sort_values().reset_index(drop=True)
    ts_min = unique_ts.iloc[0]
    ts_max = unique_ts.iloc[-1]

    embargo_td = timedelta(hours=embargo_bars)
    train_td = timedelta(days=train_days)
    val_td = timedelta(days=val_days)
    test_td = timedelta(days=test_days)
    step_td = timedelta(days=step_days)

    folds: list[dict] = []
    fold_id = 0
    cursor = ts_min

    while True:
        train_start = cursor
        train_end = train_start + train_td
        val_start = train_end + embargo_td
        val_end = val_start + val_td
        test_start = val_end + embargo_td
        test_end = test_start + test_td

        if test_end > ts_max:
            break

        folds.append({"fold_id": fold_id, "split": "train", "start": train_start, "end": train_end})
        folds.append({"fold_id": fold_id, "split": "val", "start": val_start, "end": val_end})
        folds.append({"fold_id": fold_id, "split": "test", "start": test_start, "end": test_end})

        fold_id += 1
        cursor += step_td

    df = pd.DataFrame(folds)
    log.info("Generated %d folds", fold_id)
    return df


def save_fold_definitions(fold_df: pd.DataFrame, out_dir: str = "data/artifacts/folds") -> None:
    ensure_dir(out_dir)
    save_parquet(fold_df, f"{out_dir}/fold_definitions.parquet")
    log.info("Fold definitions saved to %s", out_dir)
