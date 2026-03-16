"""I/O helpers for parquet and CSV."""
from pathlib import Path
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist, return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False, engine="pyarrow")
    return p


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path), engine="pyarrow")


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, **kwargs)
    return p


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(Path(path), **kwargs)
