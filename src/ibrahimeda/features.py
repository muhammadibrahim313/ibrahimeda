from __future__ import annotations
import pandas as pd

def frequency_encode(df: pd.DataFrame, col: str, suffix: str | None = None):
    """Add frequency encoded column for a categorical feature. Returns (new_df, mapping)."""
    counts = df[col].value_counts(dropna=False, normalize=True)
    mapping = counts.to_dict()
    new = df.copy()
    new_col = f"{col}_freq" if suffix is None else f"{col}_{suffix}"
    new[new_col] = new[col].map(mapping).fillna(0.0)
    return new, mapping

def one_hot_small(df: pd.DataFrame, cols: list[str], max_unique: int = 10) -> pd.DataFrame:
    """One hot encode only columns with small cardinality."""
    small = [c for c in cols if df[c].nunique(dropna=True) <= max_unique]
    return pd.get_dummies(df, columns=small, dummy_na=False)
