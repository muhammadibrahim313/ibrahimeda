from __future__ import annotations
import pandas as pd
import numpy as np

def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert strings that look like numbers to numeric, leave others as original."""
    if series.dtype == object:
        converted = pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")
        keep = converted.notna().mean() > 0.9
        return converted if keep else series
    return series

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert object to category when sensible."""
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if s.dtype == object:
            s = to_numeric_safe(s)
            if s.dtype == object:
                if s.nunique(dropna=True) > 0 and s.nunique(dropna=True) <= max(20, int(0.1 * len(s))):
                    s = s.astype("category")
            out[col] = s
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = pd.to_numeric(s, downcast="float")
    return out

def train_valid_test_split(df: pd.DataFrame, test_size: float = 0.2, valid_size: float = 0.1, random_state: int = 42):
    """Simple random split without stratification. Returns df_train, df_valid, df_test."""
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_test = int(n * test_size)
    n_valid = int(n * valid_size)
    test_idx = idx[:n_test]
    valid_idx = idx[n_test:n_test+n_valid]
    train_idx = idx[n_test+n_valid:]
    return df.iloc[train_idx].copy(), df.iloc[valid_idx].copy(), df.iloc[test_idx].copy()
