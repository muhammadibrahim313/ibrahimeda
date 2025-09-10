from __future__ import annotations
import pandas as pd

def add_lags(df: pd.DataFrame, col: str, lags: list[int] = [1,2,3]) -> pd.DataFrame:
    out = df.copy()
    for k in lags:
        out[f"{col}_lag{k}"] = out[col].shift(k)
    return out

def rolling_stats(df: pd.DataFrame, col: str, windows: list[int] = [3,7,14]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"{col}_r{w}_mean"] = out[col].rolling(w).mean()
        out[f"{col}_r{w}_std"] = out[col].rolling(w).std()
    return out
