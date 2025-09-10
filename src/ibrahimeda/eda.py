from typing import List
import pandas as pd

def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted report of missing values and basic stats per column."""
    rows = []
    total = len(df)
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        pct = (miss / total * 100.0) if total else 0.0
        samples = list(s.dropna().astype(str).unique()[:3])
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing": miss,
            "percent": round(pct, 2),
            "nunique": int(s.nunique(dropna=True)),
            "sample_values": samples
        })
    out = pd.DataFrame(rows).sort_values(["percent", "missing"], ascending=False).reset_index(drop=True)
    return out

def quick_stats(df: pd.DataFrame, percentiles: List[float] | None = None) -> pd.DataFrame:
    """Like df.describe(include='all') with consistent percentiles."""
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    return df.describe(percentiles=percentiles, include="all").transpose().reset_index().rename(columns={"index":"column"})
