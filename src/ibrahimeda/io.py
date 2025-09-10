from __future__ import annotations
import os
import pandas as pd

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if ext in [".parquet", ".pq"]:
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise ImportError("Install pyarrow: pip install 'ibrahimeda[io]'") from e
    if ext in [".jsonl", ".json"]:
        return pd.read_json(path, lines=ext.endswith("jsonl"))
    raise ValueError(f"Unsupported extension: {ext}")

def memory_usage(df: pd.DataFrame) -> dict:
    b = int(df.memory_usage(deep=True).sum())
    return {"bytes": b, "mb": round(b / (1024**2), 3)}

def sample_rows(df: pd.DataFrame, n: int = 5, seed: int = 42) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
