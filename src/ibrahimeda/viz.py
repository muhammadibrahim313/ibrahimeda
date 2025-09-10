from __future__ import annotations
import pandas as pd

def _get_backend(backend: str):
    backend = (backend or "matplotlib").lower()
    if backend == "plotly":
        try:
            import plotly.express as px
            return ("plotly", px)
        except Exception as e:
            raise ImportError("Install plotly: pip install 'ibrahimeda[plot]'") from e
    import matplotlib.pyplot as plt
    return ("matplotlib", plt)

def quick_hist(df: pd.DataFrame, col: str, bins: int = 30, backend: str = "matplotlib"):
    kind, lib = _get_backend(backend)
    s = df[col].dropna()
    if kind == "plotly":
        fig = lib.histogram(s, x=col, nbins=bins)
        fig.show()
        return fig
    fig = lib.figure()
    lib.hist(s, bins=bins)
    lib.title(f"Histogram of {col}")
    lib.xlabel(col)
    lib.ylabel("count")
    lib.show()
    return fig

def quick_corr(df: pd.DataFrame, method: str = "pearson", backend: str = "matplotlib"):
    kind, lib = _get_backend(backend)
    corr = df.select_dtypes(include=["number"]).corr(method=method)
    if kind == "plotly":
        fig = lib.imshow(corr)
        fig.show()
        return fig
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.gca()
    cax = ax.imshow(corr, interpolation="nearest")
    plt.title("Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()
    return fig
