import pandas as pd
from ibrahimeda import eda, prep, features, metrics

def test_missing_report():
    df = pd.DataFrame({"a":[1, None, 3], "b":["x","y","y"]})
    rep = eda.missing_report(df)
    assert set(rep.columns) >= {"column","missing","percent"}

def test_optimize():
    df = pd.DataFrame({"n":[1,2,3], "s":["1","2","3"]})
    out = prep.optimize_dtypes(df)
    assert str(out["n"].dtype).startswith("int")

def test_freq_encode():
    df = pd.DataFrame({"c":["a","a","b", None]})
    new, mapping = features.frequency_encode(df, "c")
    assert "c_freq" in new.columns

def test_metrics():
    acc, prec, rec, f1 = metrics.classification_basic([1,0,1,0],[1,0,0,0])
    assert 0 <= acc <= 1
