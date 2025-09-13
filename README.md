# ibrahimeda - pypy module

Small, dependable utilities you reuse in data projects. Simple API, quick results.

## Highlights
- EDA: missing report, quick stats, type summary
- Prep: safe dtype optimization, simple impute, random splits
- Features: frequency encode, small one-hot
- Metrics: classification and regression basics
- Time series: lag and rolling features
- Text: lower-punct-space cleanup and word counts
- IO: read_any for csv/parquet/jsonl and quick sampling
- Viz: quick histograms and correlation plots
- CLI: `ibrahimeda report`, `ibrahimeda skim`, `ibrahimeda split`, `ibrahimeda describe`

## Quick start
```python
import pandas as pd
from ibrahimeda import eda, prep, features, metrics, timeseries, text, io

df = pd.read_csv("train.csv")
eda.missing_report(df).head()
opt = prep.optimize_dtypes(df)
opt2, mapping = features.frequency_encode(opt, "city")
lagged = timeseries.add_lags(opt2, col="sales", lags=[1,7])
clean = text.basic_clean("Hello, World! 123")
```

## CLI
```bash
ibrahimeda report data.csv
ibrahimeda skim data.csv
ibrahimeda describe data.csv
ibrahimeda split data.csv out/
```

## Install
```bash
pip install ibrahimeda
# extras
pip install "ibrahimeda[ml,plot,io,dev]"
```

## License
MIT
