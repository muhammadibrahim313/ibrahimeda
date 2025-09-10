import argparse
import os
import pandas as pd
from .eda import missing_report, quick_stats
from .prep import train_valid_test_split

def _read(path: str) -> pd.DataFrame:
    from .io import read_any
    return read_any(path)

def cmd_report(args):
    df = _read(args.path)
    rep = missing_report(df)
    with pd.option_context("display.max_rows", 200, "display.max_columns", 20):
        print(rep.to_string(index=False))

def cmd_skim(args):
    df = _read(args.path)
    miss = missing_report(df)[["column","missing","percent"]]
    print(f"rows={len(df)} cols={df.shape[1]}")
    print("\ndtypes:\n", df.dtypes)
    print("\nmissing:\n", miss.to_string(index=False))

def cmd_describe(args):
    df = _read(args.path)
    print(quick_stats(df).to_string(index=False))

def cmd_split(args):
    df = _read(args.path)
    os.makedirs(args.outdir, exist_ok=True)
    tr, va, te = train_valid_test_split(df, test_size=args.test, valid_size=args.valid, random_state=args.seed)
    tr.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    va.to_csv(os.path.join(args.outdir, "valid.csv"), index=False)
    te.to_csv(os.path.join(args.outdir, "test.csv"), index=False)
    print("Wrote:", args.outdir)

def main():
    p = argparse.ArgumentParser(prog="ibrahimeda", description="ibrahimeda CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("report", help="missing report for CSV/Parquet/JSONL")
    p1.add_argument("path")
    p1.set_defaults(func=cmd_report)

    p2 = sub.add_parser("skim", help="shape, dtypes, missing summary")
    p2.add_argument("path")
    p2.set_defaults(func=cmd_skim)

    p3 = sub.add_parser("describe", help="stats summary")
    p3.add_argument("path")
    p3.set_defaults(func=cmd_describe)

    p4 = sub.add_parser("split", help="random 80-10-10 split and write CSVs")
    p4.add_argument("path")
    p4.add_argument("outdir")
    p4.add_argument("--test", type=float, default=0.2)
    p4.add_argument("--valid", type=float, default=0.1)
    p4.add_argument("--seed", type=int, default=42)
    p4.set_defaults(func=cmd_split)

    args = p.parse_args()
    args.func(args)
