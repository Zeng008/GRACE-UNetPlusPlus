"""Summarize five-fold outputs, explicitly distinguishing slice and case aggregation."""
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("runs"))
    p.add_argument("--aggregation",choices=("slice","case"),required=True)
    p.add_argument("--output",type=Path,default=Path("summary"))
    args=p.parse_args()
    paths=sorted(args.root.glob(f"*/fold_*/*/evaluation/{args.aggregation}_metrics.csv"))
    if not paths: raise FileNotFoundError(f"No evaluation CSVs under {args.root}")
    frame=pd.concat([pd.read_csv(path) for path in paths],ignore_index=True)
    columns=[c for c in frame if c.startswith(("Dice_class_","HD95_grid_class_","ASD_grid_class_","Mean_"))]
    folds=frame.groupby(["dataset","method","fold"])[columns].mean()
    summary=folds.groupby(["dataset","method"]).agg(["mean","std","count"])
    args.output.mkdir(parents=True,exist_ok=True)
    folds.to_csv(args.output/f"{args.aggregation}_fold_metrics.csv")
    summary.to_csv(args.output/f"{args.aggregation}_fivefold_summary.csv")
    print(summary.to_string())

if __name__=="__main__": main()
