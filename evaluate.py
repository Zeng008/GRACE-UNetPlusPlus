"""Evaluate one supplied GRACE checkpoint on its recorded outer test fold."""
import argparse
import json
from pathlib import Path
from grace_unetpp import GRACEUNetPlusPlus
from grace_unetpp.ablation import NoGFMRefineUNetPP
from grace_unetpp.data import make_source
from grace_unetpp.splits import loaders_for_fold, make_fold_manifest
from grace_unetpp.training import load_checkpoint
from grace_unetpp.evaluation import evaluate_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",choices=("acdc","msd"),required=True)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--fold",type=int,choices=range(5),required=True)
    p.add_argument("--without-gfm",action="store_true")
    p.add_argument("--output",type=Path,default=Path("evaluation"))
    p.add_argument("--device",default=None)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    root=Path(__file__).resolve().parent
    cfg=json.loads((root/"configs"/f"{args.dataset}.json").read_text())
    source=make_source(args.dataset,args.data_root,args.device)
    manifest=make_fold_manifest(args.dataset,source,root/"splits"/f"{args.dataset}_5fold.json")
    row=next(r for r in manifest["fold_assignments"] if r["fold"]==args.fold)
    *_,test_loader=loaders_for_fold(args.dataset,source,row,731+args.fold,workers=args.workers)
    model=(NoGFMRefineUNetPP(source,weights=None) if args.without_gfm else
           GRACEUNetPlusPlus(cfg["in_channels"],cfg["classes"],None))
    load_checkpoint(model,args.checkpoint,device="cpu",strict=True)
    evaluate_model(source,args.dataset,model,test_loader,args.output,args.fold,"grace")

if __name__=="__main__": main()
