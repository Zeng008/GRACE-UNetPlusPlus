"""Train GRACE-UNet++ and the component/refinement ablations in the paper."""
import argparse
import gc
import json
from pathlib import Path
import torch
from grace_unetpp import GRACEUNetPlusPlus
from grace_unetpp.ablation import NoGFMRefineUNetPP
from grace_unetpp.data import make_source
from grace_unetpp.splits import loaders_for_fold, make_fold_manifest
from grace_unetpp.training import seed_everything, train_stage
from grace_unetpp.evaluation import evaluate_model

ROOT = Path(__file__).resolve().parent
PROTOCOLS = ("full", "without_gfm", "without_cgrr", "cgrr_only", "cgrr_unified")

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("acdc","msd"), required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs"))
    p.add_argument("--protocol", choices=PROTOCOLS, default="full")
    p.add_argument("--fold-indices", type=int, nargs="+", choices=range(5), default=list(range(5)))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--weights", choices=("imagenet","none"), default="imagenet")
    p.add_argument("--epochs", type=int)
    p.add_argument("--refine-epochs", type=int)
    p.add_argument("--batch-size", type=int)
    return p.parse_args()

def run_fold(args, cfg, source, row):
    fold = int(row["fold"])
    seed = cfg["train_seed"] + fold
    out = args.output / args.dataset / f"fold_{fold}" / args.protocol
    run_config = {"config":cfg,"fold":fold,"seed":seed,"protocol":args.protocol,
                  "split":row,"data_root":str(args.data_root.resolve()),
                  "weights":args.weights,"epochs":args.epochs or cfg["stage1"]["epochs"],
                  "refine_epochs":args.refine_epochs or cfg["stage2"]["epochs"],
                  "batch_size":args.batch_size or cfg["batch_size"]}
    out.mkdir(parents=True,exist_ok=True)
    config_path=out/"config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise RuntimeError(f"Run settings changed; choose a new --output directory: {out}")
    if (out/"complete.json").exists():
        print(f"Skipping completed fold: {out}")
        return
    config_path.write_text(json.dumps(run_config,indent=2))
    seed_everything(seed)
    loaders = loaders_for_fold(args.dataset,source,row,seed,
                              batch_size=run_config["batch_size"],workers=args.workers)
    train_loader,val_loader,test_loader=loaders
    weights=None if args.weights=="none" else "imagenet"
    model = (NoGFMRefineUNetPP(source,weights=weights) if args.protocol=="without_gfm"
             else GRACEUNetPlusPlus(cfg["in_channels"],cfg["classes"],weights))
    if args.protocol=="without_gfm":
        for p in model.parameters(): p.requires_grad=True
        for p in model.refiner.parameters(): p.requires_grad=False
    else:
        model.set_stage("gfm")
    stage1=cfg["stage1"]
    train_stage(source,model,train_loader,val_loader,out/"stage_gfm",
                epochs=run_config["epochs"],learning_rate=stage1["lr"],run_seed=seed,
                patience=stage1["patience"],min_epochs=min(stage1["min_epochs"],run_config["epochs"]),
                extra={"protocol":args.protocol,"fold":fold,"stage":"gfm"})
    stage2=cfg["stage2"]
    def refine(phase,subdir):
        model.set_trainable(phase)
        train_stage(source,model,train_loader,val_loader,out/subdir,
                    epochs=run_config["refine_epochs"],learning_rate=stage2["lr"],run_seed=seed,
                    patience=stage2["patience"],min_epochs=min(stage2["min_epochs"],run_config["refine_epochs"]),
                    initialize_best=True,extra={"protocol":args.protocol,"fold":fold,"stage":subdir})
    if args.protocol=="cgrr_only":
        refine("head","stage_cgrr_head")
    elif args.protocol=="cgrr_unified":
        refine("head","stage_cgrr_head")
        refine("tail","stage_cgrr")
    elif args.protocol in {"full","without_gfm"}:
        refine("tail","stage_cgrr")
    evaluate_model(source,args.dataset,model,test_loader,out/"evaluation",fold,args.protocol)
    (out/"complete.json").write_text(json.dumps({"status":"complete","fold":fold,"protocol":args.protocol},indent=2))
    del model,loaders,train_loader,val_loader,test_loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    args=parse_args()
    for value in [args.epochs,args.refine_epochs,args.batch_size]:
        if value is not None and value<1: raise ValueError("Epochs and batch size must be positive")
    cfg=json.loads((ROOT/"configs"/f"{args.dataset}.json").read_text())
    source=make_source(args.dataset,args.data_root,args.device)
    manifest=make_fold_manifest(args.dataset,source,ROOT/"splits"/f"{args.dataset}_5fold.json")
    for row in manifest["fold_assignments"]:
        if int(row["fold"]) in args.fold_indices: run_fold(args,cfg,source,row)

if __name__=="__main__": main()
