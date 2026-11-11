"""Run a checkpoint on an already preprocessed CHW/NCHW NumPy tensor."""
import argparse
from pathlib import Path
import numpy as np
import torch
from grace_unetpp import GRACEUNetPlusPlus
from grace_unetpp.training import load_checkpoint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--dataset",choices=("acdc","msd"),required=True)
    p.add_argument("--output",type=Path,default=Path("prediction.npz"))
    p.add_argument("--device",default="cpu")
    args=p.parse_args()
    channels,classes=(1,4) if args.dataset=="acdc" else (3,2)
    image=np.load(args.input,allow_pickle=False).astype(np.float32)
    if image.ndim==3: image=image[None]
    if image.ndim!=4 or image.shape[1:]!=(channels,256,256) or not np.isfinite(image).all():
        raise ValueError(f"Expected finite N x {channels} x 256 x 256 normalized input")
    model=GRACEUNetPlusPlus(channels,classes,None).to(args.device).eval()
    load_checkpoint(model,args.checkpoint,device="cpu",strict=True)
    with torch.no_grad():
        refined,base,correction,ambiguity=model.forward_with_diagnostics(torch.from_numpy(image).to(args.device))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,prediction=refined.argmax(1).cpu().numpy(),
                        logits=refined.cpu().numpy(),base_logits=base.cpu().numpy(),
                        correction=correction.cpu().numpy(),ambiguity=ambiguity.cpu().numpy())

if __name__=="__main__": main()
