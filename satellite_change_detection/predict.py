from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint


def preprocess(img, mean, std):
    arr=np.array(img.convert("RGB")).astype(np.float32)/255.0
    arr=(arr-np.asarray(mean,dtype=np.float32).reshape(1,1,3))/np.asarray(std,dtype=np.float32).reshape(1,1,3)
    return torch.from_numpy(arr.transpose(2,0,1)).float()


def pad_patch(x, ps):
    c,h,w=x.shape
    ph=max(0,ps-h); pw=max(0,ps-w)
    if ph or pw: x=torch.nn.functional.pad(x,(0,pw,0,ph))
    return x


def predict_pair(model, t1, t2, patch, stride, device, threshold):
    assert t1.shape == t2.shape
    _, h, w = t1.shape
    prob=np.zeros((h,w),dtype=np.float32); count=np.zeros((h,w),dtype=np.float32)
    ys=list(range(0,max(h-patch,0)+1,stride)); xs=list(range(0,max(w-patch,0)+1,stride))
    if not ys or ys[-1] != max(h-patch,0): ys.append(max(h-patch,0))
    if not xs or xs[-1] != max(w-patch,0): xs.append(max(w-patch,0))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                a=pad_patch(t1[:,y:y+patch,x:x+patch],patch).unsqueeze(0).to(device)
                b=pad_patch(t2[:,y:y+patch,x:x+patch],patch).unsqueeze(0).to(device)
                p=torch.sigmoid(model(a,b))[0,0].cpu().numpy()
                hh=min(patch,h-y); ww=min(patch,w-x)
                prob[y:y+hh,x:x+ww]+=p[:hh,:ww]
                count[y:y+hh,x:x+ww]+=1
    prob/=np.maximum(count,1)
    mask=(prob>=threshold).astype(np.uint8)*255
    return prob, mask


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--checkpoint",required=True); p.add_argument("--t1",required=True); p.add_argument("--t2",required=True); p.add_argument("--output",default="predictions/prediction.png"); p.add_argument("--patch-size",type=int,default=None); p.add_argument("--stride",type=int,default=None); p.add_argument("--threshold",type=float,default=None); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")); ckpt=load_checkpoint(a.checkpoint,map_location="cpu"); model_cfg=ckpt.get("config",{}).get("model",cfg["model"]); model=build_model(model_cfg); model.load_state_dict(ckpt["model_state_dict"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    d=cfg["data"]; t1=preprocess(Image.open(a.t1),d["mean"],d["std"]); t2=preprocess(Image.open(a.t2),d["mean"],d["std"])
    prob,mask=predict_pair(model,t1,t2,a.patch_size or d["patch_size"],a.stride or d["stride"],device,a.threshold if a.threshold is not None else cfg["evaluation"]["threshold"])
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(mask).save(out); Image.fromarray((prob*255).clip(0,255).astype(np.uint8)).save(out.with_name(out.stem+"_prob.png")); print(f"Saved {out}")
if __name__=="__main__": main()
