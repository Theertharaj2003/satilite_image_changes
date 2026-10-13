from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.datasets import LEVIRCDDataset
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.training.evaluator import evaluate_model, save_metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--output", default=None)
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ckpt=load_checkpoint(args.checkpoint, map_location="cpu")
    model_cfg=ckpt.get("config",{}).get("model",cfg["model"])
    model=build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    d=cfg["data"]
    ds=LEVIRCDDataset(d["root"], args.split, d["patch_size"], d["stride"], False, d["mean"], d["std"])
    loader=DataLoader(ds,batch_size=d["batch_size"],shuffle=False,num_workers=d["num_workers"],pin_memory=d["pin_memory"],persistent_workers=d["num_workers"]>0)
    metrics=evaluate_model(model,loader,device,cfg["evaluation"]["threshold"])
    print(metrics)
    out=args.output or str(Path(args.checkpoint).with_name(f"{args.split}_metrics.json"))
    save_metrics(metrics,out)
    print(f"Saved metrics to {out}")

if __name__=="__main__": main()
