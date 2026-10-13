from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.datasets import LEVIRCDDataset
from src.utils.visualization import save_sample_figure

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/config.yaml"); p.add_argument("--split",default="train"); p.add_argument("--index",type=int,default=0); p.add_argument("--output",default="results/sample.png"); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")); d=cfg["data"]
    ds=LEVIRCDDataset(d["root"],a.split,d["patch_size"],d["stride"],False,d["mean"],d["std"])
    if not (0 <= a.index < len(ds)): raise IndexError(f"index must be between 0 and {len(ds)-1}")
    save_sample_figure(ds[a.index],a.output); print(f"Saved {a.output}")
if __name__=="__main__": main()
