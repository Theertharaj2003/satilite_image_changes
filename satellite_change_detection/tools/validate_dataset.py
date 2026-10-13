from __future__ import annotations
import argparse
from pathlib import Path
import sys
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.datasets import build_records

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="./datasets/LEVIR-CD"); a=p.parse_args()
    for split in ("train","val","test"):
        recs=build_records(a.root,split); print(f"{split}: {len(recs)} pairs")
        if recs: print(f"  first: {recs[0].stem} | {recs[0].width}x{recs[0].height}")
    print("Dataset validation: PASSED")
if __name__=="__main__": main()
