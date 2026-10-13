from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from src.datasets import LEVIRCDDataset
from src.losses import BCEDiceLoss
from src.metrics import ConfusionAccumulator
from src.models.factory import build_model


def make_fake_dataset(root: Path):
    rng=np.random.default_rng(123)
    for split,n in [("train",2),("val",1),("test",1)]:
        for sub in ["T1","T2","label"]: (root/split/sub).mkdir(parents=True,exist_ok=True)
        for i in range(n):
            t1=(rng.random((64,64,3))*255).astype(np.uint8)
            t2=t1.copy(); t2[16:32,16:32]=255-t2[16:32,16:32]
            mask=np.zeros((64,64),np.uint8); mask[16:32,16:32]=255
            Image.fromarray(t1).save(root/split/"T1"/f"sample_{i}.png")
            Image.fromarray(t2).save(root/split/"T2"/f"sample_{i}.png")
            Image.fromarray(mask).save(root/split/"label"/f"sample_{i}.png")


def main():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/"LEVIR-CD"; make_fake_dataset(root)
        ds=LEVIRCDDataset(root,"train",patch_size=64,stride=64,augment=True)
        loader=DataLoader(ds,batch_size=1,shuffle=False)
        batch=next(iter(loader))
        assert batch["image_t1"].shape==(1,3,64,64)
        assert batch["image_t2"].shape==(1,3,64,64)
        assert batch["mask"].shape==(1,1,64,64)
        device=torch.device("cpu")
        x1,x2,y=batch["image_t1"],batch["image_t2"],batch["mask"]
        for name in ["siamese_cnn","temporal_transformer","proposed"]:
            cfg={"type":name,"channels":[8,16,32,64],"heads":4,"layers":1,"dropout":0.1,"use_temporal_self_attention":True,"use_cross_attention":True,"use_multiscale":True}
            model=build_model(cfg).to(device)
            logits=model(x1,x2)
            assert logits.shape==y.shape, (name,logits.shape,y.shape)
            loss=BCEDiceLoss()(logits,y); loss.backward()
            meter=ConfusionAccumulator(); meter.update(logits.detach(),y)
            metrics=meter.compute()
            print(f"{name}: output={tuple(logits.shape)} loss={loss.item():.4f} F1={metrics['f1']:.4f}")
        print("SMOKE TEST PASSED")

if __name__=="__main__": main()
