from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from ..metrics import ConfusionAccumulator


def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()
    meter = ConfusionAccumulator()
    loss_sum = 0.0
    criterion = None
    with torch.no_grad():
        for batch in tqdm(loader, desc="test", leave=False):
            x1 = batch["image_t1"].to(device)
            x2 = batch["image_t2"].to(device)
            y = batch["mask"].to(device)
            logits = model(x1, x2)
            meter.update(logits, y, threshold)
    return meter.compute()


def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
