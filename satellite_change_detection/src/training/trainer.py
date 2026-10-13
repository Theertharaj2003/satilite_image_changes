from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import time

import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from ..metrics import ConfusionAccumulator
from ..utils.checkpoint import save_checkpoint


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, config, output_dir: str | Path):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.amp_enabled)
        self.grad_clip = float(config["training"].get("grad_clip", 1.0))
        self.threshold = float(config["evaluation"].get("threshold", 0.5))

    def _run_loader(self, loader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        meter = ConfusionAccumulator()
        loss_sum = 0.0
        count = 0
        desc = "train" if train else "val"
        pbar = tqdm(loader, desc=desc, leave=False)
        for batch in pbar:
            x1 = batch["image_t1"].to(self.device, non_blocking=True)
            x2 = batch["image_t2"].to(self.device, non_blocking=True)
            y = batch["mask"].to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.amp_enabled):
                logits = self.model(x1, x2)
                loss = self.criterion(logits, y)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            loss_sum += float(loss.item()) * x1.shape[0]
            count += x1.shape[0]
            meter.update(logits.detach(), y.detach(), self.threshold)
        metrics = meter.compute()
        metrics["loss"] = loss_sum / max(count, 1)
        return metrics

    def fit(self, train_loader, val_loader, epochs: int, save_every: int = 1) -> Dict[str, Any]:
        best_f1 = -1.0
        best_epoch = -1
        history = []
        patience = int(self.config["training"].get("early_stopping_patience", 10))
        stale = 0
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._run_loader(train_loader, train=True)
            with torch.no_grad():
                val_metrics = self._run_loader(val_loader, train=False)
            if self.scheduler is not None:
                self.scheduler.step(val_metrics["loss"])

            row = {"epoch": epoch, "seconds": time.time() - t0}
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            history.append(row)
            print(f"Epoch {epoch:03d} | train F1={train_metrics['f1']:.4f} | val F1={val_metrics['f1']:.4f} | val IoU={val_metrics['iou']:.4f}")

            save_checkpoint(self.output_dir / "last.pt", self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config)
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_epoch = epoch
                stale = 0
                save_checkpoint(self.output_dir / "best.pt", self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config)
            else:
                stale += 1

            if save_every > 0 and epoch % save_every == 0:
                save_checkpoint(self.output_dir / f"epoch_{epoch:03d}.pt", self.model, self.optimizer, self.scheduler, epoch, val_metrics, self.config)

            (self.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val F1={best_f1:.4f}")
                break
        return {"history": history, "best_f1": best_f1, "best_epoch": best_epoch}
