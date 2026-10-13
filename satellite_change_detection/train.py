from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
from typing import Any

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.datasets import LEVIRCDDataset
from src.losses import BCEDiceLoss
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.utils.seed import set_seed


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    if path.name.lower() == "config.yaml":
        return cfg

    base_path = path.parent / "config.yaml"
    if base_path.is_file():
        base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Base configuration must be a YAML mapping: {base_path}")
        return deep_merge(base, cfg)

    return cfg


def build_loaders(cfg):
    d = cfg["data"]
    train_ds = LEVIRCDDataset(d["root"], d["train_split"], d["patch_size"], d["stride"], True, d["mean"], d["std"])
    val_ds = LEVIRCDDataset(d["root"], d["val_split"], d["patch_size"], d["stride"], False, d["mean"], d["std"])
    train_loader = DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True, num_workers=d["num_workers"], pin_memory=d["pin_memory"], persistent_workers=d["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=d["batch_size"], shuffle=False, num_workers=d["num_workers"], pin_memory=d["pin_memory"], persistent_workers=d["num_workers"] > 0)
    return train_loader, val_loader, train_ds, val_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--override-model", default=None, choices=["siamese_cnn", "temporal_transformer", "proposed"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.override_model:
        cfg["model"]["type"] = args.override_model
    set_seed(cfg["project"]["seed"], cfg["project"].get("deterministic", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, train_ds, val_ds = build_loaders(cfg)
    print(f"Train patches: {len(train_ds)} | Validation patches: {len(val_ds)}")

    model = build_model(cfg["model"])
    criterion = BCEDiceLoss(**cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg["scheduler"]["factor"], patience=cfg["scheduler"]["patience"], min_lr=cfg["scheduler"]["min_lr"])

    output_dir = Path(cfg["output"]["root"])
    trainer = Trainer(model, optimizer, scheduler, criterion, device, cfg, output_dir)
    summary = trainer.fit(train_loader, val_loader, cfg["training"]["epochs"], cfg["training"]["save_every"])
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
