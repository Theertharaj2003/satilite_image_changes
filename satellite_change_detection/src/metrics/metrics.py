from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math

import torch


@dataclass
class ConfusionAccumulator:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self):
        return self.tp + self.tn + self.fp + self.fn

    def update(self, logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> None:
        pred = (torch.sigmoid(logits) >= threshold).to(torch.bool)
        target = target.to(torch.bool)
        self.tp += int((pred & target).sum().item())
        self.tn += int((~pred & ~target).sum().item())
        self.fp += int((pred & ~target).sum().item())
        self.fn += int((~pred & target).sum().item())

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        oa = (tp + tn) / (self.total + eps)
        specificity = tn / (tn + fp + eps)
        pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / ((self.total + eps) ** 2)
        kappa = (oa - pe) / (1 - pe + eps)
        fpr = fp / (fp + tn + eps)
        fnr = fn / (fn + tp + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "oa": oa,
            "kappa": kappa,
            "specificity": specificity,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }


def dice_from_confusion(m: Dict[str, float]) -> float:
    return m["f1"]
