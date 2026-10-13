from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_image(t: torch.Tensor, mean=None, std=None) -> np.ndarray:
    x = t.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = x.permute(1, 2, 0)
    x = x.numpy()
    if mean is not None and std is not None and x.ndim == 3 and x.shape[-1] == len(mean):
        x = x * np.asarray(std) + np.asarray(mean)
    return np.clip(x, 0, 1)


def save_sample_figure(sample: dict, output_path: str | Path, prediction: Optional[torch.Tensor] = None) -> None:
    """Save T1, T2, target and optional prediction in one figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ncols = 4 if prediction is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(_to_image(sample["image_t1"]))
    axes[0].set_title("T1")
    axes[1].imshow(_to_image(sample["image_t2"]))
    axes[1].set_title("T2")
    axes[2].imshow(sample["mask"].detach().cpu().squeeze().numpy(), cmap="gray")
    axes[2].set_title("Ground Truth")
    if prediction is not None:
        pred = prediction.detach().cpu().squeeze().numpy()
        axes[3].imshow(pred, cmap="gray")
        axes[3].set_title("Prediction")

    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
