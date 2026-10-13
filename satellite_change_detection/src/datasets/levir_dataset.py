from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PairRecord:
    split: str
    stem: str
    t1_path: Path
    t2_path: Path
    mask_path: Path
    width: int
    height: int


def _find_dir(split_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = split_dir / name
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Could not find any of {list(candidates)} under {split_dir}. "
        "Expected folders such as T1/T2/label or A/B/label."
    )


def _file_map(folder: Path) -> dict[str, Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(f"No image files found in {folder}")
    return {p.stem: p for p in files}


def build_records(root: str | Path, split: str) -> List[PairRecord]:
    root = Path(root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    t1_dir = _find_dir(split_dir, ("T1", "t1", "A", "a", "image1", "img1"))
    t2_dir = _find_dir(split_dir, ("T2", "t2", "B", "b", "image2", "img2"))
    mask_dir = _find_dir(split_dir, ("label", "labels", "Label", "mask", "masks", "gt", "GT"))

    t1, t2, mask = _file_map(t1_dir), _file_map(t2_dir), _file_map(mask_dir)
    common = sorted(set(t1) & set(t2) & set(mask))
    if not common:
        raise RuntimeError(
            f"No matching image pairs found in {split_dir}. "
            "Make sure T1, T2 and label use matching filenames."
        )

    missing_t2 = sorted(set(t1) - set(t2))
    missing_mask = sorted(set(t1) - set(mask))
    if missing_t2 or missing_mask:
        raise RuntimeError(
            f"Unmatched files in {split}: missing T2={missing_t2[:5]}, missing mask={missing_mask[:5]}"
        )

    records: List[PairRecord] = []
    for stem in common:
        with Image.open(t1[stem]) as im:
            width, height = im.size
        records.append(PairRecord(split, stem, t1[stem], t2[stem], mask[stem], width, height))
    return records


def _pad_array(arr: np.ndarray, target_h: int, target_w: int, fill: int = 0) -> np.ndarray:
    h, w = arr.shape[:2]
    pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=fill)
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=fill)


def _sync_augment(t1: np.ndarray, t2: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply only spatially synchronized transforms to prevent label mismatch."""
    if random.random() < 0.5:
        t1 = np.flip(t1, axis=1).copy(); t2 = np.flip(t2, axis=1).copy(); mask = np.flip(mask, axis=1).copy()
    if random.random() < 0.5:
        t1 = np.flip(t1, axis=0).copy(); t2 = np.flip(t2, axis=0).copy(); mask = np.flip(mask, axis=0).copy()
    k = random.randint(0, 3)
    if k:
        t1 = np.rot90(t1, k, axes=(0, 1)).copy()
        t2 = np.rot90(t2, k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()
    return t1, t2, mask


class LEVIRCDDataset(Dataset):
    """Grid-patch LEVIR-CD dataset with synchronized augmentation.

    Supports split folders named train/val/test and modality folders named
    T1/T2/label or A/B/label. Filenames must match by stem.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int = 256,
        stride: int = 256,
        augment: bool = False,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        self.root = Path(root)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        if np.any(self.std <= 0):
            raise ValueError("std values must be > 0")

        self.records = build_records(root, split)
        self.index: List[tuple[int, int, int]] = []
        for rec_idx, rec in enumerate(self.records):
            for y in range(0, max(rec.height - 1, 0), stride):
                for x in range(0, max(rec.width - 1, 0), stride):
                    self.index.append((rec_idx, y, x))
                    if x + patch_size >= rec.width:
                        break
                if y + patch_size >= rec.height:
                    break
            if rec.width == 0 or rec.height == 0:
                continue
        if not self.index:
            raise RuntimeError(f"No patches generated for split={split}")

    def __len__(self) -> int:
        return len(self.index)

    def _load_patch(self, rec: PairRecord, y: int, x: int):
        with Image.open(rec.t1_path) as im:
            t1 = np.array(im.convert("RGB"))
        with Image.open(rec.t2_path) as im:
            t2 = np.array(im.convert("RGB"))
        with Image.open(rec.mask_path) as im:
            mask = np.array(im.convert("L"))

        ps = self.patch_size
        h, w = t1.shape[:2]
        y0, x0 = min(y, max(h - ps, 0)), min(x, max(w - ps, 0))
        t1 = t1[y0:y0 + ps, x0:x0 + ps]
        t2 = t2[y0:y0 + ps, x0:x0 + ps]
        mask = mask[y0:y0 + ps, x0:x0 + ps]

        t1 = _pad_array(t1, ps, ps, 0)
        t2 = _pad_array(t2, ps, ps, 0)
        mask = _pad_array(mask, ps, ps, 0)
        return t1, t2, mask

    def __getitem__(self, idx: int) -> dict:
        rec_idx, y, x = self.index[idx]
        rec = self.records[rec_idx]
        t1, t2, mask = self._load_patch(rec, y, x)

        if self.augment:
            t1, t2, mask = _sync_augment(t1, t2, mask)

        t1 = t1.astype(np.float32) / 255.0
        t2 = t2.astype(np.float32) / 255.0
        t1 = (t1 - self.mean) / self.std
        t2 = (t2 - self.mean) / self.std
        t1 = torch.from_numpy(t1.transpose(2, 0, 1)).float()
        t2 = torch.from_numpy(t2.transpose(2, 0, 1)).float()
        mask = torch.from_numpy((mask > 127).astype(np.float32))[None, ...]

        return {
            "image_t1": t1,
            "image_t2": t2,
            "mask": mask,
            "id": rec.stem,
            "x": x,
            "y": y,
        }
