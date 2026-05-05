#!/usr/bin/env python3
"""
Train a CNN model on Martian junction patches.

Key requirements implemented:
- Architecture: ShallowCNN_GAP or DeeperCNN_GAP with AdaptiveAvgPool2d + Linear head
- Loss: CrossEntropyLoss(label_smoothing=0.03) with class-balanced sampling
- Save best checkpoint by validation loss to models/classifiers/CNN_ft_gauss40.pt
"""

from __future__ import annotations

import argparse
import copy
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) for multi-class classification.

    Downweights well-classified examples so the model focuses on hard ones.
    With gamma=0 this is equivalent to standard cross-entropy.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,  # type: ignore[arg-type]
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)  # probability of correct class
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


TASK_CLASS_NAMES = {
    "multiclass": ["N", "T", "X", "Y"],
    "gate": ["N", "P"],
    "type": ["T", "X", "Y"],
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"relpath", "label", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest must include columns {required}. Found: {reader.fieldnames}")
        for row in reader:
            rows.append({k: str(v) for k, v in row.items()})
    return rows


def build_label_maps(labels: Sequence[str]) -> Tuple[Dict[str, int], List[str]]:
    idx_to_label = sorted(set(labels))
    label_to_idx = {label: i for i, label in enumerate(idx_to_label)}
    return label_to_idx, idx_to_label


def maybe_strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_init_state(
    model: nn.Module,
    init_model_path: Path,
    reset_head: bool,
) -> Tuple[List[str], List[str]]:
    if not init_model_path.exists():
        raise FileNotFoundError(f"--init-model not found: {init_model_path}")
    payload = torch.load(init_model_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint payload at {init_model_path}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected state_dict dict at {init_model_path}, got {type(state_dict)}")

    cast_state = {str(k): v for k, v in state_dict.items()}
    cast_state = maybe_strip_module_prefix(cast_state)
    if reset_head:
        cast_state = {k: v for k, v in cast_state.items() if not k.startswith("head.")}
    strict = not reset_head
    missing, unexpected = model.load_state_dict(cast_state, strict=strict)
    return list(missing), list(unexpected)


def _mask_other_junctions(skel_u8: np.ndarray, mask_radius: int = 8) -> np.ndarray:
    """Zero out non-central junction regions in a patch.

    Detects junction clusters (degree>=3 pixel groups), identifies the one closest
    to patch center, and masks all others with a disk of mask_radius pixels.
    Branches between junctions are preserved.
    """
    from scipy import ndimage as _ndi

    bw = (skel_u8 > 0).astype(np.uint8)
    h, w = bw.shape
    cy, cx = h // 2, w // 2

    # Degree map
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    deg = cv2.filter2D(bw.astype(np.float32), -1, kernel.astype(np.float32),
                       borderType=cv2.BORDER_CONSTANT).astype(np.int32)

    junc_pixels = (bw == 1) & (deg >= 3)
    if not np.any(junc_pixels):
        return skel_u8

    # Cluster junction pixels (small dilation to connect adjacent deg>=3 pixels)
    dilated = cv2.dilate(junc_pixels.astype(np.uint8), np.ones((5, 5), np.uint8))
    labeled, n_comp = _ndi.label(dilated)
    if n_comp <= 1:
        return skel_u8  # only one junction cluster — nothing to mask

    # Find centroids
    centroids = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(labeled == comp_id)
        centroids.append((comp_id, float(xs.mean()), float(ys.mean())))

    # Identify central cluster (closest centroid to patch center)
    central_id = min(centroids, key=lambda c: (c[1] - cx) ** 2 + (c[2] - cy) ** 2)[0]

    # Build mask: 1 everywhere, 0 in disk(mask_radius) around non-central centroids
    mask = np.ones((h, w), dtype=np.float32)
    for comp_id, ccx, ccy in centroids:
        if comp_id == central_id:
            continue
        # Zero out a disk around this junction
        yy, xx = np.ogrid[0:h, 0:w]
        dist2 = (xx - ccx) ** 2 + (yy - ccy) ** 2
        mask[dist2 <= mask_radius ** 2] = 0.0

    return (skel_u8.astype(np.float32) * mask).astype(np.uint8)


class PatchDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Tuple[Path, int, Path | None, float]],
        patch_size: int = 96,
        in_channels: int = 2,
        train_shift_max: int = 0,
        train_shift_prob: float = 0.0,
        train_rotate: bool = False,
        train_flip: bool = False,
        center_gaussian_sigma: float = 0.0,
        mask_other_junctions: int = 0,
    ):
        self.items = list(items)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.train_shift_max = max(0, int(train_shift_max))
        self.train_shift_prob = float(max(0.0, min(1.0, train_shift_prob)))
        self.train_rotate = bool(train_rotate)
        self.train_flip = bool(train_flip)
        self.mask_other_junctions_radius = int(mask_other_junctions)

        # Pre-compute Gaussian center mask if enabled
        self.gaussian_mask: np.ndarray | None = None
        if center_gaussian_sigma > 0:
            h = w = patch_size
            cy, cx = h / 2.0, w / 2.0
            y_grid, x_grid = np.mgrid[0:h, 0:w]
            sigma = float(center_gaussian_sigma)
            self.gaussian_mask = np.exp(
                -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
            ).astype(np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        skel_path, y, context_path, _sample_weight = self.items[idx]

        skel = np.asarray(Image.open(skel_path).convert("L"), dtype=np.float32)
        if skel.shape != (self.patch_size, self.patch_size):
            skel = cv2.resize(skel, (self.patch_size, self.patch_size),
                              interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Mask out non-central junctions if enabled (before normalization)
        if self.mask_other_junctions_radius > 0:
            skel = _mask_other_junctions(skel.astype(np.uint8), mask_radius=self.mask_other_junctions_radius).astype(np.float32)

        skel_f32 = skel / 255.0

        # Build 3-pixel mask channel in-memory from the 1-pixel skeleton.
        mask_bool = binary_dilation(skel_f32 > 0.5, structure=np.ones((3, 3), dtype=bool))
        mask_f32 = mask_bool.astype(np.float32)

        channels: List[np.ndarray] = [skel_f32, mask_f32]
        if self.in_channels >= 3:
            context_f32: np.ndarray | None = None
            if context_path is not None and context_path.exists():
                context = np.asarray(Image.open(context_path).convert("L"), dtype=np.float32)
                if context.shape != (self.patch_size, self.patch_size):
                    context = cv2.resize(context, (self.patch_size, self.patch_size),
                                         interpolation=cv2.INTER_NEAREST).astype(np.float32)
                context_f32 = context / 255.0
            if context_f32 is None:
                context_f32 = mask_f32
            channels.append(context_f32.astype(np.float32, copy=False))

        while len(channels) < self.in_channels:
            channels.append(np.zeros_like(skel_f32, dtype=np.float32))

        if self.train_shift_max > 0 and self.train_shift_prob > 0.0 and random.random() < self.train_shift_prob:
            dy = random.randint(-self.train_shift_max, self.train_shift_max)
            dx = random.randint(-self.train_shift_max, self.train_shift_max)
            if dx != 0 or dy != 0:
                shifted: List[np.ndarray] = []
                for ch in channels[: self.in_channels]:
                    out = np.zeros_like(ch, dtype=np.float32)
                    h, w = ch.shape
                    if dx >= 0:
                        src_x0, src_x1 = 0, w - dx
                        dst_x0, dst_x1 = dx, w
                    else:
                        src_x0, src_x1 = -dx, w
                        dst_x0, dst_x1 = 0, w + dx
                    if dy >= 0:
                        src_y0, src_y1 = 0, h - dy
                        dst_y0, dst_y1 = dy, h
                    else:
                        src_y0, src_y1 = -dy, h
                        dst_y0, dst_y1 = 0, h + dy
                    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
                        out[dst_y0:dst_y1, dst_x0:dst_x1] = ch[src_y0:src_y1, src_x0:src_x1]
                    shifted.append(out)
                channels = shifted

        # Random 90-degree rotation augmentation (0, 90, 180, 270)
        if self.train_rotate:
            k = random.randint(0, 3)
            if k > 0:
                channels = [np.rot90(ch, k).copy() for ch in channels]

        # Random flip augmentation (horizontal and/or vertical)
        if self.train_flip:
            if random.random() < 0.5:
                channels = [np.fliplr(ch).copy() for ch in channels]
            if random.random() < 0.5:
                channels = [np.flipud(ch).copy() for ch in channels]

        # Apply Gaussian center mask to attenuate peripheral skeleton structure
        if self.gaussian_mask is not None:
            channels = [ch * self.gaussian_mask for ch in channels]

        x = torch.from_numpy(np.stack(channels[: self.in_channels], axis=0))
        return x, int(y)


class ShallowCNN_GAP(nn.Module):
    def __init__(self, num_classes: int, c1: int = 32, c2: int = 64, c3: int = 128, in_channels: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class DeeperCNN_GAP(nn.Module):
    """
    5-conv architecture with effective receptive field ~54px (vs ~22px for ShallowCNN_GAP).
    Covers enough of the 96x96 patch to see full branch angular geometry for T/Y discrimination.
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        # Block 1: 96 -> 48
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2: 48 -> 24
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Block 3: 24 -> 24 (no pool — keep resolution for RF growth)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Block 4: 24 -> 12
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Block 5: 12 -> 6
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 24
        x = F.relu(self.bn3(self.conv3(x)))               # -> 24 (no pool)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # -> 12
        x = self.pool(F.relu(self.bn5(self.conv5(x))))   # -> 6
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.head(x)


class DeeperCNN_GAP_v2(nn.Module):
    """
    6-conv architecture with effective receptive field ~86px (vs 54px for DeeperCNN_GAP).
    Adds a 6th conv block (no pool) at the 6x6 spatial level so each GAP position
    sees 90% of the 96x96 patch — sufficient to capture full branch geometry.

    RF calculation:
      conv1(3x3)+pool → 4 | conv2(3x3)+pool → 10 | conv3(3x3) → 18 |
      conv4(3x3)+pool → 30 | conv5(3x3)+pool → 54 | conv6(3x3) → 86
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        # Block 1: 96 -> 48
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Block 2: 48 -> 24
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Block 3: 24 -> 24 (no pool)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Block 4: 24 -> 12
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        # Block 5: 12 -> 6
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        # Block 6: 6 -> 6 (no pool — pure RF expansion, channel reduction)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 48, RF=4
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 24, RF=10
        x = F.relu(self.bn3(self.conv3(x)))               # -> 24, RF=18
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # -> 12, RF=30
        x = self.pool(F.relu(self.bn5(self.conv5(x))))   # -> 6,  RF=54
        x = F.relu(self.bn6(self.conv6(x)))               # -> 6,  RF=86
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        return self.head(x)


class DeeperCNN_Flatten_v2(nn.Module):
    """
    Same conv backbone as DeeperCNN_GAP_v2, but replaces GAP with flatten.
    The 6x6x128 feature map is flattened to 4,608 dims, preserving spatial info.
    A small MLP head reduces to num_classes.
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        # 6x6x128 = 4608 -> bigger 2-layer head for spatial pattern capacity
        self.head = nn.Sequential(
            nn.Linear(128 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 24
        x = F.relu(self.bn3(self.conv3(x)))               # -> 24
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # -> 12
        x = self.pool(F.relu(self.bn5(self.conv5(x))))   # -> 6
        x = F.relu(self.bn6(self.conv6(x)))               # -> 6
        x = torch.flatten(x, 1)                           # -> 4608
        x = self.drop(x)
        return self.head(x)


class DeeperCNN_SPP_v2(nn.Module):
    """
    Same conv backbone as DeeperCNN_GAP_v2, but uses Spatial Pyramid Pooling.
    Pools at 1x1 (128), 2x2 (512), 4x4 (2048) = 2,688 total features.
    Preserves multi-scale spatial structure.
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        # SPP levels: 1x1 + 2x2 + 3x3 = 128*(1+4+9) = 1792
        # (use 3x3 instead of 4x4 because 6x6 feature map isn't divisible by 4 on MPS)
        self.spp1 = nn.AdaptiveAvgPool2d((1, 1))
        self.spp2 = nn.AdaptiveAvgPool2d((2, 2))
        self.spp3 = nn.AdaptiveAvgPool2d((3, 3))
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(128 * (1 + 4 + 9), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 24
        x = F.relu(self.bn3(self.conv3(x)))               # -> 24
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # -> 12
        x = self.pool(F.relu(self.bn5(self.conv5(x))))   # -> 6
        x = F.relu(self.bn6(self.conv6(x)))               # -> 6, 128ch
        s1 = torch.flatten(self.spp1(x), 1)               # -> 128
        s2 = torch.flatten(self.spp2(x), 1)               # -> 512
        s3 = torch.flatten(self.spp3(x), 1)               # -> 1152
        x = torch.cat([s1, s2, s3], dim=1)                # -> 1792
        x = self.drop(x)
        return self.head(x)


class DeeperCNN_Attn_v2(nn.Module):
    """
    Same conv backbone as DeeperCNN_GAP_v2, but uses learned attention pooling.
    A 1x1 conv produces per-position attention weights, then features are
    weighted-averaged instead of uniformly averaged (as in GAP).
    Learns WHICH spatial positions matter for classification.
    """

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        # Attention: 1x1 conv -> scalar per spatial position
        self.attn_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # -> 48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # -> 24
        x = F.relu(self.bn3(self.conv3(x)))               # -> 24
        x = self.pool(F.relu(self.bn4(self.conv4(x))))   # -> 12
        x = self.pool(F.relu(self.bn5(self.conv5(x))))   # -> 6
        x = F.relu(self.bn6(self.conv6(x)))               # -> 6, 128ch
        # Attention weights: (B, 1, 6, 6) -> softmax over spatial dims
        attn = self.attn_conv(x)                           # (B, 1, 6, 6)
        attn = attn.view(attn.size(0), 1, -1)             # (B, 1, 36)
        attn = F.softmax(attn, dim=2)                      # normalize over positions
        attn = attn.view(attn.size(0), 1, 6, 6)           # (B, 1, 6, 6)
        # Weighted average: attention * features, sum over spatial
        x = (x * attn).sum(dim=(2, 3))                    # (B, 128)
        x = self.drop(x)
        return self.head(x)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * len(xb)
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_count += int(len(xb))

    avg_loss = total_loss / max(1, total_count)
    acc = total_correct / max(1, total_count)
    return avg_loss, acc


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    scheduler_type: str = "plateau",
) -> Tuple[Dict[str, torch.Tensor], int, float]:
    """
    Train loop with early stopping. Supports multiple scheduler types:
    - plateau: ReduceLROnPlateau (step with val_loss)
    - cosine/cosine_warm: CosineAnnealing* (step without metric)
    """
    best_val_loss = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1
    stale = 0

    import time as _time
    t0 = _time.time()

    print()
    print(f"  {'Ep':>3}  {'train_loss':>10} {'train_acc':>9}  {'val_loss':>10} {'val_acc':>9}  {'lr':>9}  {''}  {'time':>5}")
    print(f"  {'---':>3}  {'----------':>10} {'---------':>9}  {'----------':>10} {'---------':>9}  {'---------':>9}  {''}  {'-----':>5}")

    for epoch in range(1, epochs + 1):
        ep_t0 = _time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        if scheduler_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        ep_sec = _time.time() - ep_t0

        is_best = val_loss < best_val_loss
        marker = " *" if is_best else ""

        print(
            f"  {epoch:3d}  {train_loss:10.4f} {train_acc:9.4f}  "
            f"{val_loss:10.4f} {val_acc:9.4f}  "
            f"{current_lr:9.6f}  {marker:2s}  {ep_sec:5.1f}s"
        )

        if is_best:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  --- Early stop (no improvement for {patience} epochs)")
                break

    total_sec = _time.time() - t0
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    print(f"  --- Best: epoch {best_epoch}  val_loss={best_val_loss:.4f}  ({total_sec:.0f}s total)")
    return best_state, best_epoch, best_val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ShallowCNN_GAP on synthetic manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to training manifest CSV.")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to manifest parent.")
    parser.add_argument("--output-model", type=Path, default=Path("models/classifiers/CNN_ft_gauss40.pt"))
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--scheduler-patience", type=int, default=3,
                        help="ReduceLROnPlateau patience (epochs without val improvement before LR halves).")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw"],
        help="Optimizer: adam (L2 reg) or adamw (decoupled weight decay, recommended).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["plateau", "cosine", "cosine_warm"],
        help="LR scheduler: plateau (ReduceLROnPlateau), cosine (CosineAnnealingLR), "
             "cosine_warm (CosineAnnealingWarmRestarts with T_0=10, T_mult=2).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear LR warmup for this many epochs (0=disabled). Recommended 3-5 for fine-tuning.",
    )
    parser.add_argument(
        "--discriminative-lr",
        type=float,
        default=0.0,
        help="Per-block LR decay factor for fine-tuning. E.g., 0.1 means each earlier block "
             "gets 10x lower LR. 0=disabled (uniform LR). Only used with --init-model.",
    )
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes. 2-4 recommended for M1 MPS.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c1", type=int, default=32)
    parser.add_argument("--c2", type=int, default=64)
    parser.add_argument("--c3", type=int, default=128)
    parser.add_argument("--in-channels", type=int, default=2, help="CNN input channels. Default keeps current 2-channel mode.")
    parser.add_argument("--context-column", type=str, default="context_image", help="Optional manifest column for context image path.")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument(
        "--sample-weight-mode",
        type=str,
        default="none",
        choices=["none", "manifest"],
        help="Optional train sampling reweighting using manifest sample_weight column.",
    )
    parser.add_argument("--sample-weight-col", type=str, default="sample_weight")
    parser.add_argument("--x-class-weight", type=float, default=1.0, help="Optional multiplier for X-class train samples.")
    parser.add_argument("--n-class-weight", type=float, default=1.0,
                        help="Optional multiplier for N-class (reject/false-positive) train samples. "
                        "Values <1.0 reduce N emphasis in favor of positive classes.")
    parser.add_argument("--t-class-weight", type=float, default=1.0,
                        help="Optional multiplier for T-class train samples. "
                        "Values >1.0 oversample T to improve T recall.")
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "gate", "type"],
        help="Training task: multiclass N/T/X/Y, gate N/P, or type T/X/Y (N filtered).",
    )
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Optional class-balanced CE weighting computed from train split labels.",
    )
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default="auto",
        choices=["auto", "sampler", "loss", "both"],
        help=(
            "How to apply imbalance handling: "
            "auto=use sampler only when class weights are disabled; "
            "sampler=use sampler only; loss=use class-weighted CE only; both=use both."
        ),
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=0.5,
        help=(
            "Power transform applied to normalized class weights when class-weight-mode=balanced. "
            "Values in (0,1) soften aggressive weighting."
        ),
    )
    parser.add_argument(
        "--train-shift-max",
        type=int,
        default=0,
        help="Max random pixel shift for train-time patch augmentation (applied to all channels). 0 disables.",
    )
    parser.add_argument(
        "--train-shift-prob",
        type=float,
        default=0.0,
        help="Probability of applying random train-time shift augmentation to each train patch.",
    )
    parser.add_argument(
        "--train-rotate",
        action="store_true",
        help="Enable random 90-degree rotation augmentation (0/90/180/270) during training.",
    )
    parser.add_argument(
        "--train-flip",
        action="store_true",
        help="Enable random horizontal/vertical flip augmentation during training.",
    )
    parser.add_argument(
        "--center-gaussian-sigma",
        type=float,
        default=0.0,
        help="Apply Gaussian center mask to patches (sigma in pixels). "
             "Attenuates peripheral skeleton to focus CNN on the target junction. 0=disabled.",
    )
    parser.add_argument(
        "--mask-other-junctions",
        type=int,
        default=0,
        help="Mask out non-central junctions in patches with a disk of this radius (pixels). "
             "Removes competing junction structure while preserving branches. 0=disabled.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="Loss function: ce (cross-entropy) or focal (focal loss, focuses on hard examples).",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for focal loss. Higher values focus more on hard examples.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="shallow",
        choices=["shallow", "deeper", "deeper_v2", "flatten_v2", "spp_v2", "attn_v2"],
        help=(
            "CNN architecture: shallow (3 conv, RF~22px, 94K params), "
            "deeper (5 conv, RF~54px, ~538K params, dropout), "
            "deeper_v2 (6 conv+GAP, RF~86px, ~833K params), "
            "flatten_v2 (6 conv+flatten, spatial-preserving), "
            "spp_v2 (6 conv+SPP, multi-scale spatial), or "
            "attn_v2 (6 conv+attention pooling, learned spatial weighting)."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability before classifier head (deeper arch only).",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize weights before training (transfer fine-tuning).",
    )
    parser.add_argument(
        "--init-reset-head",
        action="store_true",
        help="When --init-model is provided, ignore classifier head weights and reinitialize the head.",
    )
    parser.add_argument(
        "--freeze-blocks",
        type=int,
        default=0,
        help="Freeze the first N conv blocks (1-5 for deeper, 1-6 for deeper_v2). "
             "Frozen blocks have requires_grad=False. Head is never frozen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.patch_size != 96 and args.arch in ("flatten_v2", "spp_v2"):
        raise ValueError(
            f"Architecture {args.arch} has a fixed spatial head and requires --patch-size 96. "
            f"Use a GAP architecture (deeper, gap_v2, attn_v2, shallow_gap) for other sizes."
        )

    data_root = args.data_root if args.data_root is not None else args.manifest.parent
    device = choose_device()

    print()
    print("=" * 62)
    print(f"  CNN Training  |  task: {args.task}  |  arch: {args.arch}")
    print("=" * 62)

    rows = read_manifest(args.manifest)

    def remap_label(label: str) -> str | None:
        t = str(args.task).strip().lower()
        raw = str(label).strip().upper()
        if t == "multiclass":
            return raw
        if t == "gate":
            return "N" if raw == "N" else "P"
        if t == "type":
            return None if raw == "N" else raw
        raise ValueError(f"Unsupported task: {args.task}")

    mapped_labels = [m for m in (remap_label(str(r.get("label", ""))) for r in rows) if m is not None]
    if not mapped_labels:
        raise RuntimeError(f"No labels available after task mapping for task={args.task}")
    label_to_idx, idx_to_label = build_label_maps(mapped_labels)

    def parse_weight(row: Dict[str, str], label: str) -> float:
        w = 1.0
        if args.sample_weight_mode == "manifest":
            raw = str(row.get(args.sample_weight_col, "")).strip()
            if raw != "":
                try:
                    parsed = float(raw)
                    if np.isfinite(parsed) and parsed > 0:
                        w = float(parsed)
                except Exception:
                    pass
        x_w = float(args.x_class_weight)
        if label == "X" and x_w > 0 and x_w != 1.0:
            w *= x_w
        n_w = float(args.n_class_weight)
        if label == "N" and n_w > 0 and n_w != 1.0:
            w *= n_w
        t_w = float(args.t_class_weight)
        if label == "T" and t_w > 0 and t_w != 1.0:
            w *= t_w
        return float(w)

    def make_items(split: str) -> List[Tuple[Path, int, Path | None, float]]:
        items: List[Tuple[Path, int, Path | None, float]] = []
        for row in rows:
            if row["split"] != split:
                continue
            mapped = remap_label(row["label"])
            if mapped is None:
                continue
            context_raw = str(row.get(args.context_column, "")).strip()
            context_path = (data_root / context_raw) if context_raw else None
            items.append(
                (
                    data_root / row["relpath"],
                    label_to_idx[mapped],
                    context_path,
                    parse_weight(row, row["label"]),
                )
            )
        return items

    train_items = make_items(str(args.train_split))
    val_items = make_items(str(args.val_split))
    test_items = make_items(str(args.test_split))

    if not train_items or not val_items or not test_items:
        raise RuntimeError("One or more splits are empty. Check manifest and split labels.")

    # Class distribution per split
    train_label_counts = Counter(idx_to_label[item[1]] for item in train_items)
    val_label_counts = Counter(idx_to_label[item[1]] for item in val_items)

    print(f"  Device:    {device}")
    print(f"  Manifest:  {args.manifest.name}")
    print(f"  Classes:   {' | '.join(idx_to_label)}")
    print(f"  Splits:    train={len(train_items):,}  val={len(val_items):,}  test={len(test_items):,}")
    dist_parts = [f"{lbl}={train_label_counts.get(lbl, 0):,}" for lbl in idx_to_label]
    print(f"  Train dist: {' | '.join(dist_parts)}")
    print("-" * 62)

    _gauss_sigma = float(args.center_gaussian_sigma) if hasattr(args, 'center_gaussian_sigma') else 0.0
    _mask_junc = int(args.mask_other_junctions) if hasattr(args, 'mask_other_junctions') else 0
    train_ds = PatchDataset(
        train_items,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        train_shift_max=int(args.train_shift_max),
        train_shift_prob=float(args.train_shift_prob),
        train_rotate=bool(args.train_rotate),
        train_flip=bool(args.train_flip),
        center_gaussian_sigma=_gauss_sigma,
        mask_other_junctions=_mask_junc,
    )
    use_pin_memory = device.type == "cuda"  # MPS does not support pin_memory
    nw = max(0, int(args.num_workers))
    train_loader_kwargs = {
        "dataset": train_ds,
        "batch_size": args.batch_size,
        "num_workers": nw,
        "pin_memory": use_pin_memory,
        "persistent_workers": nw > 0,
    }
    per_sample_weights = np.asarray([float(item[3]) for item in train_items], dtype=np.float32)
    has_sample_weights = bool(np.any(per_sample_weights != 1.0))
    wants_class_weights = str(args.class_weight_mode).lower() == "balanced"
    strategy = str(args.imbalance_strategy).strip().lower()
    if strategy == "auto":
        use_sampler = has_sample_weights and not wants_class_weights
        use_class_weight_loss = wants_class_weights
    elif strategy == "sampler":
        use_sampler = True  # Always use sampler with class-balanced weights
        use_class_weight_loss = False
    elif strategy == "loss":
        use_sampler = False
        use_class_weight_loss = wants_class_weights
    elif strategy == "both":
        use_sampler = True
        use_class_weight_loss = wants_class_weights
    else:
        raise ValueError(f"Unsupported imbalance strategy: {args.imbalance_strategy}")

    if use_sampler:
        # Compute class-inverse-frequency weights so each class is sampled equally,
        # then multiply by any per-sample weights (manifest weights, x_class_weight).
        train_label_indices = [int(item[1]) for item in train_items]
        label_counts = Counter(train_label_indices)
        num_classes_present = len(label_counts)
        total_samples = len(train_label_indices)
        class_inv_freq = {
            cls_idx: float(total_samples) / (float(num_classes_present) * float(count))
            for cls_idx, count in label_counts.items()
        }
        sampler_weights = np.asarray(
            [class_inv_freq[li] for li in train_label_indices], dtype=np.float32
        )
        sampler_weights *= per_sample_weights
        sampler = WeightedRandomSampler(
            weights=sampler_weights.tolist(), num_samples=len(sampler_weights), replacement=True
        )
        train_loader_kwargs["sampler"] = sampler
        train_loader_kwargs["shuffle"] = False
        class_balance_info = {idx_to_label[k]: f"{v:.2f}" for k, v in class_inv_freq.items()}
        bal_parts = [f"{k}={v}" for k, v in class_balance_info.items()]
        print(f"  Sampler:   {' | '.join(bal_parts)}")
    else:
        train_loader_kwargs["shuffle"] = True
    print(
        f"  Strategy:  {strategy}  sampler={'ON' if use_sampler else 'off'}  "
        f"class_wt={'ON' if use_class_weight_loss else 'off'}"
    )

    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(
        PatchDataset(
            val_items,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            train_shift_max=0,
            train_shift_prob=0.0,
            center_gaussian_sigma=_gauss_sigma,
        mask_other_junctions=_mask_junc,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin_memory,
        persistent_workers=nw > 0,
    )
    test_loader = DataLoader(
        PatchDataset(
            test_items,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            train_shift_max=0,
            train_shift_prob=0.0,
            center_gaussian_sigma=_gauss_sigma,
        mask_other_junctions=_mask_junc,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_pin_memory,
        persistent_workers=nw > 0,
    )

    arch = str(args.arch).strip().lower()
    _deeper_args = dict(num_classes=len(idx_to_label), in_channels=args.in_channels, dropout=float(args.dropout))
    if arch == "flatten_v2":
        model = DeeperCNN_Flatten_v2(**_deeper_args).to(device)
    elif arch == "spp_v2":
        model = DeeperCNN_SPP_v2(**_deeper_args).to(device)
    elif arch == "attn_v2":
        model = DeeperCNN_Attn_v2(**_deeper_args).to(device)
    elif arch == "deeper_v2":
        model = DeeperCNN_GAP_v2(**_deeper_args).to(device)
    elif arch == "deeper":
        model = DeeperCNN_GAP(**_deeper_args).to(device)
    else:
        model = ShallowCNN_GAP(
            num_classes=len(idx_to_label),
            c1=args.c1,
            c2=args.c2,
            c3=args.c3,
            in_channels=args.in_channels,
        ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model:     {arch}  ({n_params:,} params)  lr={args.lr}  wd={args.weight_decay}")
    if arch in ("deeper", "deeper_v2", "flatten_v2", "spp_v2", "attn_v2"):
        print(f"  Dropout:   {args.dropout}  label_smooth={args.label_smoothing}")
    aug_parts = []
    if args.train_shift_max > 0:
        aug_parts.append(f"shift±{args.train_shift_max}")
    if args.train_rotate:
        aug_parts.append("rot90")
    if args.train_flip:
        aug_parts.append("flip")
    if aug_parts:
        print(f"  Augment:   {' + '.join(aug_parts)}")
    init_model_path = args.init_model.resolve() if args.init_model is not None else None
    init_missing: List[str] = []
    init_unexpected: List[str] = []
    if init_model_path is not None:
        init_missing, init_unexpected = load_init_state(
            model=model,
            init_model_path=init_model_path,
            reset_head=bool(args.init_reset_head),
        )
        print(f"  Transfer:  {init_model_path.name}  reset_head={bool(args.init_reset_head)}")

    if int(args.freeze_blocks) > 0:
        freeze_n = int(args.freeze_blocks)
        frozen_count = 0
        for block_idx in range(1, freeze_n + 1):
            for attr in (f"conv{block_idx}", f"bn{block_idx}"):
                layer = getattr(model, attr, None)
                if layer is not None:
                    for p in layer.parameters():
                        p.requires_grad = False
                    frozen_count += sum(1 for p in layer.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Frozen:    blocks 1-{freeze_n} ({total - trainable:,} params frozen, {trainable:,} trainable)")

    class_weight_tensor: torch.Tensor | None = None
    class_weight_map: Dict[str, float] | None = None
    if use_class_weight_loss:
        train_label_names = [idx_to_label[int(item[1])] for item in train_items]
        c = Counter(train_label_names)
        present = [lbl for lbl in idx_to_label if c.get(lbl, 0) > 0]
        if present:
            total = float(len(train_label_names))
            raw = {lbl: total / (float(len(present)) * float(c[lbl])) for lbl in present}
            mean_raw = sum(raw.values()) / float(len(raw))
            class_weight_map = {lbl: float(raw[lbl] / mean_raw) for lbl in raw}
            power = max(0.0, float(args.class_weight_power))
            if power != 1.0:
                softened = {lbl: float(w ** power) for lbl, w in class_weight_map.items()}
                mean_soft = sum(softened.values()) / float(len(softened))
                class_weight_map = {lbl: float(softened[lbl] / mean_soft) for lbl in softened}
            class_weight_tensor = torch.ones(len(idx_to_label), dtype=torch.float32, device=device)
            for idx, lbl in enumerate(idx_to_label):
                class_weight_tensor[idx] = float(class_weight_map.get(lbl, 1.0))

    if str(args.loss).lower() == "focal":
        criterion = FocalLoss(
            weight=class_weight_tensor,
            gamma=float(args.focal_gamma),
            label_smoothing=float(args.label_smoothing),
        )
        print(f"  Loss:      focal (gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, label_smoothing=float(args.label_smoothing))
    # --- Build parameter groups (discriminative LR for fine-tuning) ---
    if float(args.discriminative_lr) > 0 and args.init_model is not None:
        factor = float(args.discriminative_lr)
        groups = []
        max_block = 6 if arch in ("deeper_v2", "attn_v2") else 5
        for block_idx in range(1, max_block + 1):
            block_params = []
            for attr in (f"conv{block_idx}", f"bn{block_idx}"):
                layer = getattr(model, attr, None)
                if layer is not None:
                    for p in layer.parameters():
                        if p.requires_grad:
                            block_params.append(p)
            if block_params:
                block_lr = args.lr * (factor ** (max_block - block_idx))
                groups.append({"params": block_params, "lr": block_lr})
        # Head + any remaining trainable params (attn_conv, drop, etc.) at full LR
        grouped_ids = {id(p) for g in groups for p in g["params"]}
        remaining = [p for p in model.parameters() if p.requires_grad and id(p) not in grouped_ids]
        if remaining:
            groups.append({"params": remaining, "lr": args.lr})
        params = groups
        print(f"  DiscrimLR: factor={factor}, {len(groups)} groups")
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    # --- Optimizer ---
    opt_name = str(args.optimizer).lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # --- Scheduler ---
    sched_name = str(args.scheduler).lower()
    if sched_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - int(args.warmup_epochs)), eta_min=1e-6,
        )
    elif sched_name == "cosine_warm":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6,
        )
    else:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=int(args.scheduler_patience),
        )

    if int(args.warmup_epochs) > 0 and sched_name != "plateau":
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=int(args.warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, main_scheduler],
            milestones=[int(args.warmup_epochs)],
        )
        sched_type_for_train = "sequential"
        print(f"  Schedule:  {sched_name} + {args.warmup_epochs}ep warmup")
    else:
        scheduler = main_scheduler
        sched_type_for_train = sched_name
    print(f"  Optimizer: {opt_name}  Scheduler: {sched_name}")

    best_state, best_epoch, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        scheduler_type=sched_type_for_train,
    )

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device, optimizer=None)

    print()
    print("-" * 62)
    print(f"  RESULTS")
    print(f"  Test loss: {test_loss:.4f}   Test acc: {test_acc:.4f}")
    print(f"  Best epoch: {best_epoch}   Best val_loss: {best_val_loss:.4f}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in best_state.items()},
        "config": {
            "arch": arch,
            "patch_size": int(args.patch_size),
            "in_channels": int(args.in_channels),
            "c1": int(args.c1) if arch == "shallow" else 0,
            "c2": int(args.c2) if arch == "shallow" else 0,
            "c3": int(args.c3) if arch == "shallow" else 0,
            "dropout": float(args.dropout) if arch in ("deeper", "deeper_v2", "flatten_v2", "spp_v2", "attn_v2") else 0.0,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "loss": f"cross_entropy_label_smoothing_{float(args.label_smoothing):.4f}",
            "sample_weight_mode": str(args.sample_weight_mode),
            "x_class_weight": float(args.x_class_weight),
            "n_class_weight": float(args.n_class_weight),
            "t_class_weight": float(args.t_class_weight),
            "class_weight_mode": str(args.class_weight_mode),
            "imbalance_strategy": str(args.imbalance_strategy),
            "class_weight_power": float(args.class_weight_power),
            "use_sampler": bool(use_sampler),
            "use_class_weight_loss": bool(use_class_weight_loss),
            "train_shift_max": int(args.train_shift_max),
            "train_shift_prob": float(args.train_shift_prob),
            "train_rotate": bool(args.train_rotate),
            "train_flip": bool(args.train_flip),
            "scheduler_patience": int(args.scheduler_patience),
            "loss_fn": str(args.loss),
            "focal_gamma": float(args.focal_gamma) if str(args.loss) == "focal" else 0.0,
            "task": str(args.task),
            "transfer_init_model": str(init_model_path) if init_model_path is not None else "",
            "transfer_init_reset_head": bool(args.init_reset_head),
            "transfer_init_missing_keys": [str(k) for k in init_missing],
            "transfer_init_unexpected_keys": [str(k) for k in init_unexpected],
        },
        "idx_to_label": idx_to_label,
        "label_to_idx": label_to_idx,
    }
    torch.save(payload, args.output_model)
    print(f"  Saved:     {args.output_model}")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()
