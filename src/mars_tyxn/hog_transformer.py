#!/usr/bin/env python3
"""
Scikit-learn compatible feature transformer for 96x96 skeleton patches.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.ndimage import binary_dilation, convolve, label
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog


class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        image_size: int = 96,
        orientations: int = 9,
        pixels_per_cell: tuple[int, int] = (8, 8),
        cells_per_block: tuple[int, int] = (2, 2),
        block_norm: str = "L2-Hys",
        feature_set: str = "legacy",
        center_window: int = 40,
    ):
        self.image_size = image_size
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.feature_set = feature_set
        self.center_window = center_window

    def __setstate__(self, state):
        # Backward compatibility for older pickles created before feature_set/center_window existed.
        self.__dict__.update(state)
        if "feature_set" not in self.__dict__:
            self.feature_set = "legacy"
        if "center_window" not in self.__dict__:
            self.center_window = 40

    def fit(self, X, y=None):
        return self

    def _hog(self, img: np.ndarray) -> np.ndarray:
        return hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            feature_vector=True,
        ).astype(np.float32, copy=False)

    def _center_stats(self, skel: np.ndarray) -> np.ndarray:
        """
        Cheap topology cues that help disambiguate offset duplicate hypotheses.
        """
        size = int(skel.shape[0])
        win = int(max(8, min(self.center_window, size)))
        x0 = (size - win) // 2
        y0 = (size - win) // 2
        x1 = x0 + win
        y1 = y0 + win

        center = skel[y0:y1, x0:x1]
        center_fg = float(np.mean(center > 0))

        # 8-neighbor degree on full image.
        kernel = np.ones((3, 3), dtype=np.int16)
        kernel[1, 1] = 0
        deg = convolve((skel > 0).astype(np.int16), kernel, mode="constant", cval=0)
        deg_center = deg[y0:y1, x0:x1]
        skel_center = skel[y0:y1, x0:x1] > 0
        endpoint_frac = float(np.mean((deg_center == 1) & skel_center))
        branch_frac = float(np.mean((deg_center >= 3) & skel_center))

        cc_count = int(label(skel_center.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))[1])
        cc_norm = float(min(cc_count, 10) / 10.0)

        # Coarse angle-gap cues from center neighborhood.
        cx = (size - 1) * 0.5
        cy = (size - 1) * 0.5
        ys, xs = np.where(skel > 0)
        angles: List[float] = []
        for x, y in zip(xs.tolist(), ys.tolist()):
            dx = float(x) - cx
            dy = float(y) - cy
            r = math.hypot(dx, dy)
            if r < 4.0 or r > float(win) * 0.6:
                continue
            ang = math.degrees(math.atan2(dy, dx)) % 360.0
            angles.append(ang)

        if len(angles) >= 3:
            bins = sorted({int(round(a / 12.0)) for a in angles})
            if len(bins) >= 2:
                degs = [float(b * 12) for b in bins]
                gaps = []
                for i in range(len(degs)):
                    a0 = degs[i]
                    a1 = degs[(i + 1) % len(degs)]
                    gap = (a1 - a0) % 360.0
                    gaps.append(gap)
                min_gap = float(min(gaps))
                max_gap = float(max(gaps))
            else:
                min_gap = 0.0
                max_gap = 0.0
        else:
            min_gap = 0.0
            max_gap = 0.0

        return np.asarray(
            [
                center_fg,
                endpoint_frac,
                branch_frac,
                cc_norm,
                min_gap / 180.0,
                max_gap / 360.0,
            ],
            dtype=np.float32,
        )

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (N, D), got shape={X.shape}")

        n_samples, n_features = X.shape
        expected = self.image_size * self.image_size
        if n_features != expected:
            raise ValueError(
                f"Expected flattened patches with {expected} features, got {n_features}."
            )

        imgs = X.reshape(n_samples, self.image_size, self.image_size)
        feat_set = str(getattr(self, "feature_set", "legacy")).strip().lower()
        if feat_set == "legacy":
            feats = [self._hog(img) for img in imgs]
            return np.asarray(feats, dtype=np.float32)
        if feat_set != "hog_mask_center":
            raise ValueError(f"Unsupported feature_set={self.feature_set!r}. Use 'legacy' or 'hog_mask_center'.")

        out: List[np.ndarray] = []
        for img in imgs:
            skel = (img > 0.5).astype(np.float32)
            mask = binary_dilation(skel > 0.5, structure=np.ones((3, 3), dtype=bool)).astype(np.float32)
            f1 = self._hog(skel)
            f2 = self._hog(mask)
            stats = self._center_stats(skel)
            out.append(np.concatenate([f1, f2, stats], axis=0).astype(np.float32, copy=False))
        return np.asarray(out, dtype=np.float32)
