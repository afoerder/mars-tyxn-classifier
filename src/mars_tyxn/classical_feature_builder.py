#!/usr/bin/env python3
"""
Shared input feature assembly for classical TYXN models.

Supports three regimes:
- image_only: HOG over patch pixels only (legacy behavior)
- geom_only: geometric feature vector only
- image_plus_geom: HOG(patch) concatenated with geometric features
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mars_tyxn.hog_transformer import HOGTransformer
from mars_tyxn.junction_geometry import compute_patch_geometry_details

FEATURE_REGIMES = ("image_only", "geom_only", "image_plus_geom")
GEOMETRY_FEATURE_NAMES = [
    "geometry_branch_count",
    "geometry_premerge_branch_count",
    "geometry_min_gap_deg",
    "geometry_mid_gap_deg",
    "geometry_max_gap_deg",
    "geometry_gap_std_deg",
    "geometry_angle_0_deg",
    "geometry_angle_1_deg",
    "geometry_angle_2_deg",
    "geometry_length_ratio_21",
    "geometry_length_ratio_31",
    "geometry_length_ratio_32",
    "geometry_length_cv",
    "geometry_t_gap_rmse",
    "geometry_y_gap_rmse",
    "geometry_t_likeness",
    "geometry_y_likeness",
    "geometry_perp_error_deg",
    "geometry_axis_symmetry_deg",
    "geometry_collinearity_r12",
    "geometry_collinearity_r16",
    "geometry_collinearity_r20",
    "geometry_collinearity_r25",
    "geometry_collinearity_r40",
    "geometry_collinearity_r60",
    "geometry_collinearity_r80",
    "geometry_collinearity_n_arms",
    "geometry_col_n_strong",
    "geometry_col_consistency",
    "geometry_col_min_large",
    "geometry_collinear_branch_lr",
]


def normalize_feature_regime(value: Any) -> str:
    regime = str(value).strip().lower()
    if regime not in FEATURE_REGIMES:
        raise ValueError(f"Unsupported feature regime: {value}. Expected one of {FEATURE_REGIMES}")
    return regime


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    s = str(value).strip()
    if s == "":
        return float(default)
    try:
        v = float(s)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _rmse3(values: Sequence[float], target: Sequence[float]) -> float:
    v = np.asarray(list(values), dtype=np.float32)
    t = np.asarray(list(target), dtype=np.float32)
    if v.size != 3 or t.size != 3:
        return 180.0
    return float(np.sqrt(np.mean((v - t) ** 2)))


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return float(min(d, 360.0 - d))


def _preferred_anchor_from_row(row: Dict[str, Any] | None) -> tuple[float, float] | None:
    if row is None:
        return None
    x = _safe_float(row.get("local_x"), default=np.nan)
    y = _safe_float(row.get("local_y"), default=np.nan)
    if np.isfinite(x) and np.isfinite(y):
        return float(x), float(y)
    return None


def _branch_angles_on_ring(binary: np.ndarray, cy: float, cx: float, radius: int) -> list:
    """Find branch direction angles (radians) by scanning a circle at given radius."""
    h, w = binary.shape
    n_points = max(64, int(2 * np.pi * radius * 2))
    angles_arr = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    vals = []
    for a in angles_arr:
        py = int(round(cy + radius * np.sin(a)))
        px = int(round(cx + radius * np.cos(a)))
        if 0 <= py < h and 0 <= px < w:
            vals.append(binary[py, px])
        else:
            vals.append(0)
    runs = []
    in_run = False
    run_start = None
    for i in range(len(vals)):
        if vals[i] > 0 and not in_run:
            in_run = True
            run_start = i
        elif vals[i] == 0 and in_run:
            in_run = False
            runs.append((run_start, i - 1))
    if in_run:
        runs.append((run_start, len(vals) - 1))
    if len(runs) >= 2 and vals[0] > 0 and vals[-1] > 0:
        first = runs[0]
        last = runs[-1]
        runs = runs[1:-1]
        runs.append((last[0], first[1] + len(vals)))
    branch_angles = []
    for start, end in runs:
        center_idx = (start + end) / 2.0
        angle_rad = (center_idx / n_points) * 2 * np.pi
        branch_angles.append(angle_rad)
    return branch_angles


def _collinearity_deviation(branch_angles_rad: list) -> float:
    """For 3+ branch angles, find the pair closest to collinear (180 apart).
    Returns min deviation from 180 degrees. Lower = more T-like."""
    n = len(branch_angles_rad)
    if n < 3:
        return 90.0
    min_dev = 180.0
    for i in range(n):
        for j in range(i + 1, n):
            diff_deg = abs(np.degrees(branch_angles_rad[i] - branch_angles_rad[j])) % 360
            if diff_deg > 180:
                diff_deg = 360 - diff_deg
            dev = abs(diff_deg - 180)
            if dev < min_dev:
                min_dev = dev
    return float(min_dev)


def _compute_collinearity_features(
    patch_f32: np.ndarray,
    anchor_x: int | None = None,
    anchor_y: int | None = None,
) -> dict:
    """Compute collinearity deviation at multiple radii on a patch.
    Returns dict with collinearity_r12/r16/r20/r25 and n_arms."""
    bw = (patch_f32 > 0.5).astype(np.uint8)
    h, w = bw.shape
    if anchor_x is None:
        anchor_x = int(round((w - 1) / 2.0))
    if anchor_y is None:
        anchor_y = int(round((h - 1) / 2.0))

    result = {"r12": 90.0, "r16": 90.0, "r20": 90.0, "r25": 90.0,
              "r40": 90.0, "r60": 90.0, "r80": 90.0, "n_arms": 0}
    best_n_arms = 0
    for radius, key in [(12, "r12"), (16, "r16"), (20, "r20"), (25, "r25"),
                         (40, "r40"), (60, "r60"), (80, "r80")]:
        if radius >= min(anchor_x, anchor_y, w - 1 - anchor_x, h - 1 - anchor_y):
            continue  # radius exceeds patch bounds at anchor
        br_angles = _branch_angles_on_ring(bw, float(anchor_y), float(anchor_x), radius)
        n = len(br_angles)
        if n >= 3:
            result[key] = _collinearity_deviation(br_angles)
            if n > best_n_arms:
                best_n_arms = n
        elif n > best_n_arms:
            best_n_arms = n
    result["n_arms"] = best_n_arms
    return result


def extract_geometry_feature_vector(
    patch_f32: np.ndarray,
    row: Dict[str, Any] | None = None,
    trace_len: int = 6,
    merge_deg: float = 20.0,
    prefer_radius: float = 10.0,
    use_local_anchor: bool = True,
) -> np.ndarray:
    details = compute_patch_geometry_details(
        patch_f32=patch_f32,
        preferred_anchor=_preferred_anchor_from_row(row) if bool(use_local_anchor) else None,
        trace_len=int(trace_len),
        merge_deg=float(merge_deg),
        prefer_radius=float(prefer_radius),
    )

    branch_count = int(details.get("branch_count", 0) or 0)
    premerge_branch_count = int(details.get("premerge_branch_count", 0) or 0)
    angles = sorted([float(v) for v in details.get("angles_deg", []) if np.isfinite(v)])
    lengths = [float(v) for v in details.get("branch_lengths_px", []) if np.isfinite(v) and float(v) > 0.0]
    gaps = [float(v) for v in details.get("gaps_deg", []) if np.isfinite(v) and float(v) >= 0.0]

    if gaps:
        sg = sorted(gaps)
        min_gap = float(sg[0])
        mid_gap = float(np.median(np.asarray(sg, dtype=np.float32)))
        max_gap = float(sg[-1])
        gap_std = float(np.std(np.asarray(sg, dtype=np.float32)))
    else:
        min_gap = 0.0
        mid_gap = 0.0
        max_gap = 0.0
        gap_std = 0.0

    angles3 = list(angles[:3]) + [0.0] * max(0, 3 - len(angles[:3]))

    sl = sorted(lengths, reverse=True)
    l1 = float(sl[0]) if len(sl) >= 1 else 0.0
    l2 = float(sl[1]) if len(sl) >= 2 else 0.0
    l3 = float(sl[2]) if len(sl) >= 3 else 0.0
    eps = 1e-6
    len_ratio_21 = float(l2 / (l1 + eps)) if l1 > 0.0 else 0.0
    len_ratio_31 = float(l3 / (l1 + eps)) if l1 > 0.0 else 0.0
    len_ratio_32 = float(l3 / (l2 + eps)) if l2 > 0.0 else 0.0
    len_cv = float(np.std(np.asarray(sl[:3], dtype=np.float32)) / (np.mean(np.asarray(sl[:3], dtype=np.float32)) + eps)) if sl else 0.0

    gap_triplet = [min_gap, mid_gap, max_gap]
    t_gap_rmse = _rmse3(gap_triplet, [90.0, 90.0, 180.0])
    y120_gap_rmse = _rmse3(gap_triplet, [120.0, 120.0, 120.0])
    y100130_gap_rmse = _rmse3(gap_triplet, [100.0, 130.0, 130.0])
    y_gap_rmse = float(min(y120_gap_rmse, y100130_gap_rmse))
    t_likeness = float(np.exp(-t_gap_rmse / 25.0))
    y_likeness = float(np.exp(-y_gap_rmse / 25.0))

    perp_error = 90.0
    axis_symmetry = 180.0
    if len(angles) >= 3 and len(lengths) >= 3:
        pairs = list(zip(angles, lengths))
        dominant_idx = int(np.argmax(np.asarray([p[1] for p in pairs], dtype=np.float32)))
        dominant_angle = float(pairs[dominant_idx][0])
        other_pairs = [pairs[i] for i in range(len(pairs)) if i != dominant_idx]
        other_pairs.sort(key=lambda p: p[1], reverse=True)
        offsets = [_angle_diff_deg(p[0], dominant_angle) for p in other_pairs[:2]]
        if offsets:
            perp_error = float(np.mean([abs(o - 90.0) for o in offsets]))
        if len(offsets) >= 2:
            axis_symmetry = float(abs(offsets[0] - offsets[1]))

    # Collinearity features at multiple radii
    col = _compute_collinearity_features(patch_f32)

    # Derived collinearity features (T/Y discrimination)
    col_values = [col["r12"], col["r16"], col["r20"], col["r25"],
                  col["r40"], col["r60"], col["r80"]]
    col_n_strong = float(sum(1 for v in col_values if v < 30.0))
    col_non_default = [v for v in col_values if v < 89.0]
    col_consistency = float(np.std(col_non_default)) if len(col_non_default) >= 2 else 90.0
    col_large = [v for v in col_values[4:] if v < 89.0]  # r40, r60, r80
    col_min_large = float(min(col_large)) if col_large else 90.0

    # Collinear branch length ratio (density-invariant T/Y discriminator)
    collinear_lr = 0.0
    if len(angles) >= 2 and len(lengths) >= 2:
        for _i in range(len(angles)):
            for _j in range(_i + 1, len(angles)):
                _diff = abs(angles[_i] - angles[_j]) % 360.0
                if _diff > 180.0:
                    _diff = 360.0 - _diff
                _dev = abs(_diff - 180.0)
                if _dev < 30.0 and _i < len(lengths) and _j < len(lengths):
                    _li, _lj = lengths[_i], lengths[_j]
                    _ratio = min(_li, _lj) / max(_li, _lj) if max(_li, _lj) > 0 else 0.0
                    if _ratio > collinear_lr:
                        collinear_lr = _ratio

    values = np.asarray(
        [
            float(branch_count),
            float(premerge_branch_count),
            float(min_gap),
            float(mid_gap),
            float(max_gap),
            float(gap_std),
            float(angles3[0]),
            float(angles3[1]),
            float(angles3[2]),
            float(len_ratio_21),
            float(len_ratio_31),
            float(len_ratio_32),
            float(len_cv),
            float(t_gap_rmse),
            float(y_gap_rmse),
            float(t_likeness),
            float(y_likeness),
            float(perp_error),
            float(axis_symmetry),
            float(col["r12"]),
            float(col["r16"]),
            float(col["r20"]),
            float(col["r25"]),
            float(col["r40"]),
            float(col["r60"]),
            float(col["r80"]),
            float(col["n_arms"]),
            float(col_n_strong),
            float(col_consistency),
            float(col_min_large),
            float(collinear_lr),
        ],
        dtype=np.float32,
    )
    return values


def build_classical_input_matrix(
    patch_flat: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    feature_regime: str,
    patch_size: int = 96,
    geometry_trace_len: int = 6,
    geometry_merge_deg: float = 20.0,
    geometry_prefer_radius: float = 10.0,
    geometry_use_local_anchor: bool = True,
) -> np.ndarray:
    regime = normalize_feature_regime(feature_regime)
    X_patch = np.asarray(patch_flat, dtype=np.float32)
    if X_patch.ndim != 2:
        raise ValueError(f"Expected patch matrix with shape [N, D], got {X_patch.shape}")
    patch_dim = int(patch_size) * int(patch_size)
    if X_patch.shape[1] != patch_dim:
        raise ValueError(f"Expected patch dim {patch_dim}, got {X_patch.shape[1]}")

    if regime == "image_only":
        return X_patch

    if len(rows) != X_patch.shape[0]:
        raise ValueError(f"Rows/features length mismatch: rows={len(rows)} vs patches={X_patch.shape[0]}")

    geom_feats = np.zeros((X_patch.shape[0], len(GEOMETRY_FEATURE_NAMES)), dtype=np.float32)
    for i in range(X_patch.shape[0]):
        patch = X_patch[i].reshape((int(patch_size), int(patch_size)))
        geom_feats[i] = extract_geometry_feature_vector(
            patch_f32=patch,
            row=rows[i] if i < len(rows) else None,
            trace_len=int(geometry_trace_len),
            merge_deg=float(geometry_merge_deg),
            prefer_radius=float(geometry_prefer_radius),
            use_local_anchor=bool(geometry_use_local_anchor),
        )

    if regime == "geom_only":
        return geom_feats
    return np.concatenate([X_patch, geom_feats], axis=1)


def build_classical_input_vector(
    patch_f32: np.ndarray,
    row: Dict[str, Any],
    feature_regime: str,
    patch_size: int = 96,
    geometry_trace_len: int = 6,
    geometry_merge_deg: float = 20.0,
    geometry_prefer_radius: float = 10.0,
    geometry_use_local_anchor: bool = True,
) -> np.ndarray:
    regime = normalize_feature_regime(feature_regime)
    if patch_f32.shape != (int(patch_size), int(patch_size)):
        raise ValueError(f"Expected patch shape {(int(patch_size), int(patch_size))}, got {patch_f32.shape}")

    patch_vec = patch_f32.reshape(-1).astype(np.float32, copy=False)
    if regime == "image_only":
        return patch_vec

    geom_vec = extract_geometry_feature_vector(
        patch_f32=patch_f32,
        row=row,
        trace_len=int(geometry_trace_len),
        merge_deg=float(geometry_merge_deg),
        prefer_radius=float(geometry_prefer_radius),
        use_local_anchor=bool(geometry_use_local_anchor),
    )
    if regime == "geom_only":
        return geom_vec
    return np.concatenate([patch_vec, geom_vec], axis=0)


class PatchFeatureAssembler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_regime: str = "image_only", feature_set: str = "legacy", patch_size: int = 96):
        self.feature_regime = feature_regime
        self.feature_set = feature_set
        self.patch_size = patch_size
        self._hog: HOGTransformer | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "PatchFeatureAssembler":
        regime = normalize_feature_regime(self.feature_regime)
        Xv = np.asarray(X, dtype=np.float32)
        if Xv.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {Xv.shape}")
        patch_dim = int(self.patch_size) * int(self.patch_size)
        if regime == "image_only":
            if Xv.shape[1] != patch_dim:
                raise ValueError(f"image_only expects dim={patch_dim}, got {Xv.shape[1]}")
            self._hog = HOGTransformer(feature_set=self.feature_set)
            self._hog.fit(Xv, y)
        elif regime == "image_plus_geom":
            if Xv.shape[1] <= patch_dim:
                raise ValueError(f"image_plus_geom expects >{patch_dim} dims, got {Xv.shape[1]}")
            self._hog = HOGTransformer(feature_set=self.feature_set)
            self._hog.fit(Xv[:, :patch_dim], y)
        else:
            # geom_only has no HOG component.
            self._hog = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        regime = normalize_feature_regime(self.feature_regime)
        Xv = np.asarray(X, dtype=np.float32)
        if Xv.ndim != 2:
            raise ValueError(f"Expected 2D input, got shape {Xv.shape}")
        patch_dim = int(self.patch_size) * int(self.patch_size)

        if regime == "geom_only":
            return Xv.astype(np.float32, copy=False)

        if self._hog is None:
            raise RuntimeError("PatchFeatureAssembler must be fit before transform for image-based regimes.")

        if regime == "image_only":
            return self._hog.transform(Xv).astype(np.float32, copy=False)

        if Xv.shape[1] <= patch_dim:
            raise ValueError(f"image_plus_geom expects >{patch_dim} dims, got {Xv.shape[1]}")
        hog_feats = self._hog.transform(Xv[:, :patch_dim]).astype(np.float32, copy=False)
        geom_feats = Xv[:, patch_dim:].astype(np.float32, copy=False)
        return np.concatenate([hog_feats, geom_feats], axis=1)
