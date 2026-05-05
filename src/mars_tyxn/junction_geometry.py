#!/usr/bin/env python3
"""
Local junction geometry utilities shared by extraction-time bridge validation
and ensemble-time deterministic geometry gating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage


_NEIGHBOR_KERNEL = np.ones((3, 3), dtype=np.uint8)
_NEIGHBOR_KERNEL[1, 1] = 0


def degree_map(binary: np.ndarray) -> np.ndarray:
    bw = (binary > 0).astype(np.uint8)
    return ndimage.convolve(bw, _NEIGHBOR_KERNEL, mode="constant", cval=0)


def _neighbor_coords(x: int, y: int, binary: np.ndarray) -> List[Tuple[int, int]]:
    h, w = binary.shape
    out: List[Tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h and binary[ny, nx] == 1:
                out.append((nx, ny))
    return out


def _trace_branch_vector(
    binary: np.ndarray,
    degree: np.ndarray,
    anchor: Tuple[int, int],
    seed: Tuple[int, int],
    max_steps: int,
) -> Tuple[np.ndarray, float]:
    ax, ay = anchor
    prev = anchor
    cur = seed
    visited = {anchor, seed}

    for _ in range(max(0, max_steps - 1)):
        nbrs = [p for p in _neighbor_coords(cur[0], cur[1], binary) if p != prev]
        if not nbrs:
            break
        if len(nbrs) == 1:
            nxt = nbrs[0]
        else:
            d_cur = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=np.float32)
            best = nbrs[0]
            best_score = -2.0
            for qx, qy in nbrs:
                d_next = np.array([qx - cur[0], qy - cur[1]], dtype=np.float32)
                denom = float(np.linalg.norm(d_cur) * np.linalg.norm(d_next))
                score = float(np.dot(d_cur, d_next) / denom) if denom > 0 else -1.0
                if score > best_score:
                    best_score = score
                    best = (qx, qy)
            nxt = best

        if nxt in visited:
            break
        visited.add(nxt)
        prev, cur = cur, nxt

        # Keep branch tangents local: stop if we hit another branching center.
        if degree[cur[1], cur[0]] >= 3:
            break

    vec = np.array([cur[0] - ax, cur[1] - ay], dtype=np.float32)
    vlen = float(np.linalg.norm(vec))
    return vec, vlen


def _merge_direction_vectors(
    vectors: Sequence[Tuple[np.ndarray, float]],
    merge_deg: float = 20.0,
) -> List[Tuple[float, float]]:
    kept: List[Tuple[float, float]] = []
    ordered = sorted(vectors, key=lambda t: t[1], reverse=True)
    for vec, vlen in ordered:
        theta = float((np.degrees(np.arctan2(vec[1], vec[0])) + 360.0) % 360.0)
        duplicate = False
        for k_theta, _ in kept:
            d = abs(theta - k_theta)
            d = min(d, 360.0 - d)
            if d < merge_deg:
                duplicate = True
                break
        if not duplicate:
            kept.append((theta, float(vlen)))
    kept.sort(key=lambda t: t[0])
    return kept


def circular_gap_stats(angles_deg: Sequence[float]) -> Tuple[float | None, float | None, List[float]]:
    if len(angles_deg) < 2:
        return None, None, []

    vals = sorted(float(a) % 360.0 for a in angles_deg)
    wrapped = vals + [vals[0] + 360.0]
    gaps = [wrapped[i + 1] - wrapped[i] for i in range(len(vals))]
    return float(min(gaps)), float(max(gaps)), [float(g) for g in gaps]


def analyze_local_junction(
    binary: np.ndarray,
    anchor_x: int,
    anchor_y: int,
    trace_len: int = 6,
    merge_deg: float = 20.0,
) -> Dict[str, Any]:
    bw = (binary > 0).astype(np.uint8)
    h, w = bw.shape
    if not (0 <= anchor_x < w and 0 <= anchor_y < h):
        return {
            "branch_count": 0,
            "premerge_branch_count": 0,
            "angles_deg": [],
            "min_gap_deg": None,
            "max_gap_deg": None,
        }

    if bw[anchor_y, anchor_x] == 0:
        best: Tuple[int, int] | None = None
        best_d = float("inf")
        for yy in range(max(0, anchor_y - 2), min(h, anchor_y + 3)):
            for xx in range(max(0, anchor_x - 2), min(w, anchor_x + 3)):
                if bw[yy, xx] == 0:
                    continue
                d = float(np.hypot(xx - anchor_x, yy - anchor_y))
                if d < best_d:
                    best = (xx, yy)
                    best_d = d
        if best is None:
            return {
                "branch_count": 0,
                "premerge_branch_count": 0,
                "angles_deg": [],
                "min_gap_deg": None,
                "max_gap_deg": None,
            }
        anchor_x, anchor_y = best

    deg = degree_map(bw)
    seeds = _neighbor_coords(anchor_x, anchor_y, bw)

    vectors: List[Tuple[np.ndarray, float]] = []
    for sx, sy in seeds:
        vec, vlen = _trace_branch_vector(
            binary=bw,
            degree=deg,
            anchor=(anchor_x, anchor_y),
            seed=(sx, sy),
            max_steps=trace_len,
        )
        if vlen >= 1.0:
            vectors.append((vec, vlen))

    merged = _merge_direction_vectors(vectors, merge_deg=merge_deg)
    angles = [float(a) for a, _ in merged]
    lengths = [float(l) for _, l in merged]
    min_gap, max_gap, _ = circular_gap_stats(angles)
    gaps = []
    if len(angles) >= 2:
        vals = sorted(float(a) % 360.0 for a in angles)
        wrapped = vals + [vals[0] + 360.0]
        gaps = [float(wrapped[i + 1] - wrapped[i]) for i in range(len(vals))]
    return {
        "branch_count": int(len(angles)),
        "premerge_branch_count": int(len(vectors)),
        "angles_deg": angles,
        "branch_lengths_px": lengths,
        "gaps_deg": gaps,
        "min_gap_deg": min_gap,
        "max_gap_deg": max_gap,
    }


def classify_geometry_label(
    branch_count: int,
    min_gap_deg: float | None,
    max_gap_deg: float | None,
) -> str:
    if branch_count != 3 or min_gap_deg is None or max_gap_deg is None:
        return "Unknown"
    if 150.0 <= max_gap_deg <= 210.0 and min_gap_deg >= 35.0:
        return "T"
    if max_gap_deg >= 150.0 and min_gap_deg < 35.0:
        return "Y_arrowhead"
    return "Y_balanced"


def _pick_anchor(
    bw: np.ndarray,
    preferred_anchor: Tuple[float, float] | None,
    prefer_radius: float,
) -> Tuple[int, int] | None:
    deg = degree_map(bw)
    junction_mask = (bw == 1) & (deg >= 3)
    if not np.any(junction_mask):
        return None

    ys, xs = np.where(junction_mask)
    center_x = (bw.shape[1] - 1) / 2.0
    center_y = (bw.shape[0] - 1) / 2.0

    if preferred_anchor is not None:
        px, py = preferred_anchor
        d_pref = np.hypot(xs - float(px), ys - float(py))
        i_pref = int(np.argmin(d_pref))
        if float(d_pref[i_pref]) <= float(prefer_radius):
            return int(xs[i_pref]), int(ys[i_pref])

    i_center = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return int(xs[i_center]), int(ys[i_center])


def compute_patch_geometry(
    patch_f32: np.ndarray,
    preferred_anchor: Tuple[float, float] | None = None,
    trace_len: int = 6,
    merge_deg: float = 20.0,
    prefer_radius: float = 10.0,
) -> Dict[str, Any]:
    details = compute_patch_geometry_details(
        patch_f32=patch_f32,
        preferred_anchor=preferred_anchor,
        trace_len=trace_len,
        merge_deg=merge_deg,
        prefer_radius=prefer_radius,
    )
    return {
        "geometry_label": str(details["geometry_label"]),
        "branch_count": int(details["branch_count"]),
        "min_gap_deg": details["min_gap_deg"],
        "max_gap_deg": details["max_gap_deg"],
        "anchor_x": int(details["anchor_x"]),
        "anchor_y": int(details["anchor_y"]),
    }


def compute_patch_geometry_details(
    patch_f32: np.ndarray,
    preferred_anchor: Tuple[float, float] | None = None,
    trace_len: int = 6,
    merge_deg: float = 20.0,
    prefer_radius: float = 10.0,
) -> Dict[str, Any]:
    bw = (patch_f32 > 0.5).astype(np.uint8)
    h, w = bw.shape
    fallback_anchor = (
        int(round(float(preferred_anchor[0]))) if preferred_anchor is not None else int(round((w - 1) / 2.0)),
        int(round(float(preferred_anchor[1]))) if preferred_anchor is not None else int(round((h - 1) / 2.0)),
    )

    anchor = _pick_anchor(bw, preferred_anchor=preferred_anchor, prefer_radius=prefer_radius)
    if anchor is None:
        return {
            "geometry_label": "Unknown",
            "branch_count": 0,
            "premerge_branch_count": 0,
            "angles_deg": [],
            "branch_lengths_px": [],
            "gaps_deg": [],
            "min_gap_deg": None,
            "max_gap_deg": None,
            "anchor_x": int(np.clip(fallback_anchor[0], 0, w - 1)),
            "anchor_y": int(np.clip(fallback_anchor[1], 0, h - 1)),
        }

    ax, ay = anchor
    local = analyze_local_junction(
        binary=bw,
        anchor_x=ax,
        anchor_y=ay,
        trace_len=trace_len,
        merge_deg=merge_deg,
    )
    label = classify_geometry_label(
        branch_count=int(local["branch_count"]),
        min_gap_deg=local["min_gap_deg"],
        max_gap_deg=local["max_gap_deg"],
    )
    return {
        "geometry_label": label,
        "branch_count": int(local["branch_count"]),
        "premerge_branch_count": int(local["premerge_branch_count"]),
        "angles_deg": [float(a) for a in local["angles_deg"]],
        "branch_lengths_px": [float(v) for v in local["branch_lengths_px"]],
        "gaps_deg": [float(v) for v in local["gaps_deg"]],
        "min_gap_deg": local["min_gap_deg"],
        "max_gap_deg": local["max_gap_deg"],
        "anchor_x": int(ax),
        "anchor_y": int(ay),
    }
