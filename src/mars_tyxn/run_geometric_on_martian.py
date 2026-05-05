#!/usr/bin/env python3
"""
Run the collinearity-based geometric classifier on martian skeletons and
evaluate against GT labels. Also includes collinearity override for ML
predictions (Workstream C1) and parameter sweep.

Usage:
    python run_geometric_on_martian.py \
        --skel-dir /path/to/skeletons \
        --labels-dir /path/to/labels

    # Collinearity override on existing ML predictions:
    python run_geometric_on_martian.py \
        --skel-dir /path/to/skeletons \
        --labels-dir /path/to/labels \
        --override-dir /path/to/predictions_runI_XGB \
        --override-threshold 35

    # Parameter sweep:
    python run_geometric_on_martian.py \
        --skel-dir /path/to/skeletons \
        --labels-dir /path/to/labels \
        --sweep
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.morphology import skeletonize as sk_skeletonize

# ---------------------------------------------------------------------------
# Reuse core functions from run_glyph_benchmark.py
# ---------------------------------------------------------------------------

def _crossing_number(binary: np.ndarray, cy: float, cx: float, radius: int) -> int:
    h, w = binary.shape
    n_points = max(32, int(2 * np.pi * radius * 2))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    vals = []
    for a in angles:
        py = int(round(cy + radius * np.sin(a)))
        px = int(round(cx + radius * np.cos(a)))
        if 0 <= py < h and 0 <= px < w:
            vals.append(binary[py, px])
        else:
            vals.append(0)
    return sum(1 for i in range(len(vals)) if vals[i] > 0 and vals[i - 1] == 0)


def _branch_angles_on_ring(binary: np.ndarray, cy: float, cx: float, radius: int) -> list[float]:
    h, w = binary.shape
    n_points = max(64, int(2 * np.pi * radius * 2))
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    vals = []
    for a in angles:
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


def _collinearity_deviation(branch_angles_rad: list[float]) -> tuple[float, float | None]:
    n = len(branch_angles_rad)
    if n < 3:
        return 90.0, None

    min_dev = 180.0
    for i in range(n):
        for j in range(i + 1, n):
            diff_deg = abs(np.degrees(branch_angles_rad[i] - branch_angles_rad[j])) % 360
            if diff_deg > 180:
                diff_deg = 360 - diff_deg
            dev = abs(diff_deg - 180)
            if dev < min_dev:
                min_dev = dev

    angles_deg = sorted(np.degrees(a) % 360 for a in branch_angles_rad)
    gaps = [(angles_deg[(i + 1) % n] - angles_deg[i]) % 360 for i in range(n)]
    max_gap = max(gaps)

    return min_dev, max_gap


def _nms_by_score(detections: list, nms_dist: int = 20) -> list:
    if not detections:
        return []
    coords = np.array([[d["y"], d["x"]] for d in detections])
    scores = np.array([d["score"] for d in detections])
    order = np.argsort(-scores)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(detections[i])
        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        for j in range(len(detections)):
            if j != i and dists[j] < nms_dist:
                suppressed.add(j)
    return keep


# ---------------------------------------------------------------------------
# Geometric classifier (collinearity + branch-tracing, no template_matcher)
# ---------------------------------------------------------------------------

def geometric_classify(
    skel_u8: np.ndarray,
    nms_distance: int = 20,
    cn_radii: tuple = (8, 12),
    min_crossings: int = 3,
    collinearity_radii: tuple = (16, 20, 12),
    collinearity_threshold: float = 41.0,
    t_min_largest_angle: float = 150.0,
    branch_trace_len: int = 8,
) -> list[dict]:
    """Detect and classify junctions using collinearity + branch-tracing only."""
    # Import junction_geometry for branch-tracing
    try:
        from mars_tyxn.junction_geometry import analyze_local_junction
        HAS_JG = True
    except ImportError:
        HAS_JG = False

    binary = (skel_u8 > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbr_count = ndimage.convolve(binary, kernel, mode="constant", cval=0)

    # Step 1: degree>=3 clusters
    branch_mask = (binary > 0) & (nbr_count >= 3)
    labeled, n_clusters = ndimage.label(branch_mask, structure=np.ones((3, 3)))

    # Step 2: crossing-number filter
    raw_dets = []
    for c in range(1, n_clusters + 1):
        cluster_pixels = np.argwhere(labeled == c)
        cy, cx = cluster_pixels.mean(axis=0)
        cns = [_crossing_number(binary, cy, cx, r) for r in cn_radii]
        max_cn = max(cns)
        if max_cn < min_crossings:
            continue
        raw_dets.append({
            "x": float(cx), "y": float(cy),
            "cn": max_cn, "score": max_cn * len(cluster_pixels),
        })

    # Step 3: NMS
    detections = _nms_by_score(raw_dets, nms_dist=nms_distance)

    # Step 4: skeletonize for classification
    thinned = sk_skeletonize(binary > 0).astype(np.uint8)

    results = []
    for det in detections:
        ix = int(round(det["x"]))
        iy = int(round(det["y"]))

        # Signal 1: Collinearity at larger radii
        col_dev = None
        col_n_arms = 0
        for r in collinearity_radii:
            br_angles = _branch_angles_on_ring(thinned, iy, ix, r)
            if len(br_angles) == 3:
                col_dev, _ = _collinearity_deviation(br_angles)
                col_n_arms = 3
                break
            elif len(br_angles) >= 4 and col_n_arms == 0:
                col_n_arms = len(br_angles)

        # Signal 2: Branch-tracing geometry
        jg_branches = 0
        jg_max_gap = None
        if HAS_JG:
            jg = analyze_local_junction(
                binary=thinned, anchor_x=ix, anchor_y=iy,
                trace_len=branch_trace_len,
            )
            jg_branches = jg["branch_count"]
            jg_max_gap = jg["max_gap_deg"]

        # Classification decision (collinearity + branch-tracing path)
        if col_n_arms >= 4 or jg_branches >= 4:
            det_type = "X"
        elif jg_branches == 3 and jg_max_gap is not None:
            if jg_max_gap >= t_min_largest_angle:
                det_type = "T"
            elif (col_dev is not None and col_n_arms == 3
                  and col_dev <= collinearity_threshold):
                det_type = "T"
            else:
                det_type = "Y"
        elif col_n_arms == 3 and col_dev is not None:
            det_type = "T" if col_dev <= collinearity_threshold else "Y"
        else:
            det_type = "Y"

        results.append({"x": det["x"], "y": det["y"], "type": det_type})

    return results


# ---------------------------------------------------------------------------
# Collinearity override for ML predictions (Workstream C1)
# ---------------------------------------------------------------------------

def collinearity_override(
    predictions: list[dict],
    skel_u8: np.ndarray,
    threshold: float = 35.0,
    radii: tuple = (16, 20, 25),
    confidence_gate: float | None = None,
) -> tuple[list[dict], int]:
    """For each Y prediction, compute collinearity. If < threshold, flip to T.

    If confidence_gate is set, only flip when the ML model's P(T) >= confidence_gate.
    This uses the ML model's own uncertainty as an additional gate — junctions where
    the model was already somewhat suspicious of T are better override candidates.
    Predictions must have a 'probs' dict with class probabilities for this to work.
    """
    thinned = sk_skeletonize((skel_u8 > 0).astype(np.uint8) > 0).astype(np.uint8)
    corrected = []
    flips = 0
    for pred in predictions:
        p = dict(pred)
        if p.get("probs"):
            p["probs"] = dict(p["probs"])
        if p["type"] == "Y":
            # Confidence gate: skip if ML is very confident about Y
            if confidence_gate is not None:
                p_t = float(p.get("probs", {}).get("T", 0.0))
                if p_t < confidence_gate:
                    corrected.append(p)
                    continue

            ix = int(round(p["x"]))
            iy = int(round(p["y"]))
            for r in radii:
                br_angles = _branch_angles_on_ring(thinned, iy, ix, r)
                if len(br_angles) == 3:
                    col_dev, _ = _collinearity_deviation(br_angles)
                    if col_dev <= threshold:
                        p["type"] = "T"
                        flips += 1
                    break
        corrected.append(p)
    return corrected, flips


# ---------------------------------------------------------------------------
# T-Likeness Score
# ---------------------------------------------------------------------------

def compute_t_likeness(
    skel_u8: np.ndarray,
    x: int,
    y: int,
    radii: tuple = (12, 16, 20, 25),
) -> float:
    """Compute T-likeness score in [0, 1] from multi-radius collinearity.

    Based on the T→Y node twisting continuum (Silver et al., 2025):
      1.0 = unambiguous T (collinear through-pair, col_dev ≈ 0°)
      0.0 = unambiguous Y (symmetric 120° splay, col_dev ≈ 60°)
      0.3-0.7 = transition zone (partially twisted junction)

    Uses the median of collinearity measurements at multiple radii for
    robustness against local skeleton noise.
    """
    binary = (skel_u8 > 0).astype(np.uint8)
    scores = []
    for r in radii:
        br_angles = _branch_angles_on_ring(binary, float(y), float(x), r)
        if len(br_angles) == 3:
            col_dev, _ = _collinearity_deviation(br_angles)
            score = max(0.0, min(1.0, 1.0 - col_dev / 60.0))
            scores.append(score)
    if not scores:
        return 0.5  # undetermined
    return float(np.median(scores))


def compute_t_likeness_for_predictions(
    predictions: list[dict],
    skel_u8: np.ndarray,
    snap_radius: int = 5,
) -> list[dict]:
    """Add t_likeness score to each T/Y prediction."""
    S = np.where(skel_u8 > 0, 255, 0).astype(np.uint8)
    out = []
    for pred in predictions:
        p = dict(pred)
        if p.get("probs"):
            p["probs"] = dict(p["probs"])
        if p["type"] in ("T", "Y"):
            ix, iy = int(round(p["x"])), int(round(p["y"]))
            snapped = _snap_to_skeleton(S, ix, iy, snap_radius)
            if snapped:
                p["t_likeness"] = compute_t_likeness(S, snapped[0], snapped[1])
            else:
                p["t_likeness"] = 0.5
        else:
            p["t_likeness"] = None
        out.append(p)
    return out


# ---------------------------------------------------------------------------
# Fracture continuity override (Workstream C2 revised)
# ---------------------------------------------------------------------------

def _trace_branch_coords(
    S: np.ndarray, x0: int, y0: int, sx: int, sy: int, max_pixels: int = 60,
    cluster_exit_dist: int = 6,
) -> list[tuple[int, int]]:
    """Trace a branch pixel-by-pixel and return coordinate list.

    Near the starting junction (within cluster_exit_dist pixels), we tolerate
    high-degree pixels since they're part of the same junction cluster.
    We only stop at degree>=3 once we're clearly past the cluster.
    """
    coords = [(sx, sy)]
    prev = (x0, y0)
    cur = (sx, sy)
    visited = {prev, cur}
    while len(coords) < max_pixels:
        nbrs = _fg_neighbors(S, cur[0], cur[1])
        dist = ((cur[0] - x0) ** 2 + (cur[1] - y0) ** 2) ** 0.5

        # Endpoint: always stop
        if len(nbrs) <= 1:
            break

        # High-degree pixel far from start: another junction — stop
        if len(nbrs) >= 3 and dist > cluster_exit_dist:
            break

        # Find unvisited neighbors to continue
        forward = [p for p in nbrs if p not in visited]
        if not forward:
            break
        if len(forward) > 1:
            if dist > cluster_exit_dist:
                break  # branching far from start = another junction
            # Within cluster: pick neighbor furthest from start center
            forward.sort(key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2, reverse=True)

        nxt = forward[0]
        prev = cur
        cur = nxt
        visited.add(cur)
        coords.append(cur)
    return coords


def _fit_branch_line(coords: list[tuple[int, int]]) -> tuple[float | None, float, int]:
    """Fit a line to branch coords via PCA. Returns (angle_deg, r_squared, length)."""
    n = len(coords)
    if n < 3:
        return None, 0.0, n
    pts = np.array(coords, dtype=np.float64)
    centroid = pts.mean(axis=0)
    pts_c = pts - centroid
    try:
        _, S, Vt = np.linalg.svd(pts_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, 0.0, n
    direction = Vt[0]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0]))) % 360
    total_var = float(np.sum(S ** 2))
    r_sq = float(S[0] ** 2 / total_var) if total_var > 1e-12 else 0.0
    return angle, r_sq, n


def _pair_continuity_score(
    angle_i: float, r2_i: float, len_i: int,
    angle_j: float, r2_j: float, len_j: int,
) -> float:
    """Score how much two branches look like a single through-fracture.

    High score = the two branches are nearly collinear (180 apart),
    both are straight (high R²), and both have sufficient trace length.
    """
    diff = abs(angle_i - angle_j)
    if diff > 180:
        diff = 360 - diff
    deviation = abs(diff - 180)  # 0 = perfectly collinear

    dev_score = max(0.0, 1.0 - deviation / 90.0)  # 1.0 at 0°, 0.0 at 90°
    min_r2 = min(r2_i, r2_j)
    len_score = min(1.0, min(len_i, len_j) / 20.0)

    return dev_score * min_r2 * len_score


def fracture_continuity_override(
    predictions: list[dict],
    skel_u8: np.ndarray,
    score_threshold: float = 0.55,
    dominance_ratio: float = 1.3,
    trace_pixels: int = 60,
    snap_radius: int = 5,
) -> tuple[list[dict], int]:
    """For each Y prediction, compute fracture continuity. If one branch pair
    has a strongly dominant continuity score, override Y→T.

    score_threshold: minimum best-pair score to consider T
    dominance_ratio: best pair must exceed 2nd-best by this ratio
    trace_pixels: how far to trace each branch
    """
    S = np.where(skel_u8 > 0, 255, 0).astype(np.uint8)

    corrected = []
    flips = 0
    for pred in predictions:
        p = dict(pred)
        if p.get("probs"):
            p["probs"] = dict(p["probs"])

        if p["type"] != "Y":
            corrected.append(p)
            continue

        ix = int(round(p["x"]))
        iy = int(round(p["y"]))

        snapped = _snap_to_skeleton(S, ix, iy, snap_radius)
        if snapped is None:
            corrected.append(p)
            continue
        sx, sy = snapped

        starts = _fg_neighbors(S, sx, sy)
        if len(starts) < 3:
            corrected.append(p)
            continue

        # Trace each branch and fit a line
        branch_fits = []
        for bx, by in starts:
            coords = _trace_branch_coords(S, sx, sy, bx, by, trace_pixels)
            angle, r2, length = _fit_branch_line(coords)
            if angle is not None:
                branch_fits.append((angle, r2, length))

        if len(branch_fits) < 3:
            corrected.append(p)
            continue

        # Compute continuity score for each pair
        pair_scores = []
        for i in range(len(branch_fits)):
            for j in range(i + 1, len(branch_fits)):
                a_i, r2_i, l_i = branch_fits[i]
                a_j, r2_j, l_j = branch_fits[j]
                score = _pair_continuity_score(a_i, r2_i, l_i, a_j, r2_j, l_j)
                pair_scores.append(score)

        pair_scores.sort(reverse=True)
        best = pair_scores[0]
        second = pair_scores[1] if len(pair_scores) > 1 else 0.0

        # T if: best pair is strong AND dominates the 2nd-best pair
        if best >= score_threshold and (second < 1e-9 or best / second >= dominance_ratio):
            p["type"] = "T"
            flips += 1

        corrected.append(p)

    return corrected, flips


# ---------------------------------------------------------------------------
# Network context correction (Workstream C2 — dead-end approach, for sparse networks)
# ---------------------------------------------------------------------------

def _fg_neighbors(S: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    """8-connected foreground neighbors of pixel (x, y)."""
    h, w = S.shape[:2]
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and S[ny, nx] > 0:
                out.append((nx, ny))
    return out


def _snap_to_skeleton(S: np.ndarray, x: int, y: int, radius: int = 5) -> tuple[int, int] | None:
    """Snap (x, y) to the nearest foreground pixel within radius."""
    h, w = S.shape[:2]
    if 0 <= x < w and 0 <= y < h and S[y, x] > 0:
        return x, y
    best, best_d2 = None, float("inf")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and S[ny, nx] > 0:
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best, best_d2 = (nx, ny), d2
    return best


def _trace_branch_terminus(
    S: np.ndarray, x0: int, y0: int, sx: int, sy: int, max_trace: int = 200,
) -> tuple[str, int]:
    """Trace a branch from junction (x0,y0) via first step (sx,sy).

    Returns (terminus, length) where terminus is:
      'junction' - reached another high-degree pixel (degree >= 3)
      'endpoint' - reached a dead-end (degree <= 1)
      'loop'     - traced back to a visited pixel
      'long'     - exceeded max_trace without reaching terminus
    """
    prev = (x0, y0)
    cur = (sx, sy)
    visited = {prev, cur}
    length = 1
    while True:
        nbrs = _fg_neighbors(S, cur[0], cur[1])
        degree = len(nbrs)
        if degree >= 3:
            return "junction", length
        if degree <= 1:
            return "endpoint", length
        forward = [p for p in nbrs if p != prev]
        if not forward:
            return "endpoint", length
        if len(forward) > 1:
            return "junction", length
        nxt = forward[0]
        if nxt in visited:
            return "loop", length
        prev = cur
        cur = nxt
        visited.add(cur)
        length += 1
        if length >= max_trace:
            return "long", length


def _bridge_gaps(S: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Minimal gap bridging: connect nearby endpoints."""
    S_bin = np.where(S > 0, 255, 0).astype(np.uint8)
    fg = (S_bin > 0).astype(np.uint8)
    h, w = fg.shape
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    N_p = cv2.filter2D(fg, ddepth=cv2.CV_16S, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    endpoints = np.argwhere((fg == 1) & (N_p == 1))
    fg_ref = fg.copy()
    S_out = S_bin.copy()
    r = int(max_gap)
    r2 = float(r * r)
    for y_i, x_i in endpoints:
        immediate = set(_fg_neighbors(fg_ref, int(x_i), int(y_i)))
        if not immediate:
            continue
        best_xy, best_d2 = None, float("inf")
        for yy in range(max(0, y_i - r), min(h, y_i + r + 1)):
            for xx in range(max(0, x_i - r), min(w, x_i + r + 1)):
                if fg_ref[yy, xx] == 0 or (xx == x_i and yy == y_i):
                    continue
                if (xx, yy) in immediate:
                    continue
                d2 = (xx - x_i) ** 2 + (yy - y_i) ** 2
                if d2 <= r2 and d2 < best_d2:
                    best_d2 = d2
                    best_xy = (xx, yy)
        if best_xy is not None:
            cv2.line(S_out, (int(x_i), int(y_i)), best_xy, 255, 1, cv2.LINE_8)
    return np.where(S_out > 0, 255, 0).astype(np.uint8)


def network_context_correction(
    predictions: list[dict],
    skeleton_u8: np.ndarray,
    min_dead_end_length: int = 15,
    max_trace_length: int = 200,
    bridge_gaps: bool = True,
    snap_radius: int = 5,
) -> tuple[list[dict], int, int]:
    """Post-classification correction using skeleton network context.

    For each junction, traces branches to determine if they reach another
    junction ('through') or terminate ('dead-end'). Reclassifies:
      - Y with 1 dead-end + 2 through → T
      - T with 0 dead-ends (all through) → Y

    Returns (corrected_predictions, y_to_t_flips, t_to_y_flips).
    """
    S = np.where(skeleton_u8 > 0, 255, 0).astype(np.uint8)
    if bridge_gaps:
        S = _bridge_gaps(S, max_gap=3)

    corrected = []
    y_to_t = 0
    t_to_y = 0

    for pred in predictions:
        p = dict(pred)
        if p.get("probs"):
            p["probs"] = dict(p["probs"])

        if p["type"] not in ("T", "Y"):
            corrected.append(p)
            continue

        ix = int(round(p["x"]))
        iy = int(round(p["y"]))

        # Snap to nearest skeleton pixel
        snapped = _snap_to_skeleton(S, ix, iy, snap_radius)
        if snapped is None:
            corrected.append(p)
            continue
        sx, sy = snapped

        # Get branch starts
        starts = _fg_neighbors(S, sx, sy)
        if len(starts) < 2:
            corrected.append(p)
            continue

        # Trace each branch
        branch_info = []
        for bx, by in starts:
            terminus, length = _trace_branch_terminus(S, sx, sy, bx, by, max_trace_length)
            branch_info.append((terminus, length))

        # Count branch types
        dead_ends = sum(
            1 for t, l in branch_info
            if t == "endpoint" and l >= min_dead_end_length
        )
        through = sum(
            1 for t, l in branch_info
            if t in ("junction", "long")
        )
        short_spurs = sum(
            1 for t, l in branch_info
            if t == "endpoint" and l < min_dead_end_length
        )

        # Reclassification rules (conservative)
        if p["type"] == "Y" and dead_ends == 1 and through >= 2:
            p["type"] = "T"
            y_to_t += 1
        elif p["type"] == "T" and dead_ends == 0 and through >= 3:
            p["type"] = "Y"
            t_to_y += 1

        corrected.append(p)

    return corrected, y_to_t, t_to_y


# ---------------------------------------------------------------------------
# Evaluation (reused from Cell 7 logic)
# ---------------------------------------------------------------------------

MATCH_RADIUS = 15.0
CLASSES = ["T", "X", "Y"]


def parse_darwin_labels(json_path: Path) -> list[tuple[str, float, float]]:
    with open(json_path) as f:
        data = json.load(f)
    gt_points = []
    for ann in data.get("annotations", []):
        cls = ann.get("name", "").upper()
        if cls not in CLASSES:
            continue
        bb = ann.get("bounding_box", {})
        if bb:
            cx = bb["x"] + bb["w"] / 2
            cy = bb["y"] + bb["h"] / 2
            gt_points.append((cls, cx, cy))
    return gt_points


def hungarian_match(gt_points, pred_points, radius=15.0):
    if not gt_points or not pred_points:
        return [], list(range(len(gt_points))), list(range(len(pred_points)))
    n_gt, n_pred = len(gt_points), len(pred_points)
    cost = np.full((n_gt, n_pred), 1e6)
    for i, (_, gx, gy) in enumerate(gt_points):
        for j, (_, px, py) in enumerate(pred_points):
            d = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)
            if d <= radius:
                cost[i, j] = d
    row_ind, col_ind = linear_sum_assignment(cost)
    matches, matched_gt, matched_pred = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e5:
            matches.append((r, c, cost[r, c]))
            matched_gt.add(r)
            matched_pred.add(c)
    return (matches,
            [i for i in range(n_gt) if i not in matched_gt],
            [j for j in range(n_pred) if j not in matched_pred])


def evaluate(pred_dir_or_list, label_dir: Path, model_name: str, from_list=False):
    """Evaluate predictions against GT. Returns (f1_scores, macro_f1)."""
    confusion = defaultdict(lambda: defaultdict(int))
    total_gt = defaultdict(int)
    total_pred = defaultdict(int)
    total_tp = defaultdict(int)

    print(f'\n{"=" * 60}')
    print(f"  {model_name}")
    print(f'{"=" * 60}')

    for label_path in sorted(label_dir.glob("*.json")):
        stem = label_path.stem
        gt = parse_darwin_labels(label_path)

        if from_list:
            # pred_dir_or_list is a dict: {stem: [{"type","x","y"}, ...]}
            preds_raw = pred_dir_or_list.get(stem, [])
            preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
        else:
            pred_path = Path(pred_dir_or_list) / f"{stem}_predictions.json"
            if not pred_path.exists():
                print(f"  SKIP {stem}: no prediction file")
                continue
            with open(pred_path) as f:
                data = json.load(f)
            preds = [(d["type"], d["x"], d["y"]) for d in data.get("detections", [])]

        matches, unmatched_gt, unmatched_pred = hungarian_match(gt, preds, MATCH_RADIUS)

        for gt_idx, pred_idx, _ in matches:
            gt_cls = gt[gt_idx][0]
            pred_cls = preds[pred_idx][0]
            confusion[gt_cls][pred_cls] += 1
            if gt_cls == pred_cls:
                total_tp[gt_cls] += 1
        for cls, _, _ in gt:
            total_gt[cls] += 1
        for cls, _, _ in preds:
            total_pred[cls] += 1

        print(f"  {stem[:2]}: GT={len(gt)} Pred={len(preds)} Matched={len(matches)} "
              f"Missed_GT={len(unmatched_gt)} FP={len(unmatched_pred)}")

    print(f"\n  Per-class metrics (radius={MATCH_RADIUS}px):")
    print(f'  {"Class":>5} {"GT":>5} {"Pred":>5} {"TP":>5} {"Prec":>7} {"Recall":>7} {"F1":>7}')
    print(f'  {"-"*5:>5} {"-"*5:>5} {"-"*5:>5} {"-"*5:>5} {"-"*7:>7} {"-"*7:>7} {"-"*7:>7}')

    f1_scores = {}
    for cls in CLASSES:
        gt_n = total_gt.get(cls, 0)
        pred_n = total_pred.get(cls, 0)
        tp = total_tp.get(cls, 0)
        prec = tp / pred_n if pred_n > 0 else 0.0
        rec = tp / gt_n if gt_n > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores[cls] = f1
        print(f"  {cls:>5} {gt_n:>5} {pred_n:>5} {tp:>5} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

    valid = [f1_scores[c] for c in CLASSES if total_gt.get(c, 0) > 0]
    macro_f1 = float(np.mean(valid)) if valid else 0.0
    print(f"\n  Macro F1: {macro_f1:.3f}")

    if confusion:
        print(f"\n  Confusion (rows=GT, cols=Pred):")
        header = "       " + "".join(f"{c:>6}" for c in CLASSES)
        print(f"  {header}")
        for gt_cls in CLASSES:
            row = f"  {gt_cls:>5} "
            for pred_cls in CLASSES:
                row += f"{confusion[gt_cls][pred_cls]:>6}"
            print(row)

    return f1_scores, macro_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skel-dir", type=Path, required=True,
                    help="Directory with *_skel.png skeleton images")
    p.add_argument("--labels-dir", type=Path, required=True,
                    help="Directory with GT JSON labels")
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Output directory for prediction JSONs")
    p.add_argument("--sweep", action="store_true",
                    help="Run parameter sweep")
    p.add_argument("--override-dir", type=Path, default=None,
                    help="Directory with ML prediction JSONs to apply collinearity override")
    p.add_argument("--override-threshold", type=float, default=35.0,
                    help="Collinearity threshold for Y->T override")
    p.add_argument("--confidence-gate", type=float, default=None,
                    help="Min P(T) from ML model to allow override (requires probs in JSON)")
    p.add_argument("--continuity-dir", type=Path, default=None,
                    help="Directory with ML prediction JSONs to apply fracture continuity override")
    p.add_argument("--continuity-threshold", type=float, default=0.55,
                    help="Min continuity score to override Y->T")
    p.add_argument("--continuity-dominance", type=float, default=1.3,
                    help="Best pair must exceed 2nd-best by this ratio")
    p.add_argument("--trace-pixels", type=int, default=60,
                    help="How far to trace each branch for continuity fitting")
    p.add_argument("--network-context", type=Path, default=None,
                    help="Directory with ML prediction JSONs to apply network context correction")
    p.add_argument("--min-dead-end", type=int, default=15,
                    help="Min branch length to count as dead-end (not spur)")
    p.add_argument("--max-trace", type=int, default=200,
                    help="Max pixels to trace before treating branch as through")
    p.add_argument("--no-bridge", action="store_true",
                    help="Disable gap bridging before network context analysis")
    p.add_argument("--t-likeness", type=Path, default=None,
                    help="Directory with ML prediction JSONs to compute T-likeness scores")
    p.add_argument("--collinearity-threshold", type=float, default=41.0)
    p.add_argument("--collinearity-radii", type=str, default="16,20,12")
    p.add_argument("--t-min-largest-angle", type=float, default=150.0)
    p.add_argument("--nms-distance", type=int, default=20)
    p.add_argument("--branch-trace-len", type=int, default=8)
    return p.parse_args()


def run_geometric(args) -> dict[str, list[dict]]:
    """Run geometric classifier on all skeletons. Returns {stem: [detections]}."""
    col_radii = tuple(int(x) for x in args.collinearity_radii.split(","))
    all_results = {}

    for skel_path in sorted(args.skel_dir.glob("*_skel.png")):
        skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
        if skel is None:
            continue
        detections = geometric_classify(
            skel,
            nms_distance=args.nms_distance,
            collinearity_radii=col_radii,
            collinearity_threshold=args.collinearity_threshold,
            t_min_largest_angle=args.t_min_largest_angle,
            branch_trace_len=args.branch_trace_len,
        )
        all_results[skel_path.stem] = detections

        # Save prediction JSON if output_dir specified
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_path = args.output_dir / f"{skel_path.stem}_predictions.json"
            payload = {"image_path": skel_path.name, "detections": detections}
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2)

    return all_results


def run_override(args, confidence_gate: float | None = None) -> dict[str, list[dict]]:
    """Apply collinearity override to existing ML predictions."""
    all_results = {}
    total_flips = 0
    _warned_no_probs = False

    for pred_path in sorted(args.override_dir.glob("*_predictions.json")):
        stem = pred_path.stem.replace("_predictions", "")
        skel_path = args.skel_dir / f"{stem}.png"
        if not skel_path.exists():
            continue

        with open(pred_path) as f:
            data = json.load(f)
        predictions = data.get("detections", [])

        # Warn once if confidence gating requested but predictions lack probs
        if confidence_gate is not None and not _warned_no_probs:
            has_probs = any(d.get("probs") for d in predictions)
            if not has_probs and predictions:
                print("  WARNING: Predictions lack 'probs' field — confidence gate disabled.")
                print("  Re-run Cell 6 with updated predict_benchmark.py to get probabilities.")
                _warned_no_probs = True

        skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
        corrected, flips = collinearity_override(
            predictions, skel,
            threshold=args.override_threshold,
            radii=(16, 20, 25),
            confidence_gate=confidence_gate,
        )
        total_flips += flips
        all_results[stem] = corrected

    gate_str = f", P(T)>={confidence_gate}" if confidence_gate else ""
    print(f"\nCollinearity override: {total_flips} Y->T flips "
          f"(col<={args.override_threshold}{gate_str})")
    return all_results


def run_sweep(args):
    """Parameter sweep for geometric classifier."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP")
    print("=" * 70)

    best_macro = -1.0
    best_params = {}
    results_table = []

    col_thresholds = [30, 35, 38, 41, 45, 50]
    col_radii_options = ["16,20,12", "12,16,20", "20,25,16"]
    t_angles = [140, 145, 150, 155]

    for col_t in col_thresholds:
        for col_r in col_radii_options:
            for t_ang in t_angles:
                args.collinearity_threshold = col_t
                args.collinearity_radii = col_r
                args.t_min_largest_angle = t_ang
                args.output_dir = None  # don't save during sweep

                all_results = run_geometric(args)

                # Quick evaluate without printing
                total_gt_cls = defaultdict(int)
                total_tp_cls = defaultdict(int)
                total_pred_cls = defaultdict(int)
                for label_path in sorted(args.labels_dir.glob("*.json")):
                    stem = label_path.stem
                    gt = parse_darwin_labels(label_path)
                    preds_raw = all_results.get(stem, [])
                    preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
                    matches, _, _ = hungarian_match(gt, preds, MATCH_RADIUS)
                    for gt_idx, pred_idx, _ in matches:
                        gt_cls = gt[gt_idx][0]
                        pred_cls = preds[pred_idx][0]
                        if gt_cls == pred_cls:
                            total_tp_cls[gt_cls] += 1
                    for cls, _, _ in gt:
                        total_gt_cls[cls] += 1
                    for cls, _, _ in preds:
                        total_pred_cls[cls] += 1

                f1s = {}
                for cls in CLASSES:
                    gt_n = total_gt_cls.get(cls, 0)
                    pred_n = total_pred_cls.get(cls, 0)
                    tp = total_tp_cls.get(cls, 0)
                    p = tp / pred_n if pred_n > 0 else 0.0
                    r = tp / gt_n if gt_n > 0 else 0.0
                    f1s[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                valid = [f1s[c] for c in CLASSES if total_gt_cls.get(c, 0) > 0]
                macro = float(np.mean(valid)) if valid else 0.0

                results_table.append({
                    "col_t": col_t, "col_r": col_r, "t_ang": t_ang,
                    "T_F1": f1s["T"], "Y_F1": f1s["Y"], "X_F1": f1s["X"],
                    "macro": macro,
                })
                if macro > best_macro:
                    best_macro = macro
                    best_params = {"col_t": col_t, "col_r": col_r, "t_ang": t_ang}

    # Print results table
    print(f"\n  {'col_t':>5} {'col_radii':>12} {'t_ang':>5} {'T F1':>7} {'Y F1':>7} {'Macro':>7}")
    print(f"  {'-'*5:>5} {'-'*12:>12} {'-'*5:>5} {'-'*7:>7} {'-'*7:>7} {'-'*7:>7}")
    for row in sorted(results_table, key=lambda r: -r["macro"]):
        print(f"  {row['col_t']:>5} {row['col_r']:>12} {row['t_ang']:>5} "
              f"{row['T_F1']:>7.3f} {row['Y_F1']:>7.3f} {row['macro']:>7.3f}")

    print(f"\n  Best: col_t={best_params['col_t']}, col_r={best_params['col_r']}, "
          f"t_ang={best_params['t_ang']} -> macro F1={best_macro:.3f}")


def main():
    args = parse_args()

    # Add src directory to path for junction_geometry import
    src_dir = Path(__file__).parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    if args.sweep:
        run_sweep(args)
        return

    if args.override_dir:
        # Collinearity override mode — run with specified params
        all_results = run_override(args, confidence_gate=args.confidence_gate)
        gate_str = f", P(T)>={args.confidence_gate}" if args.confidence_gate else ""
        evaluate(all_results, args.labels_dir,
                 f"ML + Override (col<={args.override_threshold}{gate_str})",
                 from_list=True)

        # 2D sweep: collinearity threshold x confidence gate
        print("\n\n  Override sweep (collinearity threshold x confidence gate):")
        print(f"  {'col_t':>5} {'P(T)>=':>7} {'T F1':>7} {'Y F1':>7} {'Macro':>7}")
        print(f"  {'-'*5:>5} {'-'*7:>7} {'-'*7:>7} {'-'*7:>7} {'-'*7:>7}")

        sweep_results = []
        col_thresholds = [25, 30, 35, 40, 45]
        conf_gates = [None, 0.05, 0.10, 0.15, 0.20, 0.30]

        for col_t in col_thresholds:
            for cg in conf_gates:
                args.override_threshold = col_t
                results = run_override(args, confidence_gate=cg)
                # Quick eval
                total_gt_cls = defaultdict(int)
                total_tp_cls = defaultdict(int)
                total_pred_cls = defaultdict(int)
                for label_path in sorted(args.labels_dir.glob("*.json")):
                    stem = label_path.stem
                    gt = parse_darwin_labels(label_path)
                    preds_raw = results.get(stem, [])
                    preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
                    matches, _, _ = hungarian_match(gt, preds, MATCH_RADIUS)
                    for gt_idx, pred_idx, _ in matches:
                        if gt[gt_idx][0] == preds[pred_idx][0]:
                            total_tp_cls[gt[gt_idx][0]] += 1
                    for cls, _, _ in gt:
                        total_gt_cls[cls] += 1
                    for cls, _, _ in preds:
                        total_pred_cls[cls] += 1
                f1s = {}
                for cls in CLASSES:
                    gn = total_gt_cls.get(cls, 0)
                    pn = total_pred_cls.get(cls, 0)
                    tp = total_tp_cls.get(cls, 0)
                    p = tp / pn if pn > 0 else 0.0
                    r = tp / gn if gn > 0 else 0.0
                    f1s[cls] = 2*p*r/(p+r) if (p+r) > 0 else 0.0
                valid = [f1s[c] for c in CLASSES if total_gt_cls.get(c, 0) > 0]
                macro = float(np.mean(valid)) if valid else 0.0
                cg_str = f"{cg:.2f}" if cg is not None else " none"
                sweep_results.append((col_t, cg, f1s, macro))

        # Sort by macro F1 descending and print
        sweep_results.sort(key=lambda r: -r[3])
        for col_t, cg, f1s, macro in sweep_results:
            cg_str = f"{cg:.2f}" if cg is not None else " none"
            print(f"  {col_t:>5} {cg_str:>7} {f1s['T']:>7.3f} {f1s['Y']:>7.3f} {macro:>7.3f}")
    elif args.t_likeness:
        # T-likeness scoring mode
        all_scores = []
        all_results = {}

        for pred_path in sorted(args.t_likeness.glob("*_predictions.json")):
            stem = pred_path.stem.replace("_predictions", "")
            skel_path = args.skel_dir / f"{stem}.png"
            if not skel_path.exists():
                continue
            with open(pred_path) as f:
                data = json.load(f)
            predictions = data.get("detections", [])
            skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
            scored = compute_t_likeness_for_predictions(predictions, skel)
            all_results[stem] = scored
            for d in scored:
                if d.get("t_likeness") is not None:
                    all_scores.append(d["t_likeness"])

        # Report distribution
        scores_arr = np.array(all_scores)
        print(f"\nT-Likeness Score Distribution ({len(scores_arr)} junctions):")
        print(f"  Mean:   {scores_arr.mean():.3f}")
        print(f"  Median: {np.median(scores_arr):.3f}")
        print(f"  Std:    {scores_arr.std():.3f}")

        # Histogram buckets
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(scores_arr, bins=bins)
        print(f"\n  {'Range':>10} {'Count':>6} {'Interp':>15}")
        labels_interp = ['Clear Y', 'Clear Y', 'Clear Y', 'Trans', 'Trans',
                         'Trans', 'Trans', 'Clear T', 'Clear T', 'Clear T']
        for i in range(len(hist)):
            lo, hi = bins[i], bins[i + 1]
            print(f"  {lo:.1f}-{min(hi,1.0):.1f}    {hist[i]:>6}   {labels_interp[i]:>15}")

        # Soft proportion estimate
        t_prop = float(scores_arr.mean())
        print(f"\n  Soft T proportion: {t_prop:.3f} (= mean T-likeness)")
        print(f"  Soft Y proportion: {1 - t_prop:.3f}")

        # Evaluate with hard threshold sweep
        print(f"\n  Hard classification at various T-likeness thresholds:")
        print(f"  {'Thresh':>6} {'T_pred':>6} {'T F1':>7} {'Y F1':>7} {'Macro':>7}")
        print(f"  {'-'*6:>6} {'-'*6:>6} {'-'*7:>7} {'-'*7:>7} {'-'*7:>7}")

        for t_thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            # Reclassify based on T-likeness threshold
            mod_results = {}
            t_count = 0
            for stem, scored in all_results.items():
                reclassed = []
                for d in scored:
                    p = dict(d)
                    if p.get("t_likeness") is not None:
                        p["type"] = "T" if p["t_likeness"] >= t_thresh else "Y"
                    reclassed.append(p)
                    if p["type"] == "T":
                        t_count += 1
                mod_results[stem] = reclassed

            # Quick eval
            total_gt_cls = defaultdict(int)
            total_tp_cls = defaultdict(int)
            total_pred_cls = defaultdict(int)
            for label_path in sorted(args.labels_dir.glob("*.json")):
                stem = label_path.stem
                gt = parse_darwin_labels(label_path)
                preds_raw = mod_results.get(stem, [])
                preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
                matches, _, _ = hungarian_match(gt, preds, MATCH_RADIUS)
                for gt_idx, pred_idx, _ in matches:
                    if gt[gt_idx][0] == preds[pred_idx][0]:
                        total_tp_cls[gt[gt_idx][0]] += 1
                for cls, _, _ in gt:
                    total_gt_cls[cls] += 1
                for cls, _, _ in preds:
                    total_pred_cls[cls] += 1
            f1s = {}
            for cls in CLASSES:
                gn = total_gt_cls.get(cls, 0)
                pn = total_pred_cls.get(cls, 0)
                tp = total_tp_cls.get(cls, 0)
                p_val = tp / pn if pn > 0 else 0.0
                r_val = tp / gn if gn > 0 else 0.0
                f1s[cls] = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val) > 0 else 0.0
            valid = [f1s[c] for c in CLASSES if total_gt_cls.get(c, 0) > 0]
            macro = float(np.mean(valid)) if valid else 0.0
            print(f"  {t_thresh:>6.2f} {t_count:>6} {f1s['T']:>7.3f} {f1s['Y']:>7.3f} {macro:>7.3f}")

    elif args.continuity_dir:
        # Fracture continuity override mode
        all_results = {}
        total_flips = 0
        for pred_path in sorted(args.continuity_dir.glob("*_predictions.json")):
            stem = pred_path.stem.replace("_predictions", "")
            skel_path = args.skel_dir / f"{stem}.png"
            if not skel_path.exists():
                continue
            with open(pred_path) as f:
                data = json.load(f)
            predictions = data.get("detections", [])
            skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
            corrected, flips = fracture_continuity_override(
                predictions, skel,
                score_threshold=args.continuity_threshold,
                dominance_ratio=args.continuity_dominance,
                trace_pixels=args.trace_pixels,
            )
            total_flips += flips
            all_results[stem] = corrected

        print(f"\nFracture continuity: {total_flips} Y->T flips "
              f"(score>={args.continuity_threshold}, dom>={args.continuity_dominance}, "
              f"trace={args.trace_pixels}px)")
        evaluate(all_results, args.labels_dir,
                 f"ML + Continuity Override (s>={args.continuity_threshold})",
                 from_list=True)

        # Sweep: score_threshold x dominance_ratio
        print("\n\n  Continuity override sweep:")
        print(f"  {'score':>5} {'domin':>5} {'trace':>5} {'flips':>5} "
              f"{'T F1':>7} {'Y F1':>7} {'Macro':>7}")
        print(f"  {'-'*5:>5} {'-'*5:>5} {'-'*5:>5} {'-'*5:>5} "
              f"{'-'*7:>7} {'-'*7:>7} {'-'*7:>7}")

        sweep_rows = []
        for s_thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
            for dom in [1.0, 1.2, 1.3, 1.5, 2.0]:
                for tpx in [40, 60]:
                    results = {}
                    sw_flips = 0
                    for pred_path in sorted(args.continuity_dir.glob("*_predictions.json")):
                        stem = pred_path.stem.replace("_predictions", "")
                        skel_path = args.skel_dir / f"{stem}.png"
                        if not skel_path.exists():
                            continue
                        with open(pred_path) as f:
                            data = json.load(f)
                        preds_list = data.get("detections", [])
                        skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
                        corr, flips = fracture_continuity_override(
                            preds_list, skel,
                            score_threshold=s_thresh,
                            dominance_ratio=dom,
                            trace_pixels=tpx,
                        )
                        sw_flips += flips
                        results[stem] = corr

                    # Quick eval
                    total_gt_cls = defaultdict(int)
                    total_tp_cls = defaultdict(int)
                    total_pred_cls = defaultdict(int)
                    for label_path in sorted(args.labels_dir.glob("*.json")):
                        stem = label_path.stem
                        gt = parse_darwin_labels(label_path)
                        preds_raw = results.get(stem, [])
                        preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
                        matches, _, _ = hungarian_match(gt, preds, MATCH_RADIUS)
                        for gt_idx, pred_idx, _ in matches:
                            if gt[gt_idx][0] == preds[pred_idx][0]:
                                total_tp_cls[gt[gt_idx][0]] += 1
                        for cls, _, _ in gt:
                            total_gt_cls[cls] += 1
                        for cls, _, _ in preds:
                            total_pred_cls[cls] += 1
                    f1s = {}
                    for cls in CLASSES:
                        gn = total_gt_cls.get(cls, 0)
                        pn = total_pred_cls.get(cls, 0)
                        tp = total_tp_cls.get(cls, 0)
                        p_val = tp / pn if pn > 0 else 0.0
                        r_val = tp / gn if gn > 0 else 0.0
                        f1s[cls] = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val) > 0 else 0.0
                    valid = [f1s[c] for c in CLASSES if total_gt_cls.get(c, 0) > 0]
                    macro = float(np.mean(valid)) if valid else 0.0
                    sweep_rows.append((s_thresh, dom, tpx, sw_flips, f1s, macro))

        sweep_rows.sort(key=lambda r: -r[5])
        for s_thresh, dom, tpx, sw_flips, f1s, macro in sweep_rows:
            print(f"  {s_thresh:>5.2f} {dom:>5.1f} {tpx:>5} {sw_flips:>5} "
                  f"{f1s['T']:>7.3f} {f1s['Y']:>7.3f} {macro:>7.3f}")

    elif args.network_context:
        # Network context correction mode
        all_results = {}
        total_y2t = 0
        total_t2y = 0

        for pred_path in sorted(args.network_context.glob("*_predictions.json")):
            stem = pred_path.stem.replace("_predictions", "")
            skel_path = args.skel_dir / f"{stem}.png"
            if not skel_path.exists():
                continue
            with open(pred_path) as f:
                data = json.load(f)
            predictions = data.get("detections", [])
            skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
            corrected, y2t, t2y = network_context_correction(
                predictions, skel,
                min_dead_end_length=args.min_dead_end,
                max_trace_length=args.max_trace,
                bridge_gaps=not args.no_bridge,
            )
            total_y2t += y2t
            total_t2y += t2y
            all_results[stem] = corrected

        print(f"\nNetwork context: {total_y2t} Y->T, {total_t2y} T->Y "
              f"(min_dead_end={args.min_dead_end}, max_trace={args.max_trace}, "
              f"bridge={'off' if args.no_bridge else 'on'})")
        evaluate(all_results, args.labels_dir,
                 f"ML + Network Context (dead_end>={args.min_dead_end})",
                 from_list=True)

        # Sweep min_dead_end_length
        print("\n\n  Network context sweep (min_dead_end_length):")
        print(f"  {'min_de':>6} {'bridge':>6} {'Y->T':>5} {'T->Y':>5} "
              f"{'T F1':>7} {'Y F1':>7} {'Macro':>7}")
        print(f"  {'-'*6:>6} {'-'*6:>6} {'-'*5:>5} {'-'*5:>5} "
              f"{'-'*7:>7} {'-'*7:>7} {'-'*7:>7}")

        for min_de in [5, 10, 15, 20, 30, 50]:
            for do_bridge in [True, False]:
                results = {}
                sw_y2t, sw_t2y = 0, 0
                for pred_path in sorted(args.network_context.glob("*_predictions.json")):
                    stem = pred_path.stem.replace("_predictions", "")
                    skel_path = args.skel_dir / f"{stem}.png"
                    if not skel_path.exists():
                        continue
                    with open(pred_path) as f:
                        data = json.load(f)
                    preds_list = data.get("detections", [])
                    skel = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
                    corr, y2t, t2y = network_context_correction(
                        preds_list, skel,
                        min_dead_end_length=min_de,
                        max_trace_length=args.max_trace,
                        bridge_gaps=do_bridge,
                    )
                    sw_y2t += y2t
                    sw_t2y += t2y
                    results[stem] = corr

                # Quick eval
                total_gt_cls = defaultdict(int)
                total_tp_cls = defaultdict(int)
                total_pred_cls = defaultdict(int)
                for label_path in sorted(args.labels_dir.glob("*.json")):
                    stem = label_path.stem
                    gt = parse_darwin_labels(label_path)
                    preds_raw = results.get(stem, [])
                    preds = [(d["type"], d["x"], d["y"]) for d in preds_raw]
                    matches, _, _ = hungarian_match(gt, preds, MATCH_RADIUS)
                    for gt_idx, pred_idx, _ in matches:
                        if gt[gt_idx][0] == preds[pred_idx][0]:
                            total_tp_cls[gt[gt_idx][0]] += 1
                    for cls, _, _ in gt:
                        total_gt_cls[cls] += 1
                    for cls, _, _ in preds:
                        total_pred_cls[cls] += 1
                f1s = {}
                for cls in CLASSES:
                    gn = total_gt_cls.get(cls, 0)
                    pn = total_pred_cls.get(cls, 0)
                    tp = total_tp_cls.get(cls, 0)
                    p_val = tp / pn if pn > 0 else 0.0
                    r_val = tp / gn if gn > 0 else 0.0
                    f1s[cls] = 2*p_val*r_val/(p_val+r_val) if (p_val+r_val) > 0 else 0.0
                valid = [f1s[c] for c in CLASSES if total_gt_cls.get(c, 0) > 0]
                macro = float(np.mean(valid)) if valid else 0.0
                br_str = "  on" if do_bridge else " off"
                print(f"  {min_de:>6} {br_str:>6} {sw_y2t:>5} {sw_t2y:>5} "
                      f"{f1s['T']:>7.3f} {f1s['Y']:>7.3f} {macro:>7.3f}")

    else:
        # Geometric classifier mode
        all_results = run_geometric(args)
        evaluate(all_results, args.labels_dir,
                 "Geometric (collinearity + branch-tracing)",
                 from_list=True)


if __name__ == "__main__":
    main()
