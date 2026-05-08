#!/usr/bin/env python3
"""
Orientation-aware virtual bridge proposal generation for skeleton junction recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation
from skimage.draw import line

from mars_tyxn.junction_geometry import analyze_local_junction, degree_map


@dataclass
class BridgeSearchConfig:
    gap_radii: Tuple[int, ...] = (3, 5, 7)
    endpoint_tangent_len: int = 6
    proposal_cone_deg: float = 55.0
    bridge_roi_size: int = 31
    bridge_max_side_contacts: int = 2
    border_margin: int = 8


@dataclass
class VirtualBridgeStats:
    endpoints: int = 0
    candidates_considered: int = 0
    rejected_cone: int = 0
    rejected_corridor: int = 0
    rejected_local_validation: int = 0
    accepted: int = 0


@dataclass
class _BridgeCandidate:
    endpoint: Tuple[int, int]
    target: Tuple[int, int]
    target_is_endpoint: bool
    target_component: int
    gap_len: float
    radius_used: int
    alignment_deg: float
    incident_pixels: set[Tuple[int, int]]


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


def _trace_incident_branch(
    binary: np.ndarray,
    degree: np.ndarray,
    endpoint: Tuple[int, int],
    max_len: int,
) -> List[Tuple[int, int]]:
    ex, ey = endpoint
    path: List[Tuple[int, int]] = [(ex, ey)]
    prev: Tuple[int, int] | None = None
    cur = (ex, ey)

    for _ in range(max(1, max_len * 2)):
        nbrs = _neighbor_coords(cur[0], cur[1], binary)
        if prev is not None:
            nbrs = [p for p in nbrs if p != prev]
        if not nbrs:
            break
        if len(nbrs) == 1:
            nxt = nbrs[0]
        else:
            if prev is None:
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

        path.append(nxt)
        prev, cur = cur, nxt

        if len(path) >= max_len + 1:
            break
        if degree[cur[1], cur[0]] != 2:
            break

    return path


def _estimate_tangent(endpoint: Tuple[int, int], path: List[Tuple[int, int]], k: int) -> np.ndarray | None:
    if len(path) < 2:
        return None
    idx = min(max(1, k), len(path) - 1)
    px, py = path[idx]
    ex, ey = endpoint
    vec = np.array([ex - px, ey - py], dtype=np.float32)
    nrm = float(np.linalg.norm(vec))
    if nrm <= 1e-6:
        return None
    return vec / nrm


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom <= 1e-6:
        return 180.0
    c = float(np.dot(v1, v2) / denom)
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _corridor_side_contacts(
    raw_skel: np.ndarray,
    endpoint: Tuple[int, int],
    target: Tuple[int, int],
) -> int:
    """Count crowding contacts in/adjacent to the (endpoint, target) line corridor.

    Returns the same integer as the bit-equivalent global-array reference
    implementation, but operates on a clipped ``line_bbox + pad`` subarray
    instead of an HxW canvas. For HxW=768x768 and gap-radius lines (<=10 px),
    this is ~5x faster per call and dominates overall Stage 3 wall time
    (per-tile profile: 95% of ``collect_virtual_bridge_proposals``).

    Pad math: line is at line_bbox+0; corridor's outer boundary is line_bbox+1
    (after one 3x3 dilation); ring's outer boundary is line_bbox+2 (after a
    second 3x3 dilation). The dilation kernel reads ±1 beyond, so the subarray
    must extend ≥2 beyond line_bbox for the local dilation result to match the
    global one. ``pad=4`` leaves a 2-pixel safety margin against off-by-one.

    Parity: see tests/test_corridor_side_contacts_parity.py for a frozen
    reference impl + 60+ synthetic fixtures + a 10K random-trial stress test.
    """
    h, w = raw_skel.shape
    pad = 4
    x_lo = max(0, min(endpoint[0], target[0]) - pad)
    x_hi = min(w - 1, max(endpoint[0], target[0]) + pad)
    y_lo = max(0, min(endpoint[1], target[1]) - pad)
    y_hi = min(h - 1, max(endpoint[1], target[1]) + pad)
    sub_h = y_hi - y_lo + 1
    sub_w = x_hi - x_lo + 1

    line_mask = np.zeros((sub_h, sub_w), dtype=bool)
    rr, cc = line(endpoint[1], endpoint[0], target[1], target[0])
    # Bresenham line stays within bbox of endpoints; rr/cc shifted into
    # subarray coords are guaranteed in-range.
    line_mask[rr - y_lo, cc - x_lo] = True

    sub_skel = raw_skel[y_lo : y_hi + 1, x_lo : x_hi + 1]
    corridor = binary_dilation(line_mask, structure=np.ones((3, 3), dtype=bool))

    contacts_core = ((sub_skel > 0) & corridor).astype(bool)
    contacts_core[line_mask] = False
    # Add a thin ring penalty to catch crowded parallel lanes adjacent to the bridge corridor.
    ring = binary_dilation(corridor, structure=np.ones((3, 3), dtype=bool)) & (~corridor)
    contacts_ring = ((sub_skel > 0) & ring).astype(bool)

    for px, py in (endpoint, target):
        x0 = max(0, px - 2 - x_lo)
        x1 = min(sub_w - 1, px + 2 - x_lo)
        y0 = max(0, py - 2 - y_lo)
        y1 = min(sub_h - 1, py + 2 - y_lo)
        if x0 <= x1 and y0 <= y1:
            contacts_core[y0 : y1 + 1, x0 : x1 + 1] = False
            contacts_ring[y0 : y1 + 1, x0 : x1 + 1] = False

    return int(np.count_nonzero(contacts_core) + np.count_nonzero(contacts_ring))


def _extract_roi(binary: np.ndarray, center_x: int, center_y: int, roi_size: int) -> Tuple[np.ndarray, int, int]:
    h, w = binary.shape
    roi = np.zeros((roi_size, roi_size), dtype=np.uint8)
    r = roi_size // 2
    x0 = center_x - r
    y0 = center_y - r
    x1 = x0 + roi_size
    y1 = y0 + roi_size

    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)

    if sx0 < sx1 and sy0 < sy1:
        rx0 = sx0 - x0
        ry0 = sy0 - y0
        roi[ry0 : ry0 + (sy1 - sy0), rx0 : rx0 + (sx1 - sx0)] = binary[sy0:sy1, sx0:sx1]
    return roi, x0, y0


def _select_branch_cluster_center(
    repaired_roi: np.ndarray,
    bridge_mid_local: Tuple[float, float],
) -> Tuple[int, int] | None:
    deg = degree_map(repaired_roi)
    branch_mask = (repaired_roi == 1) & (deg >= 3)
    if not np.any(branch_mask):
        return None

    labeled, n_components = ndimage.label(branch_mask.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))
    if n_components <= 0:
        return None

    bx, by = bridge_mid_local
    best_comp = -1
    best_dist = float("inf")
    for comp_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_id)
        if xs.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        d = float(np.hypot(cx - bx, cy - by))
        if d < best_dist:
            best_dist = d
            best_comp = comp_id

    if best_comp < 0:
        return None

    ys, xs = np.where(labeled == best_comp)
    if xs.size == 0:
        return None

    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    i = int(np.argmin((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2))
    chosen = (int(xs[i]), int(ys[i]))

    # Must stay local to the bridge center.
    if float(np.hypot(chosen[0] - bx, chosen[1] - by)) > 6.0:
        return None
    return chosen


def _validate_local_bridge(
    raw_skel: np.ndarray,
    endpoint: Tuple[int, int],
    target: Tuple[int, int],
    roi_size: int,
) -> Dict[str, Any]:
    mx = int(round((endpoint[0] + target[0]) / 2.0))
    my = int(round((endpoint[1] + target[1]) / 2.0))
    roi, x0, y0 = _extract_roi(raw_skel, center_x=mx, center_y=my, roi_size=roi_size)

    ex_l = endpoint[0] - x0
    ey_l = endpoint[1] - y0
    tx_l = target[0] - x0
    ty_l = target[1] - y0
    if not (0 <= ex_l < roi_size and 0 <= ey_l < roi_size):
        return {"accepted": False, "reason": "endpoint_outside_roi"}
    if not (0 <= tx_l < roi_size and 0 <= ty_l < roi_size):
        return {"accepted": False, "reason": "target_outside_roi"}

    repaired = roi.copy()
    rr, cc = line(ey_l, ex_l, ty_l, tx_l)
    repaired[rr, cc] = 1

    center = _select_branch_cluster_center(
        repaired_roi=repaired,
        bridge_mid_local=((ex_l + tx_l) / 2.0, (ey_l + ty_l) / 2.0),
    )
    if center is None:
        return {"accepted": False, "reason": "no_local_branch_cluster"}

    cx, cy = center
    local = analyze_local_junction(
        binary=repaired,
        anchor_x=cx,
        anchor_y=cy,
        trace_len=6,
        merge_deg=20.0,
    )

    branch_count = int(local["branch_count"])
    min_gap = local["min_gap_deg"]
    max_gap = local["max_gap_deg"]
    if branch_count != 3:
        # Strict was previously computed unconditionally before this branch
        # but its result is unused on this return path. Lazy-defer to save
        # ~50% of strict-analyze calls in the typical post-Win-#1 profile.
        # analyze_local_junction is pure w.r.t. its `binary` input
        # (junction_geometry.py:121 makes a local copy via astype), so
        # reordering preserves output bit-for-bit.
        return {"accepted": False, "reason": "branch_count_not_3"}

    strict = analyze_local_junction(
        binary=repaired,
        anchor_x=cx,
        anchor_y=cy,
        trace_len=6,
        merge_deg=15.0,
    )
    if int(strict["branch_count"]) >= 4:
        return {"accepted": False, "reason": "fourth_branch_detected"}
    if min_gap is None or float(min_gap) < 20.0:
        return {"accepted": False, "reason": "min_gap_too_small"}

    return {
        "accepted": True,
        "node_x": int(x0 + cx),
        "node_y": int(y0 + cy),
        "branch_count": branch_count,
        "min_gap_deg": float(min_gap),
        "max_gap_deg": float(max_gap) if max_gap is not None else None,
    }


def _proposal_score(gap_len: float, alignment_deg: float, side_contacts: int) -> float:
    alignment_term = max(0.0, float(np.cos(np.deg2rad(alignment_deg))))
    gap_term = 1.0 / (1.0 + float(gap_len))
    crowd_term = 1.0 / (1.0 + float(side_contacts))
    # Interpretable deterministic score: shorter + aligned + clean corridor + validated.
    return float(2.5 * gap_term + 2.0 * alignment_term + 1.5 * crowd_term + 1.0)


def _deduplicate_proposals(rows: List[Dict[str, Any]], radius: float = 2.0) -> List[Dict[str, Any]]:
    if len(rows) <= 1:
        return rows
    kept: List[Dict[str, Any]] = []
    ordered = sorted(rows, key=lambda r: float(r.get("proposal_score", 0.0)), reverse=True)
    for row in ordered:
        x = int(row["node_x"])
        y = int(row["node_y"])
        if any(float(np.hypot(x - int(k["node_x"]), y - int(k["node_y"]))) <= radius for k in kept):
            continue
        kept.append(row)
    return kept


def collect_virtual_bridge_proposals(
    raw_skel: np.ndarray,
    config: BridgeSearchConfig,
) -> Tuple[List[Dict[str, Any]], VirtualBridgeStats]:
    bw = (raw_skel > 0).astype(np.uint8)
    h, w = bw.shape
    deg = degree_map(bw)
    labels, _ = ndimage.label(bw.astype(np.uint8), structure=np.ones((3, 3), dtype=np.uint8))

    endpoint_yx = np.argwhere((bw == 1) & (deg == 1))
    stats = VirtualBridgeStats(endpoints=int(endpoint_yx.shape[0]))
    proposals: List[Dict[str, Any]] = []

    for ey, ex in endpoint_yx:
        endpoint = (int(ex), int(ey))
        incident_path = _trace_incident_branch(
            binary=bw,
            degree=deg,
            endpoint=endpoint,
            max_len=max(2, int(config.endpoint_tangent_len)),
        )
        tangent = _estimate_tangent(endpoint, incident_path, k=int(config.endpoint_tangent_len))
        if tangent is None:
            continue
        incident_pixels = set(incident_path)

        best_per_target: Dict[Tuple[str, int, int], _BridgeCandidate] = {}
        radii = sorted({int(r) for r in config.gap_radii if int(r) > 1})
        for radius in radii:
            x0 = max(0, endpoint[0] - radius)
            x1 = min(w - 1, endpoint[0] + radius)
            y0 = max(0, endpoint[1] - radius)
            y1 = min(h - 1, endpoint[1] + radius)

            for yy in range(y0, y1 + 1):
                for xx in range(x0, x1 + 1):
                    if bw[yy, xx] == 0:
                        continue
                    if (xx, yy) == endpoint:
                        continue
                    if max(abs(xx - endpoint[0]), abs(yy - endpoint[1])) <= 1:
                        continue
                    if (xx, yy) in incident_pixels:
                        continue

                    d = float(np.hypot(xx - endpoint[0], yy - endpoint[1]))
                    if d > float(radius) or d <= 1.0:
                        continue

                    disp = np.array([xx - endpoint[0], yy - endpoint[1]], dtype=np.float32)
                    stats.candidates_considered += 1
                    angle = _angle_deg(tangent, disp)
                    if angle > float(config.proposal_cone_deg):
                        stats.rejected_cone += 1
                        continue

                    target_component = int(labels[yy, xx])
                    if target_component <= 0:
                        continue
                    target_is_endpoint = bool(deg[yy, xx] == 1)
                    key = (
                        "endpoint" if target_is_endpoint else "component",
                        int(xx) if target_is_endpoint else int(target_component),
                        int(yy) if target_is_endpoint else 0,
                    )

                    cand = _BridgeCandidate(
                        endpoint=endpoint,
                        target=(int(xx), int(yy)),
                        target_is_endpoint=target_is_endpoint,
                        target_component=target_component,
                        gap_len=d,
                        radius_used=radius,
                        alignment_deg=float(angle),
                        incident_pixels=incident_pixels,
                    )
                    prev = best_per_target.get(key)
                    if prev is None or cand.gap_len < prev.gap_len - 1e-6 or (
                        abs(cand.gap_len - prev.gap_len) <= 1e-6 and cand.alignment_deg < prev.alignment_deg
                    ):
                        best_per_target[key] = cand

        for cand in best_per_target.values():
            side_contacts = _corridor_side_contacts(
                raw_skel=bw,
                endpoint=cand.endpoint,
                target=cand.target,
            )
            if side_contacts > int(config.bridge_max_side_contacts):
                stats.rejected_corridor += 1
                continue

            valid = _validate_local_bridge(
                raw_skel=bw,
                endpoint=cand.endpoint,
                target=cand.target,
                roi_size=int(config.bridge_roi_size),
            )
            if not bool(valid.get("accepted", False)):
                stats.rejected_local_validation += 1
                continue

            node_x = int(valid["node_x"])
            node_y = int(valid["node_y"])
            border_flag = int(
                node_x < int(config.border_margin)
                or node_y < int(config.border_margin)
                or node_x >= (w - int(config.border_margin))
                or node_y >= (h - int(config.border_margin))
            )
            score = _proposal_score(
                gap_len=cand.gap_len,
                alignment_deg=cand.alignment_deg,
                side_contacts=side_contacts,
            )

            proposals.append(
                {
                    "node_x": node_x,
                    "node_y": node_y,
                    "proposal_source": "virtual_bridge",
                    "proposal_type": (
                        "virtual_gap_endpoint_endpoint"
                        if cand.target_is_endpoint
                        else "virtual_gap_endpoint_segment"
                    ),
                    "gap_len_px": float(cand.gap_len),
                    "gap_radius_used": int(cand.radius_used),
                    "endpoint_x": int(cand.endpoint[0]),
                    "endpoint_y": int(cand.endpoint[1]),
                    "target_x": int(cand.target[0]),
                    "target_y": int(cand.target[1]),
                    "proposal_score": float(score),
                    "border_flag": border_flag,
                }
            )
            stats.accepted += 1

    proposals = _deduplicate_proposals(proposals, radius=2.0)
    stats.accepted = int(len(proposals))
    return proposals, stats
