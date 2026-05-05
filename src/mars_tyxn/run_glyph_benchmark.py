#!/usr/bin/env python3
"""
run_glyph_benchmark.py

Run the glyph-based template matching classifier on skeleton images and
produce a prediction CSV compatible with evaluate_ground_truth.py.

This bridges the glyph detector (from Glyph_Detr_Clsfr/template_matcher.py)
into the same evaluation pipeline used for the learned models (CNN, XGB, etc.).

Usage:
    python run_glyph_benchmark.py \
        --images-dir data/evaluation_martian/images \
        --detector-repo /path/to/Glyph_Detr_Clsfr \
        --output-csv predictions/glyph.csv \
        --detection-mode hybrid
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run glyph template matcher and produce evaluation-compatible CSV."
    )
    p.add_argument(
        "--images-dir", type=Path, required=True,
        help="Directory containing skeleton PNG images.",
    )
    p.add_argument(
        "--image-pattern", type=str, default="*_skel.png",
        help="Glob pattern for skeleton images (default: *_skel.png).",
    )
    p.add_argument(
        "--detector-repo", type=Path, default=Path("."),
        help="Path to directory containing template_matcher.py (default: current dir).",
    )
    p.add_argument(
        "--template-dir", type=str, default=None,
        help="Path to template directory. If not set, uses <detector-repo>/assets/templates_tyx_exact/gray_128.",
    )
    p.add_argument(
        "--detection-mode", type=str, default="hybrid",
        choices=["template", "graph", "hybrid", "improved_graph"],
        help="Detection mode (default: hybrid). 'improved_graph' uses "
             "degree≥3 clusters + crossing-number filter + NMS for "
             "detection, then geometry analysis for classification.",
    )
    p.add_argument(
        "--output-csv", type=Path, required=True,
        help="Output prediction CSV path.",
    )
    p.add_argument(
        "--nms-distance", type=int, default=5,
        help="Non-maximum suppression distance in pixels (default: 5).",
    )
    p.add_argument(
        "--classifier-mode", type=str, default="robust",
        choices=["symmetric", "robust", "robust_vote"],
        help="Three-arm classifier mode (default: robust).",
    )
    p.add_argument(
        "--t-min-largest-angle", type=float, default=150.0,
        help="T guardrail: largest inter-arm angle must be >= this (default: 150).",
    )
    p.add_argument(
        "--t-max-side-angle", type=float, default=124.0,
        help="T guardrail: both smaller angles must be <= this (default: 124).",
    )
    p.add_argument(
        "--branch-trace-len", type=int, default=8,
        help="Branch trace length for junction_geometry cross-check (default: 8).",
    )
    p.add_argument(
        "--collinearity-radii", type=str, default="16,20,12",
        help="Comma-separated radii for collinearity measurement (default: 16,20,12). "
             "Tries each in order until 3 branches are found.",
    )
    p.add_argument(
        "--collinearity-threshold", type=float, default=41.0,
        help="Max collinear deviation (degrees) to classify as T (default: 41). "
             "Lower = stricter T criterion.",
    )
    p.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _crossing_number(binary: np.ndarray, cy: float, cx: float, radius: int) -> int:
    """Count distinct foreground crossings on a circle of given radius."""
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
    """Find branch direction angles (radians) by scanning a circle at given radius."""
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

    # Find foreground runs and compute their center angles
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

    # Handle wrap-around: merge first and last run if both touch the boundary
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
    """For 3+ branch angles, find the pair closest to collinear (180° apart).

    Returns (min_deviation_from_180_degrees, max_gap_degrees).
    Lower deviation = more T-like (two arms form a straight line).
    """
    n = len(branch_angles_rad)
    if n < 3:
        return 90.0, None  # no collinearity measurable

    min_dev = 180.0
    for i in range(n):
        for j in range(i + 1, n):
            diff_deg = abs(np.degrees(branch_angles_rad[i] - branch_angles_rad[j])) % 360
            if diff_deg > 180:
                diff_deg = 360 - diff_deg
            dev = abs(diff_deg - 180)
            if dev < min_dev:
                min_dev = dev

    # Also compute max angular gap
    angles_deg = sorted(np.degrees(a) % 360 for a in branch_angles_rad)
    gaps = [(angles_deg[(i + 1) % n] - angles_deg[i]) % 360 for i in range(n)]
    max_gap = max(gaps)

    return min_dev, max_gap


def _nms_by_score(detections: list, nms_dist: int = 20) -> list:
    """Non-maximum suppression: keep highest-score detection within nms_dist."""
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


def improved_graph_detect(
    image_u8: np.ndarray,
    local_geometry_fn,
    nms_distance: int = 20,
    cn_radii: tuple = (8, 12),
    min_crossings: int = 3,
    geom_inner: int = 8,
    geom_outer: int = 15,
    geom_snap: int = 10,
    classifier_mode: str = "robust",
    t_min_largest_angle: float = 150.0,
    t_max_side_angle: float = 124.0,
    branch_trace_len: int = 8,
    collinearity_radii: tuple = (16, 20, 12),
    collinearity_threshold: float = 41.0,
) -> list[dict]:
    """
    Improved junction detection using degree>=3 pixel clusters, crossing-number
    filtering, NMS, and multi-feature classification.

    Classification uses collinearity at larger radii as the primary T/Y
    discriminator (two arms forming a straight line = T), supplemented by
    ring-based geometry and branch-tracing.

    Returns list of dicts with keys: x, y, type.
    """
    from skimage.morphology import skeletonize
    from mars_tyxn.junction_geometry import analyze_local_junction

    binary = (image_u8 > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbr_count = ndimage.convolve(binary, kernel, mode="constant", cval=0)

    # Step 1: find connected components of degree>=3 pixels
    branch_mask = (binary > 0) & (nbr_count >= 3)
    labeled, n_clusters = ndimage.label(branch_mask, structure=np.ones((3, 3)))

    # Step 2: filter by crossing number
    raw_dets = []
    for c in range(1, n_clusters + 1):
        cluster_pixels = np.argwhere(labeled == c)
        cy, cx = cluster_pixels.mean(axis=0)
        cns = [_crossing_number(binary, cy, cx, r) for r in cn_radii]
        max_cn = max(cns)
        if max_cn < min_crossings:
            continue
        score = max_cn * len(cluster_pixels)
        raw_dets.append({
            "x": float(cx), "y": float(cy),
            "cn": max_cn, "score": score,
        })

    # Step 3: NMS
    detections = _nms_by_score(raw_dets, nms_dist=nms_distance)

    # Step 4: classify using multiple geometry signals on thinned skeleton
    thinned = skeletonize(binary > 0).astype(np.uint8)

    results = []
    for det in detections:
        ix = int(round(det["x"]))
        iy = int(round(det["y"]))

        # --- Signal 1: Collinearity at larger radii ---
        # Measures whether two branches form a straight line through the
        # junction (T-like) vs all branches splay equally (Y-like).
        # Larger radii see past local branch curvature near the junction.
        col_dev = None
        col_max_gap = None
        col_n_arms = 0
        for r in collinearity_radii:
            br_angles = _branch_angles_on_ring(thinned, iy, ix, r)
            if len(br_angles) == 3:
                col_dev, col_max_gap = _collinearity_deviation(br_angles)
                col_n_arms = 3
                break
            elif len(br_angles) >= 4 and col_n_arms == 0:
                col_n_arms = len(br_angles)

        # --- Signal 2: Ring-based geometry from template_matcher ---
        geom = local_geometry_fn(
            thin_binary=thinned,
            y=iy,
            x=ix,
            inner_radius=geom_inner,
            outer_radius=geom_outer,
            snap_radius=geom_snap,
            ring_bridge_radius=3,
            classifier_mode=classifier_mode,
            t_min_largest_angle=t_min_largest_angle,
            t_max_side_angle=t_max_side_angle,
        )
        ring_label = geom.get("label", "")
        arm_count = geom.get("arm_count", 0)

        # --- Signal 3: Branch-tracing geometry from junction_geometry ---
        jg = analyze_local_junction(
            binary=thinned,
            anchor_x=ix,
            anchor_y=iy,
            trace_len=branch_trace_len,
        )
        jg_branches = jg["branch_count"]
        jg_max_gap = jg["max_gap_deg"]
        jg_min_gap = jg["min_gap_deg"]

        # --- Classification decision ---
        # Ring geometry is primary; collinearity upgrades Y→T when strong.
        #
        # 4+ arms at any signal → X
        if col_n_arms >= 4 or arm_count >= 4 or jg_branches >= 4:
            det_type = "X"

        # 3-arm: ring geometry primary, collinearity as upgrade
        elif ring_label in ("T", "Y", "X") and arm_count >= 3:
            det_type = ring_label

            # If ring says Y, check if collinearity strongly suggests T.
            # Require collinearity AND branch-tracing agreement to override.
            if det_type == "Y" and col_dev is not None and col_n_arms == 3:
                col_says_t = col_dev <= collinearity_threshold
                jg_says_t = (
                    jg_branches == 3
                    and jg_max_gap is not None
                    and jg_max_gap >= t_min_largest_angle
                )
                if col_says_t and jg_says_t:
                    det_type = "T"

        # Fallback: branch-tracing + collinearity (ring unavailable)
        elif jg_branches == 3 and jg_max_gap is not None:
            if jg_max_gap >= t_min_largest_angle:
                det_type = "T"
            elif (
                col_dev is not None
                and col_n_arms == 3
                and col_dev <= collinearity_threshold
            ):
                det_type = "T"
            else:
                det_type = "Y"

        else:
            # Last resort: crossing number
            cn_med = int(np.median([
                _crossing_number(binary, det["y"], det["x"], r)
                for r in (8, 10, 15)
            ]))
            det_type = "X" if cn_med >= 4 else "Y"

        results.append({"x": det["x"], "y": det["y"], "type": det_type})

    return results


def load_detector(detector_repo: Path):
    """Import template_matcher functions, adding detector_repo to sys.path if needed."""
    detector_repo = detector_repo.expanduser().resolve()
    if str(detector_repo) not in sys.path:
        sys.path.insert(0, str(detector_repo))
    try:
        import template_matcher
    except ImportError:
        raise FileNotFoundError(
            f"template_matcher.py not found in {detector_repo}. "
            "Set --detector-repo to the directory containing template_matcher.py."
        )
    detect_fn = getattr(template_matcher, "detect_junctions", None)
    if detect_fn is None:
        raise RuntimeError("template_matcher.detect_junctions not found")
    geom_fn = getattr(template_matcher, "local_geometry_analysis", None)
    return detect_fn, geom_fn


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Resolve template directory
    if args.template_dir:
        template_dir = Path(args.template_dir)
    else:
        # Try archive layout first, then original Glyph_Detr_Clsfr layout
        _here = Path(__file__).resolve().parent
        archive_tpl = _here / ".." / "data" / "templates" / "templates_tyx_exact" / "gray_128"
        legacy_tpl = args.detector_repo / "assets" / "templates_tyx_exact" / "gray_128"
        template_dir = archive_tpl if archive_tpl.exists() else legacy_tpl
    if not template_dir.exists():
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    # Find skeleton images
    images = sorted(args.images_dir.glob(args.image_pattern))
    if not images:
        raise FileNotFoundError(
            f"No images matching '{args.image_pattern}' in {args.images_dir}"
        )
    logging.info("Found %d skeleton images in %s", len(images), args.images_dir)

    # Load detector
    detect_fn, geom_fn = load_detector(args.detector_repo)
    logging.info(
        "Loaded glyph detector: mode=%s, templates=%s",
        args.detection_mode, template_dir,
    )

    # Run detection on each image and collect predictions
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_image", "node_x", "node_y", "consensus", "agreement"]

    all_rows = []
    for img_path in images:
        source_image = img_path.name
        logging.info("Processing: %s", source_image)

        # Load skeleton as grayscale
        pil_img = Image.open(img_path).convert("L")
        skel_arr = np.array(pil_img)

        # Convert to binary uint8 (0 or 255) — same as extract_inference_patches.py
        image_u8 = (skel_arr > 0).astype(np.uint8) * 255

        h, w = skel_arr.shape
        img_rows = 0

        if args.detection_mode == "improved_graph":
            # Improved detection: degree≥3 clusters + crossing-number + NMS,
            # then geometry-based classification via local_geometry_analysis.
            if geom_fn is None:
                raise RuntimeError(
                    "local_geometry_analysis not found in template_matcher.py; "
                    "required for improved_graph mode"
                )
            col_radii = tuple(int(r) for r in args.collinearity_radii.split(","))
            det_list = improved_graph_detect(
                image_u8=image_u8,
                local_geometry_fn=geom_fn,
                nms_distance=args.nms_distance,
                classifier_mode=args.classifier_mode,
                t_min_largest_angle=args.t_min_largest_angle,
                t_max_side_angle=args.t_max_side_angle,
                branch_trace_len=args.branch_trace_len,
                collinearity_radii=col_radii,
                collinearity_threshold=args.collinearity_threshold,
            )
            for det in det_list:
                raw_x, raw_y = det["x"], det["y"]
                node_x = int(np.clip(raw_x, 0, w - 1))
                node_y = int(np.clip(raw_y, 0, h - 1))
                det_type = det["type"]
                all_rows.append({
                    "source_image": source_image,
                    "node_x": node_x,
                    "node_y": node_y,
                    "consensus": det_type,
                    "agreement": 1,
                })
                img_rows += 1
            logging.info("  %s: %d detections", source_image, img_rows)
        else:
            # Original modes: template, graph, hybrid
            # skip_thinning=True because input is already a skeleton;
            # re-thinning destroys junction topology at branch meeting points.
            detections, counts, _ = detect_fn(
                image=image_u8,
                template_dir=str(template_dir),
                labels=["T", "Y", "X"],
                detection_mode=args.detection_mode,
                use_topology_gate=False,
                skip_thinning=True,
                nms_distance=args.nms_distance,
            )

            for det in detections:
                raw_x = det.get("x", -1)
                raw_y = det.get("y", -1)
                if raw_x < 0 or raw_y < 0:
                    continue
                node_x = int(np.clip(raw_x, 0, w - 1))
                node_y = int(np.clip(raw_y, 0, h - 1))
                det_type = str(det.get("type", "")).strip().upper()
                if det_type not in ("T", "Y", "X"):
                    logging.warning(
                        "Skipping detection with unknown type '%s' at (%d, %d)",
                        det_type, node_x, node_y,
                    )
                    continue

                all_rows.append({
                    "source_image": source_image,
                    "node_x": node_x,
                    "node_y": node_y,
                    "consensus": det_type,
                    "agreement": 1,
                })
                img_rows += 1

            logging.info(
                "  %s: %d detections (counts: %s)",
                source_image, img_rows,
                {k: int(v) for k, v in (counts or {}).items()},
            )

    # Write prediction CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logging.info("Wrote %d predictions to %s", len(all_rows), args.output_csv)


if __name__ == "__main__":
    main()
