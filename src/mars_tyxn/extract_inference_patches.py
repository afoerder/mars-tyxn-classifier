#!/usr/bin/env python3
"""
extract_inference_patches.py

Extract 96x96 inference patches around proposal-stage junction candidates from
large binary fracture images.

Proposal streams:
1) Base topology proposals from spur-pruned skeleton.
2) Virtual bridge proposals from raw skeleton (orientation-aware, locally validated).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, dilation, disk, skeletonize, square

from mars_tyxn.junction_proposals import BridgeSearchConfig, VirtualBridgeStats, collect_virtual_bridge_proposals


PATCH_SIZE = 96
JITTER_MIN = 28
JITTER_MAX = 67  # inclusive
SPUR_MIN_LEN = 12

MANIFEST_COLUMNS = [
    "patch_filename",
    "source_image",
    "node_x",
    "node_y",
    "local_x",
    "local_y",
    "proposal_source",
    "proposal_type",
    "gap_len_px",
    "gap_radius_used",
    "endpoint_x",
    "endpoint_y",
    "target_x",
    "target_y",
    "proposal_score",
    "border_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract jittered 96x96 patches around skeleton junction proposals."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing large fracture images.",
    )
    parser.add_argument(
        "--image-pattern",
        type=str,
        default="*_skel.png",
        help="Glob pattern for input images (default: *_skel.png).",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="auto",
        choices=["auto", "skeleton", "mask"],
        help="Input interpretation mode. auto infers skeleton for '*_skel*' paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_patches"),
        help="Directory to write extracted patch PNGs.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("inference_manifest.csv"),
        help="Output CSV manifest path.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Binarization threshold for grayscale input (pixel > threshold is foreground).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=96,
        help="Requested zero-padding in pixels on all sides (minimum enforced automatically).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for jitter placement reproducibility.",
    )
    parser.add_argument(
        "--neighbors-threshold",
        type=int,
        default=2,
        help="Junction criterion: node pixels satisfy neighbor_count > this value (default: >2).",
    )
    parser.add_argument(
        "--proposal-mode",
        type=str,
        default="current",
        choices=["current", "detector", "hybrid"],
        help=(
            "Proposal source mode: current=base+virtual, "
            "detector=graph detector only, hybrid=union(detector,current)."
        ),
    )
    parser.add_argument(
        "--detector-repo",
        type=Path,
        default=Path("."),
        help="Path to directory containing template_matcher.py (default: current dir).",
    )
    parser.add_argument(
        "--detector-detection-mode",
        type=str,
        default="graph",
        choices=["graph", "template", "hybrid"],
        help="Detection mode passed into template_matcher.detect_junctions.",
    )
    parser.add_argument(
        "--detector-template-dir",
        type=Path,
        default=None,
        help="Template directory for detector template/hybrid mode (optional for graph mode).",
    )
    parser.add_argument(
        "--detector-labels",
        type=str,
        default="T,Y,X",
        help="Comma-separated label set for detector.",
    )
    parser.add_argument(
        "--proposal-merge-radius",
        type=float,
        default=6.0,
        help="Dedup radius (px) for detector/current hybrid proposal merging.",
    )
    parser.add_argument(
        "--gap-radii",
        type=int,
        nargs="+",
        default=[3, 5, 7],
        help="Search radii for endpoint-driven virtual bridge proposals.",
    )
    parser.add_argument(
        "--endpoint-tangent-len",
        type=int,
        default=6,
        help="Backward trace length (px) used for endpoint tangent estimation.",
    )
    parser.add_argument(
        "--proposal-cone-deg",
        type=float,
        default=55.0,
        help="Max endpoint tangent-to-displacement angle for candidate targets.",
    )
    parser.add_argument(
        "--bridge-roi-size",
        type=int,
        default=31,
        help="Local ROI side length used for virtual-bridge validation.",
    )
    parser.add_argument(
        "--bridge-max-side-contacts",
        type=int,
        default=2,
        help="Max allowed corridor side contacts for a candidate bridge.",
    )
    parser.add_argument(
        "--border-margin",
        type=int,
        default=8,
        help="Pixels from image border considered low-trust border region.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_binary_image(path: Path, threshold: int) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return (gray > threshold).astype(np.uint8)


def heal_and_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    healed = closing(binary_mask.astype(bool), square(5))
    healed = binary_fill_holes(healed)
    skel = skeletonize(healed)
    return skel.astype(np.uint8)


def _neighbor_coords(x: int, y: int, skel: np.ndarray) -> List[Tuple[int, int]]:
    h, w = skel.shape
    out: List[Tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < w and 0 <= ny < h and skel[ny, nx] == 1:
                out.append((nx, ny))
    return out


def prune_spurs(binary_skel: np.ndarray, min_spur_len: int = SPUR_MIN_LEN, max_iters: int = 8) -> np.ndarray:
    if binary_skel.ndim != 2:
        raise ValueError(f"Expected 2D skeleton, got shape {binary_skel.shape}")

    skel = (binary_skel > 0).astype(np.uint8).copy()
    if not np.any(skel):
        return skel

    for _ in range(max_iters):
        changed = False
        kernel = np.ones((3, 3), dtype=np.uint8)
        kernel[1, 1] = 0
        deg = ndimage.convolve(skel, kernel, mode="constant", cval=0)
        endpoints = np.argwhere((skel == 1) & (deg == 1))
        if endpoints.size == 0:
            break

        for y0, x0 in endpoints:
            if skel[y0, x0] == 0:
                continue
            start = (int(x0), int(y0))
            path: List[Tuple[int, int]] = [start]
            prev: Tuple[int, int] | None = None
            cur = start
            terminal_kind = "unknown"

            while True:
                nbrs = _neighbor_coords(cur[0], cur[1], skel)
                if prev is not None:
                    nbrs = [p for p in nbrs if p != prev]
                if len(nbrs) == 0:
                    terminal_kind = "endpoint"
                    break
                if len(nbrs) > 1:
                    terminal_kind = "branch"
                    break

                nxt = nbrs[0]
                prev = cur
                cur = nxt
                path.append(cur)

                dcur = len(_neighbor_coords(cur[0], cur[1], skel))
                if dcur >= 3:
                    terminal_kind = "junction"
                    break
                if dcur <= 1:
                    terminal_kind = "endpoint"
                    break

            if terminal_kind == "junction":
                branch_len = len(path) - 1
                if branch_len < min_spur_len:
                    for px, py in path[:-1]:
                        skel[py, px] = 0
                    changed = True

        if not changed:
            break

    return skel


def detect_junction_nodes(binary_skel: np.ndarray, neighbors_threshold: int = 2) -> List[Tuple[float, float]]:
    if binary_skel.ndim != 2:
        raise ValueError(f"Expected 2D binary image, got shape {binary_skel.shape}")
    skel = (binary_skel > 0).astype(np.uint8)
    if not np.any(skel):
        return []

    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = ndimage.convolve(skel, kernel, mode="constant", cval=0)
    node_pixels = (skel == 1) & (neighbor_count > neighbors_threshold)
    if not np.any(node_pixels):
        return []

    merged_nodes = dilation(node_pixels, disk(6))
    labeled, n_components = ndimage.label(merged_nodes, structure=np.ones((3, 3), dtype=np.uint8))

    nodes: List[Tuple[float, float]] = []
    for comp_id in range(1, n_components + 1):
        ys, xs = np.where(labeled == comp_id)
        if xs.size == 0:
            continue
        nodes.append((float(xs.mean()), float(ys.mean())))

    if len(nodes) <= 1:
        return nodes

    threshold = 12.0
    remaining = list(nodes)
    collapsed: List[Tuple[float, float]] = []
    while remaining:
        seed_x, seed_y = remaining.pop(0)
        group = [(seed_x, seed_y)]
        grew = True
        while grew:
            grew = False
            keep: List[Tuple[float, float]] = []
            for cx, cy in remaining:
                if any(np.hypot(cx - gx, cy - gy) <= threshold for gx, gy in group):
                    group.append((cx, cy))
                    grew = True
                else:
                    keep.append((cx, cy))
            remaining = keep
        collapsed.append((float(np.mean([p[0] for p in group])), float(np.mean([p[1] for p in group]))))
    return collapsed


def extract_robust_junctions(mask_path: str | Path) -> List[Tuple[float, float]]:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    binary = load_binary_image(mask_path, threshold=0)
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D mask image, got shape {binary.shape}")
    if not np.any(binary):
        return []
    skel = heal_and_skeletonize(binary)
    if not np.any(skel):
        return []
    pruned = prune_spurs(skel, min_spur_len=SPUR_MIN_LEN)
    if not np.any(pruned):
        return []
    return detect_junction_nodes(pruned, neighbors_threshold=2)


def effective_padding(requested_pad: int) -> int:
    min_required = max(48, JITTER_MAX, (PATCH_SIZE - 1) - JITTER_MIN)
    return max(requested_pad, min_required)


def extract_jittered_patch(
    padded_binary: np.ndarray,
    node_x: int,
    node_y: int,
    pad: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int]:
    local_x = int(rng.integers(JITTER_MIN, JITTER_MAX + 1))
    local_y = int(rng.integers(JITTER_MIN, JITTER_MAX + 1))
    px = node_x + pad
    py = node_y + pad
    x0 = px - local_x
    y0 = py - local_y
    patch = padded_binary[y0 : y0 + PATCH_SIZE, x0 : x0 + PATCH_SIZE]
    if patch.shape != (PATCH_SIZE, PATCH_SIZE):
        raise RuntimeError(f"Patch extraction failed: expected {(PATCH_SIZE, PATCH_SIZE)}, got {patch.shape}")
    return patch, local_x, local_y


def unique_patch_name(output_dir: Path, base_name: str) -> str:
    candidate = output_dir / f"{base_name}.png"
    if not candidate.exists():
        return candidate.name
    k = 1
    while True:
        alt = output_dir / f"{base_name}_{k}.png"
        if not alt.exists():
            return alt.name
        k += 1


def find_images(input_dir: Path, pattern: str) -> List[Path]:
    return sorted([p for p in input_dir.glob(pattern) if p.is_file()])


def resolve_input_mode(image_path: Path, image_pattern: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    name = image_path.name.lower()
    pattern = image_pattern.lower()
    if "_skel" in name or "skel" in pattern:
        return "skeleton"
    return "mask"


def _is_border(node_x: int, node_y: int, w: int, h: int, border_margin: int) -> int:
    return int(
        node_x < border_margin
        or node_y < border_margin
        or node_x >= (w - border_margin)
        or node_y >= (h - border_margin)
    )


def _base_proposals(
    pruned_skel: np.ndarray,
    neighbors_threshold: int,
    border_margin: int,
) -> List[Dict[str, Any]]:
    h, w = pruned_skel.shape
    nodes = detect_junction_nodes(pruned_skel, neighbors_threshold=neighbors_threshold)
    out: List[Dict[str, Any]] = []
    for cx, cy in nodes:
        node_x = int(np.clip(round(cx), 0, w - 1))
        node_y = int(np.clip(round(cy), 0, h - 1))
        out.append(
            {
                "node_x": node_x,
                "node_y": node_y,
                "proposal_source": "base_topology",
                "proposal_type": "base_junction",
                "gap_len_px": "",
                "gap_radius_used": "",
                "endpoint_x": "",
                "endpoint_y": "",
                "target_x": "",
                "target_y": "",
                "proposal_score": 1.0,
                "border_flag": _is_border(node_x, node_y, w=w, h=h, border_margin=border_margin),
            }
        )
    return out


def _filter_virtual_near_base(
    base_rows: Sequence[Dict[str, Any]],
    virtual_rows: Sequence[Dict[str, Any]],
    distance_px: float = 5.0,
) -> List[Dict[str, Any]]:
    if not base_rows:
        return list(virtual_rows)
    base_xy = [(int(r["node_x"]), int(r["node_y"])) for r in base_rows]
    out: List[Dict[str, Any]] = []
    for row in virtual_rows:
        vx = int(row["node_x"])
        vy = int(row["node_y"])
        if any(float(np.hypot(vx - bx, vy - by)) <= distance_px for bx, by in base_xy):
            continue
        out.append(row)
    return out


def _blank_to_csv(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _load_detector_api(detector_repo: Path):
    detector_repo = detector_repo.expanduser().resolve()
    module_path = detector_repo / "template_matcher.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Detector module not found: {module_path}. "
            "Set --detector-repo to a directory containing template_matcher.py."
        )
    if str(detector_repo) not in sys.path:
        sys.path.insert(0, str(detector_repo))
    module = importlib.import_module("template_matcher")
    detect_fn = getattr(module, "detect_junctions", None)
    parse_labels_fn = getattr(module, "parse_labels", None)
    if detect_fn is None:
        raise RuntimeError("template_matcher.detect_junctions was not found.")
    return detect_fn, parse_labels_fn


def _detector_proposals(
    raw_skel: np.ndarray,
    detector_fn: Any,
    labels: Sequence[str],
    detection_mode: str,
    template_dir: Path | None,
    border_margin: int,
) -> List[Dict[str, Any]]:
    h, w = raw_skel.shape
    image_u8 = (raw_skel > 0).astype(np.uint8) * 255
    dets, counts, _ = detector_fn(
        image=image_u8,
        template_dir=str(template_dir) if template_dir else None,
        labels=list(labels),
        detection_mode=str(detection_mode),
        use_topology_gate=False,
    )
    out: List[Dict[str, Any]] = []
    for det in dets:
        raw_x = int(det.get("x", -1))
        raw_y = int(det.get("y", -1))
        if raw_x < 0 or raw_y < 0:
            continue
        node_x = int(np.clip(raw_x, 0, w - 1))
        node_y = int(np.clip(raw_y, 0, h - 1))
        source_type = str(det.get("source_type", "graph")).strip().lower()
        det_type = str(det.get("type", "")).strip().upper()
        out.append(
            {
                "node_x": node_x,
                "node_y": node_y,
                # Keep base_topology semantics so downstream base guards still apply.
                "proposal_source": "base_topology",
                "proposal_type": f"detector_junction_{source_type}_{det_type or 'UNK'}",
                "gap_len_px": "",
                "gap_radius_used": "",
                "endpoint_x": "",
                "endpoint_y": "",
                "target_x": "",
                "target_y": "",
                "proposal_score": float(det.get("score", 1.0)),
                "border_flag": _is_border(node_x, node_y, w=w, h=h, border_margin=border_margin),
            }
        )
    logging.info(
        "Detector proposals: mode=%s labels=%s detections=%d counts=%s",
        detection_mode,
        ",".join(labels),
        len(out),
        {k: int(v) for k, v in dict(counts or {}).items()},
    )
    return out


def _merge_proposals_by_distance(
    rows: Sequence[Dict[str, Any]],
    radius_px: float,
) -> List[Dict[str, Any]]:
    if radius_px <= 0.0:
        return list(rows)

    def _priority(row: Dict[str, Any]) -> Tuple[int, float]:
        ptype = str(row.get("proposal_type", "")).strip()
        psrc = str(row.get("proposal_source", "")).strip()
        if ptype.startswith("detector_junction"):
            # Prefer detector nodes in local conflicts.
            return (0, -float(row.get("proposal_score", 0.0) or 0.0))
        if psrc == "virtual_bridge" or ptype.startswith("virtual_gap"):
            return (2, -float(row.get("proposal_score", 0.0) or 0.0))
        return (1, -float(row.get("proposal_score", 0.0) or 0.0))

    ordered = sorted(list(rows), key=_priority)
    kept: List[Dict[str, Any]] = []
    for row in ordered:
        x = float(row["node_x"])
        y = float(row["node_y"])
        if any(float(np.hypot(x - float(k["node_x"]), y - float(k["node_y"]))) <= radius_px for k in kept):
            continue
        kept.append(row)
    return kept


def extract_junctions(
    raw_skel: np.ndarray,
    pruned_skel: np.ndarray,
    image_path: Path,
    output_dir: Path,
    pad: int,
    rng: np.random.Generator,
    neighbors_threshold: int,
    bridge_config: BridgeSearchConfig,
    proposal_mode: str,
    detector_fn: Any | None,
    detector_labels: Sequence[str],
    detector_detection_mode: str,
    detector_template_dir: Path | None,
    proposal_merge_radius: float,
) -> List[Dict[str, Any]]:
    h, w = raw_skel.shape
    padded = np.pad((raw_skel > 0).astype(np.uint8), ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    base_rows: List[Dict[str, Any]] = []
    virtual_rows: List[Dict[str, Any]] = []
    stats = VirtualBridgeStats()

    if proposal_mode in {"current", "hybrid"}:
        base_rows = _base_proposals(
            pruned_skel=pruned_skel,
            neighbors_threshold=neighbors_threshold,
            border_margin=int(bridge_config.border_margin),
        )
        virtual_rows, stats = collect_virtual_bridge_proposals(raw_skel=raw_skel, config=bridge_config)
        virtual_rows = _filter_virtual_near_base(base_rows=base_rows, virtual_rows=virtual_rows, distance_px=5.0)
    detector_rows: List[Dict[str, Any]] = []
    if proposal_mode in {"detector", "hybrid"}:
        if detector_fn is None:
            raise RuntimeError("Detector proposal mode requested but detector API was not loaded.")
        detector_rows = _detector_proposals(
            raw_skel=raw_skel,
            detector_fn=detector_fn,
            labels=detector_labels,
            detection_mode=detector_detection_mode,
            template_dir=detector_template_dir,
            border_margin=int(bridge_config.border_margin),
        )

    if proposal_mode == "current":
        proposals = base_rows + virtual_rows
    elif proposal_mode == "detector":
        proposals = detector_rows
    else:
        proposals = _merge_proposals_by_distance(
            detector_rows + base_rows + virtual_rows,
            radius_px=float(proposal_merge_radius),
        )

    manifest_rows: List[Dict[str, Any]] = []
    stem = image_path.stem
    for proposal in proposals:
        node_x = int(np.clip(int(proposal["node_x"]), 0, w - 1))
        node_y = int(np.clip(int(proposal["node_y"]), 0, h - 1))

        patch_binary, local_x, local_y = extract_jittered_patch(
            padded_binary=padded,
            node_x=node_x,
            node_y=node_y,
            pad=pad,
            rng=rng,
        )
        patch_u8 = (patch_binary * 255).astype(np.uint8)
        base_name = f"{stem}_x{node_x}_y{node_y}"
        patch_filename = unique_patch_name(output_dir, base_name)
        patch_path = output_dir / patch_filename
        Image.fromarray(patch_u8, mode="L").save(patch_path)

        row = {
            "patch_filename": patch_filename,
            "source_image": image_path.name,
            "node_x": node_x,
            "node_y": node_y,
            "local_x": local_x,
            "local_y": local_y,
            "proposal_source": str(proposal.get("proposal_source", "")),
            "proposal_type": str(proposal.get("proposal_type", "")),
            "gap_len_px": _blank_to_csv(proposal.get("gap_len_px", "")),
            "gap_radius_used": _blank_to_csv(proposal.get("gap_radius_used", "")),
            "endpoint_x": _blank_to_csv(proposal.get("endpoint_x", "")),
            "endpoint_y": _blank_to_csv(proposal.get("endpoint_y", "")),
            "target_x": _blank_to_csv(proposal.get("target_x", "")),
            "target_y": _blank_to_csv(proposal.get("target_y", "")),
            "proposal_score": _blank_to_csv(proposal.get("proposal_score", "")),
            "border_flag": int(proposal.get("border_flag", 0)),
        }
        manifest_rows.append(row)

    logging.info(
        "%s: proposal_mode=%s proposals=%d (detector=%d, base=%d, virtual=%d) | endpoints=%d candidates=%d cone_rej=%d corridor_rej=%d local_rej=%d accepted_virtual=%d",
        image_path.name,
        proposal_mode,
        len(manifest_rows),
        len(detector_rows),
        len(base_rows),
        len(virtual_rows),
        stats.endpoints,
        stats.candidates_considered,
        stats.rejected_cone,
        stats.rejected_corridor,
        stats.rejected_local_validation,
        stats.accepted,
    )
    return manifest_rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    images = find_images(args.input_dir, args.image_pattern)
    if not images:
        raise RuntimeError(f"No images found in {args.input_dir} matching pattern '{args.image_pattern}'.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    pad = effective_padding(args.pad)
    rng = np.random.default_rng(args.seed)
    bridge_roi_size = int(args.bridge_roi_size)
    if bridge_roi_size % 2 == 0:
        bridge_roi_size += 1

    bridge_config = BridgeSearchConfig(
        gap_radii=tuple(int(r) for r in args.gap_radii),
        endpoint_tangent_len=int(args.endpoint_tangent_len),
        proposal_cone_deg=float(args.proposal_cone_deg),
        bridge_roi_size=bridge_roi_size,
        bridge_max_side_contacts=int(args.bridge_max_side_contacts),
        border_margin=int(args.border_margin),
    )
    detector_fn = None
    detector_labels: List[str] = ["T", "Y", "X"]
    detector_mode = str(args.detector_detection_mode).strip().lower()
    detector_template_dir = args.detector_template_dir
    if detector_template_dir is not None:
        detector_template_dir = detector_template_dir.expanduser().resolve()
    if args.proposal_mode in {"detector", "hybrid"}:
        detector_fn, parse_labels_fn = _load_detector_api(args.detector_repo)
        if parse_labels_fn is not None:
            detector_labels = list(parse_labels_fn(args.detector_labels))
        else:
            detector_labels = [token.strip().upper() for token in str(args.detector_labels).split(",") if token.strip()]
        if detector_mode in {"template", "hybrid"} and detector_template_dir is None:
            raise ValueError(
                "--detector-template-dir is required when --detector-detection-mode is template or hybrid."
            )
        logging.info(
            "Loaded detector API from %s (mode=%s labels=%s)",
            args.detector_repo,
            detector_mode,
            ",".join(detector_labels),
        )

    logging.info("Found %d image(s)", len(images))
    logging.info("Patch geometry: size=%d, jitter_range=[%d,%d], pad=%d", PATCH_SIZE, JITTER_MIN, JITTER_MAX, pad)
    logging.info("Proposal mode: %s", args.proposal_mode)
    logging.info(
        "Virtual bridge params: gap_radii=%s tangent_len=%d cone=%.1f roi=%d max_side_contacts=%d border_margin=%d",
        bridge_config.gap_radii,
        bridge_config.endpoint_tangent_len,
        bridge_config.proposal_cone_deg,
        bridge_config.bridge_roi_size,
        bridge_config.bridge_max_side_contacts,
        bridge_config.border_margin,
    )

    manifest_rows: List[Dict[str, Any]] = []
    for image_path in images:
        binary = load_binary_image(image_path, threshold=args.threshold)
        mode = resolve_input_mode(image_path=image_path, image_pattern=args.image_pattern, requested_mode=args.input_mode)

        if mode == "skeleton":
            raw_skel = (binary > 0).astype(np.uint8)
            logging.debug("%s: input_mode=skeleton (no global healing)", image_path.name)
        else:
            logging.info("%s: input_mode=mask (global closing/fill/skeletonize enabled)", image_path.name)
            raw_skel = heal_and_skeletonize(binary)

        if not np.any(raw_skel):
            logging.warning("%s: empty foreground after preprocessing; skipping", image_path.name)
            continue

        pruned_skel = prune_spurs(raw_skel, min_spur_len=SPUR_MIN_LEN)
        if not np.any(pruned_skel):
            logging.warning("%s: empty skeleton after spur pruning; skipping", image_path.name)
            continue

        image_rows = extract_junctions(
            raw_skel=raw_skel,
            pruned_skel=pruned_skel,
            image_path=image_path,
            output_dir=args.output_dir,
            pad=pad,
            rng=rng,
            neighbors_threshold=int(args.neighbors_threshold),
            bridge_config=bridge_config,
            proposal_mode=str(args.proposal_mode).strip().lower(),
            detector_fn=detector_fn,
            detector_labels=detector_labels,
            detector_detection_mode=detector_mode,
            detector_template_dir=detector_template_dir,
            proposal_merge_radius=float(args.proposal_merge_radius),
        )
        manifest_rows.extend(image_rows)

    with args.manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    logging.info("Done. Patches saved: %d", len(manifest_rows))
    logging.info("Patches directory: %s", args.output_dir)
    logging.info("Manifest: %s", args.manifest_path)


if __name__ == "__main__":
    main()
