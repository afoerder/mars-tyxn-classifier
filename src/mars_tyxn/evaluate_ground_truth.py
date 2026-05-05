#!/usr/bin/env python3
"""
evaluate_ground_truth.py

Compare manual JSON labels (ground truth) against ensemble CSV predictions using
spatial matching with scipy.spatial.cKDTree.

Ground truth JSON format assumptions:
- top-level key: "shapes"
- each shape has:
    - "label": class string in {N, T, X, Y}
    - "points": list containing at least one [x, y]

Prediction CSV requirements:
- columns: source_image, node_x, node_y, consensus, agreement

Outputs printed to terminal:
1) Per-image matching summary
2) Global confusion matrix (True Label vs Consensus Prediction)
3) Per-class Precision/Recall/F1 for classes [N, T, X, Y]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree


VALID_CLASSES = ["N", "T", "X", "Y"]
MISSED = "MISSED"
NO_GT = "NO_GT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble predictions against manual JSON labels.")
    parser.add_argument(
        "--pred-csv",
        type=Path,
        default=Path("final_ensemble_results.csv"),
        help="Path to predictions CSV.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory containing manual JSON label files.",
    )
    parser.add_argument(
        "--match-radius",
        type=float,
        default=15.0,
        help="Spatial match radius in pixels.",
    )
    parser.add_argument(
        "--save-detailed-csv",
        type=Path,
        default=None,
        help="Optional path to save detailed match records.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional image directory for coordinate-system consistency checks.",
    )
    parser.add_argument(
        "--strict-coordinate-check",
        action="store_true",
        help="Fail evaluation if coordinate systems appear mismatched (slot/image dim mismatch, out-of-bounds GT).",
    )
    parser.add_argument(
        "--auto-rescale-gt-to-image",
        action="store_true",
        help="If slot dims differ from image dims, rescale GT coordinates to image coordinates.",
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
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def norm_label(value: object) -> str:
    """Normalize labels to uppercase canonical strings."""
    return str(value).strip().upper()


def load_predictions(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"source_image", "node_x", "node_y", "consensus", "agreement"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")

    # Normalize and coerce types.
    df = df.copy()
    df["source_image"] = df["source_image"].astype(str)
    df["consensus"] = df["consensus"].map(norm_label)
    df["node_x"] = pd.to_numeric(df["node_x"], errors="coerce")
    df["node_y"] = pd.to_numeric(df["node_y"], errors="coerce")
    df["agreement"] = pd.to_numeric(df["agreement"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["node_x", "node_y", "source_image", "consensus"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        logging.warning("Dropped %d prediction rows with invalid coordinates/labels.", dropped)

    return df


def candidate_image_names_from_json(json_path: Path, payload: dict) -> List[str]:
    """
    Build robust candidate source-image names from a JSON file.
    """
    candidates: List[str] = []

    image_path = payload.get("imagePath", None)
    if isinstance(image_path, str) and image_path.strip():
        candidates.append(Path(image_path).name)
        candidates.append(f"{Path(image_path).stem}.png")

    stem = json_path.stem
    candidates.append(f"{stem}.png")

    # Common naming conversion seen in GT exports.
    if stem.endswith("__gt_kp"):
        candidates.append(f"{stem.replace('__gt_kp', '_skel')}.png")
    if stem.endswith("_gt"):
        candidates.append(f"{stem[:-3]}.png")

    # Deduplicate while preserving order.
    out: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_source_image(json_path: Path, payload: dict, available_source_images: Sequence[str]) -> str | None:
    """
    Resolve which source_image in predictions corresponds to this JSON.
    """
    avail = list(available_source_images)
    avail_set = set(avail)
    avail_by_stem: Dict[str, List[str]] = {}
    for name in avail:
        avail_by_stem.setdefault(Path(name).stem, []).append(name)

    # 1) Exact filename match.
    for c in candidate_image_names_from_json(json_path, payload):
        if c in avail_set:
            return c

    # 2) Stem match.
    for c in candidate_image_names_from_json(json_path, payload):
        stem = Path(c).stem
        matches = avail_by_stem.get(stem, [])
        if len(matches) == 1:
            return matches[0]

    # 3) Unique substring fallback.
    for c in candidate_image_names_from_json(json_path, payload):
        stem = Path(c).stem
        matches = [name for name in avail if stem in Path(name).stem]
        if len(matches) == 1:
            return matches[0]

    return None


def get_slot_dims(payload: dict) -> Tuple[int, int] | None:
    item = payload.get("item", {})
    if not isinstance(item, dict):
        return None
    slots = item.get("slots", [])
    if not isinstance(slots, list) or not slots:
        return None
    slot0 = slots[0]
    if not isinstance(slot0, dict):
        return None
    w = slot0.get("width", None)
    h = slot0.get("height", None)
    try:
        wi = int(w)
        hi = int(h)
    except Exception:
        return None
    if wi <= 0 or hi <= 0:
        return None
    return wi, hi


def load_json_labels(json_path: Path) -> pd.DataFrame:
    """
    Load Darwin V7 labels and return a DataFrame with columns:
      ['true_label', 'x', 'y'].

    Parsing order:
      1) bounding_box center (x + w/2, y + h/2)
      2) point (x, y)
      3) polygon.path first point
    """
    out_cols = ["true_label", "x", "y"]

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return pd.DataFrame(columns=out_cols)

    annotations = payload.get("annotations", [])
    if not isinstance(annotations, list):
        return pd.DataFrame(columns=out_cols)

    rows: List[Dict[str, object]] = []
    valid_labels = set(VALID_CLASSES)

    def _xy_from_obj(obj: object) -> Tuple[float, float] | None:
        if isinstance(obj, dict):
            try:
                return float(obj["x"]), float(obj["y"])
            except Exception:
                return None
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            try:
                return float(obj[0]), float(obj[1])
            except Exception:
                return None
        return None

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        label = norm_label(ann.get("name", ""))
        if label not in valid_labels:
            continue

        xy: Tuple[float, float] | None = None

        # Primary Darwin geometry: bounding box center.
        bb = ann.get("bounding_box")
        if isinstance(bb, dict):
            try:
                x_center = float(bb["x"]) + (float(bb["w"]) / 2.0)
                y_center = float(bb["y"]) + (float(bb["h"]) / 2.0)
                xy = (x_center, y_center)
            except Exception:
                xy = None

        # Fallback 1: keypoint (Darwin v7 keypoint annotation)
        if xy is None and "keypoint" in ann:
            xy = _xy_from_obj(ann.get("keypoint"))

        # Fallback 2: point
        if xy is None and "point" in ann:
            xy = _xy_from_obj(ann.get("point"))

        # Fallback 3: polygon.path first point
        if xy is None and isinstance(ann.get("polygon"), dict):
            path = ann["polygon"].get("path", [])
            if isinstance(path, list) and len(path) > 0:
                xy = _xy_from_obj(path[0])

        if xy is None:
            continue

        rows.append({"true_label": label, "x": float(xy[0]), "y": float(xy[1])})

    if not rows:
        return pd.DataFrame(columns=out_cols)

    return pd.DataFrame(rows, columns=out_cols)


def match_points_ckdtree(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    radius: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """
    One-to-one spatial matching using cKDTree candidate generation + greedy assignment.

    Returns:
      matched_pairs: list of (gt_idx, pred_idx, distance)
      unmatched_gt: list of gt indices with no assigned prediction
      unmatched_pred: list of prediction indices with no assigned gt
    """
    n_gt = len(gt_points)
    n_pred = len(pred_points)

    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    tree = cKDTree(pred_points)

    # Build all candidate pairs within radius.
    candidates: List[Tuple[float, int, int]] = []
    for gi, gpt in enumerate(gt_points):
        near_pred = tree.query_ball_point(gpt, r=radius)
        for pj in near_pred:
            dist = float(np.linalg.norm(gpt - pred_points[pj]))
            candidates.append((dist, gi, int(pj)))

    # Greedy one-to-one by ascending distance.
    candidates.sort(key=lambda t: t[0])
    used_gt = set()
    used_pred = set()
    matches: List[Tuple[int, int, float]] = []

    for dist, gi, pj in candidates:
        if gi in used_gt or pj in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pj)
        matches.append((gi, pj, dist))

    unmatched_gt = [i for i in range(n_gt) if i not in used_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in used_pred]
    return matches, unmatched_gt, unmatched_pred


def compute_class_metrics(y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str]) -> pd.DataFrame:
    """
    Compute per-class precision/recall/F1 manually so MISSED/NO_GT are handled correctly.
    """
    rows = []
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)

    for c in classes:
        tp = int(np.sum((y_true_arr == c) & (y_pred_arr == c)))
        fp = int(np.sum((y_true_arr != c) & (y_pred_arr == c)))
        fn = int(np.sum((y_true_arr == c) & (y_pred_arr != c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class": c,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.labels_dir.exists() or not args.labels_dir.is_dir():
        raise FileNotFoundError(f"Labels directory not found: {args.labels_dir}")
    if args.images_dir is not None and (not args.images_dir.exists() or not args.images_dir.is_dir()):
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

    pred_df = load_predictions(args.pred_csv)
    source_images = sorted(pred_df["source_image"].astype(str).unique().tolist())

    json_files = sorted([p for p in args.labels_dir.glob("*.json") if p.is_file()])
    if not json_files:
        raise RuntimeError(f"No JSON files found in {args.labels_dir}")

    logging.info("Loaded predictions: %d rows across %d source image(s)", len(pred_df), len(source_images))
    logging.info("Found %d JSON label file(s)", len(json_files))

    # Detailed records for confusion/metrics.
    detailed_records: List[Dict[str, object]] = []

    print("\nPer-image summary")
    print("source_image, gt_total, pred_total, matched, missed_gt, unmatched_pred")

    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        source_image = resolve_source_image(json_path, payload, source_images)
        if source_image is None:
            logging.warning("Could not map JSON to source_image, skipping: %s", json_path.name)
            continue

        gt_df = load_json_labels(json_path)
        gt_labels = gt_df["true_label"].astype(str).tolist() if len(gt_df) > 0 else []

        pred_sub = pred_df[pred_df["source_image"] == source_image].reset_index(drop=True)
        pred_labels = [norm_label(v) for v in pred_sub["consensus"].tolist()]

        image_w: int | None = None
        image_h: int | None = None
        slot_dims = get_slot_dims(payload)
        if args.images_dir is not None:
            image_path = args.images_dir / source_image
            if not image_path.exists():
                msg = f"Resolved source image not found in --images-dir: {image_path}"
                if args.strict_coordinate_check:
                    raise FileNotFoundError(msg)
                logging.warning(msg)
            else:
                with Image.open(image_path) as im:
                    image_w, image_h = im.size

        # If labeling slot dimensions differ from actual image dimensions, either fail fast
        # (strict mode) or explicitly rescale GT coordinates if requested.
        if slot_dims is not None and image_w is not None and image_h is not None:
            slot_w, slot_h = slot_dims
            if slot_w != image_w or slot_h != image_h:
                msg = (
                    f"Coordinate-space mismatch for {source_image}: "
                    f"label slot=({slot_w},{slot_h}) vs image=({image_w},{image_h})"
                )
                if args.auto_rescale_gt_to_image and len(gt_df) > 0:
                    sx = float(image_w) / float(slot_w)
                    sy = float(image_h) / float(slot_h)
                    gt_df = gt_df.copy()
                    gt_df["x"] = gt_df["x"] * sx
                    gt_df["y"] = gt_df["y"] * sy
                    logging.warning("%s -> rescaled GT by sx=%.6f sy=%.6f", msg, sx, sy)
                elif args.strict_coordinate_check:
                    raise RuntimeError(msg)
                else:
                    logging.warning(msg)

        gt_points = gt_df[["x", "y"]].to_numpy(dtype=np.float64) if len(gt_df) > 0 else np.zeros((0, 2), dtype=np.float64)
        pred_points = pred_sub[["node_x", "node_y"]].to_numpy(dtype=np.float64)

        # Range check for coordinate sanity in image space (if dimensions available).
        if image_w is not None and image_h is not None:
            if len(gt_points) > 0:
                gt_oob = int(
                    np.sum(
                        (gt_points[:, 0] < 0)
                        | (gt_points[:, 0] >= float(image_w))
                        | (gt_points[:, 1] < 0)
                        | (gt_points[:, 1] >= float(image_h))
                    )
                )
                if gt_oob > 0:
                    msg = f"{source_image}: {gt_oob}/{len(gt_points)} GT points are out-of-bounds for image size {image_w}x{image_h}"
                    if args.strict_coordinate_check:
                        raise RuntimeError(msg)
                    logging.warning(msg)
            if len(pred_points) > 0:
                pred_oob = int(
                    np.sum(
                        (pred_points[:, 0] < 0)
                        | (pred_points[:, 0] >= float(image_w))
                        | (pred_points[:, 1] < 0)
                        | (pred_points[:, 1] >= float(image_h))
                    )
                )
                if pred_oob > 0:
                    logging.warning(
                        "%s: %d/%d prediction points are out-of-bounds for image size %dx%d",
                        source_image,
                        pred_oob,
                        len(pred_points),
                        image_w,
                        image_h,
                    )

        matches, unmatched_gt, unmatched_pred = match_points_ckdtree(
            gt_points=gt_points,
            pred_points=pred_points,
            radius=args.match_radius,
        )

        # Matched pairs.
        for gi, pj, dist in matches:
            detailed_records.append(
                {
                    "source_image": source_image,
                    "true_label": gt_labels[gi],
                    "pred_label": pred_labels[pj],
                    "status": "MATCHED",
                    "distance": dist,
                    "gt_x": float(gt_points[gi, 0]),
                    "gt_y": float(gt_points[gi, 1]),
                    "pred_x": float(pred_points[pj, 0]),
                    "pred_y": float(pred_points[pj, 1]),
                }
            )

        # Missed detections: GT nodes with no prediction within radius.
        for gi in unmatched_gt:
            detailed_records.append(
                {
                    "source_image": source_image,
                    "true_label": gt_labels[gi],
                    "pred_label": MISSED,
                    "status": "MISSED_DETECTION",
                    "distance": np.nan,
                    "gt_x": float(gt_points[gi, 0]),
                    "gt_y": float(gt_points[gi, 1]),
                    "pred_x": np.nan,
                    "pred_y": np.nan,
                }
            )

        # Unmatched predictions: false detections (no nearby GT).
        for pj in unmatched_pred:
            detailed_records.append(
                {
                    "source_image": source_image,
                    "true_label": NO_GT,
                    "pred_label": pred_labels[pj],
                    "status": "UNMATCHED_PREDICTION",
                    "distance": np.nan,
                    "gt_x": np.nan,
                    "gt_y": np.nan,
                    "pred_x": float(pred_points[pj, 0]),
                    "pred_y": float(pred_points[pj, 1]),
                }
            )

        print(
            f"{source_image}, {len(gt_points)}, {len(pred_points)}, {len(matches)}, "
            f"{len(unmatched_gt)}, {len(unmatched_pred)}"
        )

    if not detailed_records:
        raise RuntimeError("No evaluation records were generated. Check source-image mapping and inputs.")

    detailed_df = pd.DataFrame(detailed_records)
    detailed_df["true_label"] = detailed_df["true_label"].map(norm_label)
    detailed_df["pred_label"] = detailed_df["pred_label"].map(norm_label)

    # Global confusion matrix includes MISSED and NO_GT to expose misses/false detections.
    row_order = VALID_CLASSES + [NO_GT]
    col_order = VALID_CLASSES + [MISSED]

    confusion = pd.crosstab(
        detailed_df["true_label"],
        detailed_df["pred_label"],
        rownames=["True Label"],
        colnames=["Consensus Prediction"],
        dropna=False,
    )

    # Reindex for stable printable order.
    confusion = confusion.reindex(index=row_order, columns=col_order, fill_value=0)

    print("\nGlobal confusion matrix (includes NO_GT row and MISSED column)")
    print(confusion.to_string())

    # Metrics only for target classes N/T/X/Y.
    metrics_df = compute_class_metrics(
        y_true=detailed_df["true_label"].tolist(),
        y_pred=detailed_df["pred_label"].tolist(),
        classes=VALID_CLASSES,
    )

    print("\nPer-class Precision/Recall/F1")
    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
            },
        )
    )

    missed_total = int((detailed_df["status"] == "MISSED_DETECTION").sum())
    logging.info("Total missed detections (no nearby prediction): %d", missed_total)

    if args.save_detailed_csv is not None:
        args.save_detailed_csv.parent.mkdir(parents=True, exist_ok=True)
        detailed_df.to_csv(args.save_detailed_csv, index=False)
        logging.info("Saved detailed match table: %s", args.save_detailed_csv)


if __name__ == "__main__":
    main()
