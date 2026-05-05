#!/usr/bin/env python3
"""
run_improved_detection_benchmark.py

Use improved junction detection (degree≥3 + crossing-number + NMS) as proposals,
then classify each proposal with a trained ML model (CNN, XGB, RF, etc.).

This decouples detection from classification:
  - Detection: our improved topological method (high recall, no training needed)
  - Classification: trained ML model (good T/Y discrimination)

Usage:
    python run_improved_detection_benchmark.py \
        --images-dir data/evaluation_martian/images \
        --models-dir models \
        --heads cnn xgb rf \
        --output-csv predictions/improved_det_cnn.csv \
        --recrop-window 96
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


# ---------------------------------------------------------------------------
# Detection helpers (same as run_glyph_benchmark.py improved_graph)
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


def detect_junctions(
    image_u8: np.ndarray,
    nms_distance: int = 20,
    cn_radii: tuple = (8, 12),
    min_crossings: int = 3,
) -> list[dict]:
    """
    Find junction candidates: degree≥3 pixel clusters + crossing-number filter + NMS.
    Returns list of dicts with x, y, cn (crossing number), score.
    """
    binary = (image_u8 > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbr_count = ndimage.convolve(binary, kernel, mode="constant", cval=0)

    branch_mask = (binary > 0) & (nbr_count >= 3)
    labeled, n_clusters = ndimage.label(branch_mask, structure=np.ones((3, 3)))

    raw_dets = []
    for c in range(1, n_clusters + 1):
        cluster_pixels = np.argwhere(labeled == c)
        cy, cx = cluster_pixels.mean(axis=0)
        cns = [_crossing_number(binary, cy, cx, r) for r in cn_radii]
        max_cn = max(cns)
        if max_cn < min_crossings:
            continue
        score = max_cn * len(cluster_pixels)
        raw_dets.append({"x": float(cx), "y": float(cy), "cn": max_cn, "score": score})

    return _nms_by_score(raw_dets, nms_dist=nms_distance)


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

PATCH_H = PATCH_W = 96


def crop_patch(source_u8: np.ndarray, cx: float, cy: float, window: int) -> np.ndarray:
    """Crop a square patch centered at (cx, cy), resize to PATCH_H×PATCH_W."""
    win = max(1, int(window))
    half = float(win) / 2.0
    x0 = int(np.floor(float(cx) - half))
    y0 = int(np.floor(float(cy) - half))
    x1 = x0 + win
    y1 = y0 + win
    h, w = source_u8.shape
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        source_u8 = np.pad(
            source_u8,
            ((pad_t, pad_b), (pad_l, pad_r)),
            mode="constant",
            constant_values=0,
        )
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t
    patch = source_u8[y0:y1, x0:x1]
    if patch.shape != (PATCH_H, PATCH_W):
        patch = np.asarray(
            Image.fromarray(patch).resize((PATCH_W, PATCH_H), resample=Image.NEAREST),
            dtype=np.uint8,
        )
    return patch


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_cnn_model(model_path: Path, device: str = "cpu"):
    """Load a trained CNN model."""
    import torch
    from mars_tyxn import train_cnn

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
        config = checkpoint.get("config", {})
    else:
        state = checkpoint
        config = {}

    arch = config.get("arch", "deeper")
    in_channels = int(config.get("in_channels", 1))
    dropout = float(config.get("dropout", 0.3))

    # Map arch name to class
    arch_map = {
        "shallow": train_cnn.ShallowCNN_GAP,
        "deeper": train_cnn.DeeperCNN_GAP,
    }
    # Check for newer architectures
    for attr in ["DeeperCNN_Flatten_v2", "DeeperCNN_SPP_v2", "DeeperCNN_Attn_v2"]:
        cls = getattr(train_cnn, attr, None)
        if cls is not None:
            key = attr.replace("DeeperCNN_", "").lower()
            arch_map[key] = cls

    ModelClass = arch_map.get(arch)
    if ModelClass is None:
        raise RuntimeError(f"Unknown architecture '{arch}' in checkpoint")

    model = ModelClass(num_classes=4, in_channels=in_channels, dropout=dropout)
    model.load_state_dict(state)
    model.eval()
    model._in_channels = in_channels  # Store for later use
    return model, device


def classify_cnn(model, device, patches_f32: np.ndarray, in_channels: int = 2) -> list[str]:
    """Classify patches using CNN. patches_f32: (N, 96, 96) float32 [0,1]."""
    import torch
    from scipy.ndimage import binary_dilation

    LABEL_MAP = {0: "N", 1: "T", 2: "X", 3: "Y"}

    # Build multi-channel input
    batch = []
    for patch in patches_f32:
        skel = patch.astype(np.float32)
        mask = binary_dilation(skel > 0.5, structure=np.ones((3, 3), dtype=bool)).astype(np.float32)
        channels = [skel, mask]
        # Add more channels if needed
        while len(channels) < in_channels:
            channels.append(np.zeros_like(skel))
        channels = channels[:in_channels]
        batch.append(np.stack(channels, axis=0))

    batch_arr = np.array(batch)

    with torch.no_grad():
        tensor = torch.from_numpy(batch_arr).float().to(device)
        logits = model(tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
    return [LABEL_MAP[int(p)] for p in preds]


def load_classical_model(model_path: Path):
    """Load a trained classical model dict (contains pipeline, class_names, feature_spec)."""
    import joblib
    model_dict = joblib.load(model_path)
    return model_dict


def classify_classical(model_dict: dict, patches_f32: np.ndarray) -> list[str]:
    """Classify patches using a classical model pipeline.

    Uses build_classical_input_vector from classical_feature_builder to match
    exactly what was used during training.
    """
    from mars_tyxn.classical_feature_builder import build_classical_input_vector

    pipeline = model_dict["pipeline"]
    class_names = model_dict["class_names"]
    feature_regime = model_dict.get("feature_regime", "image_only")
    patch_size = int(model_dict.get("patch_size", PATCH_H))
    geometry_trace_len = int(model_dict.get("geometry_trace_len", 6))
    geometry_merge_deg = float(model_dict.get("geometry_merge_deg", 20.0))
    geometry_prefer_radius = float(model_dict.get("geometry_prefer_radius", 10.0))
    geometry_use_local_anchor = bool(model_dict.get("geometry_use_local_anchor", False))

    preds = []
    for patch in patches_f32:
        # Build the feature vector (patch pixels + geometry features)
        x_vec = build_classical_input_vector(
            patch_f32=patch,
            row={},  # No metadata row — geometry will use patch center as anchor
            feature_regime=feature_regime,
            patch_size=patch_size,
            geometry_trace_len=geometry_trace_len,
            geometry_merge_deg=geometry_merge_deg,
            geometry_prefer_radius=geometry_prefer_radius,
            geometry_use_local_anchor=geometry_use_local_anchor,
        )
        x = x_vec.reshape(1, -1).astype(np.float32, copy=False)
        pred = pipeline.predict(x)
        # Decode prediction
        raw = pred[0]
        if isinstance(raw, (int, np.integer)):
            label = class_names[int(raw)] if 0 <= int(raw) < len(class_names) else "N"
        else:
            label = str(raw).strip().upper()
            if label not in ("T", "Y", "X", "N"):
                label = "N"
        preds.append(label)

    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Improved detection + ML classification benchmark."
    )
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--image-pattern", type=str, default="*_skel.png")
    p.add_argument("--models-dir", type=Path, required=True,
                   help="Directory containing trained model files.")
    p.add_argument("--head", type=str, required=True,
                   choices=["cnn", "xgb", "rf", "svm", "mlp"],
                   help="Which model head to use for classification.")
    p.add_argument("--model-file", type=str, default=None,
                   help="Specific model filename. If not set, uses default naming.")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--recrop-window", type=int, default=96)
    p.add_argument("--nms-distance", type=int, default=20)
    p.add_argument("--keep-n", action="store_true",
                   help="Keep N predictions (don't filter them out). "
                        "If set, N predictions become Y (trusting detection).")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Find images
    images = sorted(args.images_dir.glob(args.image_pattern))
    if not images:
        raise FileNotFoundError(f"No images matching '{args.image_pattern}' in {args.images_dir}")
    logging.info("Found %d skeleton images", len(images))

    # Load model
    if args.model_file:
        model_path = args.models_dir / args.model_file
    else:
        default_names = {
            "cnn": "CNN_ft_gauss40.pt",
            "xgb": "XGB_32_d6.joblib",
            "rf": "RF_32.joblib",
            "svm": "SVM_32.joblib",
            "mlp": "MLP_32.joblib",
        }
        model_path = args.models_dir / default_names[args.head]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.head == "cnn":
        model, device = load_cnn_model(model_path)
        n_ch = getattr(model, "_in_channels", 2)
        classify_fn = lambda patches: classify_cnn(model, device, patches, in_channels=n_ch)
    else:
        model_dict = load_classical_model(model_path)
        classify_fn = lambda patches: classify_classical(model_dict, patches)

    logging.info("Loaded %s model from %s", args.head, model_path)

    # Process each image
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source_image", "node_x", "node_y", "consensus", "agreement"]
    all_rows = []

    for img_path in images:
        source_image = img_path.name
        logging.info("Processing: %s", source_image)

        skel_arr = np.array(Image.open(img_path).convert("L"))
        image_u8 = (skel_arr > 0).astype(np.uint8) * 255
        h, w = skel_arr.shape

        # Step 1: Detect junctions
        detections = detect_junctions(image_u8, nms_distance=args.nms_distance)
        logging.info("  Detected %d junction candidates", len(detections))

        if not detections:
            continue

        # Step 2: Crop patches
        patches = []
        valid_dets = []
        for det in detections:
            cx, cy_det = det["x"], det["y"]
            patch = crop_patch(image_u8, cx, cy_det, args.recrop_window)
            patches.append(patch.astype(np.float32) / 255.0)
            valid_dets.append(det)

        patches_arr = np.array(patches)

        # Step 3: Classify
        labels = classify_fn(patches_arr)

        # Step 4: Collect results
        img_rows = 0
        for det, label in zip(valid_dets, labels):
            if label == "N":
                if args.keep_n:
                    label = "Y"  # Trust detection, default to Y
                else:
                    continue  # Filter out N predictions
            node_x = int(np.clip(det["x"], 0, w - 1))
            node_y = int(np.clip(det["y"], 0, h - 1))
            all_rows.append({
                "source_image": source_image,
                "node_x": node_x,
                "node_y": node_y,
                "consensus": label,
                "agreement": 1,
            })
            img_rows += 1

        logging.info("  %s: %d detections -> %d classified junctions",
                     source_image, len(detections), img_rows)

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logging.info("Wrote %d predictions to %s", len(all_rows), args.output_csv)


if __name__ == "__main__":
    main()
