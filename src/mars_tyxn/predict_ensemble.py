#!/usr/bin/env python3
"""
predict_ensemble.py

Run ensemble inference on unlabelled 96x96 Martian junction patches.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
from tqdm import tqdm

from mars_tyxn.classical_feature_builder import (
    GEOMETRY_FEATURE_NAMES,
    build_classical_input_vector,
    normalize_feature_regime,
)
from mars_tyxn.junction_geometry import compute_patch_geometry
from mars_tyxn.meta_features import CLASS_NAMES as META_CLASS_NAMES, row_to_meta_features


DEFAULT_CLASS_NAMES = ["N", "T", "X", "Y"]
PATCH_H = 96
PATCH_W = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble inference on 96x96 Martian patches.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("inference_manifest_images.csv"),
        help="Input manifest CSV.",
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        default=Path("inference_patches/images"),
        help="Directory containing patch PNG files.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("final_ensemble_results.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--geometry-trace-len",
        type=int,
        default=6,
        help="Local branch trace length (px) used by geometry gate.",
    )
    parser.add_argument(
        "--cnn-context-col",
        type=str,
        default="context_image",
        help="Optional manifest column for extra CNN context channel.",
    )
    parser.add_argument(
        "--cnn-model-file",
        type=str,
        default="CNN_ft_gauss40.pt",
        help="CNN checkpoint filename inside --models-dir.",
    )
    parser.add_argument(
        "--cnn-source-image-dir",
        type=Path,
        default=None,
        help="Optional directory for source skeleton images used by --cnn-recrop-window.",
    )
    parser.add_argument(
        "--cnn-recrop-window",
        type=int,
        default=0,
        help=(
            "If >0, recrop this window size around (node_x,node_y) from source images "
            "and resize to 96x96 for CNN input."
        ),
    )
    parser.add_argument(
        "--classical-source-image-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for source skeleton images used by --classical-recrop-window. "
            "Defaults to --cnn-source-image-dir when not set."
        ),
    )
    parser.add_argument(
        "--classical-recrop-window",
        type=int,
        default=0,
        help=(
            "If >0, recrop this window size around (node_x,node_y) from source images "
            "and resize to 96x96 for MLP/SVM/XGB inputs."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--positive-only-output",
        action="store_true",
        help="If set, drop rows with final consensus=N from the written CSV.",
    )
    parser.add_argument(
        "--virtual-t-min-agreement-output",
        type=int,
        default=0,
        help="Optional post-filter: drop virtual-bridge rows with consensus=T and agreement below this value.",
    )
    parser.add_argument(
        "--drop-border-virtual-output",
        action="store_true",
        help="If set, drop virtual-bridge rows where border_flag is true.",
    )
    parser.add_argument(
        "--t-demotion-min-gap-floor",
        type=float,
        default=0.0,
        help="Stage-1 arbitration: demote T->Y when geometry_min_gap_deg is below this floor. 0 disables.",
    )
    parser.add_argument(
        "--y-rescue-max-gap",
        type=float,
        default=0.0,
        help="Stage-2 arbitration: promote Y->T when geometry_max_gap_deg exceeds this threshold. 0 disables.",
    )
    parser.add_argument(
        "--y-rescue-min-agreement",
        type=int,
        default=0,
        help="Minimum agreement required for Y->T rescue rules.",
    )
    parser.add_argument(
        "--y-rescue-vseg-unknown-b2",
        action="store_true",
        help="Enable proposal-aware rescue: vseg + geometry Unknown + branch_count=2 + agreement>=--y-rescue-min-agreement.",
    )
    parser.add_argument(
        "--t-endpoint-endpoint-mode",
        type=str,
        default="allow",
        choices=["allow", "rescue_only", "veto"],
        help="Policy for virtual_gap_endpoint_endpoint rows that are labeled T after arbitration.",
    )
    parser.add_argument(
        "--local-mixed-arbitration-radius",
        type=float,
        default=0.0,
        help="Cluster radius (px) for mixed T/Y arbitration. 0 disables.",
    )
    parser.add_argument(
        "--same-label-nms-radius",
        type=float,
        default=0.0,
        help="Optional same-label NMS radius (px) over positive predictions. 0 disables.",
    )
    parser.add_argument(
        "--rescue-min-gap-floor",
        type=float,
        default=0.0,
        help="Optional additional min-gap floor required by curved-T rescue signature. 0 disables.",
    )
    parser.add_argument(
        "--x-mode",
        type=str,
        default="veto",
        choices=["veto", "monitor", "enabled"],
        help="X handling mode: veto(default), monitor(veto but keep raw-X diagnostics), enabled(allow X through synthesis).",
    )
    parser.add_argument(
        "--label-head",
        type=str,
        default="ensemble",
        choices=[
            "ensemble",
            "cnn",
            "mlp",
            "svm",
            "xgb",
            "rf",
            "cnn_cascade",
            "mlp_cascade",
            "svm_cascade",
            "xgb_cascade",
            "rf_cascade",
            "meta",
        ],
        help="Final label source. Use a single head to avoid group-vote labeling.",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.50,
        help="Base positive gate threshold for *_cascade heads. Positive probability below this is rejected to N.",
    )
    parser.add_argument(
        "--virtual-gate-threshold",
        type=float,
        default=0.60,
        help="Minimum gate threshold for virtual proposals in *_cascade heads.",
    )
    parser.add_argument(
        "--border-virtual-gate-threshold",
        type=float,
        default=0.70,
        help="Minimum gate threshold for border-flagged virtual proposals in *_cascade heads.",
    )
    parser.add_argument(
        "--geometry-voter-mode",
        type=str,
        default="on",
        choices=["off", "on"],
        help="Optional T/Y geometric voter for *_cascade heads.",
    )
    parser.add_argument(
        "--geometry-t-min-gap-low",
        type=float,
        default=75.0,
        help="Geometric T vote lower bound for min gap (deg).",
    )
    parser.add_argument(
        "--geometry-t-min-gap-high",
        type=float,
        default=95.0,
        help="Geometric T vote upper bound for min gap (deg).",
    )
    parser.add_argument(
        "--geometry-t-max-gap-low",
        type=float,
        default=155.0,
        help="Geometric T vote lower bound for max gap (deg).",
    )
    parser.add_argument(
        "--geometry-t-max-gap-high",
        type=float,
        default=205.0,
        help="Geometric T vote upper bound for max gap (deg).",
    )
    parser.add_argument(
        "--geometry-y-min-gap",
        type=float,
        default=100.0,
        help="Geometric Y vote lower bound for min gap (deg).",
    )
    parser.add_argument(
        "--meta-model-path",
        type=Path,
        default=None,
        help="Optional learned meta-classifier joblib (train_meta_classifier.py).",
    )
    parser.add_argument(
        "--meta-min-confidence",
        type=float,
        default=0.0,
        help="If >0, only apply meta override when max meta probability >= this threshold.",
    )
    parser.add_argument(
        "--cluster-arbitration-radius",
        type=float,
        default=0.0,
        help="If >0, perform cluster-level arbitration and duplicate suppression within this radius (px).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def choose_device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_class_names(raw: Any) -> List[str]:
    if raw is None:
        return list(DEFAULT_CLASS_NAMES)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        vals = [str(x) for x in raw]
        return vals if vals else list(DEFAULT_CLASS_NAMES)
    return list(DEFAULT_CLASS_NAMES)


def normalize_idx_to_label(raw: Any) -> List[str]:
    if raw is None:
        return list(DEFAULT_CLASS_NAMES)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        labels = [str(x) for x in raw]
        return labels if labels else list(DEFAULT_CLASS_NAMES)
    if isinstance(raw, dict):
        idx_key_items: List[Tuple[int, str]] = []
        for k, v in raw.items():
            try:
                idx_key_items.append((int(k), str(v)))
            except Exception:
                idx_key_items = []
                break
        if idx_key_items:
            idx_key_items.sort(key=lambda kv: kv[0])
            return [label for _, label in idx_key_items]

        idx_val_items: List[Tuple[int, str]] = []
        for k, v in raw.items():
            try:
                idx_val_items.append((int(v), str(k)))
            except Exception:
                idx_val_items = []
                break
        if idx_val_items:
            idx_val_items.sort(key=lambda kv: kv[0])
            return [label for _, label in idx_val_items]
    return list(DEFAULT_CLASS_NAMES)


def maybe_strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def load_cnn_model(models_dir: Path, device: Any, filename: str = "CNN_ft_gauss40.pt") -> Tuple[Any, List[str], int]:
    import torch
    from mars_tyxn.train_cnn import (
        ShallowCNN_GAP, DeeperCNN_GAP, DeeperCNN_GAP_v2,
        DeeperCNN_Flatten_v2, DeeperCNN_SPP_v2, DeeperCNN_Attn_v2,
    )

    model_path = models_dir / str(filename)
    if not model_path.exists():
        raise FileNotFoundError(f"CNN checkpoint not found: {model_path}")

    payload = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        config = payload.get("config", {}) if isinstance(payload.get("config", {}), dict) else {}
        idx_to_label = normalize_idx_to_label(payload.get("idx_to_label"))
    else:
        state_dict = payload
        config = {}
        idx_to_label = list(DEFAULT_CLASS_NAMES)

    state_dict = maybe_strip_module_prefix(state_dict)
    arch = str(config.get("arch", "shallow")).strip().lower()
    in_channels = int(config.get("in_channels", 2))

    _arch_map = {
        "flatten_v2": DeeperCNN_Flatten_v2,
        "spp_v2": DeeperCNN_SPP_v2,
        "attn_v2": DeeperCNN_Attn_v2,
        "deeper_v2": DeeperCNN_GAP_v2,
    }
    if arch in _arch_map:
        dropout = float(config.get("dropout", 0.3))
        model = _arch_map[arch](
            num_classes=len(idx_to_label),
            in_channels=in_channels,
            dropout=dropout,
        )
    elif arch == "deeper":
        dropout = float(config.get("dropout", 0.3))
        model = DeeperCNN_GAP(
            num_classes=len(idx_to_label),
            in_channels=in_channels,
            dropout=dropout,
        )
    else:
        c1 = int(config.get("c1", 32))
        c2 = int(config.get("c2", 64))
        c3 = int(config.get("c3", 128))
        model = ShallowCNN_GAP(
            num_classes=len(idx_to_label),
            in_channels=in_channels,
            c1=c1,
            c2=c2,
            c3=c3,
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    logging.info(
        "Loaded CNN model: %s | arch=%s | classes=%s | in_channels=%d | device=%s",
        model_path.name,
        arch,
        idx_to_label,
        in_channels,
        device,
    )
    return model, idx_to_label, in_channels


def load_classical_model(models_dir: Path, filename: str, model_tag: str) -> Tuple[Any, List[str], Dict[str, Any]]:
    import joblib

    model_path = models_dir / filename
    if not model_path.exists():
        raise FileNotFoundError(f"{model_tag} model not found: {model_path}")

    payload = joblib.load(model_path)
    feature_spec: Dict[str, Any] = {
        "feature_regime": "image_only",
        "patch_size": int(PATCH_H),
        "geometry_trace_len": 6,
        "geometry_merge_deg": 20.0,
        "geometry_prefer_radius": 10.0,
        "geometry_use_local_anchor": False,
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
    }
    if isinstance(payload, dict) and "pipeline" in payload:
        pipeline = payload["pipeline"]
        class_names = normalize_class_names(payload.get("class_names"))
        if "feature_regime" in payload:
            feature_spec["feature_regime"] = normalize_feature_regime(payload.get("feature_regime"))
        if "patch_size" in payload:
            feature_spec["patch_size"] = int(payload.get("patch_size"))
        if "geometry_trace_len" in payload:
            feature_spec["geometry_trace_len"] = int(payload.get("geometry_trace_len"))
        if "geometry_merge_deg" in payload:
            feature_spec["geometry_merge_deg"] = float(payload.get("geometry_merge_deg"))
        if "geometry_prefer_radius" in payload:
            feature_spec["geometry_prefer_radius"] = float(payload.get("geometry_prefer_radius"))
        if "geometry_use_local_anchor" in payload:
            feature_spec["geometry_use_local_anchor"] = bool(payload.get("geometry_use_local_anchor"))
        gf = payload.get("geometry_feature_names")
        if isinstance(gf, (list, tuple)) and gf:
            feature_spec["geometry_feature_names"] = [str(v) for v in gf]
    else:
        pipeline = payload
        class_names = list(DEFAULT_CLASS_NAMES)
    feature_spec["feature_regime"] = normalize_feature_regime(feature_spec["feature_regime"])

    logging.info(
        "Loaded %s model: %s | classes=%s | regime=%s | geom_anchor=%s",
        model_tag,
        model_path.name,
        class_names,
        str(feature_spec["feature_regime"]),
        "local_x/local_y" if bool(feature_spec.get("geometry_use_local_anchor", False)) else "patch-center",
    )
    return pipeline, class_names, feature_spec


def load_meta_model(meta_path: Path | None) -> Dict[str, Any] | None:
    if meta_path is None:
        return None
    import joblib

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta model not found: {meta_path}")
    payload = joblib.load(meta_path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError(f"Unexpected meta model payload: {meta_path}")
    class_names = normalize_class_names(payload.get("class_names"))
    payload["class_names"] = class_names
    logging.info("Loaded meta model: %s | classes=%s", meta_path.name, class_names)
    return payload


def decode_classical_prediction(pred_value: Any, class_names: Sequence[str]) -> str:
    if isinstance(pred_value, str):
        if pred_value in class_names:
            return pred_value
        try:
            idx = int(pred_value)
            if 0 <= idx < len(class_names):
                return str(class_names[idx])
        except Exception:
            pass
        raise ValueError(f"Unrecognized string prediction: {pred_value}")

    try:
        idx = int(pred_value)
    except Exception as exc:
        raise ValueError(f"Could not interpret prediction as class index: {pred_value}") from exc
    if idx < 0 or idx >= len(class_names):
        raise IndexError(f"Predicted class index {idx} out of range for class_names={class_names}")
    return str(class_names[idx])


def load_patch_f32(patch_path: Path) -> np.ndarray:
    if not patch_path.exists():
        raise FileNotFoundError(f"Patch not found: {patch_path}")
    arr = np.asarray(Image.open(patch_path).convert("L"), dtype=np.float32)
    if arr.shape != (PATCH_H, PATCH_W):
        raise ValueError(f"Expected patch shape {(PATCH_H, PATCH_W)}, got {arr.shape} for {patch_path.name}")
    return arr / 255.0


def _crop_resize_from_source_u8(source_u8: np.ndarray, cx: float, cy: float, window: int) -> np.ndarray:
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
    return patch.astype(np.float32, copy=False) / 255.0


def maybe_recrop_patch(
    patch_f32: np.ndarray,
    row: Dict[str, str],
    recrop_window: int,
    source_dir: Path | None,
    source_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, bool]:
    if int(recrop_window) <= 0 or source_dir is None:
        return patch_f32, False

    source_name = str(row.get("source_image", "")).strip()
    if not source_name:
        return patch_f32, False

    try:
        cx = float(str(row.get("node_x", "")).strip())
        cy = float(str(row.get("node_y", "")).strip())
    except Exception:
        return patch_f32, False
    if not np.isfinite(cx) or not np.isfinite(cy):
        return patch_f32, False

    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = Path(source_dir) / source_path
    source_key = str(source_path)

    source_u8 = source_cache.get(source_key)
    if source_u8 is None:
        if not source_path.exists():
            return patch_f32, False
        source_u8 = np.asarray(Image.open(source_path).convert("L"), dtype=np.uint8)
        source_cache[source_key] = source_u8

    return _crop_resize_from_source_u8(source_u8, cx=cx, cy=cy, window=int(recrop_window)), True


def maybe_recrop_cnn_patch(
    patch_f32: np.ndarray,
    row: Dict[str, str],
    args: argparse.Namespace,
    source_cache: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, bool]:
    return maybe_recrop_patch(
        patch_f32=patch_f32,
        row=row,
        recrop_window=int(args.cnn_recrop_window),
        source_dir=args.cnn_source_image_dir,
        source_cache=source_cache,
    )


def _load_optional_context_channel(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if arr.shape != (PATCH_H, PATCH_W):
        return None
    return arr / 255.0


def _resolve_context_path(row: Dict[str, str], patch_dir: Path, preferred_col: str) -> Path | None:
    candidates = [preferred_col, "context_image", "context_path", "context_relpath"]
    seen = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        raw = str(row.get(col, "")).strip()
        if not raw:
            continue
        p = Path(raw)
        if p.is_absolute():
            return p
        return patch_dir / p
    return None


def build_cnn_input(
    patch_f32: np.ndarray,
    row: Dict[str, str],
    patch_dir: Path,
    expected_channels: int,
    context_col: str,
) -> np.ndarray:
    skel = patch_f32.astype(np.float32, copy=False)
    mask = binary_dilation(skel > 0.5, structure=np.ones((3, 3), dtype=bool)).astype(np.float32)
    channels: List[np.ndarray] = [skel, mask]

    if expected_channels >= 3:
        context_path = _resolve_context_path(row=row, patch_dir=patch_dir, preferred_col=context_col)
        context = _load_optional_context_channel(context_path) if context_path is not None else None
        if context is None:
            context = mask  # Backward-compatible fallback when no context evidence exists.
        channels.append(context.astype(np.float32, copy=False))

    while len(channels) < expected_channels:
        channels.append(np.zeros_like(skel, dtype=np.float32))
    if len(channels) > expected_channels:
        channels = channels[:expected_channels]
    return np.stack(channels, axis=0)


def _empty_prob_map(labels: Sequence[str]) -> Dict[str, float]:
    return {str(lbl): 0.0 for lbl in labels}


def _prob_map_from_vector(labels: Sequence[str], probs: Sequence[float]) -> Dict[str, float]:
    out = _empty_prob_map(labels)
    for i, lbl in enumerate(labels):
        p = float(probs[i]) if i < len(probs) else 0.0
        if np.isfinite(p):
            out[str(lbl)] = max(0.0, min(1.0, p))
    return out


def predict_cnn_with_proba(
    model: Any,
    idx_to_label: Sequence[str],
    patch_f32: np.ndarray,
    row: Dict[str, str],
    patch_dir: Path,
    expected_channels: int,
    context_col: str,
    device: Any,
) -> str:
    import torch

    stacked = build_cnn_input(
        patch_f32=patch_f32,
        row=row,
        patch_dir=patch_dir,
        expected_channels=expected_channels,
        context_col=context_col,
    )
    x = torch.from_numpy(stacked).to(torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        pred_idx = int(np.argmax(probs))
    if pred_idx < 0 or pred_idx >= len(idx_to_label):
        raise IndexError(f"CNN predicted index {pred_idx} out of range for labels {idx_to_label}")
    pred_label = str(idx_to_label[pred_idx])
    prob_map = _prob_map_from_vector(idx_to_label, probs)
    if pred_label not in prob_map:
        prob_map[pred_label] = 1.0
    return pred_label, prob_map


def predict_cnn(
    model: Any,
    idx_to_label: Sequence[str],
    patch_f32: np.ndarray,
    row: Dict[str, str],
    patch_dir: Path,
    expected_channels: int,
    context_col: str,
    device: Any,
) -> str:
    pred_label, _ = predict_cnn_with_proba(
        model=model,
        idx_to_label=idx_to_label,
        patch_f32=patch_f32,
        row=row,
        patch_dir=patch_dir,
        expected_channels=expected_channels,
        context_col=context_col,
        device=device,
    )
    return pred_label


def predict_classical_with_proba(
    pipeline: Any,
    class_names: Sequence[str],
    patch_f32: np.ndarray,
    row: Dict[str, str],
    feature_spec: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, float]]:
    spec = feature_spec or {}
    x_vec = build_classical_input_vector(
        patch_f32=patch_f32,
        row=row,
        feature_regime=spec.get("feature_regime", "image_only"),
        patch_size=int(spec.get("patch_size", PATCH_H)),
        geometry_trace_len=int(spec.get("geometry_trace_len", 6)),
        geometry_merge_deg=float(spec.get("geometry_merge_deg", 20.0)),
        geometry_prefer_radius=float(spec.get("geometry_prefer_radius", 10.0)),
        geometry_use_local_anchor=bool(spec.get("geometry_use_local_anchor", False)),
    )
    x = x_vec.reshape(1, -1).astype(np.float32, copy=False)
    pred = pipeline.predict(x)
    if len(pred) == 0:
        raise RuntimeError("Classical model returned empty prediction array.")
    pred_label = decode_classical_prediction(pred[0], class_names)

    prob_map = _empty_prob_map(class_names)
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(x)
            if probs is not None and len(probs) > 0:
                prob_map = _prob_map_from_vector(class_names, probs[0])
        except Exception:
            pass
    if pred_label not in prob_map:
        prob_map[pred_label] = 1.0
    if float(sum(prob_map.values())) <= 1e-6:
        prob_map[pred_label] = 1.0
    return pred_label, prob_map


def predict_classical(
    pipeline: Any,
    class_names: Sequence[str],
    patch_f32: np.ndarray,
    row: Dict[str, str],
    feature_spec: Dict[str, Any] | None = None,
) -> str:
    pred_label, _ = predict_classical_with_proba(
        pipeline=pipeline,
        class_names=class_names,
        patch_f32=patch_f32,
        row=row,
        feature_spec=feature_spec,
    )
    return pred_label


def ensemble_consensus(votes: Sequence[str]) -> Tuple[str, int]:
    counts = Counter(votes)
    consensus, agreement = counts.most_common(1)[0]
    return str(consensus), int(agreement)


def _second_best_class(votes: Sequence[str], exclude: str = "X") -> str:
    counts = Counter(votes)
    ranked = counts.most_common()
    alternatives = [label for label, _ in ranked if label != exclude]
    if alternatives:
        return str(alternatives[0])
    if "Y" in counts:
        return "Y"
    if "T" in counts:
        return "T"
    if "N" in counts:
        return "N"
    return "Y"


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_optional_int(value: Any) -> int | None:
    v = _parse_optional_float(value)
    if v is None:
        return None
    return int(round(v))


def _parse_bool_flag(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n", ""}:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return False


def resolve_proposal_metadata(row: Dict[str, str], patch_filename: str) -> Tuple[str, str]:
    proposal_source = str(row.get("proposal_source", "")).strip()
    proposal_type = str(row.get("proposal_type", "")).strip()
    if proposal_source:
        return proposal_source, proposal_type
    if proposal_type:
        if proposal_type.startswith("virtual_gap"):
            return "virtual_bridge", proposal_type
        return "base_topology", proposal_type
    if "_gap" in patch_filename:
        return "virtual_bridge", "virtual_gap_legacy_filename"
    return "base_topology", "base_junction_legacy"


def apply_final_synthesis_filter(
    votes: Sequence[str],
    consensus: str,
    agreement: int,
    geometry_label: str,
    proposal_source: str,
    proposal_type: str,
    border_flag: bool,
    mlp_pred: str,
    xgb_pred: str,
    x_mode: str,
) -> Tuple[str, int]:
    filtered = str(consensus)
    agreement_i = int(agreement)
    raw_consensus = str(consensus)
    trusted_virtual_t = False
    trusted_virtual_y = False
    x_mode_norm = str(x_mode).strip().lower()
    if x_mode_norm not in {"veto", "monitor", "enabled"}:
        x_mode_norm = "veto"

    # 1) Absolute X-veto.
    if x_mode_norm in {"veto", "monitor"} and filtered == "X":
        filtered = _second_best_class(votes, exclude="X")

    # 2) Geometry gate with arrowhead safeguard.
    if filtered in {"Y", "X"} and geometry_label == "T":
        filtered = "T"
    if filtered == "T" and geometry_label != "T":
        filtered = "Y"
    if geometry_label == "Y_arrowhead" and filtered == "T":
        filtered = "Y"

    # 3) Weighted trust, stricter for virtual-bridge proposals.
    is_virtual = proposal_source == "virtual_bridge" or proposal_type.startswith("virtual_gap")
    weak_agreement = agreement_i < 3

    if is_virtual:
        if filtered == "Y":
            trusted_virtual_y = xgb_pred == "Y" and agreement_i >= 2 and geometry_label != "T"
            if weak_agreement and not trusted_virtual_y:
                filtered = "N"
        elif filtered == "T":
            # Virtual T proposals are high-variance; require stronger multi-model support.
            trusted_virtual_t = mlp_pred == "T" and agreement_i >= 3 and geometry_label == "T"
            if weak_agreement and not trusted_virtual_t:
                filtered = "N"
        elif filtered == "X" and x_mode_norm in {"veto", "monitor"}:
            filtered = "N"

        allow_weak_virtual = (filtered == "T" and trusted_virtual_t) or (
            filtered == "Y" and trusted_virtual_y
        )
        if weak_agreement and filtered in {"Y", "T", "X"} and not allow_weak_virtual:
            filtered = "N"

    if border_flag and weak_agreement and filtered in {"Y", "T", "X"} and not trusted_virtual_t:
        filtered = "N"

    # Guard in X-veto mode: raw-X sites should not be rescued directly to T.
    if x_mode_norm in {"veto", "monitor"} and raw_consensus == "X" and filtered == "T":
        if "Y" in votes:
            filtered = "Y"
        elif "N" in votes:
            filtered = "N"
        else:
            filtered = "Y"

    # 4) Base-proposal Y precision guard:
    # base Y proposals are noisy unless XGB agrees on Y.
    if proposal_source == "base_topology" and filtered == "Y" and xgb_pred != "Y":
        filtered = "N"

    # 5) T rescue for rare unanimous-T vote cases that geometry may miss.
    if filtered == "Y" and all(v == "T" for v in votes):
        filtered = "T"

    return filtered, agreement_i


def read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"patch_filename", "source_image", "node_x", "node_y"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        return [row for row in reader]


def write_results_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
        "geometry_label",
        "geometry_branch_count",
        "geometry_min_gap_deg",
        "geometry_max_gap_deg",
        "geometry_anchor_x",
        "geometry_anchor_y",
        "cnn_pred",
        "mlp_pred",
        "svm_pred",
        "xgb_pred",
        "rf_pred",
        "cnn_prob_N",
        "cnn_prob_T",
        "cnn_prob_X",
        "cnn_prob_Y",
        "mlp_prob_N",
        "mlp_prob_T",
        "mlp_prob_X",
        "mlp_prob_Y",
        "svm_prob_N",
        "svm_prob_T",
        "svm_prob_X",
        "svm_prob_Y",
        "xgb_prob_N",
        "xgb_prob_T",
        "xgb_prob_X",
        "xgb_prob_Y",
        "rf_prob_N",
        "rf_prob_T",
        "rf_prob_X",
        "rf_prob_Y",
        "avg_prob_N",
        "avg_prob_T",
        "avg_prob_X",
        "avg_prob_Y",
        "meta_pred",
        "meta_confidence",
        "meta_prob_N",
        "meta_prob_T",
        "meta_prob_X",
        "meta_prob_Y",
        "gate_source_head",
        "gate_prob_positive",
        "gate_threshold_used",
        "gate_decision",
        "geometry_vote",
        "geometry_vote_applied",
        "cluster_id",
        "cluster_size",
        "cluster_label",
        "cluster_score",
        "consensus",
        "agreement",
        "raw_consensus",
        "raw_agreement",
        "raw_x_votes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_output_row_filters(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Optional output-stage filters; disabled by default for backward compatibility."""
    keep_rows: List[Dict[str, Any]] = []
    dropped_n = 0
    dropped_virtual_t = 0
    dropped_virtual_border = 0

    vt_min = max(0, int(args.virtual_t_min_agreement_output))

    for row in rows:
        consensus = str(row.get("consensus", "")).strip().upper()
        proposal_source = str(row.get("proposal_source", "")).strip()
        proposal_type = str(row.get("proposal_type", "")).strip()
        is_virtual = proposal_source == "virtual_bridge" or proposal_type.startswith("virtual_gap")

        if args.positive_only_output and consensus == "N":
            dropped_n += 1
            continue

        if vt_min > 0 and is_virtual and consensus == "T":
            agreement = _parse_optional_int(row.get("agreement"))
            agreement_i = int(agreement) if agreement is not None else 0
            if agreement_i < vt_min:
                dropped_virtual_t += 1
                continue

        if args.drop_border_virtual_output and is_virtual and _parse_bool_flag(row.get("border_flag")):
            dropped_virtual_border += 1
            continue

        keep_rows.append(row)

    if args.positive_only_output or vt_min > 0 or args.drop_border_virtual_output:
        logging.info(
            "Output filters: kept=%d dropped_n=%d dropped_virtual_t=%d dropped_virtual_border=%d",
            len(keep_rows),
            dropped_n,
            dropped_virtual_t,
            dropped_virtual_border,
        )

    return keep_rows


def _row_consensus(row: Dict[str, Any]) -> str:
    return str(row.get("consensus", "")).strip().upper()


def _row_is_virtual(row: Dict[str, Any]) -> bool:
    proposal_source = str(row.get("proposal_source", "")).strip()
    proposal_type = str(row.get("proposal_type", "")).strip()
    return proposal_source == "virtual_bridge" or proposal_type.startswith("virtual_gap")


def _row_agreement_int(row: Dict[str, Any]) -> int:
    v = _parse_optional_int(row.get("agreement"))
    return int(v) if v is not None else 0


def _row_coord_xy(row: Dict[str, Any]) -> Tuple[float, float] | None:
    x = _parse_optional_float(row.get("node_x"))
    y = _parse_optional_float(row.get("node_y"))
    if x is None or y is None:
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def _row_proposal_score(row: Dict[str, Any]) -> float:
    v = _parse_optional_float(row.get("proposal_score"))
    if v is None or not np.isfinite(v):
        return 0.0
    return float(v)


def _passes_curved_t_rescue_signature(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    max_gap_thr = float(args.y_rescue_max_gap)
    min_agree = max(0, int(args.y_rescue_min_agreement))

    if max_gap_thr <= 0.0:
        return False
    if _row_agreement_int(row) < min_agree:
        return False

    max_gap = _parse_optional_float(row.get("geometry_max_gap_deg"))
    if max_gap is None or max_gap <= max_gap_thr:
        return False

    min_gap_floor = float(args.rescue_min_gap_floor)
    if min_gap_floor > 0.0:
        min_gap = _parse_optional_float(row.get("geometry_min_gap_deg"))
        if min_gap is None or min_gap < min_gap_floor:
            return False

    return True


def _passes_vseg_unknown_b2_rescue(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(args.y_rescue_vseg_unknown_b2):
        return False
    if str(row.get("proposal_type", "")).strip() != "virtual_gap_endpoint_segment":
        return False
    if str(row.get("geometry_label", "")).strip() != "Unknown":
        return False
    branch_count = _parse_optional_int(row.get("geometry_branch_count"))
    if branch_count != 2:
        return False
    min_agree = max(0, int(args.y_rescue_min_agreement))
    return _row_agreement_int(row) >= min_agree


def apply_local_t_y_arbitration(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Two-stage T gate + optional proposal-type-aware endpoint-endpoint policy."""
    demoted_t = 0
    rescued_y = 0
    rescued_y_vseg = 0
    endpoint_demotions = 0

    min_gap_floor = float(args.t_demotion_min_gap_floor)
    for row in rows:
        label = _row_consensus(row)
        if label == "T" and min_gap_floor > 0.0:
            min_gap = _parse_optional_float(row.get("geometry_min_gap_deg"))
            if min_gap is None or min_gap < min_gap_floor:
                row["consensus"] = "Y"
                demoted_t += 1
                label = "Y"

        if label == "Y":
            rescue_a = _passes_curved_t_rescue_signature(row, args)
            rescue_b = False
            if not rescue_a:
                rescue_b = _passes_vseg_unknown_b2_rescue(row, args)
            if rescue_a or rescue_b:
                row["consensus"] = "T"
                rescued_y += 1
                if rescue_b:
                    rescued_y_vseg += 1

    mode = str(args.t_endpoint_endpoint_mode).strip().lower()
    if mode in {"rescue_only", "veto"}:
        for row in rows:
            if _row_consensus(row) != "T":
                continue
            if str(row.get("proposal_type", "")).strip() != "virtual_gap_endpoint_endpoint":
                continue
            if mode == "rescue_only" and _passes_curved_t_rescue_signature(row, args):
                continue
            row["consensus"] = "Y"
            endpoint_demotions += 1

    if min_gap_floor > 0.0 or float(args.y_rescue_max_gap) > 0.0 or bool(args.y_rescue_vseg_unknown_b2) or mode != "allow":
        logging.info(
            "Local T/Y arbitration: demoted_t=%d rescued_y=%d rescued_y_vseg=%d endpoint_demotions=%d mode=%s",
            demoted_t,
            rescued_y,
            rescued_y_vseg,
            endpoint_demotions,
            mode,
        )


def apply_mixed_label_cluster_arbitration(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    """Within local mixed T/Y clusters, prefer Y unless T passes curved-T rescue signature."""
    radius = float(args.local_mixed_arbitration_radius)
    if radius <= 0.0:
        return

    by_image: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        label = _row_consensus(row)
        if label not in {"T", "Y"}:
            continue
        if _row_coord_xy(row) is None:
            continue
        src = str(row.get("source_image", ""))
        by_image.setdefault(src, []).append(idx)

    changed = 0
    mixed_clusters = 0
    total_clusters = 0

    for _, indices in by_image.items():
        if len(indices) <= 1:
            continue
        pts = np.asarray([_row_coord_xy(rows[i]) for i in indices], dtype=np.float64)
        tree = cKDTree(pts)
        neighbors = tree.query_ball_tree(tree, r=radius)

        visited = np.zeros(len(indices), dtype=bool)
        for seed in range(len(indices)):
            if visited[seed]:
                continue
            stack = [seed]
            comp_local: List[int] = []
            visited[seed] = True
            while stack:
                cur = stack.pop()
                comp_local.append(cur)
                for nxt in neighbors[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)

            if len(comp_local) <= 1:
                continue
            total_clusters += 1
            comp_global = [indices[j] for j in comp_local]
            labels = {_row_consensus(rows[g]) for g in comp_global}
            if not ({"T", "Y"} <= labels):
                continue
            mixed_clusters += 1
            for g in comp_global:
                if _row_consensus(rows[g]) != "T":
                    continue
                if _passes_curved_t_rescue_signature(rows[g], args):
                    continue
                rows[g]["consensus"] = "Y"
                changed += 1

    logging.info(
        "Mixed-label arbitration: radius=%.1f total_clusters=%d mixed_clusters=%d t_to_y=%d",
        radius,
        total_clusters,
        mixed_clusters,
        changed,
    )


def apply_same_label_nms(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Greedy same-label NMS by (agreement desc, proposal_score desc)."""
    radius = float(args.same_label_nms_radius)
    if radius <= 0.0:
        return list(rows)

    r2 = radius * radius
    suppress = np.zeros(len(rows), dtype=bool)
    by_key: Dict[Tuple[str, str], List[int]] = {}

    for idx, row in enumerate(rows):
        label = _row_consensus(row)
        if label not in {"T", "Y", "X"}:
            continue
        if _row_coord_xy(row) is None:
            continue
        src = str(row.get("source_image", ""))
        by_key.setdefault((src, label), []).append(idx)

    suppressed_count = 0
    for _, indices in by_key.items():
        ranked = sorted(
            indices,
            key=lambda i: (
                _row_agreement_int(rows[i]),
                _row_proposal_score(rows[i]),
            ),
            reverse=True,
        )
        kept_xy: List[Tuple[float, float]] = []
        for idx in ranked:
            if suppress[idx]:
                continue
            xy = _row_coord_xy(rows[idx])
            if xy is None:
                continue
            x, y = xy
            too_close = False
            for kx, ky in kept_xy:
                dx = x - kx
                dy = y - ky
                if (dx * dx + dy * dy) <= r2:
                    too_close = True
                    break
            if too_close:
                suppress[idx] = True
                suppressed_count += 1
                continue
            kept_xy.append((x, y))

    kept_rows = [row for i, row in enumerate(rows) if not suppress[i]]
    logging.info("Same-label NMS: radius=%.1f suppressed=%d kept=%d", radius, suppressed_count, len(kept_rows))
    return kept_rows


def _choose_label_by_prob(prob_map: Dict[str, float], x_mode: str) -> Tuple[str, float]:
    mode = str(x_mode).strip().lower()
    ranked = sorted(prob_map.items(), key=lambda kv: float(kv[1]), reverse=True)
    if not ranked:
        return "N", 0.0
    if mode in {"veto", "monitor"}:
        ranked_non_x = [(k, v) for k, v in ranked if k != "X"]
        if ranked_non_x:
            return str(ranked_non_x[0][0]), float(ranked_non_x[0][1])
    return str(ranked[0][0]), float(ranked[0][1])


def apply_meta_classifier(
    rows: Sequence[Dict[str, Any]],
    meta_payload: Dict[str, Any],
    args: argparse.Namespace,
    force_override: bool = False,
) -> None:
    if not rows:
        return
    pipe = meta_payload["pipeline"]
    class_names = normalize_class_names(meta_payload.get("class_names"))
    feat_rows = [row_to_meta_features(r) for r in rows]
    X_df = pd.DataFrame(feat_rows).fillna(0.0)
    try:
        prob = pipe.predict_proba(X_df)
        pred = pipe.predict(X_df)
    except Exception as exc:
        logging.error("Meta classifier failed; skipping meta override: %s", exc)
        return

    label_to_col = {str(lbl): i for i, lbl in enumerate(class_names)}
    min_conf = max(0.0, min(1.0, float(args.meta_min_confidence)))
    changed = 0
    for i, row in enumerate(rows):
        pmap = {cls: float(prob[i, label_to_col[cls]]) if cls in label_to_col else 0.0 for cls in META_CLASS_NAMES}
        meta_label_raw = str(pred[i])
        if meta_label_raw not in pmap:
            meta_label_raw, _ = _choose_label_by_prob(pmap, x_mode="enabled")
        meta_label, meta_conf = _choose_label_by_prob(pmap, x_mode=str(args.x_mode))

        row["meta_pred"] = meta_label
        row["meta_confidence"] = meta_conf
        for cls in META_CLASS_NAMES:
            row[f"meta_prob_{cls}"] = pmap.get(cls, 0.0)

        if (force_override or meta_conf >= min_conf) and _row_consensus(row) not in {"ERROR"}:
            if _row_consensus(row) != meta_label:
                changed += 1
            row["consensus"] = meta_label
            # Interpretability only: map confidence to pseudo-support out of 4 models.
            row["agreement"] = max(1, int(round(meta_conf * 4.0)))
    logging.info("Meta classifier override: rows=%d changed=%d min_conf=%.2f", len(rows), changed, min_conf)


def apply_cluster_arbitration(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    radius = float(args.cluster_arbitration_radius)
    if radius <= 0.0 or not rows:
        return

    by_image: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        xy = _row_coord_xy(row)
        if xy is None:
            continue
        src = str(row.get("source_image", ""))
        by_image.setdefault(src, []).append(i)
        row["cluster_id"] = ""
        row["cluster_size"] = ""
        row["cluster_label"] = ""
        row["cluster_score"] = ""

    cluster_count = 0
    suppressed = 0
    for src, indices in by_image.items():
        if len(indices) <= 1:
            idx = indices[0]
            rows[idx]["cluster_id"] = f"{src}:0"
            rows[idx]["cluster_size"] = 1
            rows[idx]["cluster_label"] = _row_consensus(rows[idx])
            rows[idx]["cluster_score"] = _parse_optional_float(rows[idx].get(f"meta_prob_{_row_consensus(rows[idx])}")) or 0.0
            continue
        pts = np.asarray([_row_coord_xy(rows[i]) for i in indices], dtype=np.float64)
        tree = cKDTree(pts)
        nbr = tree.query_ball_tree(tree, r=radius)
        visited = np.zeros(len(indices), dtype=bool)
        comp_id = 0
        for seed in range(len(indices)):
            if visited[seed]:
                continue
            stack = [seed]
            comp: List[int] = []
            visited[seed] = True
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nxt in nbr[cur]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        stack.append(nxt)

            global_ids = [indices[j] for j in comp]
            cid = f"{src}:{comp_id}"
            comp_id += 1
            cluster_count += 1

            # Cluster label by mean meta probabilities if present, else mean model probs.
            score_by_cls: Dict[str, float] = {}
            for cls in META_CLASS_NAMES:
                vals: List[float] = []
                for gi in global_ids:
                    v = _parse_optional_float(rows[gi].get(f"meta_prob_{cls}"))
                    if v is None:
                        v = _parse_optional_float(rows[gi].get(f"avg_prob_{cls}"))
                    if v is not None and np.isfinite(v):
                        vals.append(float(v))
                score_by_cls[cls] = float(np.mean(vals)) if vals else 0.0
            cluster_label, cluster_score = _choose_label_by_prob(score_by_cls, x_mode=str(args.x_mode))

            # Keep best-support row for cluster label; suppress others to N.
            best_idx = global_ids[0]
            best_val = -1.0
            for gi in global_ids:
                rv = _parse_optional_float(rows[gi].get(f"meta_prob_{cluster_label}"))
                if rv is None:
                    rv = _parse_optional_float(rows[gi].get(f"avg_prob_{cluster_label}"))
                if rv is None:
                    rv = float(_row_agreement_int(rows[gi])) / 4.0
                if float(rv) > best_val:
                    best_val = float(rv)
                    best_idx = gi

            for gi in global_ids:
                rows[gi]["cluster_id"] = cid
                rows[gi]["cluster_size"] = len(global_ids)
                rows[gi]["cluster_label"] = cluster_label
                rows[gi]["cluster_score"] = cluster_score
                if gi == best_idx:
                    rows[gi]["consensus"] = cluster_label
                    rows[gi]["agreement"] = max(1, int(round(cluster_score * 4.0)))
                else:
                    if _row_consensus(rows[gi]) != "N":
                        suppressed += 1
                    rows[gi]["consensus"] = "N"
                    rows[gi]["agreement"] = 0
    logging.info(
        "Cluster arbitration: radius=%.1f clusters=%d suppressed_to_N=%d",
        radius,
        cluster_count,
        suppressed,
    )


def _row_head_prob_map(row: Dict[str, Any], head: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cls in META_CLASS_NAMES:
        key = f"{head}_prob_{cls}"
        v = _parse_optional_float(row.get(key))
        out[cls] = float(v) if v is not None and np.isfinite(v) else 0.0
    return out


def _positive_gate_prob_from_map(prob_map: Dict[str, float]) -> float:
    # Dedicated gate models may expose positive class under these aliases.
    for key in ("P", "POS", "POSITIVE", "J", "JUNCTION"):
        if key in prob_map:
            return float(max(0.0, min(1.0, prob_map[key])))

    n_prob = float(prob_map.get("N", 0.0))
    non_n_prob = float(sum(v for k, v in prob_map.items() if k != "N"))
    if non_n_prob > 0.0:
        return float(max(0.0, min(1.0, non_n_prob)))
    return float(max(0.0, min(1.0, 1.0 - n_prob)))


def _cascade_gate_threshold(row: Dict[str, Any], args: argparse.Namespace) -> float:
    thr = float(args.gate_threshold)
    if _row_is_virtual(row):
        thr = max(thr, float(args.virtual_gate_threshold))
        if _parse_bool_flag(row.get("border_flag")):
            thr = max(thr, float(args.border_virtual_gate_threshold))
    return float(max(0.0, min(1.0, thr)))


def _choose_type_label_from_probs(prob_map: Dict[str, float], x_mode: str) -> Tuple[str, float]:
    mode = str(x_mode).strip().lower()
    labels = ["T", "Y", "X"]
    if mode in {"veto", "monitor"}:
        labels = ["T", "Y"]
    ranked = sorted(((lbl, float(prob_map.get(lbl, 0.0))) for lbl in labels), key=lambda kv: kv[1], reverse=True)
    if not ranked:
        return "Y", 0.0
    return str(ranked[0][0]), float(ranked[0][1])


def _geometry_ty_vote(row: Dict[str, Any], args: argparse.Namespace) -> str | None:
    try:
        branch_count = _parse_optional_int(row.get("geometry_branch_count"))
        min_gap = _parse_optional_float(row.get("geometry_min_gap_deg"))
        max_gap = _parse_optional_float(row.get("geometry_max_gap_deg"))
    except Exception:
        return None
    if branch_count != 3 or min_gap is None or max_gap is None:
        return None

    t_min_low = float(args.geometry_t_min_gap_low)
    t_min_high = float(args.geometry_t_min_gap_high)
    t_max_low = float(args.geometry_t_max_gap_low)
    t_max_high = float(args.geometry_t_max_gap_high)
    y_min = float(args.geometry_y_min_gap)

    if t_max_low <= max_gap <= t_max_high and t_min_low <= min_gap <= t_min_high:
        return "T"
    if min_gap >= y_min:
        return "Y"
    return None


def apply_single_head_labeling(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> None:
    label_head = str(args.label_head).strip().lower()
    is_cascade = label_head.endswith("_cascade")
    head = label_head[:-8] if is_cascade else label_head
    if head not in {"cnn", "mlp", "svm", "xgb", "rf"}:
        return
    changed = 0
    cascade_rejects = 0
    cascade_accepts = 0
    geometry_overrides = 0
    geom_voter_enabled = is_cascade and str(args.geometry_voter_mode).strip().lower() == "on"
    for row in rows:
        pred = str(row.get(f"{head}_pred", "")).strip()
        prob_map = _row_head_prob_map(row, head=head)
        row["geometry_vote"] = ""
        row["geometry_vote_applied"] = ""
        if is_cascade:
            precomputed_pos = _parse_optional_float(row.get("gate_prob_positive"))
            precomputed_head = str(row.get("gate_source_head", "")).strip().lower()
            if precomputed_pos is not None and precomputed_head == head:
                pos_prob = float(max(0.0, min(1.0, precomputed_pos)))
            else:
                pos_prob = _positive_gate_prob_from_map(prob_map)
            gate_thr = _cascade_gate_threshold(row, args)
            if pos_prob < gate_thr:
                label = "N"
                cascade_rejects += 1
                row["gate_decision"] = "reject"
            else:
                label, _ = _choose_type_label_from_probs(prob_map=prob_map, x_mode=str(args.x_mode))
                if label == "" or label not in {"T", "Y", "X"}:
                    if pred in {"T", "Y", "X"}:
                        label = pred
                    else:
                        label = "Y"
                cascade_accepts += 1
                row["gate_decision"] = "accept"
            row["gate_source_head"] = head
            row["gate_prob_positive"] = pos_prob
            row["gate_threshold_used"] = gate_thr

            if geom_voter_enabled:
                geom_vote = _geometry_ty_vote(row=row, args=args)
                if (
                    geom_vote in {"T", "Y"}
                    and str(row.get("gate_decision", "")).strip().lower() == "accept"
                    and label in {"T", "Y"}
                ):
                    row["geometry_vote"] = geom_vote
                    if label != geom_vote:
                        label = geom_vote
                        row["geometry_vote_applied"] = "override"
                        geometry_overrides += 1
        else:
            if not pred:
                continue
            label, _ = _choose_label_by_prob(prob_map=prob_map, x_mode=str(args.x_mode))
            if label == "" or label not in {"N", "T", "X", "Y"}:
                label = pred
            row["gate_source_head"] = ""
            row["gate_prob_positive"] = ""
            row["gate_threshold_used"] = ""
            row["gate_decision"] = ""
        if _row_consensus(row) != label:
            changed += 1
        row["consensus"] = label
        row["raw_consensus"] = pred if pred else label
        row["raw_agreement"] = 1
        row["agreement"] = 0 if label == "N" else 1
        row["raw_x_votes"] = int((pred if pred else label) == "X")
    if is_cascade:
        logging.info(
            "Single-head cascade labeling applied: head=%s changed=%d accepted=%d rejected=%d geom_overrides=%d",
            head,
            changed,
            cascade_accepts,
            cascade_rejects,
            geometry_overrides,
        )
    else:
        logging.info("Single-head labeling applied: head=%s changed=%d", head, changed)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    device = choose_device()
    logging.info("Using device: %s", device)
    label_head_norm = str(args.label_head).strip().lower()
    cnn_affects_output = label_head_norm in {"ensemble", "meta", "cnn", "cnn_cascade"}
    classical_affects_output = label_head_norm in {
        "ensemble",
        "meta",
        "mlp",
        "svm",
        "xgb",
        "rf",
        "mlp_cascade",
        "svm_cascade",
        "xgb_cascade",
        "rf_cascade",
    }
    effective_classical_window = int(args.classical_recrop_window)
    effective_classical_source_dir = args.classical_source_image_dir
    if effective_classical_window <= 0 and int(args.cnn_recrop_window) > 0:
        effective_classical_window = int(args.cnn_recrop_window)
        if effective_classical_source_dir is None:
            effective_classical_source_dir = args.cnn_source_image_dir

    if int(args.cnn_recrop_window) > 0 and args.cnn_source_image_dir is None and cnn_affects_output:
        logging.warning(
            "CNN recrop requested (window=%d) but --cnn-source-image-dir is unset; recrop will not be applied.",
            int(args.cnn_recrop_window),
        )
    if effective_classical_window > 0 and effective_classical_source_dir is None and classical_affects_output:
        logging.warning(
            "Classical recrop requested (window=%d) but source image dir is unset; classical recrop will not be applied.",
            effective_classical_window,
        )

    if cnn_affects_output:
        cnn_model, cnn_idx_to_label, cnn_in_channels = load_cnn_model(
            args.models_dir,
            device,
            filename=str(args.cnn_model_file),
        )
    else:
        cnn_model = None
        cnn_idx_to_label = []
        cnn_in_channels = 0
    def _try_load_classical(filename: str, tag: str):
        path = args.models_dir / filename
        if path.exists():
            return load_classical_model(args.models_dir, filename, tag)
        return None, [], {}

    if classical_affects_output:
        mlp_model, mlp_class_names, mlp_feature_spec = _try_load_classical("MLP_32.joblib", "MLP")
        svm_model, svm_class_names, svm_feature_spec = _try_load_classical("SVM_32.joblib", "SVM")
        xgb_model, xgb_class_names, xgb_feature_spec = _try_load_classical("XGB_32_d6.joblib", "XGB")
        rf_model, rf_class_names, rf_feature_spec = _try_load_classical("RF_32.joblib", "RF")
    else:
        mlp_model = svm_model = xgb_model = rf_model = None
        mlp_class_names = svm_class_names = xgb_class_names = rf_class_names = []
        mlp_feature_spec = svm_feature_spec = xgb_feature_spec = rf_feature_spec = {}
    meta_model = load_meta_model(args.meta_model_path) if label_head_norm == "meta" else None
    cascade_selected_head = label_head_norm[:-8] if label_head_norm.endswith("_cascade") else ""
    cascade_gate_model: Any = None
    cascade_type_model: Any = None
    cascade_gate_class_names: List[str] = []
    cascade_type_class_names: List[str] = []
    cascade_gate_feature_spec: Dict[str, Any] = {}
    cascade_type_feature_spec: Dict[str, Any] = {}
    cascade_gate_idx_to_label: List[str] = []
    cascade_type_idx_to_label: List[str] = []
    cascade_gate_in_channels = 0
    cascade_type_in_channels = 0
    cnn_source_cache: Dict[str, np.ndarray] = {}
    cnn_recrop_total = 0
    cnn_recrop_used = 0
    classical_recrop_total = 0
    classical_recrop_used = 0

    if cascade_selected_head in {"mlp", "svm", "xgb"}:
        prefix = cascade_selected_head.upper()
        gate_path = args.models_dir / f"{prefix}_Martian_GATE.joblib"
        type_path = args.models_dir / f"{prefix}_Martian_TYPE.joblib"
        if gate_path.exists() and type_path.exists():
            cascade_gate_model, cascade_gate_class_names, cascade_gate_feature_spec = load_classical_model(
                args.models_dir, gate_path.name, f"{prefix} gate"
            )
            cascade_type_model, cascade_type_class_names, cascade_type_feature_spec = load_classical_model(
                args.models_dir, type_path.name, f"{prefix} type"
            )
            logging.info("Using specialized cascade models for %s", cascade_selected_head)
        elif cascade_selected_head:
            logging.info("Specialized cascade models not found for %s; using multiclass probabilities fallback", cascade_selected_head)
    elif cascade_selected_head == "cnn":
        gate_ckpt = args.models_dir / "CNN_ft_gauss40_GATE.pt"
        type_ckpt = args.models_dir / "CNN_ft_gauss40_TYPE.pt"
        if gate_ckpt.exists() and type_ckpt.exists():
            cascade_gate_model, cascade_gate_idx_to_label, cascade_gate_in_channels = load_cnn_model(
                args.models_dir, device, filename=gate_ckpt.name
            )
            cascade_type_model, cascade_type_idx_to_label, cascade_type_in_channels = load_cnn_model(
                args.models_dir, device, filename=type_ckpt.name
            )
            logging.info("Using specialized cascade models for cnn")
        else:
            logging.info("Specialized cascade models not found for cnn; using multiclass probabilities fallback")

    rows = read_manifest_rows(args.manifest)
    logging.info("Loaded manifest rows: %d", len(rows))
    local_anchor_rows = 0
    off_center_rows = 0
    for row in rows:
        lx = _parse_optional_float(row.get("local_x"))
        ly = _parse_optional_float(row.get("local_y"))
        if lx is None or ly is None:
            continue
        local_anchor_rows += 1
        if abs(float(lx) - 48.0) > 4.0 or abs(float(ly) - 48.0) > 4.0:
            off_center_rows += 1
    if local_anchor_rows > 0:
        off_ratio = float(off_center_rows) / float(local_anchor_rows)
        logging.info(
            "Manifest anchor spread: local_xy_rows=%d off_center_ratio(|d|>4)=%.3f",
            local_anchor_rows,
            off_ratio,
        )
        if int(args.cnn_recrop_window) <= 0 and off_ratio >= 0.50 and cnn_affects_output:
            logging.warning(
                "CNN recrop disabled with high off-center anchors (%.1f%% off-center). "
                "This is a known source of CNN N-collapse on proposal-jittered manifests.",
                off_ratio * 100.0,
            )
        if effective_classical_window <= 0 and off_ratio >= 0.50 and classical_affects_output:
            logging.warning(
                "Classical recrop disabled with high off-center anchors (%.1f%% off-center). "
                "This can suppress MLP/SVM/XGB precision/recall on proposal-jittered manifests.",
                off_ratio * 100.0,
            )

    out_rows: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Ensemble inference", unit="patch"):
        patch_filename = row["patch_filename"]
        patch_path = args.patch_dir / patch_filename

        proposal_source, proposal_type = resolve_proposal_metadata(row=row, patch_filename=patch_filename)
        result: Dict[str, Any] = {
            "patch_filename": patch_filename,
            "source_image": row.get("source_image", ""),
            "node_x": row.get("node_x", ""),
            "node_y": row.get("node_y", ""),
            "local_x": row.get("local_x", ""),
            "local_y": row.get("local_y", ""),
            "proposal_source": proposal_source,
            "proposal_type": proposal_type,
            "gap_len_px": row.get("gap_len_px", ""),
            "gap_radius_used": row.get("gap_radius_used", ""),
            "endpoint_x": row.get("endpoint_x", ""),
            "endpoint_y": row.get("endpoint_y", ""),
            "target_x": row.get("target_x", ""),
            "target_y": row.get("target_y", ""),
            "proposal_score": row.get("proposal_score", ""),
            "border_flag": row.get("border_flag", ""),
        }

        try:
            patch_f32 = load_patch_f32(patch_path)
            cnn_recrop_total += 1
            classical_recrop_total += 1
            cnn_patch_f32, recropped = maybe_recrop_cnn_patch(
                patch_f32=patch_f32,
                row=row,
                args=args,
                source_cache=cnn_source_cache,
            )
            if recropped:
                cnn_recrop_used += 1
            classical_patch_f32, classical_recropped = maybe_recrop_patch(
                patch_f32=patch_f32,
                row=row,
                recrop_window=effective_classical_window,
                source_dir=effective_classical_source_dir,
                source_cache=cnn_source_cache,
            )
            if classical_recropped:
                classical_recrop_used += 1

            if cnn_model is not None:
                cnn_pred, cnn_prob = predict_cnn_with_proba(
                    model=cnn_model,
                    idx_to_label=cnn_idx_to_label,
                    patch_f32=cnn_patch_f32,
                    row=row,
                    patch_dir=args.patch_dir,
                    expected_channels=cnn_in_channels,
                    context_col=args.cnn_context_col,
                    device=device,
                )
            else:
                cnn_pred, cnn_prob = "N", {}
            if mlp_model is not None:
                mlp_pred, mlp_prob = predict_classical_with_proba(
                    mlp_model,
                    mlp_class_names,
                    classical_patch_f32,
                    row=row,
                    feature_spec=mlp_feature_spec,
                )
            else:
                mlp_pred, mlp_prob = "N", {}
            if svm_model is not None:
                svm_pred, svm_prob = predict_classical_with_proba(
                    svm_model,
                    svm_class_names,
                    classical_patch_f32,
                    row=row,
                    feature_spec=svm_feature_spec,
                )
            else:
                svm_pred, svm_prob = "N", {}
            if xgb_model is not None:
                xgb_pred, xgb_prob = predict_classical_with_proba(
                    xgb_model,
                    xgb_class_names,
                    classical_patch_f32,
                    row=row,
                    feature_spec=xgb_feature_spec,
                )
            else:
                xgb_pred, xgb_prob = "N", {}
            if rf_model is not None:
                rf_pred, rf_prob = predict_classical_with_proba(
                    rf_model,
                    rf_class_names,
                    classical_patch_f32,
                    row=row,
                    feature_spec=rf_feature_spec,
                )
            else:
                rf_pred, rf_prob = "N", {}

            local_x = _parse_optional_float(row.get("local_x"))
            local_y = _parse_optional_float(row.get("local_y"))
            preferred_anchor = (local_x, local_y) if local_x is not None and local_y is not None else None
            geom = compute_patch_geometry(
                patch_f32=patch_f32,
                preferred_anchor=preferred_anchor,
                trace_len=max(4, int(args.geometry_trace_len)),
                merge_deg=20.0,
            )
            geometry_label = str(geom["geometry_label"])

            votes = [cnn_pred, mlp_pred, svm_pred, xgb_pred]
            raw_consensus, raw_agreement = ensemble_consensus(votes)
            consensus, agreement = raw_consensus, raw_agreement
            consensus, agreement = apply_final_synthesis_filter(
                votes=votes,
                consensus=consensus,
                agreement=agreement,
                geometry_label=geometry_label,
                proposal_source=proposal_source,
                proposal_type=proposal_type,
                border_flag=_parse_bool_flag(row.get("border_flag")),
                mlp_pred=mlp_pred,
                xgb_pred=xgb_pred,
                x_mode=str(args.x_mode),
            )

            avg_prob = {
                cls: float(
                    np.mean(
                        [
                            float(cnn_prob.get(cls, 0.0)),
                            float(mlp_prob.get(cls, 0.0)),
                            float(svm_prob.get(cls, 0.0)),
                            float(xgb_prob.get(cls, 0.0)),
                        ]
                    )
                )
                for cls in META_CLASS_NAMES
            }

            result.update(
                {
                    "geometry_label": geometry_label,
                    "geometry_branch_count": geom.get("branch_count"),
                    "geometry_min_gap_deg": geom.get("min_gap_deg"),
                    "geometry_max_gap_deg": geom.get("max_gap_deg"),
                    "geometry_anchor_x": geom.get("anchor_x"),
                    "geometry_anchor_y": geom.get("anchor_y"),
                    "cnn_pred": cnn_pred,
                    "mlp_pred": mlp_pred,
                    "svm_pred": svm_pred,
                    "xgb_pred": xgb_pred,
                    "rf_pred": rf_pred,
                    "cnn_prob_N": cnn_prob.get("N", 0.0),
                    "cnn_prob_T": cnn_prob.get("T", 0.0),
                    "cnn_prob_X": cnn_prob.get("X", 0.0),
                    "cnn_prob_Y": cnn_prob.get("Y", 0.0),
                    "mlp_prob_N": mlp_prob.get("N", 0.0),
                    "mlp_prob_T": mlp_prob.get("T", 0.0),
                    "mlp_prob_X": mlp_prob.get("X", 0.0),
                    "mlp_prob_Y": mlp_prob.get("Y", 0.0),
                    "svm_prob_N": svm_prob.get("N", 0.0),
                    "svm_prob_T": svm_prob.get("T", 0.0),
                    "svm_prob_X": svm_prob.get("X", 0.0),
                    "svm_prob_Y": svm_prob.get("Y", 0.0),
                    "xgb_prob_N": xgb_prob.get("N", 0.0),
                    "xgb_prob_T": xgb_prob.get("T", 0.0),
                    "xgb_prob_X": xgb_prob.get("X", 0.0),
                    "xgb_prob_Y": xgb_prob.get("Y", 0.0),
                    "rf_prob_N": rf_prob.get("N", 0.0),
                    "rf_prob_T": rf_prob.get("T", 0.0),
                    "rf_prob_X": rf_prob.get("X", 0.0),
                    "rf_prob_Y": rf_prob.get("Y", 0.0),
                    "avg_prob_N": avg_prob.get("N", 0.0),
                    "avg_prob_T": avg_prob.get("T", 0.0),
                    "avg_prob_X": avg_prob.get("X", 0.0),
                    "avg_prob_Y": avg_prob.get("Y", 0.0),
                    "meta_pred": "",
                    "meta_confidence": "",
                    "meta_prob_N": "",
                    "meta_prob_T": "",
                    "meta_prob_X": "",
                    "meta_prob_Y": "",
                    "gate_source_head": "",
                    "gate_prob_positive": "",
                    "gate_threshold_used": "",
                    "gate_decision": "",
                    "cluster_id": "",
                    "cluster_size": "",
                    "cluster_label": "",
                    "cluster_score": "",
                    "consensus": consensus,
                    "agreement": agreement,
                    "raw_consensus": raw_consensus,
                    "raw_agreement": raw_agreement,
                    "raw_x_votes": int(sum(1 for v in votes if str(v) == "X")),
                }
            )

            # Optional specialized cascade models (gate/type) for a single selected head.
            if cascade_selected_head:
                specialized_used = False
                gate_prob_map: Dict[str, float] = {}
                type_prob_map: Dict[str, float] = {}
                type_pred_label = ""

                if cascade_selected_head in {"mlp", "svm", "xgb"} and cascade_gate_model is not None and cascade_type_model is not None:
                    _gate_pred, gate_prob_map = predict_classical_with_proba(
                        cascade_gate_model,
                        cascade_gate_class_names,
                        classical_patch_f32,
                        row=row,
                        feature_spec=cascade_gate_feature_spec,
                    )
                    type_pred_label, type_prob_map = predict_classical_with_proba(
                        cascade_type_model,
                        cascade_type_class_names,
                        classical_patch_f32,
                        row=row,
                        feature_spec=cascade_type_feature_spec,
                    )
                    specialized_used = True
                elif cascade_selected_head == "cnn" and cascade_gate_model is not None and cascade_type_model is not None:
                    _gate_pred, gate_prob_map = predict_cnn_with_proba(
                        model=cascade_gate_model,
                        idx_to_label=cascade_gate_idx_to_label,
                        patch_f32=cnn_patch_f32,
                        row=row,
                        patch_dir=args.patch_dir,
                        expected_channels=cascade_gate_in_channels,
                        context_col=args.cnn_context_col,
                        device=device,
                    )
                    type_pred_label, type_prob_map = predict_cnn_with_proba(
                        model=cascade_type_model,
                        idx_to_label=cascade_type_idx_to_label,
                        patch_f32=cnn_patch_f32,
                        row=row,
                        patch_dir=args.patch_dir,
                        expected_channels=cascade_type_in_channels,
                        context_col=args.cnn_context_col,
                        device=device,
                    )
                    specialized_used = True

                if specialized_used:
                    head = cascade_selected_head
                    gate_pos = _positive_gate_prob_from_map(gate_prob_map)
                    gate_n = float(gate_prob_map.get("N", max(0.0, 1.0 - gate_pos)))
                    if type_pred_label not in {"T", "Y", "X"}:
                        type_pred_label, _ = _choose_type_label_from_probs(type_prob_map, x_mode="enabled")
                    t_prob = float(type_prob_map.get("T", 0.0)) * gate_pos
                    x_prob = float(type_prob_map.get("X", 0.0)) * gate_pos
                    y_prob = float(type_prob_map.get("Y", 0.0)) * gate_pos
                    result[f"{head}_pred"] = type_pred_label
                    result[f"{head}_prob_N"] = gate_n
                    result[f"{head}_prob_T"] = t_prob
                    result[f"{head}_prob_X"] = x_prob
                    result[f"{head}_prob_Y"] = y_prob
                    result["gate_source_head"] = head
                    result["gate_prob_positive"] = gate_pos
        except Exception as exc:
            logging.error("Failed on patch %s: %s", patch_filename, exc)
            result.update(
                {
                    "geometry_label": "Unknown",
                    "geometry_branch_count": 0,
                    "geometry_min_gap_deg": "",
                    "geometry_max_gap_deg": "",
                    "geometry_anchor_x": "",
                    "geometry_anchor_y": "",
                    "cnn_pred": "ERROR",
                    "mlp_pred": "ERROR",
                    "svm_pred": "ERROR",
                    "xgb_pred": "ERROR",
                    "cnn_prob_N": "",
                    "cnn_prob_T": "",
                    "cnn_prob_X": "",
                    "cnn_prob_Y": "",
                    "mlp_prob_N": "",
                    "mlp_prob_T": "",
                    "mlp_prob_X": "",
                    "mlp_prob_Y": "",
                    "svm_prob_N": "",
                    "svm_prob_T": "",
                    "svm_prob_X": "",
                    "svm_prob_Y": "",
                    "xgb_prob_N": "",
                    "xgb_prob_T": "",
                    "xgb_prob_X": "",
                    "xgb_prob_Y": "",
                    "avg_prob_N": "",
                    "avg_prob_T": "",
                    "avg_prob_X": "",
                    "avg_prob_Y": "",
                    "meta_pred": "",
                    "meta_confidence": "",
                    "meta_prob_N": "",
                    "meta_prob_T": "",
                    "meta_prob_X": "",
                    "meta_prob_Y": "",
                    "gate_source_head": "",
                    "gate_prob_positive": "",
                    "gate_threshold_used": "",
                    "gate_decision": "",
                    "cluster_id": "",
                    "cluster_size": "",
                    "cluster_label": "",
                    "cluster_score": "",
                    "consensus": "ERROR",
                    "agreement": 0,
                    "raw_consensus": "ERROR",
                    "raw_agreement": 0,
                    "raw_x_votes": 0,
                }
            )
        out_rows.append(result)

    label_head = str(args.label_head).strip().lower()
    if label_head in {"cnn", "mlp", "svm", "xgb", "rf", "cnn_cascade", "mlp_cascade", "svm_cascade", "xgb_cascade", "rf_cascade"}:
        apply_single_head_labeling(out_rows, args)
    elif label_head == "meta":
        if meta_model is None:
            raise ValueError("--label-head meta requires --meta-model-path")
        apply_meta_classifier(out_rows, meta_payload=meta_model, args=args, force_override=True)
    else:
        apply_local_t_y_arbitration(out_rows, args)
        if meta_model is not None:
            apply_meta_classifier(out_rows, meta_payload=meta_model, args=args, force_override=False)

    apply_cluster_arbitration(out_rows, args)
    if label_head == "ensemble":
        apply_mixed_label_cluster_arbitration(out_rows, args)
        nms_rows = apply_same_label_nms(out_rows, args)
    else:
        nms_rows = list(out_rows)
    filtered_rows = apply_output_row_filters(nms_rows, args)
    if int(args.cnn_recrop_window) > 0:
        logging.info(
            "CNN recrop: window=%d source_dir=%s used=%d/%d",
            int(args.cnn_recrop_window),
            str(args.cnn_source_image_dir),
            int(cnn_recrop_used),
            int(cnn_recrop_total),
        )
        if cnn_recrop_total > 0 and cnn_recrop_used == 0 and cnn_affects_output:
            logging.warning(
                "CNN recrop configured but used on 0/%d patches. Check --cnn-source-image-dir and source_image names.",
                int(cnn_recrop_total),
            )
    if effective_classical_window > 0:
        logging.info(
            "Classical recrop: window=%d source_dir=%s used=%d/%d",
            int(effective_classical_window),
            str(effective_classical_source_dir),
            int(classical_recrop_used),
            int(classical_recrop_total),
        )
        if classical_recrop_total > 0 and classical_recrop_used == 0 and classical_affects_output:
            logging.warning(
                "Classical recrop configured but used on 0/%d patches. Check source image dir and source_image names.",
                int(classical_recrop_total),
            )
    write_results_csv(args.output_csv, filtered_rows)
    logging.info("Wrote ensemble results: %s", args.output_csv)


if __name__ == "__main__":
    main()
