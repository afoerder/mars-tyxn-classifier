#!/usr/bin/env python3
"""
Train an XGBoost pipeline on 96x96 Martian patches.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from mars_tyxn.classical_feature_builder import (
    GEOMETRY_FEATURE_NAMES,
    PatchFeatureAssembler,
    build_classical_input_matrix,
    normalize_feature_regime,
)

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("xgboost is required for train_xgb.py. Install with: pip install xgboost") from exc

CLASS_NAMES = ["N", "T", "Y", "X"]
TASK_CLASS_NAMES = {
    "multiclass": ["N", "T", "Y", "X"],
    "gate": ["N", "P"],
    "type": ["T", "Y", "X"],
}


def read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"relpath", "label", "split"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Manifest must include columns {required}. Found: {reader.fieldnames}")
        for row in reader:
            rows.append({k: str(v) for k, v in row.items()})
    return rows


def load_patch_flat(abs_png_path: Path, patch_size: int = 96) -> np.ndarray:
    arr = np.asarray(Image.open(abs_png_path).convert("L"), dtype=np.float32)
    if arr.shape != (patch_size, patch_size):
        arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((patch_size, patch_size), Image.NEAREST), dtype=np.float32)
    return (arr / 255.0).reshape(-1)


def _parse_weight(row: Dict[str, str], col: str) -> float:
    raw = str(row.get(col, "")).strip()
    if raw == "":
        return 1.0
    try:
        v = float(raw)
        if not np.isfinite(v) or v <= 0:
            return 1.0
        return float(v)
    except Exception:
        return 1.0


def _has_valid_local_anchor(row: Dict[str, str]) -> bool:
    try:
        lx = float(str(row.get("local_x", "")).strip())
        ly = float(str(row.get("local_y", "")).strip())
    except Exception:
        return False
    return bool(np.isfinite(lx) and np.isfinite(ly))


def infer_geometry_use_local_anchor(rows: Sequence[Dict[str, str]]) -> bool:
    if not rows:
        return False
    return any(_has_valid_local_anchor(r) for r in rows)


def load_split(
    rows: Sequence[Dict[str, str]],
    split_name: str,
    data_root: Path,
    patch_size: int,
    sample_weight_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]]]:
    split_rows = [row for row in rows if row.get("split", "") == split_name]
    if not split_rows:
        raise RuntimeError(f"Split '{split_name}' is empty.")

    X = np.stack(
        [load_patch_flat(data_root / str(row["relpath"]), patch_size=patch_size) for row in split_rows],
        axis=0,
    )
    y = np.asarray([str(row["label"]) for row in split_rows], dtype=object)
    w = np.asarray([_parse_weight(row, sample_weight_col) for row in split_rows], dtype=np.float32)
    return X, y, w, split_rows


def build_sample_weight(
    y_str: np.ndarray,
    manifest_w: np.ndarray,
    mode: str,
    non_n_weight: float,
    x_class_weight: float,
    t_class_weight: float = 1.0,
    class_weight_map: Dict[str, float] | None = None,
) -> np.ndarray:
    m = str(mode).strip().lower()
    if m == "none":
        w = np.ones_like(manifest_w, dtype=np.float32)
    elif m == "manifest":
        w = manifest_w.astype(np.float32, copy=False)
    elif m == "non_n_focus":
        w = np.ones_like(manifest_w, dtype=np.float32)
        w[y_str != "N"] = float(non_n_weight)
    elif m == "manifest_non_n":
        w = manifest_w.astype(np.float32, copy=True)
        w[y_str != "N"] *= float(non_n_weight)
    else:
        raise ValueError(f"Unsupported --sample-weight-mode: {mode}")

    x_w = float(x_class_weight)
    if x_w > 0 and x_w != 1.0:
        w = w.astype(np.float32, copy=True)
        w[y_str == "X"] *= x_w

    t_w = float(t_class_weight)
    if t_w > 0 and t_w != 1.0:
        w = w.astype(np.float32, copy=True)
        w[y_str == "T"] *= t_w

    if class_weight_map:
        w = w.astype(np.float32, copy=True)
        for label, c_w in class_weight_map.items():
            w[y_str == label] *= float(c_w)
    return w


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HOG+XGB model on synthetic manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to training manifest CSV.")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to manifest parent.")
    parser.add_argument("--output-model", type=Path, default=Path("models/classifiers/XGB_32_d6.joblib"))
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-set", type=str, default="legacy", choices=["legacy", "hog_mask_center"])
    parser.add_argument(
        "--feature-regime",
        type=str,
        default="image_only",
        choices=["image_only", "geom_only", "image_plus_geom"],
        help="Classical input regime: image HOG only, geometry only, or fused image+geometry.",
    )
    parser.add_argument("--geometry-trace-len", type=int, default=40)
    parser.add_argument("--geometry-merge-deg", type=float, default=20.0)
    parser.add_argument("--geometry-prefer-radius", type=float, default=10.0)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--sample-weight-mode",
        type=str,
        default="non_n_focus",
        choices=["none", "manifest", "non_n_focus", "manifest_non_n"],
    )
    parser.add_argument("--sample-weight-col", type=str, default="sample_weight")
    parser.add_argument("--non-n-weight", type=float, default=1.5)
    parser.add_argument("--x-class-weight", type=float, default=1.0, help="Optional multiplier for X samples.")
    parser.add_argument("--t-class-weight", type=float, default=1.0,
                        help="Optional multiplier for T-class samples. Values >1.0 boost T emphasis.")
    parser.add_argument(
        "--task",
        type=str,
        default="multiclass",
        choices=["multiclass", "gate", "type"],
        help="Training task: multiclass N/T/Y/X, gate N/P, or type T/Y/X (N filtered).",
    )
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Optional class balancing over train labels (multiplies sample weights).",
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help="Optional prior XGB artifact used for continuation training.",
    )
    parser.add_argument(
        "--transfer-mode",
        type=str,
        default="none",
        choices=["none", "continue_booster"],
        help="Transfer strategy: none or continue_booster via XGBoost xgb_model.",
    )
    return parser.parse_args()


def apply_task_to_split(
    X: np.ndarray,
    y_str: np.ndarray,
    w_manifest: np.ndarray,
    split_rows: Sequence[Dict[str, str]],
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]]]:
    task_norm = str(task).strip().lower()
    if task_norm == "multiclass":
        return X, y_str, w_manifest, list(split_rows)
    if task_norm == "gate":
        y_gate = np.where(y_str == "N", "N", "P").astype(object)
        return X, y_gate, w_manifest, list(split_rows)
    if task_norm == "type":
        keep = (y_str != "N")
        keep_rows = [split_rows[i] for i in np.where(keep)[0].tolist()]
        return X[keep], y_str[keep], w_manifest[keep], keep_rows
    raise ValueError(f"Unsupported task: {task}")


def to_label_predictions(pred: np.ndarray, num_classes: int) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 2:
        return np.asarray(np.argmax(arr, axis=1), dtype=np.int64)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.floating):
        if int(num_classes) <= 2:
            return np.asarray(arr >= 0.5, dtype=np.int64)
        return np.asarray(np.rint(arr), dtype=np.int64)
    return np.asarray(arr, dtype=np.int64)


def _extract_xgb_clf_from_payload(payload_obj: object) -> XGBClassifier:
    if isinstance(payload_obj, dict) and "pipeline" in payload_obj:
        pipeline = payload_obj["pipeline"]
    else:
        pipeline = payload_obj
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Init model must contain a sklearn Pipeline with an XGB classifier.")
    if "clf" not in pipeline.named_steps:
        raise ValueError("Init model pipeline has no 'clf' step.")
    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, XGBClassifier):
        raise ValueError(f"Init classifier is not XGBClassifier: {type(clf)}")
    return clf


def main() -> None:
    args = parse_args()
    feature_regime = normalize_feature_regime(args.feature_regime)
    data_root = args.data_root if args.data_root is not None else args.manifest.parent

    rows = read_manifest(args.manifest)
    X_train_patch, y_train_str, w_train_manifest, rows_train = load_split(
        rows, str(args.train_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_val_patch, y_val_str, w_val_manifest, rows_val = load_split(
        rows, str(args.val_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_test_patch, y_test_str, _, rows_test = load_split(
        rows, str(args.test_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_train_patch, y_train_str, w_train_manifest, rows_train = apply_task_to_split(
        X_train_patch, y_train_str, w_train_manifest, rows_train, args.task
    )
    X_val_patch, y_val_str, w_val_manifest, rows_val = apply_task_to_split(
        X_val_patch, y_val_str, w_val_manifest, rows_val, args.task
    )
    X_test_patch, y_test_str, _, rows_test = apply_task_to_split(
        X_test_patch, y_test_str, np.ones(len(y_test_str), dtype=np.float32), rows_test, args.task
    )
    geometry_use_local_anchor = infer_geometry_use_local_anchor(rows_train)

    X_train = build_classical_input_matrix(
        patch_flat=X_train_patch,
        rows=rows_train,
        feature_regime=feature_regime,
        patch_size=int(args.patch_size),
        geometry_trace_len=int(args.geometry_trace_len),
        geometry_merge_deg=float(args.geometry_merge_deg),
        geometry_prefer_radius=float(args.geometry_prefer_radius),
        geometry_use_local_anchor=geometry_use_local_anchor,
    )
    X_val = build_classical_input_matrix(
        patch_flat=X_val_patch,
        rows=rows_val,
        feature_regime=feature_regime,
        patch_size=int(args.patch_size),
        geometry_trace_len=int(args.geometry_trace_len),
        geometry_merge_deg=float(args.geometry_merge_deg),
        geometry_prefer_radius=float(args.geometry_prefer_radius),
        geometry_use_local_anchor=geometry_use_local_anchor,
    )
    X_test = build_classical_input_matrix(
        patch_flat=X_test_patch,
        rows=rows_test,
        feature_regime=feature_regime,
        patch_size=int(args.patch_size),
        geometry_trace_len=int(args.geometry_trace_len),
        geometry_merge_deg=float(args.geometry_merge_deg),
        geometry_prefer_radius=float(args.geometry_prefer_radius),
        geometry_use_local_anchor=geometry_use_local_anchor,
    )

    if len(y_train_str) == 0 or len(y_val_str) == 0 or len(y_test_str) == 0:
        raise RuntimeError(f"Task '{args.task}' produced an empty split.")

    seen_labels = set(np.concatenate([y_train_str, y_val_str, y_test_str], axis=0).tolist())
    expected = set(TASK_CLASS_NAMES[str(args.task).strip().lower()])
    unexpected = sorted(seen_labels.difference(expected))
    if unexpected:
        raise ValueError(f"Task {args.task} found unexpected labels: {unexpected}. Expected subset of {sorted(expected)}")

    le = LabelEncoder()
    le.fit(TASK_CLASS_NAMES[str(args.task).strip().lower()])
    y_train = le.transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)
    class_names = [str(c) for c in le.classes_]
    num_classes = len(class_names)

    class_weight_map: Dict[str, float] | None = None
    if str(args.class_weight_mode).lower() == "balanced":
        c = Counter(y_train_str.tolist())
        present = [cls for cls in class_names if c.get(cls, 0) > 0]
        if present:
            total = float(len(y_train_str))
            raw = {cls: total / (float(len(present)) * float(c[cls])) for cls in present}
            mean_raw = sum(raw.values()) / float(len(raw))
            class_weight_map = {cls: float(raw[cls] / mean_raw) for cls in raw}

    sw_train = build_sample_weight(
        y_str=y_train_str,
        manifest_w=w_train_manifest,
        mode=args.sample_weight_mode,
        non_n_weight=float(args.non_n_weight),
        x_class_weight=float(args.x_class_weight),
        t_class_weight=float(args.t_class_weight),
        class_weight_map=class_weight_map,
    )
    sw_val = build_sample_weight(
        y_str=y_val_str,
        manifest_w=w_val_manifest,
        mode=args.sample_weight_mode,
        non_n_weight=float(args.non_n_weight),
        x_class_weight=float(args.x_class_weight),
        t_class_weight=float(args.t_class_weight),
        class_weight_map=class_weight_map,
    )

    xgb_kwargs = {
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "max_depth": int(args.max_depth),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "random_state": int(args.seed),
        "tree_method": "hist",
        "early_stopping_rounds": int(args.early_stopping_rounds),
    }
    if num_classes <= 2:
        xgb_kwargs["objective"] = "binary:logistic"
        xgb_kwargs["eval_metric"] = "logloss"
    else:
        xgb_kwargs["objective"] = "multi:softprob"
        xgb_kwargs["eval_metric"] = "mlogloss"
        xgb_kwargs["num_class"] = int(num_classes)

    pipeline = Pipeline(
        steps=[
            (
                "features",
                PatchFeatureAssembler(
                    feature_regime=feature_regime,
                    feature_set=args.feature_set,
                    patch_size=int(args.patch_size),
                ),
            ),
            (
                "clf",
                XGBClassifier(**xgb_kwargs),
            ),
        ]
    )

    # Precompute transformed features for eval_set expected by XGB inside pipeline fit kwargs.
    pipeline.named_steps["features"].fit(X_train)
    X_val_feat = pipeline.named_steps["features"].transform(X_val)

    fit_kwargs = {
        "clf__eval_set": [(X_val_feat, y_val)],
        "clf__sample_weight": sw_train,
        "clf__sample_weight_eval_set": [sw_val],
        "clf__verbose": False,
    }
    init_model_path = args.init_model.resolve() if args.init_model is not None else None
    transfer_mode = str(args.transfer_mode).strip().lower()
    if init_model_path is not None:
        if transfer_mode != "continue_booster":
            raise ValueError("--init-model requires --transfer-mode continue_booster for train_xgb.py")
        init_payload = joblib.load(init_model_path)
        init_clf = _extract_xgb_clf_from_payload(init_payload)
        if isinstance(init_payload, dict):
            init_classes = [str(v) for v in init_payload.get("class_names", [])]
            if init_classes and init_classes != class_names:
                raise ValueError(
                    f"Init model class_names mismatch: init={init_classes} current={class_names}. "
                    "Use matching task/label space for continuation."
                )
            init_task = str(init_payload.get("task", "")).strip().lower()
            if init_task and init_task != str(args.task).strip().lower():
                raise ValueError(
                    f"Init model task mismatch: init={init_task} current={str(args.task).strip().lower()}"
                )
            init_regime = str(init_payload.get("feature_regime", "")).strip().lower()
            if init_regime and init_regime != str(feature_regime):
                raise ValueError(
                    f"Init model feature_regime mismatch: init={init_regime} current={feature_regime}"
                )
        fit_kwargs["clf__xgb_model"] = init_clf.get_booster()
        print(
            "Transfer init: "
            f"source={init_model_path} "
            f"mode={transfer_mode} "
            f"adds_n_estimators={int(args.n_estimators)}"
        )
    pipeline.fit(X_train, y_train, **fit_kwargs)

    val_pred = to_label_predictions(pipeline.predict(X_val), num_classes=num_classes)
    test_pred = to_label_predictions(pipeline.predict(X_test), num_classes=num_classes)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")

    print(f"Classes: {class_names}")
    print(
        f"Split sizes ({args.train_split}/{args.val_split}/{args.test_split}): "
        f"{len(X_train)}/{len(X_val)}/{len(X_test)}"
    )
    print(
        f"Task: {args.task} | Feature regime: {feature_regime} | Features: {args.feature_set} | "
        f"input_dim={X_train.shape[1]} | sample_weight_mode={args.sample_weight_mode} | "
        f"x_class_weight={float(args.x_class_weight):.3f} | class_weight_mode={args.class_weight_mode}"
    )
    if feature_regime != "image_only":
        print(f"Geometry anchor mode: {'local_x/local_y' if geometry_use_local_anchor else 'patch-center'}")
    if class_weight_map:
        print(f"Class weights (normalized): {class_weight_map}")
    print(f"Val acc: {val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "label_encoder": le,
        "class_names": class_names,
        "task": str(args.task),
        "feature_set": str(args.feature_set),
        "feature_regime": str(feature_regime),
        "patch_size": int(args.patch_size),
        "geometry_trace_len": int(args.geometry_trace_len),
        "geometry_merge_deg": float(args.geometry_merge_deg),
        "geometry_prefer_radius": float(args.geometry_prefer_radius),
        "geometry_use_local_anchor": bool(geometry_use_local_anchor),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "sample_weight_mode": str(args.sample_weight_mode),
        "x_class_weight": float(args.x_class_weight),
        "transfer_mode": transfer_mode,
        "transfer_init_model": str(init_model_path) if init_model_path is not None else "",
    }
    joblib.dump(payload, args.output_model)
    print(f"Saved model: {args.output_model}")


if __name__ == "__main__":
    main()
