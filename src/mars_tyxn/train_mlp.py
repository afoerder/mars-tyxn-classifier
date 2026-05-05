#!/usr/bin/env python3
"""
Train an MLP pipeline on 96x96 Martian patches.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mars_tyxn.classical_feature_builder import (
    GEOMETRY_FEATURE_NAMES,
    PatchFeatureAssembler,
    build_classical_input_matrix,
    normalize_feature_regime,
)


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
    parser = argparse.ArgumentParser(description="Train HOG+MLP model on synthetic manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to training manifest CSV.")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to manifest parent.")
    parser.add_argument("--output-model", type=Path, default=Path("models/classifiers/MLP_32.joblib"))
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=3e-4)
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
    parser.add_argument(
        "--sample-weight-mode",
        type=str,
        default="none",
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
        help="Optional prior MLP artifact used for warm-start continuation.",
    )
    parser.add_argument(
        "--transfer-mode",
        type=str,
        default="none",
        choices=["none", "warm_start"],
        help="Transfer strategy: none or warm_start from --init-model.",
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


def main() -> None:
    args = parse_args()
    feature_regime = normalize_feature_regime(args.feature_regime)
    data_root = args.data_root if args.data_root is not None else args.manifest.parent

    rows = read_manifest(args.manifest)
    X_train_patch, y_train_str, w_train_manifest, rows_train = load_split(
        rows, str(args.train_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_val_patch, y_val_str, _, rows_val = load_split(
        rows, str(args.val_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_test_patch, y_test_str, _, rows_test = load_split(
        rows, str(args.test_split), data_root, args.patch_size, args.sample_weight_col
    )
    X_train_patch, y_train_str, w_train_manifest, rows_train = apply_task_to_split(
        X_train_patch, y_train_str, w_train_manifest, rows_train, args.task
    )
    X_val_patch, y_val_str, _, rows_val = apply_task_to_split(
        X_val_patch, y_val_str, np.ones(len(y_val_str), dtype=np.float32), rows_val, args.task
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
    le.fit(np.concatenate([y_train_str, y_val_str, y_test_str], axis=0))
    y_train = le.transform(y_train_str)
    y_val = le.transform(y_val_str)
    y_test = le.transform(y_test_str)

    class_names = [str(c) for c in le.classes_]
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

    transfer_mode = str(args.transfer_mode).strip().lower()
    init_model_path = args.init_model.resolve() if args.init_model is not None else None
    used_sample_weight = False

    if transfer_mode == "warm_start":
        if init_model_path is None:
            raise ValueError("--transfer-mode warm_start requires --init-model")
        init_payload = joblib.load(init_model_path)
        if not isinstance(init_payload, dict) or "pipeline" not in init_payload:
            raise ValueError("Warm-start init model must be a dict payload containing 'pipeline'.")
        model = init_payload["pipeline"]
        if not isinstance(model, Pipeline):
            raise ValueError("Warm-start init payload pipeline is not a sklearn Pipeline.")
        if "clf" not in model.named_steps:
            raise ValueError("Warm-start init pipeline is missing 'clf' step.")

        init_class_names = [str(v) for v in init_payload.get("class_names", [])]
        if init_class_names and init_class_names != class_names:
            raise ValueError(
                f"Init model class_names mismatch: init={init_class_names} current={class_names}. "
                "Use matching task/label space for warm-start."
            )
        init_task = str(init_payload.get("task", "")).strip().lower()
        if init_task and init_task != str(args.task).strip().lower():
            raise ValueError(f"Init model task mismatch: init={init_task} current={str(args.task).strip().lower()}")
        init_regime = str(init_payload.get("feature_regime", "")).strip().lower()
        if init_regime and init_regime != str(feature_regime):
            raise ValueError(f"Init model feature_regime mismatch: init={init_regime} current={feature_regime}")

        clf = model.named_steps["clf"]
        if not isinstance(clf, MLPClassifier):
            raise ValueError(f"Warm-start init classifier is not MLPClassifier: {type(clf)}")
        requested_hidden = tuple(int(v) for v in args.hidden_layers)
        if tuple(int(v) for v in clf.hidden_layer_sizes) != requested_hidden:
            raise ValueError(
                f"Warm-start hidden layer mismatch: init={clf.hidden_layer_sizes} current={requested_hidden}"
            )

        clf.set_params(
            warm_start=True,
            max_iter=int(args.max_iter),
            alpha=float(args.alpha),
            learning_rate_init=float(args.learning_rate_init),
            random_state=int(args.seed),
        )

        try:
            X_train_trans = model.named_steps["scaler"].transform(model.named_steps["features"].transform(X_train))
        except Exception as exc:
            raise RuntimeError("Failed to transform training inputs with warm-start init pipeline.") from exc
        if np.any(sw_train != 1.0):
            try:
                clf.fit(X_train_trans, y_train, sample_weight=sw_train)
                used_sample_weight = True
            except TypeError:
                clf.fit(X_train_trans, y_train)
        else:
            clf.fit(X_train_trans, y_train)
        print(
            "Transfer init: "
            f"source={init_model_path} mode={transfer_mode} "
            "transform=init_features+init_scaler_frozen classifier=warm_start"
        )
    else:
        model = Pipeline(
            steps=[
                (
                    "features",
                    PatchFeatureAssembler(
                        feature_regime=feature_regime,
                        feature_set=args.feature_set,
                        patch_size=int(args.patch_size),
                    ),
                ),
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=tuple(int(v) for v in args.hidden_layers),
                        early_stopping=True,
                        max_iter=args.max_iter,
                        alpha=float(args.alpha),
                        learning_rate_init=float(args.learning_rate_init),
                        random_state=args.seed,
                    ),
                ),
            ]
        )
        if np.any(sw_train != 1.0):
            try:
                model.fit(X_train, y_train, clf__sample_weight=sw_train)
                used_sample_weight = True
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

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
        f"x_class_weight={float(args.x_class_weight):.3f} | class_weight_mode={args.class_weight_mode} | used={used_sample_weight}"
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
        "pipeline": model,
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
        "class_weight_mode": str(args.class_weight_mode),
        "transfer_mode": transfer_mode,
        "transfer_init_model": str(init_model_path) if init_model_path is not None else "",
    }
    joblib.dump(payload, args.output_model)
    print(f"Saved model: {args.output_model}")


if __name__ == "__main__":
    main()
