#!/usr/bin/env python3
"""
Shared feature extraction for learned meta-classification over ensemble outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np


CLASS_NAMES = ["N", "T", "X", "Y"]
MODEL_NAMES = ["cnn", "mlp", "svm", "xgb"]

NUMERIC_FEATURES = [
    "agreement",
    "raw_agreement",
    "raw_x_votes",
    "proposal_score",
    "border_flag",
    "geometry_branch_count",
    "geometry_min_gap_deg",
    "geometry_max_gap_deg",
    "gap_len_px",
    "gap_radius_used",
]

CATEGORICAL_FEATURES = [
    "proposal_source",
    "proposal_type",
    "geometry_label",
    "raw_consensus",
    "consensus",
    "cnn_pred",
    "mlp_pred",
    "svm_pred",
    "xgb_pred",
]


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    s = str(value).strip()
    if s == "":
        return default
    try:
        v = float(s)
    except Exception:
        return default
    if not np.isfinite(v):
        return default
    return float(v)


def _to_str(value: Any, default: str = "UNK") -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _bool_to_float(value: Any) -> float:
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return 1.0
    if s in {"0", "false", "f", "no", "n", ""}:
        return 0.0
    try:
        return float(int(float(s) > 0.0))
    except Exception:
        return 0.0


def row_to_meta_features(row: Dict[str, Any]) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}

    feat["agreement"] = _to_float(row.get("agreement"), 0.0)
    feat["raw_agreement"] = _to_float(row.get("raw_agreement"), feat["agreement"])
    feat["raw_x_votes"] = _to_float(row.get("raw_x_votes"), 0.0)
    feat["proposal_score"] = _to_float(row.get("proposal_score"), 0.0)
    feat["border_flag"] = _bool_to_float(row.get("border_flag"))
    feat["geometry_branch_count"] = _to_float(row.get("geometry_branch_count"), 0.0)
    feat["geometry_min_gap_deg"] = _to_float(row.get("geometry_min_gap_deg"), 0.0)
    feat["geometry_max_gap_deg"] = _to_float(row.get("geometry_max_gap_deg"), 0.0)
    feat["gap_len_px"] = _to_float(row.get("gap_len_px"), 0.0)
    feat["gap_radius_used"] = _to_float(row.get("gap_radius_used"), 0.0)

    for key in CATEGORICAL_FEATURES:
        feat[key] = _to_str(row.get(key), "UNK")

    for model_name in MODEL_NAMES:
        model_pred = _to_str(row.get(f"{model_name}_pred"), "UNK")
        for cls in CLASS_NAMES:
            key = f"{model_name}_prob_{cls}"
            default = 1.0 if model_pred == cls else 0.0
            feat[key] = _to_float(row.get(key), default)

    for cls in CLASS_NAMES:
        probs = [feat[f"{m}_prob_{cls}"] for m in MODEL_NAMES]
        feat[f"avg_prob_{cls}"] = float(np.mean(probs))
        feat[f"max_prob_{cls}"] = float(np.max(probs))
        feat[f"vote_count_{cls}"] = float(sum(1 for m in MODEL_NAMES if _to_str(row.get(f"{m}_pred"), "UNK") == cls))

    return feat


def rows_to_meta_features(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row_to_meta_features(row) for row in rows]

