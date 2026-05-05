"""Stage-4 ensemble-classifier wrapper for the Mars TCP Ch 3 inference pipeline.

This module exposes :func:`predict_per_junction`, a programmatic entry point
into the ensemble inference logic in ``predict_ensemble.py``. It mirrors
``predict_ensemble.py:main()`` end-to-end: model loading (lines 1611-1719),
the per-row inference loop (lines 1752-2070), and the post-loop labeling
pipeline (lines 2072-2117), but skips the CSV write step and returns rows
in-memory instead.

The :class:`EnsembleInferenceConfig` dataclass mirrors every argparse default
in ``predict_ensemble.py:parse_args()`` (lines 38-286). Calling the wrapper
with ``config=None`` (or omitting ``config``) produces output that matches
running the CLI with all flags at default on the same manifest within the
project's Tier-A (Stage 2) / Tier-C (Stage 4) acceptance tolerances, in
``consensus``, ``meta_pred``, and per-class probabilities. Note that the CLI
default ``label_head`` is ``"ensemble"`` and ``cnn_model_file`` is
``CNN_ft_gauss40.pt`` — those are the v7 publication defaults. To use the
stacking meta-classifier path described in the README's Quick Start, set
``label_head="meta"`` and pass ``meta_model_path`` explicitly.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from mars_tyxn.junction_geometry import compute_patch_geometry
from mars_tyxn.meta_features import CLASS_NAMES as META_CLASS_NAMES
from mars_tyxn.predict_ensemble import (
    _parse_bool_flag,
    _parse_optional_float,
    apply_cluster_arbitration,
    apply_final_synthesis_filter,
    apply_local_t_y_arbitration,
    apply_meta_classifier,
    apply_mixed_label_cluster_arbitration,
    apply_output_row_filters,
    apply_same_label_nms,
    apply_single_head_labeling,
    ensemble_consensus,
    load_classical_model,
    load_cnn_model,
    load_meta_model,
    load_patch_f32,
    maybe_recrop_cnn_patch,
    maybe_recrop_patch,
    predict_classical_with_proba,
    predict_cnn_with_proba,
    resolve_proposal_metadata,
)


@dataclass
class EnsembleInferenceConfig:
    """Defaults mirror ``predict_ensemble.py:parse_args()``.

    Every field with a non-``None`` default is the same value the CLI
    argparse would set when the corresponding flag is omitted.
    """

    # Manifest / IO (ignored by the wrapper, kept for parity with main()).
    # TODO(Phase 5.5): manifest/output_csv are CLI-parity placeholders only;
    # the Phase 5.5 ManifestRow schema (endpoint_x/y, target_x/y, etc.) will
    # supersede the CSV-based wiring used by the CLI helpers.
    manifest: Path = field(default_factory=lambda: Path("inference_manifest_images.csv"))
    output_csv: Path = field(default_factory=lambda: Path("final_ensemble_results.csv"))

    # Geometry gate.
    geometry_trace_len: int = 6

    # CNN.
    cnn_context_col: str = "context_image"
    cnn_model_file: str = "CNN_ft_gauss40.pt"
    cnn_source_image_dir: Optional[Path] = None
    cnn_recrop_window: int = 0

    # Classical recrop.
    classical_source_image_dir: Optional[Path] = None
    classical_recrop_window: int = 0

    # Logging.
    log_level: str = "INFO"

    # Output filters.
    positive_only_output: bool = False
    virtual_t_min_agreement_output: int = 0
    drop_border_virtual_output: bool = False

    # Arbitration / rescue.
    t_demotion_min_gap_floor: float = 0.0
    y_rescue_max_gap: float = 0.0
    y_rescue_min_agreement: int = 0
    y_rescue_vseg_unknown_b2: bool = False
    t_endpoint_endpoint_mode: str = "allow"
    local_mixed_arbitration_radius: float = 0.0
    same_label_nms_radius: float = 0.0
    rescue_min_gap_floor: float = 0.0
    x_mode: str = "veto"

    # Single-head / cascade.
    label_head: str = "ensemble"
    gate_threshold: float = 0.50
    virtual_gate_threshold: float = 0.60
    border_virtual_gate_threshold: float = 0.70

    # Geometric voter (cascade heads).
    geometry_voter_mode: str = "on"
    geometry_t_min_gap_low: float = 75.0
    geometry_t_min_gap_high: float = 95.0
    geometry_t_max_gap_low: float = 155.0
    geometry_t_max_gap_high: float = 205.0
    geometry_y_min_gap: float = 100.0

    # Meta classifier.
    meta_model_path: Optional[Path] = None
    meta_min_confidence: float = 0.0

    # Cluster arbitration.
    cluster_arbitration_radius: float = 0.0


def _config_to_args_namespace(
    config: EnsembleInferenceConfig,
    *,
    models_dir: Path,
    patch_dir: Path,
) -> argparse.Namespace:
    """Materialize an ``argparse.Namespace`` shaped exactly like the one
    ``parse_args()`` produces, so internal helpers that take ``args`` work
    unchanged."""
    ns = argparse.Namespace(**config.__dict__)
    ns.models_dir = Path(models_dir)
    ns.patch_dir = Path(patch_dir)
    return ns


def predict_per_junction(
    rows: List[Dict[str, Any]],
    *,
    models_dir: Path,
    patch_dir: Path,
    device: torch.device,
    config: Optional[EnsembleInferenceConfig] = None,
) -> List[Dict[str, Any]]:
    """Run ensemble classification on a list of manifest rows.

    Parameters
    ----------
    rows : list of dict
        Manifest rows (typically loaded from ``inference_manifest_images.csv``).
        See ``predict_ensemble.read_manifest_rows`` and the README for the
        expected columns (``patch_filename``, ``source_image``, ``node_x``,
        ``node_y``, ``local_x``, ``local_y``, optional ``proposal_*`` and
        ``border_flag`` fields).
    models_dir : Path
        Directory containing the trained model artifacts
        (``CNN_ft_gauss40.pt``, ``RF_32.joblib``, ``XGB_32_d6.joblib``,
        ``MLP_32.joblib``, ``SVM_32.joblib``).
    patch_dir : Path
        Directory containing the per-junction PNG patches referenced by each
        row's ``patch_filename`` field.
    device : torch.device
        Inference device for the CNN.
    config : EnsembleInferenceConfig, optional
        Overrides for argparse defaults. ``None`` reproduces the CLI default
        invocation.

    Returns
    -------
    list of dict
        One dict per input row carrying per-head predictions, per-class
        probabilities, geometry features, and the final ensemble ``consensus``
        (and ``meta_pred`` when ``label_head="meta"``). Output is intended to
        match ``predict_ensemble.py`` invoked with default flags on the same
        manifest within the project's Tier-A (Stage 2) / Tier-C (Stage 4)
        acceptance tolerances, save for the CSV write step which is skipped.
    """
    if config is None:
        config = EnsembleInferenceConfig()

    args = _config_to_args_namespace(config, models_dir=Path(models_dir), patch_dir=Path(patch_dir))

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

    out_rows: List[Dict[str, Any]] = []
    for row in tqdm(rows, desc="Ensemble inference", unit="patch"):
        patch_filename = row["patch_filename"]
        patch_path = args.patch_dir / patch_filename

        proposal_source, proposal_type = resolve_proposal_metadata(
            row=row, patch_filename=patch_filename
        )
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
            cnn_patch_f32, _ = maybe_recrop_cnn_patch(
                patch_f32=patch_f32,
                row=row,
                args=args,
                source_cache=cnn_source_cache,
            )
            classical_patch_f32, _ = maybe_recrop_patch(
                patch_f32=patch_f32,
                row=row,
                recrop_window=effective_classical_window,
                source_dir=effective_classical_source_dir,
                source_cache=cnn_source_cache,
            )

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
                    mlp_model, mlp_class_names, classical_patch_f32,
                    row=row, feature_spec=mlp_feature_spec,
                )
            else:
                mlp_pred, mlp_prob = "N", {}
            if svm_model is not None:
                svm_pred, svm_prob = predict_classical_with_proba(
                    svm_model, svm_class_names, classical_patch_f32,
                    row=row, feature_spec=svm_feature_spec,
                )
            else:
                svm_pred, svm_prob = "N", {}
            if xgb_model is not None:
                xgb_pred, xgb_prob = predict_classical_with_proba(
                    xgb_model, xgb_class_names, classical_patch_f32,
                    row=row, feature_spec=xgb_feature_spec,
                )
            else:
                xgb_pred, xgb_prob = "N", {}
            if rf_model is not None:
                rf_pred, rf_prob = predict_classical_with_proba(
                    rf_model, rf_class_names, classical_patch_f32,
                    row=row, feature_spec=rf_feature_spec,
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

            # Optional specialized cascade gate/type for a single selected head.
            if cascade_selected_head:
                specialized_used = False
                gate_prob_map: Dict[str, float] = {}
                type_prob_map: Dict[str, float] = {}
                type_pred_label = ""

                if (
                    cascade_selected_head in {"mlp", "svm", "xgb"}
                    and cascade_gate_model is not None
                    and cascade_type_model is not None
                ):
                    _gate_pred, gate_prob_map = predict_classical_with_proba(
                        cascade_gate_model, cascade_gate_class_names, classical_patch_f32,
                        row=row, feature_spec=cascade_gate_feature_spec,
                    )
                    type_pred_label, type_prob_map = predict_classical_with_proba(
                        cascade_type_model, cascade_type_class_names, classical_patch_f32,
                        row=row, feature_spec=cascade_type_feature_spec,
                    )
                    specialized_used = True
                elif (
                    cascade_selected_head == "cnn"
                    and cascade_gate_model is not None
                    and cascade_type_model is not None
                ):
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
                    from mars_tyxn.predict_ensemble import (
                        _choose_type_label_from_probs,
                        _positive_gate_prob_from_map,
                    )

                    head = cascade_selected_head
                    gate_pos = _positive_gate_prob_from_map(gate_prob_map)
                    gate_n = float(gate_prob_map.get("N", max(0.0, 1.0 - gate_pos)))
                    if type_pred_label not in {"T", "Y", "X"}:
                        type_pred_label, _ = _choose_type_label_from_probs(
                            type_prob_map, x_mode="enabled"
                        )
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
    if label_head in {"cnn", "mlp", "svm", "xgb", "rf",
                      "cnn_cascade", "mlp_cascade", "svm_cascade",
                      "xgb_cascade", "rf_cascade"}:
        apply_single_head_labeling(out_rows, args)
    elif label_head == "meta":
        if meta_model is None:
            raise ValueError("label_head='meta' requires meta_model_path on the config")
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
    return filtered_rows


__all__ = ["EnsembleInferenceConfig", "predict_per_junction"]
