"""Public stacking-ensemble inference for the Ch 2 v7 pipeline.

This module exposes :func:`predict_stacking`, the canonical inference path
used to produce the Ch 2 v7 paper headline numbers (Stacking T F1 = 0.62,
macro F1 = 0.77 on the martian eval set; see ``project_v7_results``). The
v7 deposit ships these numbers as **notebook code** that loads
``stacking_gauss40_best.pkl`` directly. The CLI ``predict_ensemble.py
--label-head meta`` path expects a different file format
(``{"pipeline": ..., ...}``) that does not exist on the v7 deposit, so
calling that CLI cannot reproduce the paper headline.

This module ports the notebook flow into a public Python API so downstream
consumers (e.g. MarsTCPDetection Ch 3 Phase 6) can call it without
duplicating the logic. The exact steps mirror the notebook cell at
``MarsCracks_v7_classifier_pipeline.ipynb`` "Build Stacking Ensemble":

1. Load ``stacking_gauss40_best.pkl`` (a self-contained dict of
   ``meta_classifier`` + ``rf_model`` + ``xgb_model`` + ``cnn_state_dict``
   + ``cnn_config`` + ``gaussian_sigma`` + ``patch_size`` +
   ``label_map`` + ``idx_to_label`` + ``feature_dim`` + ``description``).
   Apply the bare-module shim so joblib can deserialize the classical
   pickles (RF/XGB) that were saved under bare module paths.
2. Reconstruct the CNN with :class:`mars_tyxn.train_cnn.DeeperCNN_GAP_v2`
   from ``cnn_config`` and load the saved ``state_dict``.
3. For each candidate junction:

   a. Center-crop a ``patch_size x patch_size`` window from the
      zero-padded skeleton at ``(node_x, node_y)``.
   b. Compute geometry features via
      :func:`mars_tyxn.classical_feature_builder.extract_geometry_feature_vector`
      on the binary patch.
   c. Build the 2-channel CNN input ``[skel * gauss, mask * gauss]``
      where ``mask`` is the 3x3 dilation of the binarized skeleton and
      ``gauss = exp(-((x-c)^2 + (y-c)^2) / (2*sigma^2))``.
   d. Compute base-model probabilities (RF, XGB, CNN softmax).
   e. Stack: ``X_meta = hstack([rf_proba, xgb_proba, cnn_proba,
      geom_features])`` and call ``meta_classifier.predict_proba``.
   f. Project to ``{"node_x", "node_y", "label", "confidence", "probs",
      "geometry_label"}``.

Performance note: the canonical Phase 6 production tile carries roughly
5-20 candidates; per-candidate looping is more than fast enough. If a
future caller needs batched CNN inference, that is a clean optimization
boundary inside :func:`predict_stacking` that does not change the public
API.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "StackingHandle",
    "predict_stacking",
]


@dataclasses.dataclass
class StackingHandle:
    """Cached deserialized state for a single stacking bundle.

    Held in a module-level cache keyed by the resolved bundle path so that
    repeated calls to :func:`predict_stacking` against the same checkpoint
    do not re-deserialize the ~35 MB pickle.
    """

    rf_model: Any
    xgb_model: Any
    meta_classifier: Any
    cnn_model: Any  # torch.nn.Module
    gauss_mask: np.ndarray
    patch_size: int
    gaussian_sigma: float
    idx_to_label: Any  # list[str] or dict[int, str]
    label_map: Any  # dict[str, int]
    cnn_config: dict
    geometry_trace_len: int


# Module-level cache keyed by str(Path(bundle_path).resolve()).
_BUNDLE_CACHE: dict[str, StackingHandle] = {}


def _apply_bare_module_shim() -> None:
    """Mirror ``mars_tyxn.<sub>`` to bare ``<sub>`` for joblib pickles.

    The Ch 2 v7 classical heads (RF / XGB / etc.) were pickled under bare
    module paths (``classical_feature_builder`` etc., no ``mars_tyxn.``
    prefix). ``setdefault`` makes this idempotent and safe to call from
    multiple entry points.
    """
    import sys

    try:
        import mars_tyxn.classical_feature_builder as _cfb
        import mars_tyxn.hog_transformer as _ht
        import mars_tyxn.junction_geometry as _jg
        import mars_tyxn.junction_proposals as _jp

        sys.modules.setdefault("classical_feature_builder", _cfb)
        sys.modules.setdefault("hog_transformer", _ht)
        sys.modules.setdefault("junction_geometry", _jg)
        sys.modules.setdefault("junction_proposals", _jp)
    except ImportError:
        # If a submodule is missing, fall through. joblib.load will raise
        # the underlying error cleanly.
        pass


def _build_gauss_mask(patch_size: int, sigma: float) -> np.ndarray:
    """Build the ``patch_size x patch_size`` Gaussian center mask.

    Matches the notebook cell exactly::

        h = w = CNN_PATCH_SIZE
        y_grid, x_grid = np.mgrid[0:h, 0:w]
        gauss_mask = np.exp(-((x_grid - h//2)**2 + (y_grid - w//2)**2)
                            / (2 * GAUSSIAN_SIGMA**2)).astype(np.float32)
    """
    y_grid, x_grid = np.mgrid[0:patch_size, 0:patch_size]
    cx = patch_size // 2
    cy = patch_size // 2
    return np.exp(
        -(((x_grid - cx) ** 2) + ((y_grid - cy) ** 2)) / (2.0 * float(sigma) ** 2)
    ).astype(np.float32)


def _load_handle(
    bundle_path: Path, device: Any
) -> StackingHandle:
    """Load (or fetch from cache) a :class:`StackingHandle` for ``bundle_path``."""
    import torch

    key = str(Path(bundle_path).resolve())
    cached = _BUNDLE_CACHE.get(key)
    if cached is not None:
        # Move CNN to the requested device opportunistically. Cheap if
        # already on it.
        cached.cnn_model.to(device)
        cached.cnn_model.eval()
        return cached

    _apply_bare_module_shim()

    import joblib

    bundle = joblib.load(key)

    cnn_config = dict(bundle["cnn_config"])
    in_channels = int(cnn_config.get("in_channels", 2))
    dropout = float(cnn_config.get("dropout", 0.0))
    num_classes = int(cnn_config.get("num_classes", 4))

    # Lazy import: keep torch out of module load.
    from mars_tyxn.train_cnn import DeeperCNN_GAP_v2

    cnn_model = DeeperCNN_GAP_v2(
        num_classes=num_classes, in_channels=in_channels, dropout=dropout
    )
    cnn_model.load_state_dict(bundle["cnn_state_dict"])
    cnn_model.to(device)
    cnn_model.eval()

    patch_size = int(bundle["patch_size"])
    gaussian_sigma = float(bundle["gaussian_sigma"])
    gauss_mask = _build_gauss_mask(patch_size, gaussian_sigma)

    # Source geometry_trace_len from the bundle so the classical geometry
    # feature vector matches the meta classifier's training-time feature
    # distribution. Default 40 keeps backward compat with the v7 deposit.
    bundle_trace_len = int(bundle.get("geometry_trace_len", 40))

    handle = StackingHandle(
        rf_model=bundle["rf_model"],
        xgb_model=bundle["xgb_model"],
        meta_classifier=bundle["meta_classifier"],
        cnn_model=cnn_model,
        gauss_mask=gauss_mask,
        patch_size=patch_size,
        gaussian_sigma=gaussian_sigma,
        idx_to_label=bundle["idx_to_label"],
        label_map=bundle["label_map"],
        cnn_config=cnn_config,
        geometry_trace_len=bundle_trace_len,
    )
    _BUNDLE_CACHE[key] = handle
    return handle


def _resolve_label(idx: int, idx_to_label: Any) -> str:
    """Resolve a class index to a label string.

    The bundle's ``idx_to_label`` is a list in the v7 deposit
    (``['N', 'T', 'X', 'Y']``) but we accept a dict too so a future repickle
    that swaps to ``{0: 'N', ...}`` keeps working without code change.
    """
    if isinstance(idx_to_label, dict):
        return str(idx_to_label[int(idx)])
    return str(idx_to_label[int(idx)])


def _ordered_class_keys(idx_to_label: Any, num_classes: int) -> list[str]:
    """Return the class label string for each class index in [0, num_classes)."""
    return [_resolve_label(i, idx_to_label) for i in range(num_classes)]


def _project_probs_to_full(
    proba: np.ndarray,
    classes: Any,
    idx_to_label: Any,
) -> dict[str, float]:
    """Project a base-model proba vector to the canonical NTXY dict.

    Some sklearn estimators (RF/XGB) attach a ``classes_`` attribute that
    may be a subset of the full label set if a class was unseen at fit
    time. We map each estimator-class index to its label string via
    ``idx_to_label`` and zero-fill any missing labels in the output dict.
    """
    full = {"N": 0.0, "T": 0.0, "X": 0.0, "Y": 0.0}
    proba = np.asarray(proba, dtype=np.float64).ravel()
    if classes is None:
        # No class metadata: assume canonical NTXY order matches
        # idx_to_label for the first 4 entries.
        for i, p in enumerate(proba):
            full[_resolve_label(i, idx_to_label)] = float(p)
        return full
    classes = list(classes)
    for cls_idx, p in zip(classes, proba):
        full[_resolve_label(cls_idx, idx_to_label)] = float(p)
    return full


def _classes_for(estimator: Any) -> Any:
    """Best-effort fetch of a sklearn-style ``classes_`` attribute."""
    return getattr(estimator, "classes_", None)


def _crop_patch(
    padded: np.ndarray, node_x: int, node_y: int, patch_size: int, half: int
) -> np.ndarray:
    """Center-crop a ``patch_size x patch_size`` window from a padded skeleton.

    The padded array's coordinate frame is shifted by ``half`` relative to
    the unpadded one, so the center of the window for raw coordinate
    ``(node_x, node_y)`` lands at ``(node_x + half, node_y + half)`` in
    the padded frame.
    """
    cx = int(node_x) + half
    cy = int(node_y) + half
    y0 = cy - half
    y1 = y0 + patch_size
    x0 = cx - half
    x1 = x0 + patch_size
    return padded[y0:y1, x0:x1]


def predict_stacking(
    skeleton: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    bundle_path: Any,
    device: Any = None,
    geometry_trace_len: int | None = None,
) -> list[dict]:
    """Run the canonical Ch 2 v7 stacking ensemble on a list of candidates.

    Parameters
    ----------
    skeleton
        ``HxW`` array. Accepts ``{0, 1}`` or ``{0, 255}`` valued uint8 (or
        bool); coerced to ``{0, 255}`` early so internal patch processing
        matches the notebook's saved-patch format
        (``cv2.imread(IMREAD_GRAYSCALE)`` of PNGs that were saved as
        ``patch * 255``).
    candidates
        Per-junction candidate dicts. Each must contain integer ``node_x``
        and ``node_y`` keys. Other keys (proposal_source, gap_len_px, etc.)
        are ignored by this path -- the stacking head uses only the patch
        geometry features computed from ``skeleton``.
    bundle_path
        Path to the ``stacking_gauss40_best.pkl`` produced by the v7
        notebook. Cached per-resolved-path so repeated calls do not re-load
        the ~35 MB pickle.
    device
        Optional torch device (or string). ``None`` -> cuda if available
        else cpu. The CNN forward pass runs on this device; classical
        models stay on CPU.
    geometry_trace_len
        Branch trace length used by
        :func:`extract_geometry_feature_vector`. ``None`` (the default)
        reads the value stored in the bundle (key ``geometry_trace_len``,
        falling back to ``40`` for older bundles that pre-date the key),
        which is the value the meta classifier was trained against. Pass
        an explicit ``int`` only to override the bundle's value.

    Returns
    -------
    list[dict]
        One dict per candidate, in the same order. Schema::

            {
                "node_x": int,
                "node_y": int,
                "label": "N" | "T" | "X" | "Y",
                "confidence": float in [0, 1],
                "probs": {"N": float, "T": float, "X": float, "Y": float},
                "geometry_label": str,
            }

        Empty candidate lists return ``[]`` without loading the bundle.
    """
    # Empty short-circuit: do not load the bundle for nothing.
    if len(candidates) == 0:
        return []

    # Lazy imports: keep torch / scipy out of module load. mars_tyxn.* is
    # the consumer's lazy-import boundary too -- top-level
    # ``import mars_tyxn`` does pull these in via __init__.py, but
    # downstream callers that import this module directly should not pay
    # the torch cost until they actually call predict_stacking.
    import torch
    from scipy.ndimage import binary_dilation

    from mars_tyxn.classical_feature_builder import (
        extract_geometry_feature_vector,
    )

    # Resolve device.
    if device is None:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        torch_device = device
    else:
        torch_device = torch.device(device)

    handle = _load_handle(Path(bundle_path), torch_device)

    # Resolve effective geometry_trace_len: when the caller leaves it as
    # None, use the value the meta classifier was trained against (stored
    # in the bundle). An explicit int wins.
    if geometry_trace_len is None:
        effective_trace_len = handle.geometry_trace_len
    else:
        effective_trace_len = int(geometry_trace_len)

    patch_size = handle.patch_size
    half = patch_size // 2

    # Coerce the skeleton to {0, 255} uint8. The notebook reads patches via
    # cv2.imread(..., IMREAD_GRAYSCALE), which yields uint8 in {0, 255}.
    # Inside the CNN preprocessing the notebook divides by 255.0 to land
    # in {0, 1}; we keep that exact pipeline so numerics line up.
    skel_arr = np.asarray(skeleton)
    if skel_arr.dtype == bool:
        skel_u8 = skel_arr.astype(np.uint8) * 255
    else:
        # Treat any nonzero pixel as "on", same as the CLI's input-mode
        # skeleton branch: ``raw_skel = (binary > 0).astype(np.uint8)``.
        skel_u8 = ((skel_arr > 0).astype(np.uint8)) * 255

    h, w = skel_u8.shape
    padded = np.pad(
        skel_u8, pad_width=half, mode="constant", constant_values=0
    )

    # Class layout for proba projection. The bundle's idx_to_label is a
    # list/dict mapping classifier-class-index to label string.
    canonical_keys = ["N", "T", "X", "Y"]

    rf_classes = _classes_for(handle.rf_model)
    xgb_classes = _classes_for(handle.xgb_model)
    meta_classes = _classes_for(handle.meta_classifier)

    # Number of CNN output classes (drives the gauss-cnn proba shape used
    # in the meta input). Matches ``cnn_config['num_classes']``.
    cnn_num_classes = int(handle.cnn_config.get("num_classes", 4))
    cnn_class_keys = _ordered_class_keys(handle.idx_to_label, cnn_num_classes)

    results: list[dict] = []

    # Pre-clip candidate coords to the unpadded image bounds. This matches
    # the wrapper's old behavior and protects against off-by-one upstream
    # bugs where a candidate sits one pixel outside the tile.
    with torch.no_grad():
        for cand in candidates:
            node_x = int(np.clip(int(cand["node_x"]), 0, w - 1))
            node_y = int(np.clip(int(cand["node_y"]), 0, h - 1))

            patch_u8 = _crop_patch(padded, node_x, node_y, patch_size, half)

            # 1. Geometry features from the binary patch.
            patch_binary_f32 = (patch_u8 > 0).astype(np.float32)
            geom_features = extract_geometry_feature_vector(
                patch_binary_f32, trace_len=effective_trace_len
            ).astype(np.float32)

            # 2. CNN input: 2-channel [skel * gauss, mask * gauss].
            skel = patch_u8.astype(np.float32) / 255.0
            mask = binary_dilation(
                skel > 0.5, structure=np.ones((3, 3), dtype=bool)
            ).astype(np.float32)
            skel = skel * handle.gauss_mask
            mask = mask * handle.gauss_mask
            cnn_in = (
                torch.tensor(np.stack([skel, mask]), dtype=torch.float32)
                .unsqueeze(0)
                .to(torch_device)
            )

            # 3. CNN softmax.
            cnn_logits = handle.cnn_model(cnn_in)
            cnn_proba_vec = (
                torch.softmax(cnn_logits, dim=1)
                .cpu()
                .numpy()
                .astype(np.float64)
                .ravel()
            )

            # 4. Classical base probas (geometry features only).
            geom_2d = geom_features.reshape(1, -1)
            rf_proba_vec = np.asarray(
                handle.rf_model.predict_proba(geom_2d), dtype=np.float64
            ).ravel()
            xgb_proba_vec = np.asarray(
                handle.xgb_model.predict_proba(geom_2d), dtype=np.float64
            ).ravel()

            # 5. Stack meta input. The notebook uses the raw RF/XGB/CNN
            # proba vectors (not the canonicalized NTXY-ordered dicts) +
            # geometry features. Match that exactly: meta_classifier was
            # trained on hstack([rf_p, xgb_p, cnn_p, X_geom]).
            x_meta = np.hstack(
                [rf_proba_vec, xgb_proba_vec, cnn_proba_vec, geom_features]
            ).reshape(1, -1)

            # 6. Meta predict + predict_proba.
            meta_pred_idx = int(handle.meta_classifier.predict(x_meta)[0])
            meta_proba_vec = np.asarray(
                handle.meta_classifier.predict_proba(x_meta), dtype=np.float64
            ).ravel()
            label = _resolve_label(meta_pred_idx, handle.idx_to_label)

            # 7. Project meta proba to canonical NTXY dict.
            probs_full = _project_probs_to_full(
                meta_proba_vec, meta_classes, handle.idx_to_label
            )
            confidence = float(probs_full.get(label, 0.0))
            confidence = max(0.0, min(1.0, confidence))

            results.append(
                {
                    "node_x": node_x,
                    "node_y": node_y,
                    "label": label,
                    "confidence": confidence,
                    "probs": {k: float(probs_full[k]) for k in canonical_keys},
                    "geometry_label": "Unknown",
                }
            )

    return results
