"""Tests for :mod:`mars_tyxn.infer_stacking`.

These exercises the canonical Ch 2 v7 stacking inference path against the
real ``stacking_gauss40_best.pkl`` checkpoint shipped under
``mars-tyxn-classifier/models/classifiers/``.

The bundle's ``xgb_model`` segfaults on macOS arm64 during deserialization
(see ``feedback_marstcp_ch2_pickle_arch_lockin``), so every test that
touches the bundle gates on Linux x86_64. Running the suite on Mac yields
a clean SKIP rather than a "Python quit unexpectedly" dialog.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


CHECKPOINT_PATH = Path(__file__).resolve().parents[1] / "models" / "classifiers" / "stacking_gauss40_best.pkl"


def _bundle_loadable() -> bool:
    """Probe whether the runtime can deserialize the stacking bundle.

    The Ch 2 v7 joblib pickles depend on (a) bare module names being
    importable (no ``mars_tyxn.`` prefix) and (b) a numpy/sklearn/xgboost
    ABI that matches the training-time pickles. On macOS arm64 the xgboost
    ``__setstate__`` segfaults below the Python layer; pytest cannot catch
    it, so we skip cleanly. On Linux x86_64 with a matching xgboost wheel
    the load succeeds.
    """
    if not CHECKPOINT_PATH.is_file():
        return False
    if sys.platform == "darwin":
        return False
    try:
        # Apply the bare-module shim before joblib.load attempts to look
        # up classical_feature_builder etc. without the mars_tyxn prefix.
        import mars_tyxn.classical_feature_builder as _cfb
        import mars_tyxn.hog_transformer as _ht
        import mars_tyxn.junction_geometry as _jg
        import mars_tyxn.junction_proposals as _jp

        sys.modules.setdefault("classical_feature_builder", _cfb)
        sys.modules.setdefault("hog_transformer", _ht)
        sys.modules.setdefault("junction_geometry", _jg)
        sys.modules.setdefault("junction_proposals", _jp)

        import joblib

        joblib.load(CHECKPOINT_PATH)
        return True
    except Exception:
        return False


_BUNDLE_RUNNABLE = _bundle_loadable()
_SKIP_REASON = (
    f"Stacking bundle not loadable on this runtime "
    f"(checkpoint={CHECKPOINT_PATH}, platform={sys.platform}). "
    f"Real-checkpoint tests must run on Linux x86_64 with the xgboost "
    f"ABI matching the v7 pickle."
)


def _make_synthetic_skeleton(seed: int = 0, size: int = 256) -> np.ndarray:
    """Build a sparse-noise skeleton with at least a handful of pixels on."""
    rng = np.random.default_rng(seed)
    skel = (rng.random((size, size)) < 0.01).astype(np.uint8)
    # Add a couple of straight lines so geometry features are non-trivial.
    skel[size // 2, :] = 1
    skel[:, size // 2] = 1
    return skel


def test_empty_candidates_returns_empty_list_no_bundle_load(tmp_path: Path) -> None:
    """Empty candidate list must short-circuit before any bundle load.

    This test runs unconditionally on every platform: it deliberately
    points at a path that does NOT exist, so any bundle load attempt
    would raise. The expected behavior is a clean empty return.
    """
    from mars_tyxn import predict_stacking

    skeleton = np.zeros((128, 128), dtype=np.uint8)
    out = predict_stacking(
        skeleton,
        [],
        bundle_path=tmp_path / "does_not_exist.pkl",
        device="cpu",
    )
    assert out == []


@pytest.mark.skipif(not _BUNDLE_RUNNABLE, reason=_SKIP_REASON)
def test_predict_stacking_single_candidate_schema() -> None:
    """One candidate -> one result dict with the canonical schema."""
    from mars_tyxn import predict_stacking

    skel = _make_synthetic_skeleton(seed=0, size=256)
    cand = [{"node_x": 100, "node_y": 100}]
    out = predict_stacking(
        skel, cand, bundle_path=CHECKPOINT_PATH, device="cpu"
    )

    assert len(out) == 1
    r = out[0]
    assert set(r.keys()) >= {
        "node_x",
        "node_y",
        "label",
        "confidence",
        "probs",
        "geometry_label",
    }
    assert r["node_x"] == 100
    assert r["node_y"] == 100
    assert r["label"] in {"N", "T", "X", "Y"}
    assert 0.0 <= r["confidence"] <= 1.0
    assert set(r["probs"].keys()) == {"N", "T", "X", "Y"}


@pytest.mark.skipif(not _BUNDLE_RUNNABLE, reason=_SKIP_REASON)
def test_predict_stacking_probs_sum_to_one() -> None:
    """Per-class probs must sum to ~1.0 within 0.01."""
    from mars_tyxn import predict_stacking

    skel = _make_synthetic_skeleton(seed=1, size=256)
    cands = [
        {"node_x": 80, "node_y": 80},
        {"node_x": 120, "node_y": 130},
        {"node_x": 128, "node_y": 128},
    ]
    out = predict_stacking(
        skel, cands, bundle_path=CHECKPOINT_PATH, device="cpu"
    )
    assert len(out) == len(cands)
    for r in out:
        s = sum(r["probs"].values())
        assert abs(s - 1.0) < 0.01, f"probs sum {s:.6f} not within 1e-2 of 1.0"
        assert r["label"] in {"N", "T", "X", "Y"}


@pytest.mark.skipif(not _BUNDLE_RUNNABLE, reason=_SKIP_REASON)
def test_predict_stacking_handle_caching() -> None:
    """Repeated calls against the same bundle path reuse the cached handle."""
    from mars_tyxn import predict_stacking
    from mars_tyxn.infer_stacking import _BUNDLE_CACHE

    skel = _make_synthetic_skeleton(seed=2, size=256)
    key = str(Path(CHECKPOINT_PATH).resolve())
    _BUNDLE_CACHE.pop(key, None)

    predict_stacking(
        skel, [{"node_x": 50, "node_y": 50}],
        bundle_path=CHECKPOINT_PATH, device="cpu",
    )
    handle1 = _BUNDLE_CACHE[key]

    predict_stacking(
        skel, [{"node_x": 60, "node_y": 60}],
        bundle_path=CHECKPOINT_PATH, device="cpu",
    )
    handle2 = _BUNDLE_CACHE[key]

    assert handle1 is handle2, "Bundle handle should be cached per resolved path"
