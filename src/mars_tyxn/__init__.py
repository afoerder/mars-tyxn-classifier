"""mars_tyxn — Mars TYXN junction classifier (Ch 2 v7) as a pip-installable package.

Re-exports the public API consumed by downstream pipelines (e.g. MarsTCPDetection
Ch 3). Importing the package as a whole loads training-side modules as well, so
that ``import mars_tyxn`` is sufficient and submodule paths still work.
"""

from mars_tyxn.junction_proposals import (
    BridgeSearchConfig,
    VirtualBridgeStats,
    collect_virtual_bridge_proposals,
)
from mars_tyxn.classical_feature_builder import (
    GEOMETRY_FEATURE_NAMES,
    extract_geometry_feature_vector,
)
from mars_tyxn.junction_geometry import (
    analyze_local_junction,
    compute_patch_geometry,
    degree_map,
)
from mars_tyxn.infer_unet import infer_unet_mask
from mars_tyxn.infer_ensemble import (
    EnsembleInferenceConfig,
    predict_per_junction,
)
from mars_tyxn.infer_stacking import (
    StackingHandle,
    predict_stacking,
)

__all__ = [
    "BridgeSearchConfig",
    "EnsembleInferenceConfig",
    "GEOMETRY_FEATURE_NAMES",
    "StackingHandle",
    "VirtualBridgeStats",
    "analyze_local_junction",
    "collect_virtual_bridge_proposals",
    "compute_patch_geometry",
    "degree_map",
    "extract_geometry_feature_vector",
    "infer_unet_mask",
    "predict_per_junction",
    "predict_stacking",
]
