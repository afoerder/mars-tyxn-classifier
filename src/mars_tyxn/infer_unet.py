"""Stage-2 U-Net wrapper for the Mars TCP Ch 3 inference pipeline.

This module exposes :func:`infer_unet_mask`, a pure function that runs the
production U-Net (mit_b3 SegFormer-style SMP encoder by default) on a single
768x768 uint8 tile and returns a binary fracture mask. Behavior is intended
to match running ``predict_unet.py`` from the command line with all flags
left at their defaults (``--threshold auto``, ``--mask-no-data auto`` etc.)
within the project's Tier-A (Stage 2) / Tier-C (Stage 4) acceptance
tolerances. Loaded models are cached at module level keyed by a resolved
(model_path, device.type, device.index) tuple.

Default-flag CLI behavior preserved: threshold from metrics, optional CLAHE
per ``metrics.config.preprocess_clahe``, border-only no-data masking when
``metrics.config.ignore_no_data`` is set. Source extracted from
``predict_unet.py:main()`` lines 383-634. CLI features intentionally omitted
because they are off by default in the CLI: connectivity closing, hysteresis
thresholding, ``save_probs``, overlay write, and on-disk mask I/O.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from mars_tyxn.predict_unet import (
    SimpleUNet,
    compute_valid_region,
    resolve_model_config,
    resolve_threshold,
)


# Cache key is (resolved_path_str, device.type, device.index). See _cache_key.
_MODEL_CACHE: Dict[Tuple, Tuple[nn.Module, dict, float, int, str]] = {}


def _cache_key(model_path: Path, device: torch.device) -> tuple:
    """Robust cache key tolerant of equivalent Path/device representations.

    ``Path("/a/./b.pth")`` and ``Path("/a/b.pth")`` resolve to the same file
    but stringify differently; ``torch.device("cuda")`` and
    ``torch.device("cuda:0")`` reference the same physical device but have
    different ``.index`` (``None`` vs ``0``). Resolve the path and normalize
    the unindexed CUDA case to the current CUDA device before keying.
    """
    resolved = str(Path(model_path).resolve())
    dev = torch.device(device)
    index = dev.index
    if index is None and dev.type == "cuda" and torch.cuda.is_available():
        index = torch.cuda.current_device()
    return (resolved, dev.type, index)


def _load_model_and_metadata(
    model_path: Path,
    metrics_path: Path,
    device: torch.device,
) -> Tuple[nn.Module, dict, float, int, str]:
    """Load (or fetch from cache) the U-Net and resolve auto-config from metrics.

    Returns
    -------
    model : nn.Module
        Eval-mode model on the requested device.
    config : dict
        The ``config`` block from metrics.json (used for image_size/encoder name).
    threshold : float
        Probability threshold for binarization (``best_threshold`` from metrics.json).
    image_size : int
        Internal forward resolution used during training (e.g. 704 for mit_b3).
    encoder_type : str
        Either ``"smp"`` or ``"custom"``.
    """
    cache_key = _cache_key(model_path, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    with open(metrics_path, "r") as f:
        metrics_payload = json.load(f)

    threshold = float(resolve_threshold("auto", metrics_payload))

    config = metrics_payload.get("config", {}) or {}
    image_size = int(config.get("image_size", 384))

    (
        unet_depth,
        unet_base_channels,
        unet_decoder_dropout,
        unet_norm,
        unet_gn_groups,
        unet_upsample_mode,
        unet_deep_supervision,
    ) = resolve_model_config(
        "auto", "auto", "auto", "auto", "auto", "auto", "auto", metrics_payload
    )

    encoder_type = config.get("encoder_type", "custom")
    if encoder_type == "custom":
        # Same fallback the CLI uses: detect smp from state_dict key prefixes.
        # encoder_name is preserved from metrics.json (e.g. "mit_b3") via
        # config.get(...)'s existing-value-wins semantics; the resnet-name guess
        # below is only the fallback when metrics.json has no encoder_name. SMP
        # accepts mit_b3 directly and matches the published Ch 2 CLI behavior.
        state_peek = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if any(k.startswith("net.encoder.") for k in state_peek.keys()):
            encoder_type = "smp"
            has_layer4 = any("layer4" in k for k in state_peek.keys())
            has_layer3 = any("layer3" in k for k in state_peek.keys())
            if not has_layer3:
                detected_enc = "resnet18"
            elif not has_layer4:
                detected_enc = "resnet34"
            else:
                has_conv3 = any("conv3" in k for k in state_peek.keys())
                detected_enc = "resnet50" if has_conv3 else "resnet34"
            config["encoder_name"] = config.get("encoder_name", detected_enc)
        del state_peek

    if encoder_type == "smp":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for smp encoder models. "
                "pip install segmentation-models-pytorch"
            ) from exc

        encoder_name = config.get("encoder_name", "resnet34")
        encoder_weights = config.get("encoder_weights", "imagenet")
        imagenet_normalize = config.get("imagenet_normalize", "auto")

        class SmpGrayscaleUnet(nn.Module):
            def __init__(self, enc_name: str, enc_weights):
                super().__init__()
                self.net = smp.Unet(
                    encoder_name=enc_name,
                    encoder_weights=None,
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
                should_normalize = (
                    imagenet_normalize == "1"
                    or (
                        imagenet_normalize == "auto"
                        and enc_weights is not None
                        and str(enc_weights).lower() == "imagenet"
                    )
                )
                if should_normalize:
                    self.register_buffer(
                        "img_mean",
                        torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                    )
                    self.register_buffer(
                        "img_std",
                        torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                    )
                else:
                    self.img_mean = None
                    self.img_std = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.img_mean is not None:
                    x = (x - self.img_mean) / self.img_std
                return self.net(x)

        model = SmpGrayscaleUnet(encoder_name, encoder_weights).to(device)
    else:
        if image_size < (2 ** unet_depth):
            raise ValueError(
                f"image_size {image_size} too small for UNet depth {unet_depth}; "
                f"need >= {2 ** unet_depth}."
            )
        model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=unet_base_channels,
            depth=unet_depth,
            decoder_dropout=unet_decoder_dropout,
            norm=unet_norm,
            gn_groups=unet_gn_groups,
            upsample_mode=unet_upsample_mode,
            deep_supervision=unet_deep_supervision,
        ).to(device)

    state = torch.load(str(model_path), map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, config, threshold, image_size, encoder_type)
    return _MODEL_CACHE[cache_key]


def infer_unet_mask(
    image_u8: np.ndarray,
    model_path: Path,
    metrics_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run the production U-Net on one tile and return a binary fracture mask.

    Parameters
    ----------
    image_u8 : np.ndarray
        Single-channel (or BGR/RGB) ``uint8`` tile, typically 768x768. Multi-channel
        inputs are converted to grayscale via ``cv2.cvtColor(BGR2GRAY)``.
    model_path : Path
        Path to the trained ``.pth`` checkpoint (e.g. ``mit_b3_skelrecall.pth``).
    metrics_path : Path
        Path to the companion ``..._metrics.json`` (carries ``best_threshold``,
        ``image_size``, encoder config).
    device : torch.device
        Inference device. Cached separately per device.

    Returns
    -------
    np.ndarray
        Binary mask the same H x W as the input, ``uint8`` valued in ``{0, 255}``.

    Notes
    -----
    Default-flag CLI behavior preserved: threshold from metrics, optional
    CLAHE per ``metrics.config.preprocess_clahe``, border-only no-data
    masking when ``metrics.config.ignore_no_data`` is set (using
    ``no_data_threshold``, ``no_data_border_only``, and
    ``no_data_border_pad`` from the metrics config). The mask is thresholded
    directly from the bilinearly-upsampled probability map at the full input
    resolution. Output matches ``predict_unet.py`` invoked with all flags at
    default within the project's Tier-A (Stage 2) acceptance tolerances.

    The wrapper omits CLI features that are off by default in
    ``predict_unet.py``: connectivity closing (``--connectivity-close-iters``),
    hysteresis thresholding (``--hysteresis-low``), ``--save-probs``, and
    overlay/mask file writes.
    """
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    model, config, threshold, image_size, _encoder_type = _load_model_and_metadata(
        model_path, metrics_path, device
    )

    if image_u8.ndim == 3:
        image = cv2.cvtColor(image_u8, cv2.COLOR_BGR2GRAY)
    else:
        image = image_u8
    if image.dtype != np.uint8:
        # Match CLI percentile-stretch fallback for non-uint8 inputs.
        lo, hi = np.percentile(image, (2, 98))
        if hi <= lo:
            hi = lo + 1
        image = np.clip(
            (image.astype(np.float64) - lo) / (hi - lo) * 255.0, 0, 255
        ).astype(np.uint8)

    original_h, original_w = image.shape[:2]

    # No-data settings: match CLI ``--mask-no-data auto`` etc. semantics by
    # reading defaults straight from metrics.config (the production
    # mit_b3_skelrecall metrics has ignore_no_data=True). Compute the valid
    # region BEFORE CLAHE — CLAHE brightens nodata border pixels and breaks
    # the threshold; this mirrors predict_unet.py:574-582.
    apply_nodata_mask = bool(config.get("ignore_no_data", False))
    nodata_threshold = int(config.get("no_data_threshold", 3))
    nodata_border_only = bool(config.get("no_data_border_only", True))
    nodata_border_pad = int(config.get("no_data_border_pad", 1))
    valid = None
    if apply_nodata_mask:
        valid = compute_valid_region(
            image,
            nodata_threshold,
            border_only=nodata_border_only,
            border_pad=nodata_border_pad,
        )

    # Optional CLAHE if the training metrics enabled it (default off).
    preprocess_clahe = config.get("preprocess_clahe", False)
    if preprocess_clahe:
        clip = float(config.get("preprocess_clahe_clip", 3.0) or 3.0)
        clahe_obj = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        image = clahe_obj.apply(image)

    resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        elif isinstance(logits, dict):
            logits = logits.get("main", next(iter(logits.values())))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    probs_full = cv2.resize(
        probs, (original_w, original_h), interpolation=cv2.INTER_LINEAR
    )
    pred_mask = (probs_full >= threshold).astype(np.uint8) * 255
    if valid is not None:
        # Match predict_unet.py line 627: zero out predictions in the nodata
        # region. Done after thresholding so the valid mask doesn't influence
        # probabilities themselves.
        pred_mask[valid == 0] = 0
    return pred_mask


__all__ = ["infer_unet_mask"]
