import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import torch
from torch import nn


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DS_SCALE_ORDER = (2, 4, 8)


def build_norm_layer(num_channels, norm, gn_groups):
    if norm == "gn":
        groups = min(max(1, gn_groups), num_channels)
        while num_channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, num_channels)
    return nn.BatchNorm2d(num_channels)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm="bn", gn_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(out_channels, norm=norm, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            build_norm_layer(out_channels, norm=norm, gn_groups=gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_channels=32,
        depth=4,
        decoder_dropout=0.0,
        norm="bn",
        gn_groups=8,
        upsample_mode="transpose",
        deep_supervision=False,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("UNet depth must be >= 2.")
        if norm not in {"bn", "gn"}:
            raise ValueError("norm must be either 'bn' or 'gn'.")
        if upsample_mode not in {"transpose", "bilinear"}:
            raise ValueError("upsample_mode must be either 'transpose' or 'bilinear'.")

        self.deep_supervision = deep_supervision
        self.decoder_scales = [2 ** (depth - 1 - i) for i in range(depth)]

        channels = [base_channels * (2**i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for ch in channels:
            self.encoders.append(DoubleConv(prev_channels, ch, norm=norm, gn_groups=gn_groups))
            prev_channels = ch

        self.pool = nn.MaxPool2d(2)
        bottleneck_channels = channels[-1] * 2
        self.bottleneck = DoubleConv(channels[-1], bottleneck_channels, norm=norm, gn_groups=gn_groups)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = bottleneck_channels
        for ch in reversed(channels):
            if upsample_mode == "transpose":
                upsample = nn.ConvTranspose2d(dec_in, ch, kernel_size=2, stride=2)
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dec_in, ch, kernel_size=1, bias=False),
                )
            self.upconvs.append(upsample)
            decoder = [DoubleConv(ch * 2, ch, norm=norm, gn_groups=gn_groups)]
            if decoder_dropout > 0.0:
                decoder.append(nn.Dropout2d(decoder_dropout))
            self.decoders.append(nn.Sequential(*decoder))
            dec_in = ch

        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.side_heads = nn.ModuleDict()
        if self.deep_supervision:
            for stage_idx, ch in enumerate(reversed(channels)):
                scale = self.decoder_scales[stage_idx]
                if scale in DS_SCALE_ORDER:
                    self.side_heads[str(scale)] = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = decoder(torch.cat([x, skip], dim=1))

        return self.out_conv(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crack masks using a trained SimpleUNet checkpoint.")
    parser.add_argument("--model-path", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for predicted outputs.")
    parser.add_argument("--image-size", default="auto", help="Model input size used during training, or 'auto' from metrics JSON.")
    parser.add_argument("--encoder-type", default="auto", help="Model type: 'custom' (SimpleUNet), 'smp', or 'auto' (from metrics).")
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Probability threshold for binary mask, or 'auto' to load best threshold from metrics JSON.",
    )
    parser.add_argument(
        "--mask-no-data",
        default="auto",
        help="Set to 1/0 to enable/disable masking predictions in no-data regions, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--no-data-threshold",
        default="auto",
        help="Pixel threshold for no-data masking (image <= threshold is invalid), or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--no-data-border-only",
        default="auto",
        help="Set 1/0 to mask only border-connected low-intensity no-data regions, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--no-data-border-pad",
        default="auto",
        help="Optional dilation (pixels) for no-data regions after detection, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--unet-depth",
        default="auto",
        help="UNet encoder depth used for training, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--unet-base-channels",
        default="auto",
        help="UNet base channels used for training, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--unet-decoder-dropout",
        default="auto",
        help="UNet decoder dropout used for training, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--norm",
        default="auto",
        help="Normalization used for training: bn/gn, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--gn-groups",
        default="auto",
        help="GroupNorm groups used for training, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--upsample-mode",
        default="auto",
        help="Decoder upsample mode used for training: transpose/bilinear, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--deep-supervision",
        default="auto",
        help="Set 1/0 to match training deep supervision module layout, or 'auto' from metrics JSON.",
    )
    parser.add_argument(
        "--connectivity-close-iters",
        type=int,
        default=0,
        help="Optional binary closing iterations to connect small gaps (default: 0/off).",
    )
    parser.add_argument(
        "--hysteresis-low",
        type=float,
        default=None,
        help="Low threshold for hysteresis thresholding. When set, --threshold becomes the "
             "high threshold. Pixels above low are kept only if connected to pixels above high. "
             "Bridges gaps and captures faint fractures without noise.",
    )
    parser.add_argument(
        "--save-probs",
        action="store_true",
        help="Save the raw probability map as a .npy file alongside the mask.",
    )
    parser.add_argument("--device", default="", help="Force device: cpu/cuda/mps. Leave empty for auto.")
    return parser.parse_args()


def resolve_device(user_device: str):
    override = user_device.strip().lower()
    if override:
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is unavailable.")
        if override == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise RuntimeError("Requested --device mps, but MPS is unavailable.")
        if override not in {"cpu", "cuda", "mps"}:
            raise RuntimeError(f"Unknown --device '{user_device}'. Use cpu/cuda/mps.")
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def collect_images(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    files = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        if os.path.splitext(name)[1].lower() in VALID_EXTENSIONS:
            files.append(full)
    return files


def load_metrics_for_model(model_path: str):
    model_base = os.path.splitext(model_path)[0]
    candidates = [model_base + "_metrics.json"]
    if model_base.endswith("_best"):
        candidates.append(model_base[:-5] + "_metrics.json")

    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload, path
    return None, ""


def resolve_threshold(threshold_arg: str, metrics_payload):
    if threshold_arg.lower() != "auto":
        return float(threshold_arg)
    if metrics_payload and "best_threshold" in metrics_payload:
        return float(metrics_payload["best_threshold"])
    return 0.5


def parse_auto_bool(value: str, default: bool):
    lower = value.strip().lower()
    if lower == "auto":
        return default
    if lower in {"1", "true", "yes", "y"}:
        return True
    if lower in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean-like argument: {value}")


def resolve_nodata_settings(mask_no_data_arg: str, no_data_threshold_arg: str, metrics_payload):
    config = metrics_payload.get("config", {}) if metrics_payload else {}
    default_mask = bool(config.get("ignore_no_data", False))
    default_thr = int(config.get("no_data_threshold", 3))
    default_border_only = bool(config.get("no_data_border_only", True))
    default_border_pad = int(config.get("no_data_border_pad", 1))

    apply_mask = parse_auto_bool(mask_no_data_arg, default_mask)
    if no_data_threshold_arg.strip().lower() == "auto":
        threshold = default_thr
    else:
        threshold = int(no_data_threshold_arg)
    return apply_mask, threshold, default_border_only, default_border_pad


def resolve_nodata_refinement(no_data_border_only_arg: str, no_data_border_pad_arg: str, metrics_payload):
    config = metrics_payload.get("config", {}) if metrics_payload else {}
    default_border_only = bool(config.get("no_data_border_only", True))
    default_border_pad = int(config.get("no_data_border_pad", 1))

    border_only = parse_auto_bool(no_data_border_only_arg, default_border_only)
    if no_data_border_pad_arg.strip().lower() == "auto":
        border_pad = default_border_pad
    else:
        border_pad = int(no_data_border_pad_arg)
    border_pad = max(0, border_pad)
    return border_only, border_pad


def _resolve_auto_value(raw_value: str, default_value):
    if raw_value.strip().lower() == "auto":
        return default_value
    return raw_value


def resolve_model_config(
    unet_depth_arg: str,
    unet_base_arg: str,
    unet_dropout_arg: str,
    norm_arg: str,
    gn_groups_arg: str,
    upsample_mode_arg: str,
    deep_supervision_arg: str,
    metrics_payload,
):
    config = metrics_payload.get("config", {}) if metrics_payload else {}

    depth = int(_resolve_auto_value(unet_depth_arg, config.get("unet_depth", 4)))
    base_channels = int(_resolve_auto_value(unet_base_arg, config.get("unet_base_channels", 32)))
    decoder_dropout = float(
        _resolve_auto_value(unet_dropout_arg, config.get("unet_decoder_dropout", 0.0))
    )
    norm = str(_resolve_auto_value(norm_arg, config.get("norm", "bn"))).strip().lower()
    gn_groups = int(_resolve_auto_value(gn_groups_arg, config.get("gn_groups", 8)))
    upsample_mode = str(_resolve_auto_value(upsample_mode_arg, config.get("upsample_mode", "transpose"))).strip().lower()
    deep_supervision_default = bool(config.get("deep_supervision", False))
    deep_supervision = parse_auto_bool(deep_supervision_arg, deep_supervision_default)

    if depth < 2:
        raise ValueError(f"Invalid --unet-depth={depth}; must be >= 2.")
    if base_channels < 4:
        raise ValueError(f"Invalid --unet-base-channels={base_channels}; must be >= 4.")
    if decoder_dropout < 0.0:
        raise ValueError(f"Invalid --unet-decoder-dropout={decoder_dropout}; must be >= 0.")
    if norm not in {"bn", "gn"}:
        raise ValueError(f"Invalid --norm={norm}; use bn or gn.")
    if gn_groups < 1:
        raise ValueError(f"Invalid --gn-groups={gn_groups}; must be >= 1.")
    if upsample_mode not in {"transpose", "bilinear"}:
        raise ValueError(f"Invalid --upsample-mode={upsample_mode}; use transpose or bilinear.")

    return depth, base_channels, decoder_dropout, norm, gn_groups, upsample_mode, deep_supervision


def make_overlay(gray_image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
    color = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    crack_layer = np.zeros_like(color)
    crack_layer[:, :, 2] = binary_mask
    return cv2.addWeighted(color, 0.75, crack_layer, 0.5, 0.0)


def compute_valid_region(image, nodata_threshold, border_only=True, border_pad=0):
    low = (image <= nodata_threshold).astype(np.uint8)
    if low.max() == 0:
        return np.ones_like(low, dtype=np.uint8)

    if border_only:
        num_labels, labels = cv2.connectedComponents(low, connectivity=8)
        if num_labels <= 1:
            invalid = low
        else:
            border_labels = np.unique(
                np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
            )
            invalid = (np.isin(labels, border_labels) & (low > 0)).astype(np.uint8)
    else:
        invalid = low

    if border_pad > 0 and invalid.max() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        invalid = cv2.dilate(invalid, kernel, iterations=border_pad)

    return (invalid == 0).astype(np.uint8)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    metrics_payload, metrics_path = load_metrics_for_model(args.model_path)
    threshold = resolve_threshold(args.threshold, metrics_payload)

    # Resolve image_size from metrics or CLI
    config_for_imgsize = metrics_payload.get("config", {}) if metrics_payload else {}
    if str(args.image_size).strip().lower() == "auto":
        args.image_size = int(config_for_imgsize.get("image_size", 384))
    else:
        args.image_size = int(args.image_size)
    apply_nodata_mask, nodata_threshold, _, _ = resolve_nodata_settings(
        args.mask_no_data, args.no_data_threshold, metrics_payload
    )
    nodata_border_only, nodata_border_pad = resolve_nodata_refinement(
        args.no_data_border_only, args.no_data_border_pad, metrics_payload
    )
    (
        unet_depth,
        unet_base_channels,
        unet_decoder_dropout,
        unet_norm,
        unet_gn_groups,
        unet_upsample_mode,
        unet_deep_supervision,
    ) = resolve_model_config(
        args.unet_depth,
        args.unet_base_channels,
        args.unet_decoder_dropout,
        args.norm,
        args.gn_groups,
        args.upsample_mode,
        args.deep_supervision,
        metrics_payload,
    )
    # Detect encoder type from metrics or state dict
    config = metrics_payload.get("config", {}) if metrics_payload else {}
    encoder_type = config.get("encoder_type", "custom")
    if hasattr(args, "encoder_type") and args.encoder_type != "auto":
        encoder_type = args.encoder_type
    elif encoder_type == "custom":
        # Fallback: detect smp from state dict keys (for older metrics without encoder_type)
        state_peek = torch.load(args.model_path, map_location="cpu", weights_only=True)
        if any(k.startswith("net.encoder.") for k in state_peek.keys()):
            encoder_type = "smp"
            # Try to detect encoder name from key patterns
            has_layer4 = any("layer4" in k for k in state_peek.keys())
            has_layer3 = any("layer3" in k for k in state_peek.keys())
            if not has_layer3:
                detected_enc = "resnet18"
            elif not has_layer4:
                detected_enc = "resnet34"
            else:
                # Could be resnet34 or resnet50 — check for bottleneck blocks
                has_conv3 = any("conv3" in k for k in state_peek.keys())
                detected_enc = "resnet50" if has_conv3 else "resnet34"
            config["encoder_name"] = config.get("encoder_name", detected_enc)
            print(f"Auto-detected smp model from checkpoint keys (encoder guess: {detected_enc}).")
        del state_peek

    if encoder_type == "smp":
        try:
            import segmentation_models_pytorch as smp
        except ImportError:
            raise ImportError("segmentation_models_pytorch is required for smp encoder models. pip install segmentation-models-pytorch")

        encoder_name = config.get("encoder_name", "resnet34")
        encoder_weights = config.get("encoder_weights", "imagenet")
        imagenet_normalize = config.get("imagenet_normalize", "auto")

        class SmpGrayscaleUnet(nn.Module):
            def __init__(self, enc_name, enc_weights):
                super().__init__()
                self.net = smp.Unet(
                    encoder_name=enc_name,
                    encoder_weights=None,  # load weights from checkpoint, not hub
                    in_channels=3,
                    classes=1,
                    activation=None,
                )
                should_normalize = (
                    imagenet_normalize == "1"
                    or (imagenet_normalize == "auto"
                        and enc_weights is not None
                        and str(enc_weights).lower() == "imagenet")
                )
                if should_normalize:
                    self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                    self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
                else:
                    self.img_mean = None
                    self.img_std = None

            def forward(self, x):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if self.img_mean is not None:
                    x = (x - self.img_mean) / self.img_std
                return self.net(x)

        model = SmpGrayscaleUnet(encoder_name, encoder_weights).to(device)
        print(f"[SMP] encoder={encoder_name}, weights_from_checkpoint=True, imagenet_norm={imagenet_normalize}")
    else:
        if args.image_size < (2**unet_depth):
            raise ValueError(
                f"--image-size {args.image_size} is too small for UNet depth {unet_depth}; "
                f"use >= {2**unet_depth}."
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

    state = torch.load(args.model_path, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as err:
        model.load_state_dict(state, strict=False)
        print(f"Warning: strict checkpoint load failed ({err}); retried with strict=False.")
    model.eval()

    # Preprocessing config from metrics
    preprocess_clahe = config.get("preprocess_clahe", False)
    preprocess_clahe_clip = float(config.get("preprocess_clahe_clip", 3.0) or 3.0)
    clahe_obj = None
    if preprocess_clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=preprocess_clahe_clip, tileGridSize=(8, 8))
        print(f"Preprocessing CLAHE: enabled (clip={preprocess_clahe_clip})")

    image_paths = collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No input images found at: {args.input}")

    print(f"Using device: {device}")
    print(f"Loaded model: {args.model_path}")
    print(
        "Model config: "
        f"depth={unet_depth}, base_channels={unet_base_channels}, "
        f"decoder_dropout={unet_decoder_dropout:.3f}, norm={unet_norm}, "
        f"gn_groups={unet_gn_groups}, upsample={unet_upsample_mode}, "
        f"deep_supervision={unet_deep_supervision}"
    )
    if metrics_path:
        print(f"Loaded metrics: {metrics_path}")
    if args.threshold.lower() == "auto" and metrics_path:
        print(f"Using threshold: {threshold:.3f} (auto from metrics)")
    else:
        print(f"Using threshold: {threshold:.3f}")
    print(
        "No-data masking: "
        f"{apply_nodata_mask} (threshold <= {nodata_threshold}, "
        f"border_only={nodata_border_only}, pad={nodata_border_pad})"
    )
    print(f"Connectivity closing: iterations={max(0, args.connectivity_close_iters)}")
    hysteresis_low = args.hysteresis_low
    if hysteresis_low is not None:
        if hysteresis_low >= threshold:
            raise ValueError(
                f"--hysteresis-low ({hysteresis_low}) must be < --threshold ({threshold})."
            )
        print(f"Hysteresis thresholding: low={hysteresis_low:.3f}, high={threshold:.3f}")
    print(f"Images to process: {len(image_paths)}")

    with torch.no_grad():
        close_iters = max(0, int(args.connectivity_close_iters))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) if close_iters > 0 else None
        for idx, image_path in enumerate(image_paths, start=1):
            raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                print(f"[{idx}/{len(image_paths)}] Skipping unreadable image: {image_path}")
                continue
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            if raw.dtype != np.uint8:
                lo, hi = np.percentile(raw, (2, 98))
                if hi <= lo:
                    hi = lo + 1
                image = np.clip((raw.astype(np.float64) - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
            else:
                image = raw

            # Compute valid region BEFORE CLAHE (CLAHE can brighten nodata border pixels)
            original_h, original_w = image.shape[:2]
            if apply_nodata_mask:
                valid = compute_valid_region(
                    image,
                    nodata_threshold,
                    border_only=nodata_border_only,
                    border_pad=nodata_border_pad,
                )

            image_for_overlay = image.copy()
            if clahe_obj is not None:
                image = clahe_obj.apply(image)

            resized = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float() / 255.0
            tensor = tensor.to(device)

            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = logits.get("main", next(iter(logits.values())))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            # Resize probability map to original resolution (INTER_LINEAR for smooth values)
            probs_full = cv2.resize(probs, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            if hysteresis_low is not None:
                # Hysteresis thresholding: keep low-threshold pixels only if
                # connected to high-threshold seeds. Bridges gaps and captures
                # faint fractures without picking up isolated noise.
                from skimage.filters import apply_hysteresis_threshold
                pred_mask = apply_hysteresis_threshold(
                    probs_full, low=hysteresis_low, high=threshold,
                ).astype(np.uint8) * 255
            else:
                pred_mask = (probs_full >= threshold).astype(np.uint8) * 255

            stem = os.path.splitext(os.path.basename(image_path))[0]

            if args.save_probs:
                probs_path = os.path.join(args.output_dir, f"{stem}_probs.npy")
                np.save(probs_path, probs_full.astype(np.float32))

            if close_kernel is not None:
                pred_mask = cv2.morphologyEx(
                    pred_mask,
                    cv2.MORPH_CLOSE,
                    close_kernel,
                    iterations=close_iters,
                )
            if apply_nodata_mask:
                pred_mask[valid == 0] = 0
            mask_path = os.path.join(args.output_dir, f"{stem}_pred_mask.png")
            overlay_path = os.path.join(args.output_dir, f"{stem}_overlay.png")

            overlay = make_overlay(image_for_overlay, pred_mask)
            cv2.imwrite(mask_path, pred_mask)
            cv2.imwrite(overlay_path, overlay)
            print(f"[{idx}/{len(image_paths)}] Saved {mask_path} and {overlay_path}")


if __name__ == "__main__":
    main()
