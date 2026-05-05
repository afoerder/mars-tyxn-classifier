import itertools
import json
import logging
import os
import random
from contextlib import nullcontext
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# --- Paths and device ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_default_dataset_dir(base_dir):
    for candidate in (
        "MarsCracks186",
        "marscracks_186_prepared",
        "marscracks_183_prepared",
        "deepcrack_143_prepared",
        "deepcrack_115_prepared",
    ):
        dataset_dir = os.path.join(base_dir, candidate)
        deepcrack_imgs = os.path.join(dataset_dir, "train_images")
        deepcrack_masks = os.path.join(dataset_dir, "train_masks")
        if os.path.isdir(deepcrack_imgs) and os.path.isdir(deepcrack_masks):
            return dataset_dir
    return base_dir


def resolve_device(device_override):
    override = device_override.strip().lower()
    if override:
        if override == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested but CUDA is unavailable.")
        if override == "mps" and not (
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        ):
            raise RuntimeError("DEVICE=mps requested but MPS is unavailable in this PyTorch build/runtime.")
        return override

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


GROUP_NAMES = ("ESP", "tile_grid", "tile_numeric", "other")
GROUP_TO_INDEX = {name: idx for idx, name in enumerate(GROUP_NAMES)}


def parse_pos_weight_override(raw_value):
    text = str(raw_value).strip()
    if not text:
        return None
    value = float(text)
    if value <= 0.0:
        raise ValueError("POS_WEIGHT_OVERRIDE must be > 0 when set.")
    return value


def parse_pos_weight_by_group_json(raw_value):
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid POS_WEIGHT_BY_GROUP_JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("POS_WEIGHT_BY_GROUP_JSON must decode to an object.")

    out = {}
    for key, value in payload.items():
        if key not in GROUP_TO_INDEX:
            raise ValueError(
                f"Unknown group '{key}' in POS_WEIGHT_BY_GROUP_JSON. "
                f"Expected one of: {', '.join(GROUP_NAMES)}"
            )
        weight = float(value)
        if weight <= 0.0:
            raise ValueError("All POS_WEIGHT_BY_GROUP_JSON values must be > 0.")
        out[key] = weight
    return out


def parse_positive_float_map_json(raw_value, env_name):
    text = str(raw_value).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {env_name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{env_name} must decode to an object.")

    out = {}
    for key, value in payload.items():
        k = str(key).strip()
        if not k:
            raise ValueError(f"{env_name} has an empty key.")
        weight = float(value)
        if weight <= 0.0:
            raise ValueError(f"All {env_name} values must be > 0.")
        out[k] = weight
    return out


def parse_group_csv(raw_value):
    text = str(raw_value).strip()
    if not text:
        return set()
    out = set()
    for token in text.split(","):
        group = token.strip()
        if not group:
            continue
        if group not in GROUP_TO_INDEX:
            raise ValueError(
                f"Unknown group '{group}'. Expected one of: {', '.join(GROUP_NAMES)}"
            )
        out.add(group)
    return out


def parse_legacy_toggle_and_weight(toggle_env_name, weight_env_name, default_weight):
    toggle_raw = os.environ.get(toggle_env_name, "0").strip()
    weight_raw = os.environ.get(weight_env_name, "").strip()

    enabled = toggle_raw != "0"
    legacy_weight = None
    if toggle_raw and toggle_raw not in {"0", "1"}:
        try:
            parsed = float(toggle_raw)
        except ValueError:
            parsed = None
        if parsed is not None:
            enabled = parsed > 0.0
            if parsed > 0.0:
                legacy_weight = parsed

    if weight_raw:
        weight = float(weight_raw)
    elif legacy_weight is not None:
        # Back-compat: LOSS_* can act as a direct weight if *_WEIGHT is omitted.
        weight = legacy_weight
    else:
        weight = float(default_weight)

    return enabled, float(weight), legacy_weight


DATASET_DIR = os.environ.get("DATASET_DIR", resolve_default_dataset_dir(BASE_DIR))
IMG_SUBDIR = os.environ.get("IMG_SUBDIR", "train_images")
MASK_SUBDIR = os.environ.get("MASK_SUBDIR", "train_masks")
TRAIN_IMG_DIR = os.environ.get("TRAIN_IMG_DIR", os.path.join(DATASET_DIR, IMG_SUBDIR))
TRAIN_MASK_DIR = os.environ.get("TRAIN_MASK_DIR", os.path.join(DATASET_DIR, MASK_SUBDIR))
DEVICE = resolve_device(os.environ.get("DEVICE", ""))

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


# --- Core training config ---
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8" if DEVICE == "cuda" else "4"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-4"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", str(IMAGE_SIZE)))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "..", "models", "unet", "mit_b3_skelrecall.pth"))
SEED = int(os.environ.get("SEED", "42"))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.15"))
UNET_BASE_CHANNELS = int(os.environ.get("UNET_BASE_CHANNELS", "32"))
UNET_DEPTH = int(os.environ.get("UNET_DEPTH", "4"))
UNET_DECODER_DROPOUT = float(os.environ.get("UNET_DECODER_DROPOUT", "0.0"))
NORM = os.environ.get("NORM", "bn").strip().lower()
GN_GROUPS = int(os.environ.get("GN_GROUPS", "8"))
UPSAMPLE_MODE = os.environ.get("UPSAMPLE_MODE", "transpose").strip().lower()
DEEP_SUPERVISION = os.environ.get("DEEP_SUPERVISION", "0") != "0"
ARCHITECTURE = os.environ.get("ARCHITECTURE", "simple").strip().lower()
CYCLIC_ITERATIONS = int(os.environ.get("CYCLIC_ITERATIONS", "1"))
CYCLIC_AUX_WEIGHT = float(os.environ.get("CYCLIC_AUX_WEIGHT", "0.4"))
CYCLIC_FEEDBACK_UNFREEZE_EPOCH = int(os.environ.get("CYCLIC_FEEDBACK_UNFREEZE_EPOCH", "0"))
CYCLIC_FEEDBACK_LR_MULT = float(os.environ.get("CYCLIC_FEEDBACK_LR_MULT", "1.0"))
CYCLIC_AUX_FINAL_ONLY = os.environ.get("CYCLIC_AUX_FINAL_ONLY", "0") != "0"
CYCLIC_WARMSTART_PATH = os.environ.get("CYCLIC_WARMSTART_PATH", "").strip()
CYCLIC_WARMSTART_FREEZE_EPOCHS = int(os.environ.get("CYCLIC_WARMSTART_FREEZE_EPOCHS", "0"))

# Auto-bump iterations if feedback features require it
if ARCHITECTURE == "cyclic" and CYCLIC_ITERATIONS < 2:
    if CYCLIC_FEEDBACK_UNFREEZE_EPOCH > 0 or CYCLIC_AUX_FINAL_ONLY or CYCLIC_WARMSTART_PATH:
        print(f"NOTE: Feedback features require CYCLIC_ITERATIONS >= 2 (was {CYCLIC_ITERATIONS}). Auto-setting to 2.")
        CYCLIC_ITERATIONS = 2

# Block conflicting combos
if CYCLIC_WARMSTART_FREEZE_EPOCHS > 0 and CYCLIC_FEEDBACK_UNFREEZE_EPOCH > 0:
    raise ValueError(
        "Cannot use both CYCLIC_WARMSTART_FREEZE_EPOCHS and "
        "CYCLIC_FEEDBACK_UNFREEZE_EPOCH simultaneously."
    )

ENCODER_NAME = os.environ.get("ENCODER_NAME", "").strip()
ENCODER_WEIGHTS = os.environ.get("ENCODER_WEIGHTS", "imagenet").strip()
IMAGENET_NORMALIZE = os.environ.get("IMAGENET_NORMALIZE", "auto").strip().lower()

if ENCODER_NAME and DEEP_SUPERVISION:
    print("NOTE: Deep supervision not supported with smp encoders. Disabling.")
    DEEP_SUPERVISION = False

DS_SCALE_ORDER = (2, 4, 8)


def parse_ds_weights(raw_value):
    defaults = [0.50, 0.30, 0.20]
    weights = []
    for token in raw_value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            weights.append(max(0.0, float(token)))
        except ValueError:
            continue

    if not weights:
        weights = defaults[:]

    weights = weights[: len(DS_SCALE_ORDER)]
    if len(weights) < len(DS_SCALE_ORDER):
        weights.extend(defaults[len(weights) : len(DS_SCALE_ORDER)])

    return {scale: weights[idx] for idx, scale in enumerate(DS_SCALE_ORDER)}


def parse_nonnegative_int_env(name, default):
    value = int(os.environ.get(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")
    return value

TRAIN_USE_PATCHES = os.environ.get("TRAIN_USE_PATCHES", "1") != "0"
TRAIN_PATCHES_PER_IMAGE = int(os.environ.get("TRAIN_PATCHES_PER_IMAGE", "3"))
POSITIVE_CROP_PROB = float(os.environ.get("POSITIVE_CROP_PROB", "0.75"))
TRAIN_MASK_DILATION = parse_nonnegative_int_env("TRAIN_MASK_DILATION", 1)
TRAIN_MASK_DILATION_ESP = parse_nonnegative_int_env(
    "TRAIN_MASK_DILATION_ESP", TRAIN_MASK_DILATION
)
TRAIN_MASK_DILATION_TILE_GRID = parse_nonnegative_int_env(
    "TRAIN_MASK_DILATION_TILE_GRID", TRAIN_MASK_DILATION
)
TRAIN_MASK_DILATION_TILE_NUMERIC = parse_nonnegative_int_env(
    "TRAIN_MASK_DILATION_TILE_NUMERIC", TRAIN_MASK_DILATION
)
TRAIN_MASK_DILATION_OTHER = parse_nonnegative_int_env(
    "TRAIN_MASK_DILATION_OTHER", TRAIN_MASK_DILATION
)
TRAIN_MASK_DILATION_BY_GROUP = {
    "ESP": TRAIN_MASK_DILATION_ESP,
    "tile_grid": TRAIN_MASK_DILATION_TILE_GRID,
    "tile_numeric": TRAIN_MASK_DILATION_TILE_NUMERIC,
    "other": TRAIN_MASK_DILATION_OTHER,
}
IGNORE_NO_DATA = os.environ.get("IGNORE_NO_DATA", "1") != "0"
NO_DATA_THRESHOLD = int(os.environ.get("NO_DATA_THRESHOLD", "3"))
NO_DATA_BORDER_ONLY = os.environ.get("NO_DATA_BORDER_ONLY", "1") != "0"
NO_DATA_BORDER_PAD = max(0, int(os.environ.get("NO_DATA_BORDER_PAD", "1")))
VAL_PENALIZE_OOB = os.environ.get("VAL_PENALIZE_OOB", "1") != "0"

MIN_TRAIN_PAIRS = int(os.environ.get("MIN_TRAIN_PAIRS", "50"))
ALLOW_SMALL_DATASET = os.environ.get("ALLOW_SMALL_DATASET", "0") == "1"

# Calibration defaults tuned for sparse centerline labels.
LOSS_BCE_WEIGHT = float(os.environ.get("LOSS_BCE_WEIGHT", "0.7"))
LOSS_TVERSKY_WEIGHT = float(os.environ.get("LOSS_TVERSKY_WEIGHT", "0.3"))
TVERSKY_ALPHA = float(os.environ.get("TVERSKY_ALPHA", "0.7"))
TVERSKY_BETA = float(os.environ.get("TVERSKY_BETA", "0.3"))
POS_WEIGHT_MULTIPLIER = float(os.environ.get("POS_WEIGHT_MULTIPLIER", "0.35"))
POS_WEIGHT_MIN = float(os.environ.get("POS_WEIGHT_MIN", "1.0"))
POS_WEIGHT_CAP = float(os.environ.get("POS_WEIGHT_CAP", "40.0"))
POS_WEIGHT_OVERRIDE = parse_pos_weight_override(os.environ.get("POS_WEIGHT_OVERRIDE", ""))
POS_WEIGHT_ESTIMATE_USE_DILATED_TARGETS = (
    os.environ.get("POS_WEIGHT_ESTIMATE_USE_DILATED_TARGETS", "1") != "0"
)
POS_WEIGHT_BY_GROUP = parse_pos_weight_by_group_json(
    os.environ.get("POS_WEIGHT_BY_GROUP_JSON", "")
)
POS_WEIGHT_BY_FAMILY = parse_positive_float_map_json(
    os.environ.get("POS_WEIGHT_BY_FAMILY_JSON", ""),
    "POS_WEIGHT_BY_FAMILY_JSON",
)
POS_WEIGHT_COMBINE_MODE = os.environ.get("POS_WEIGHT_COMBINE_MODE", "group").strip().lower()
LOGIT_BIAS_INIT = float(os.environ.get("LOGIT_BIAS_INIT", "-2.0"))
LOSS_CLDICE, CLDICE_MAX, CLDICE_MAX_LEGACY_OVERRIDE = parse_legacy_toggle_and_weight(
    "LOSS_CLDICE",
    "CLDICE_MAX",
    default_weight=0.20,
)
CLDICE_ITERS = int(os.environ.get("CLDICE_ITERS", "10"))
CLDICE_WARMUP_EPOCHS = int(os.environ.get("CLDICE_WARMUP_EPOCHS", "5"))
CLDICE_RAMP_EPOCHS = int(os.environ.get("CLDICE_RAMP_EPOCHS", "15"))
LOSS_SKEL_RECALL, SKEL_RECALL_MAX, _ = parse_legacy_toggle_and_weight(
    "LOSS_SKEL_RECALL",
    "SKEL_RECALL_MAX",
    default_weight=0.15,
)
SKEL_RECALL_WARMUP_EPOCHS = int(os.environ.get("SKEL_RECALL_WARMUP_EPOCHS", "5"))
SKEL_RECALL_RAMP_EPOCHS = int(os.environ.get("SKEL_RECALL_RAMP_EPOCHS", "15"))
SKEL_RECALL_TUBE_RADIUS = int(os.environ.get("SKEL_RECALL_TUBE_RADIUS", "2"))
LOSS_OOB_NEG, OOB_NEG_WEIGHT, OOB_NEG_WEIGHT_LEGACY_OVERRIDE = parse_legacy_toggle_and_weight(
    "LOSS_OOB_NEG",
    "OOB_NEG_WEIGHT",
    default_weight=0.10,
)
DS_WEIGHTS_RAW = os.environ.get("DS_WEIGHTS", "0.50,0.30,0.20")
DS_WEIGHTS = parse_ds_weights(DS_WEIGHTS_RAW)

VAL_TOLERANCE_PX = int(os.environ.get("VAL_TOLERANCE_PX", "2"))
DEFAULT_THRESHOLDS = "0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80"
VAL_THRESHOLDS = [
    float(x.strip())
    for x in os.environ.get("VAL_THRESHOLDS", DEFAULT_THRESHOLDS).split(",")
    if x.strip()
]
if not VAL_THRESHOLDS:
    VAL_THRESHOLDS = [0.5]
BEST_MODEL_SELECTION = os.environ.get("BEST_MODEL_SELECTION", "global_tolf1").strip().lower()
VAL_GROUP_COMBO_SEARCH = os.environ.get("VAL_GROUP_COMBO_SEARCH", "0") != "0"
VAL_GROUP_COMBO_OOB_GUARDRAIL = float(
    os.environ.get("VAL_GROUP_COMBO_OOB_GUARDRAIL", "0.0025")
)
VAL_GROUP_CLOSE_ITERS = parse_nonnegative_int_env("VAL_GROUP_CLOSE_ITERS", 0)
VAL_GROUP_CLOSE_GROUPS = parse_group_csv(
    os.environ.get("VAL_GROUP_CLOSE_GROUPS", "tile_numeric,tile_grid")
)

if UNET_BASE_CHANNELS < 4:
    raise ValueError("UNET_BASE_CHANNELS must be >= 4.")
if UNET_DEPTH < 2:
    raise ValueError("UNET_DEPTH must be >= 2.")
if NORM not in {"bn", "gn"}:
    raise ValueError("NORM must be either 'bn' or 'gn'.")
if GN_GROUPS < 1:
    raise ValueError("GN_GROUPS must be >= 1.")
if UPSAMPLE_MODE not in {"transpose", "bilinear"}:
    raise ValueError("UPSAMPLE_MODE must be either 'transpose' or 'bilinear'.")
if CLDICE_ITERS < 1:
    raise ValueError("CLDICE_ITERS must be >= 1.")
if CLDICE_MAX < 0.0:
    raise ValueError("CLDICE_MAX must be >= 0.")
if CLDICE_WARMUP_EPOCHS < 0:
    raise ValueError("CLDICE_WARMUP_EPOCHS must be >= 0.")
if CLDICE_RAMP_EPOCHS < 0:
    raise ValueError("CLDICE_RAMP_EPOCHS must be >= 0.")
if OOB_NEG_WEIGHT < 0.0:
    raise ValueError("OOB_NEG_WEIGHT must be >= 0.")
if POS_WEIGHT_COMBINE_MODE not in {"group", "family", "mean", "max"}:
    raise ValueError(
        "POS_WEIGHT_COMBINE_MODE must be one of: group, family, mean, max."
    )
if BEST_MODEL_SELECTION not in {"global_tolf1", "combo_tolf1"}:
    raise ValueError("BEST_MODEL_SELECTION must be one of: global_tolf1, combo_tolf1.")
if VAL_GROUP_COMBO_OOB_GUARDRAIL < 0.0:
    raise ValueError("VAL_GROUP_COMBO_OOB_GUARDRAIL must be >= 0.")
if IMAGE_SIZE < (2**UNET_DEPTH):
    raise ValueError(
        f"IMAGE_SIZE={IMAGE_SIZE} is too small for UNET_DEPTH={UNET_DEPTH}; "
        f"need IMAGE_SIZE >= {2**UNET_DEPTH}."
    )

# Dataloader / runtime tuning.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4" if DEVICE == "cuda" else "2"))
PREFETCH_FACTOR = int(os.environ.get("PREFETCH_FACTOR", "2"))
PERSISTENT_WORKERS = os.environ.get("PERSISTENT_WORKERS", "1") != "0"
PIN_MEMORY = (os.environ.get("PIN_MEMORY", "1") != "0") and (DEVICE == "cuda")
CACHE_SAMPLES = os.environ.get("CACHE_SAMPLES", "1") != "0"
AMP = os.environ.get("AMP", "1") != "0"
GRAD_CLIP_NORM = float(os.environ.get("GRAD_CLIP_NORM", "1.0"))
BALANCE_GROUP_SAMPLING = os.environ.get("BALANCE_GROUP_SAMPLING", "0") != "0"
BALANCE_FAMILY_SAMPLING = os.environ.get("BALANCE_FAMILY_SAMPLING", "0") != "0"
FAMILY_SAMPLE_MULTIPLIER = parse_positive_float_map_json(
    os.environ.get("FAMILY_SAMPLE_MULTIPLIER_JSON", ""),
    "FAMILY_SAMPLE_MULTIPLIER_JSON",
)

# Augmentation tuning.
AUG_GEOMETRIC = os.environ.get("AUG_GEOMETRIC", "1") != "0"
AUG_PHOTOMETRIC = os.environ.get("AUG_PHOTOMETRIC", "1") != "0"
AUG_BRIGHTNESS = max(0.0, float(os.environ.get("AUG_BRIGHTNESS", "0.12")))
AUG_CONTRAST = max(0.0, float(os.environ.get("AUG_CONTRAST", "0.18")))
AUG_GAMMA = max(0.0, float(os.environ.get("AUG_GAMMA", "0.15")))
AUG_NOISE_STD = max(0.0, float(os.environ.get("AUG_NOISE_STD", "0.03")))
AUG_NOISE_PROB = min(max(float(os.environ.get("AUG_NOISE_PROB", "0.30")), 0.0), 1.0)
AUG_BLUR_PROB = min(max(float(os.environ.get("AUG_BLUR_PROB", "0.20")), 0.0), 1.0)
AUG_CONTRAST_MODE_BRANCH = os.environ.get("AUG_CONTRAST_MODE_BRANCH", "0") != "0"
AUG_CONTRAST_MODE_PROB = min(
    max(float(os.environ.get("AUG_CONTRAST_MODE_PROB", "0.60")), 0.0), 1.0
)
AUG_INVERT_PROB = min(max(float(os.environ.get("AUG_INVERT_PROB", "0.40")), 0.0), 1.0)
AUG_CLAHE_PROB = min(max(float(os.environ.get("AUG_CLAHE_PROB", "0.30")), 0.0), 1.0)
AUG_CLAHE_CLIP = max(0.1, float(os.environ.get("AUG_CLAHE_CLIP", "2.0")))
AUG_ELASTIC = os.environ.get("AUG_ELASTIC", "0") != "0"
AUG_ELASTIC_PROB = min(max(float(os.environ.get("AUG_ELASTIC_PROB", "0.30")), 0.0), 1.0)
AUG_ELASTIC_ALPHA = max(0.0, float(os.environ.get("AUG_ELASTIC_ALPHA", "120.0")))
AUG_ELASTIC_SIGMA = max(1.0, float(os.environ.get("AUG_ELASTIC_SIGMA", "12.0")))

SPLIT_FILE = os.environ.get("SPLIT_FILE", os.path.splitext(MODEL_PATH)[0] + "_split.json")
REUSE_SPLIT = os.environ.get("REUSE_SPLIT", "1") != "0"


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def compute_valid_region(image, nodata_threshold, border_only=True, border_pad=0):
    """Build a no-data mask from low-intensity regions.

    When border_only is True, only low-valued regions connected to the image border
    are treated as no-data. This avoids masking legitimately dark interior terrain.
    """
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


def random_elastic_deform(image, mask, valid, alpha, sigma):
    """Apply synchronized elastic deformation to image, mask, and valid region.

    Generates smooth random displacement fields via Gaussian-filtered noise,
    then remaps all three arrays with the same field (linear interp for image,
    nearest for binary mask/valid).
    """
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1), (0, 0), sigma
    ) * alpha
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = x + dx
    map_y = y + dy
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = cv2.remap(valid, map_x, map_y, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask, valid


def random_geometric_augment(image, mask, valid):
    # Keep geometric transforms synchronized across image, mask, and valid-region map.
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        valid = cv2.flip(valid, 1)
    if random.random() < 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        valid = cv2.flip(valid, 0)
    if random.random() < 0.5:
        k = random.randint(0, 3)
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
        valid = np.ascontiguousarray(np.rot90(valid, k))
    if random.random() < 0.5:
        h, w = image.shape
        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.warpAffine(
            valid,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    if AUG_ELASTIC and random.random() < AUG_ELASTIC_PROB:
        image, mask, valid = random_elastic_deform(
            image, mask, valid, AUG_ELASTIC_ALPHA, AUG_ELASTIC_SIGMA
        )
    return image, mask, valid


def random_photometric_augment(image):
    # Image-only perturbations improve robustness across illumination/contrast regimes.
    working = image.copy()
    if AUG_CONTRAST_MODE_BRANCH and random.random() < AUG_CONTRAST_MODE_PROB:
        if random.random() < AUG_INVERT_PROB:
            working = 255 - working
        if random.random() < AUG_CLAHE_PROB:
            clahe = cv2.createCLAHE(clipLimit=AUG_CLAHE_CLIP, tileGridSize=(8, 8))
            working = clahe.apply(working)

    out = working.astype(np.float32)

    if AUG_BRIGHTNESS > 0.0 or AUG_CONTRAST > 0.0:
        alpha = random.uniform(max(0.5, 1.0 - AUG_CONTRAST), 1.0 + AUG_CONTRAST)
        beta = random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS) * 255.0
        out = out * alpha + beta

    if AUG_GAMMA > 0.0 and random.random() < 0.5:
        gamma = random.uniform(max(0.5, 1.0 - AUG_GAMMA), 1.0 + AUG_GAMMA)
        out = np.power(np.clip(out, 0.0, 255.0) / 255.0, gamma) * 255.0

    if AUG_NOISE_STD > 0.0 and random.random() < AUG_NOISE_PROB:
        sigma = random.uniform(0.0, AUG_NOISE_STD) * 255.0
        out = out + np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)

    if AUG_BLUR_PROB > 0.0 and random.random() < AUG_BLUR_PROB:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def random_augment(image, mask, valid):
    if AUG_GEOMETRIC:
        image, mask, valid = random_geometric_augment(image, mask, valid)
    if AUG_PHOTOMETRIC:
        image = random_photometric_augment(image)
    return image, mask, valid


def pair_key(pair):
    return os.path.splitext(os.path.basename(pair[0]))[0]


def infer_group_from_key(key):
    if key.startswith("ESP_"):
        return "ESP"
    if key.startswith("tile_y"):
        return "tile_grid"
    if key.startswith("tile_"):
        return "tile_numeric"
    return "other"


def infer_family_from_key(key):
    if key.startswith("hirise_"):
        return key.split("__")[0]
    if key.startswith("ESP_"):
        return key.split("__")[0]
    if key.startswith("tile_y"):
        return "tile_grid"
    if key.startswith("tile_"):
        return "tile_numeric"
    return "other"


class FiberDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        pairs=None,
        augment=True,
        use_patches=False,
        patch_size=512,
        patches_per_image=1,
        positive_crop_prob=0.5,
        mask_dilation=0,
        group_mask_dilation=None,
        family_to_index=None,
        ignore_nodata=True,
        nodata_threshold=3,
        cache_samples=False,
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.patches_per_image = max(1, patches_per_image)
        self.positive_crop_prob = positive_crop_prob
        self.mask_dilation = max(0, mask_dilation)
        self.group_mask_dilation = {}
        if group_mask_dilation:
            for group, dilation in group_mask_dilation.items():
                self.group_mask_dilation[group] = max(0, int(dilation))
        self.ignore_nodata = ignore_nodata
        self.nodata_threshold = nodata_threshold
        self.augment = augment
        self.cache_samples = cache_samples

        self.pairs = pairs if pairs is not None else self._build_pairs()
        if family_to_index is None:
            families = sorted({infer_family_from_key(pair_key(p)) for p in self.pairs})
            self.family_to_index = {family: idx for idx, family in enumerate(families)}
        else:
            self.family_to_index = dict(family_to_index)
        self.family_unknown_index = self.family_to_index.get("other", 0)

        self._kernels = {}
        for dilation in sorted({self.mask_dilation, *self.group_mask_dilation.values()}):
            if dilation > 0:
                size = 2 * dilation + 1
                self._kernels[dilation] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            else:
                self._kernels[dilation] = None

        self._cache = {}

        print("--- Dataset Check ---")
        print(f"Image dir: {img_dir}")
        print(f"Mask dir:  {mask_dir}")
        print(f"Matched pairs: {len(self.pairs)}")
        if self.use_patches:
            print(
                f"Patch mode: size={self.patch_size}, patches_per_image={self.patches_per_image}, "
                f"positive_crop_prob={self.positive_crop_prob:.2f}"
            )
        if self.cache_samples:
            print("Caching: enabled (lazy per-sample cache)")
        if len(self.pairs) == 0:
            print("ERROR: No image/mask pairs found.")

    def _mask_lookup(self):
        lookup = {}
        for filename in os.listdir(self.mask_dir):
            lower = filename.lower()
            if not lower.endswith(VALID_EXTENSIONS):
                continue
            stem = os.path.splitext(filename)[0].lower()
            path = os.path.join(self.mask_dir, filename)
            if stem not in lookup:
                lookup[stem] = path
        return lookup

    def _build_pairs(self):
        if not os.path.isdir(self.img_dir):
            return []
        if not os.path.isdir(self.mask_dir):
            return []

        masks = self._mask_lookup()
        pairs = []
        unresolved = []

        for img_name in sorted(os.listdir(self.img_dir)):
            lower = img_name.lower()
            if not lower.endswith(VALID_EXTENSIONS):
                continue
            if "_trace" in lower:
                continue

            base = os.path.splitext(img_name)[0]
            candidates = [base.lower(), f"{base}_trace".lower()]
            mask_path = ""
            for key in candidates:
                if key in masks:
                    mask_path = masks[key]
                    break

            if not mask_path:
                unresolved.append(img_name)
                continue

            img_path = os.path.join(self.img_dir, img_name)
            pairs.append((img_path, mask_path))

        if unresolved:
            print(f"Warning: {len(unresolved)} images had no matching mask.")
            for name in unresolved[:5]:
                print(f"  - {name}")
        return pairs

    def __len__(self):
        if self.use_patches:
            return len(self.pairs) * self.patches_per_image
        return len(self.pairs)

    def _read_pair(self, pair_idx):
        if self.cache_samples and pair_idx in self._cache:
            image, mask = self._cache[pair_idx]
            return image.copy(), mask.copy()

        img_path, mask_path = self.pairs[pair_idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image {img_path} or mask {mask_path}")

        if self.cache_samples:
            self._cache[pair_idx] = (image, mask)

        return image.copy(), mask.copy()

    def _sample_patch(self, image, mask, valid):
        h, w = image.shape
        patch = min(self.patch_size, h, w)
        if patch <= 0 or (h == patch and w == patch):
            return image, mask, valid

        if mask.max() > 0 and random.random() < self.positive_crop_prob:
            ys, xs = np.where(mask > 0)
            pick = random.randrange(len(xs))
            cx = int(xs[pick])
            cy = int(ys[pick])
            jitter = patch // 4
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)
            x0 = max(0, min(w - patch, cx - patch // 2))
            y0 = max(0, min(h - patch, cy - patch // 2))
        else:
            x0 = random.randint(0, w - patch)
            y0 = random.randint(0, h - patch)

        x1 = x0 + patch
        y1 = y0 + patch
        return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], valid[y0:y1, x0:x1]

    def _prepare_mask(self, mask, group):
        dilation = self.group_mask_dilation.get(group, self.mask_dilation)
        kernel = self._kernels.get(dilation)
        if kernel is None:
            return mask
        return cv2.dilate(mask, kernel, iterations=1)

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)
        key = pair_key(self.pairs[pair_idx])
        group = infer_group_from_key(key)
        family = infer_family_from_key(key)
        group_idx = GROUP_TO_INDEX.get(group, GROUP_TO_INDEX["other"])
        family_idx = self.family_to_index.get(family, self.family_unknown_index)
        image, mask = self._read_pair(pair_idx)

        mask = ((mask > 127) * 255).astype(np.uint8)
        if self.ignore_nodata:
            valid = (
                compute_valid_region(
                    image,
                    self.nodata_threshold,
                    border_only=NO_DATA_BORDER_ONLY,
                    border_pad=NO_DATA_BORDER_PAD,
                )
                * 255
            ).astype(np.uint8)
        else:
            valid = np.full_like(mask, 255, dtype=np.uint8)

        if self.use_patches:
            image, mask, valid = self._sample_patch(image, mask, valid)

        if self.augment:
            image, mask, valid = random_augment(image, mask, valid)

        mask = self._prepare_mask(mask, group=group)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        valid = torch.from_numpy(valid).unsqueeze(0).float()
        mask = (mask > 127).float()
        valid = (valid > 127).float()
        group_tensor = torch.tensor(group_idx, dtype=torch.long)
        family_tensor = torch.tensor(family_idx, dtype=torch.long)
        return image, mask, valid, group_tensor, family_tensor


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
        logit_bias=-2.0,
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
        self.upsample_mode = upsample_mode
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

        if self.out_conv.bias is not None:
            nn.init.constant_(self.out_conv.bias, logit_bias)
        for head in self.side_heads.values():
            if head.bias is not None:
                nn.init.constant_(head.bias, logit_bias)

    def forward(self, x, return_side=False):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        side_logits = {}
        for stage_idx, (upconv, decoder, skip) in enumerate(zip(self.upconvs, self.decoders, reversed(skips))):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
            if self.deep_supervision:
                scale = self.decoder_scales[stage_idx]
                side_key = str(scale)
                if side_key in self.side_heads:
                    side_logits[scale] = self.side_heads[side_key](x)

        main_logits = self.out_conv(x)
        if return_side and self.deep_supervision:
            ordered_side = [(scale, side_logits[scale]) for scale in DS_SCALE_ORDER if scale in side_logits]
            return main_logits, ordered_side
        return main_logits


# ---------------------------------------------------------------------------
# _HalfNet: reusable encoder->bottleneck->decoder (no output head)
# ---------------------------------------------------------------------------

class _HalfNet(nn.Module):
    def __init__(self, in_channels, base_channels=32, depth=4,
                 decoder_dropout=0.0, norm="bn", gn_groups=8,
                 upsample_mode="transpose"):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        channels = [base_channels * (2 ** i) for i in range(depth)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(DoubleConv(prev, ch, norm=norm, gn_groups=gn_groups))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        bn_ch = channels[-1] * 2
        self.bottleneck = DoubleConv(channels[-1], bn_ch, norm=norm, gn_groups=gn_groups)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        dec_in = bn_ch
        for ch in reversed(channels):
            if upsample_mode == "transpose":
                up = nn.ConvTranspose2d(dec_in, ch, kernel_size=2, stride=2)
            else:
                up = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(dec_in, ch, 1, bias=False),
                )
            self.upconvs.append(up)
            layers = [DoubleConv(ch * 2, ch, norm=norm, gn_groups=gn_groups)]
            if decoder_dropout > 0:
                layers.append(nn.Dropout2d(decoder_dropout))
            self.decoders.append(nn.Sequential(*layers))
            dec_in = ch

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return x


# ---------------------------------------------------------------------------
# CyclicDualBottleneckUNet
# ---------------------------------------------------------------------------

class CyclicDualBottleneckUNet(nn.Module):
    """Two U-Net-shaped paths forming a closed loop with dual bottlenecks.

    U-Net half processes raw input; cap-Net half processes prediction features.
    Dual supervision: aux loss at U->cap handoff, main loss at cap-Net output.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, depth=4,
                 logit_bias=-2.0, decoder_dropout=0.0, norm="bn", gn_groups=8,
                 upsample_mode="transpose", n_iterations=1):
        super().__init__()
        self.n_iterations = n_iterations
        self.in_channels = in_channels
        self.u_net = _HalfNet(in_channels, base_channels, depth,
                              decoder_dropout, norm, gn_groups, upsample_mode)
        self.cap_net = _HalfNet(base_channels, base_channels, depth,
                                decoder_dropout, norm, gn_groups, upsample_mode)
        self.u_to_cap = nn.Conv2d(base_channels, base_channels, 1)
        self.cap_to_u = nn.Conv2d(base_channels, in_channels, 1)
        nn.init.zeros_(self.cap_to_u.weight)
        nn.init.zeros_(self.cap_to_u.bias)
        self.main_head = nn.Conv2d(base_channels, out_channels, 1)
        self.aux_head = nn.Conv2d(base_channels, out_channels, 1)
        for head in [self.main_head, self.aux_head]:
            if head.bias is not None:
                nn.init.constant_(head.bias, logit_bias)

    def forward(self, x, return_side=False):
        original_input = x
        all_main, all_aux = [], []
        for i in range(self.n_iterations):
            u_feat = self.u_net(x)
            aux_logits = self.aux_head(u_feat)
            handoff = self.u_to_cap(u_feat)
            cap_feat = self.cap_net(handoff)
            main_logits = self.main_head(cap_feat)
            all_main.append(main_logits)
            all_aux.append(aux_logits)
            if i < self.n_iterations - 1:
                feedback = self.cap_to_u(cap_feat)
                x = original_input + feedback
        if not self.training:
            return all_main[-1]
        if self.n_iterations == 1:
            return {"main": all_main[0], "aux": all_aux[0]}
        return {"main": all_main, "aux": all_aux}


def cldice_weight_for_epoch(epoch_idx):
    if not LOSS_CLDICE or CLDICE_MAX <= 0.0:
        return 0.0
    if epoch_idx < CLDICE_WARMUP_EPOCHS:
        return 0.0
    if CLDICE_RAMP_EPOCHS == 0:
        return CLDICE_MAX
    progress = min(1.0, (epoch_idx - CLDICE_WARMUP_EPOCHS + 1) / CLDICE_RAMP_EPOCHS)
    return CLDICE_MAX * progress


def skel_recall_weight_for_epoch(epoch_idx):
    if not LOSS_SKEL_RECALL or SKEL_RECALL_MAX <= 0.0:
        return 0.0
    if epoch_idx < SKEL_RECALL_WARMUP_EPOCHS:
        return 0.0
    if SKEL_RECALL_RAMP_EPOCHS == 0:
        return SKEL_RECALL_MAX
    progress = min(1.0, (epoch_idx - SKEL_RECALL_WARMUP_EPOCHS + 1) / SKEL_RECALL_RAMP_EPOCHS)
    return SKEL_RECALL_MAX * progress


def masked_mean(values, valid):
    denom = valid.sum().clamp_min(1.0)
    return (values * valid).sum() / denom


def tversky_loss_from_logits(logits, targets, valid, alpha, beta, eps=1e-6):
    probs = torch.sigmoid(logits) * valid
    targets = targets * valid
    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()


def soft_erode(img):
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(p1, p2)


def soft_dilate(img):
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img, iter_=20):
    img = torch.clamp(img, 0.0, 1.0)
    skel = F.relu(img - soft_open(img))
    for _ in range(iter_):
        img = soft_erode(img)
        delta = F.relu(img - soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss_from_logits(logits, targets, valid, iter_, eps=1e-6):
    probs = torch.clamp(torch.sigmoid(logits) * valid, 0.0, 1.0)
    targets = torch.clamp(targets * valid, 0.0, 1.0)
    dims = (1, 2, 3)
    has_target = targets.sum(dim=dims) > eps
    if not torch.any(has_target):
        return logits.new_tensor(0.0)

    skel_pred = soft_skeletonize(probs, iter_=iter_)
    skel_true = soft_skeletonize(targets, iter_=iter_)
    tprec = ((skel_pred * targets).sum(dim=dims) + eps) / (skel_pred.sum(dim=dims) + eps)
    tsens = ((skel_true * probs).sum(dim=dims) + eps) / (skel_true.sum(dim=dims) + eps)
    cl_dice = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    cl_loss = 1.0 - cl_dice
    return cl_loss[has_target].mean()


def skeleton_recall_loss_from_logits(logits, targets, valid, tube_radius=2, eps=1e-6):
    """Skeleton Recall Loss (ECCV 2024): penalizes predictions that miss GT skeleton pixels."""
    probs = torch.sigmoid(logits) * valid
    targets_np = (targets * valid).detach().cpu().numpy()
    tube_masks = []
    for i in range(targets_np.shape[0]):
        gt_bin = (targets_np[i, 0] > 0.5).astype(np.uint8)
        if gt_bin.sum() == 0:
            tube_masks.append(np.zeros_like(gt_bin, dtype=np.float32))
            continue
        skel = skeletonize_binary(gt_bin)
        if tube_radius > 0:
            k_size = 2 * tube_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            tube = cv2.dilate(skel, kernel, iterations=1)
        else:
            tube = skel
        tube_masks.append(tube.astype(np.float32))
    tube_tensor = torch.from_numpy(np.stack(tube_masks)[:, None, :, :]).to(logits.device)
    tube_sum = tube_tensor.sum(dim=(1, 2, 3))
    has_skeleton = tube_sum > eps
    if not torch.any(has_skeleton):
        return logits.new_tensor(0.0)
    recall = (probs * tube_tensor).sum(dim=(1, 2, 3)) / (tube_sum + eps)
    return (1.0 - recall[has_skeleton]).mean()


def _expand_pos_weight(pos_weight, logits):
    if pos_weight is None:
        return None
    if not isinstance(pos_weight, torch.Tensor):
        pos_weight = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    pw = pos_weight.to(device=logits.device, dtype=logits.dtype)
    if pw.ndim == 0:
        pw = pw.view(1, 1, 1, 1)
    elif pw.ndim == 1:
        if pw.numel() == 1:
            pw = pw.view(1, 1, 1, 1)
        elif pw.numel() == logits.shape[1]:
            pw = pw.view(1, logits.shape[1], 1, 1)
        elif pw.numel() == logits.shape[0]:
            pw = pw.view(logits.shape[0], 1, 1, 1)
        else:
            raise ValueError("Unsupported 1D pos_weight shape for current batch.")
    elif pw.ndim == 4:
        pass
    else:
        while pw.ndim < 4:
            pw = pw.unsqueeze(-1)
    if pw.shape[0] == 1 and logits.shape[0] > 1:
        pw = pw.expand(logits.shape[0], -1, -1, -1)
    return pw


def single_head_loss(logits, targets, valid, pos_weight, cldice_weight=0.0):
    bce_map = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    pw = _expand_pos_weight(pos_weight, logits)
    if pw is not None:
        pos_scale = targets * pw + (1.0 - targets)
        bce_map = bce_map * pos_scale
    bce = masked_mean(bce_map, valid)
    tversky = tversky_loss_from_logits(logits, targets, valid, TVERSKY_ALPHA, TVERSKY_BETA)
    cldice = logits.new_tensor(0.0)
    if cldice_weight > 0.0:
        cldice = soft_cldice_loss_from_logits(logits, targets, valid, iter_=CLDICE_ITERS)
    total = LOSS_BCE_WEIGHT * bce + LOSS_TVERSKY_WEIGHT * tversky + cldice_weight * cldice
    return total, bce.detach(), tversky.detach(), cldice.detach()


def oob_negative_loss_from_logits(logits, targets, valid):
    invalid = (valid < 0.5).float()
    if float(invalid.sum().item()) <= 0.0:
        return logits.new_tensor(0.0)
    oob_loss_map = F.binary_cross_entropy_with_logits(
        logits,
        torch.zeros_like(targets),
        reduction="none",
    )
    return masked_mean(oob_loss_map, invalid)


def segmentation_loss(logits, targets, valid, pos_weight, side_logits=None,
                      cldice_weight=0.0, skel_recall_weight=0.0):
    main_total, bce, tversky, cldice = single_head_loss(
        logits,
        targets,
        valid,
        pos_weight,
        cldice_weight=cldice_weight,
    )

    ds_loss = main_total.new_tensor(0.0)
    total = main_total
    if DEEP_SUPERVISION and side_logits:
        weighted_side = main_total.new_tensor(0.0)
        total_weight = 0.0
        for scale, side in side_logits:
            weight = DS_WEIGHTS.get(scale, 0.0)
            if weight <= 0.0:
                continue
            targets_ds = F.interpolate(targets, size=side.shape[-2:], mode="nearest")
            valid_ds = (F.interpolate(valid, size=side.shape[-2:], mode="nearest") > 0.5).float()
            side_total, _, _, _ = single_head_loss(
                side,
                targets_ds,
                valid_ds,
                pos_weight,
                cldice_weight=0.0,
            )
            weighted_side = weighted_side + weight * side_total
            total_weight += weight
        if total_weight > 0.0:
            ds_loss = weighted_side / total_weight
            total = total + ds_loss

    oob_neg = total.new_tensor(0.0)
    if LOSS_OOB_NEG and OOB_NEG_WEIGHT > 0.0:
        oob_neg = oob_negative_loss_from_logits(logits, targets, valid)
        total = total + (OOB_NEG_WEIGHT * oob_neg)

    skel_recall = total.new_tensor(0.0)
    if skel_recall_weight > 0.0:
        skel_recall = skeleton_recall_loss_from_logits(
            logits, targets, valid, tube_radius=SKEL_RECALL_TUBE_RADIUS)
        total = total + skel_recall_weight * skel_recall

    return total, bce, tversky, cldice, ds_loss.detach(), oob_neg.detach(), skel_recall.detach()


def estimate_pos_weight(pairs):
    pos = 0.0
    neg = 0.0
    kernels = {}
    unique_dilations = set(TRAIN_MASK_DILATION_BY_GROUP.values())
    unique_dilations.add(TRAIN_MASK_DILATION)
    for dilation in unique_dilations:
        if dilation > 0:
            size = 2 * dilation + 1
            kernels[dilation] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        else:
            kernels[dilation] = None

    for img_path, mask_path in pairs:
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        mask_bin = (mask > 127).astype(np.uint8)
        if POS_WEIGHT_ESTIMATE_USE_DILATED_TARGETS:
            group = infer_group_from_key(pair_key((img_path, mask_path)))
            dilation = TRAIN_MASK_DILATION_BY_GROUP.get(group, TRAIN_MASK_DILATION)
            kernel = kernels.get(dilation)
            if kernel is not None:
                mask_bin = cv2.dilate(mask_bin, kernel, iterations=1)

        if IGNORE_NO_DATA:
            valid = compute_valid_region(
                image,
                NO_DATA_THRESHOLD,
                border_only=NO_DATA_BORDER_ONLY,
                border_pad=NO_DATA_BORDER_PAD,
            )
        else:
            valid = np.ones_like(mask_bin, dtype=np.uint8)

        pos += float((mask_bin * valid).sum())
        neg += float(((1 - mask_bin) * valid).sum())

    if pos <= 0:
        return 1.0, 1.0, 0.0

    fg_ratio = pos / max(pos + neg, 1.0)
    raw_pos_weight = neg / pos
    adj_pos_weight = raw_pos_weight * POS_WEIGHT_MULTIPLIER
    adj_pos_weight = min(max(POS_WEIGHT_MIN, adj_pos_weight), POS_WEIGHT_CAP)
    return raw_pos_weight, adj_pos_weight, fg_ratio


def build_effective_pos_weight_by_group(default_pos_weight):
    weights = {group: float(default_pos_weight) for group in GROUP_NAMES}
    for group, value in POS_WEIGHT_BY_GROUP.items():
        weights[group] = float(value)
    return weights


def build_effective_pos_weight_by_family(default_pos_weight, family_names):
    weights = {"__default__": float(default_pos_weight)}
    for family in family_names:
        weights[family] = float(default_pos_weight)
    for family, value in POS_WEIGHT_BY_FAMILY.items():
        weights[family] = float(value)
    return weights


def make_group_pos_weight_tensor(group_weights, device):
    return torch.tensor([group_weights[group] for group in GROUP_NAMES], device=device, dtype=torch.float32)


def make_family_pos_weight_tensor(family_weights, family_names, device):
    return torch.tensor(
        [family_weights.get(name, family_weights["__default__"]) for name in family_names],
        device=device,
        dtype=torch.float32,
    )


def resolve_batch_pos_weight(
    group_ids,
    family_ids,
    default_pos_weight,
    group_weight_tensor,
    family_weight_tensor,
):
    batch_size = None
    if group_ids is not None:
        batch_size = group_ids.shape[0]
    elif family_ids is not None:
        batch_size = family_ids.shape[0]

    if batch_size is None:
        return default_pos_weight

    if isinstance(default_pos_weight, torch.Tensor) and default_pos_weight.numel() == 1:
        default_b = default_pos_weight.view(1, 1, 1, 1).expand(batch_size, 1, 1, 1)
    else:
        default_b = default_pos_weight

    group_b = None
    family_b = None
    if group_ids is not None and group_weight_tensor is not None:
        group_b = group_weight_tensor[group_ids.long()].view(-1, 1, 1, 1)
    if family_ids is not None and family_weight_tensor is not None:
        family_b = family_weight_tensor[family_ids.long()].view(-1, 1, 1, 1)

    if group_b is not None and family_b is not None:
        if POS_WEIGHT_COMBINE_MODE == "family":
            return family_b
        if POS_WEIGHT_COMBINE_MODE == "mean":
            return 0.5 * (group_b + family_b)
        if POS_WEIGHT_COMBINE_MODE == "max":
            return torch.maximum(group_b, family_b)
        return group_b
    if group_b is not None:
        return group_b
    if family_b is not None:
        return family_b
    return default_pos_weight


def hard_dice_from_numpy(probs, targets, valid, threshold, penalize_oob=True, eps=1e-6):
    pred = (probs >= threshold).astype(np.float32)
    pred_eval = pred if penalize_oob else (pred * valid)
    targets_eval = targets * valid
    inter = float((pred_eval * targets_eval).sum())
    total = float(pred_eval.sum() + targets_eval.sum())
    return (2.0 * inter + eps) / (total + eps), float(pred_eval.mean())


def tolerant_f1_from_binary(pred, gt, valid, tolerance_px, penalize_oob=True):
    if tolerance_px > 0:
        size = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    else:
        kernel = None

    tp_pred = 0.0
    pred_pos = 0.0
    tp_gt = 0.0
    gt_pos = 0.0

    for i in range(pred.shape[0]):
        valid_i = (valid[i, 0] >= 0.5).astype(np.uint8)
        pred_i = pred[i, 0].astype(np.uint8)
        # Ground truth is defined only over valid area.
        g = (gt[i, 0] * valid_i).astype(np.uint8)
        # Recall should be measured against in-bounds predictions only.
        p_recall = (pred_i * valid_i).astype(np.uint8)
        # Precision can optionally penalize out-of-bounds positives.
        p_precision = pred_i if penalize_oob else p_recall

        if kernel is not None:
            g_d = cv2.dilate(g, kernel, iterations=1)
            p_d = cv2.dilate(p_recall, kernel, iterations=1)
        else:
            g_d = g
            p_d = p_recall

        tp_pred += float((p_precision & g_d).sum())
        pred_pos += float(p_precision.sum())
        tp_gt += float((g & p_d).sum())
        gt_pos += float(g.sum())

    precision = tp_pred / max(pred_pos, 1.0)
    recall = tp_gt / max(gt_pos, 1.0)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-8)
    return precision, recall, f1


def oob_prediction_stats(pred, valid):
    valid_bin = (valid >= 0.5).astype(np.uint8)
    invalid = (1 - valid_bin).astype(np.uint8)
    oob = (pred * invalid).astype(np.uint8)
    oob_pixels = float(oob.sum())
    invalid_pixels = float(invalid.sum())
    pred_pixels = float(pred.sum())
    return (
        oob_pixels,
        oob_pixels / max(invalid_pixels, 1.0),
        oob_pixels / max(pred_pixels, 1.0),
    )


def hard_dice_from_binary(pred, targets, valid, penalize_oob=True, eps=1e-6):
    pred_eval = pred if penalize_oob else (pred * valid)
    targets_eval = targets * valid
    inter = float((pred_eval * targets_eval).sum())
    total = float(pred_eval.sum() + targets_eval.sum())
    return (2.0 * inter + eps) / (total + eps)


def group_name_from_index(group_idx):
    idx = int(group_idx)
    if 0 <= idx < len(GROUP_NAMES):
        return GROUP_NAMES[idx]
    return "other"


def morphology_close(binary, iters):
    if iters <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(
        (binary.astype(np.uint8) * 255),
        cv2.MORPH_CLOSE,
        kernel,
        iterations=iters,
    )
    return (closed > 0).astype(np.uint8)


def apply_group_close(pred, sample_groups, close_groups, close_iters):
    if close_iters <= 0 or not close_groups:
        return pred
    out = pred.copy()
    for i, group in enumerate(sample_groups):
        if group in close_groups:
            out[i, 0] = morphology_close(out[i, 0], close_iters)
    return out


def make_groupwise_pred(probs, sample_groups, group_thresholds, default_threshold):
    pred = np.zeros_like(probs, dtype=np.uint8)
    for i, group in enumerate(sample_groups):
        threshold = group_thresholds.get(group, default_threshold)
        pred[i, 0] = (probs[i, 0] >= threshold).astype(np.uint8)
    return pred


def skeletonize_binary(binary):
    binary = (binary > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary

    skel = np.zeros_like(binary, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary.copy()
    while True:
        eroded = cv2.erode(current, kernel)
        opened = cv2.dilate(eroded, kernel)
        residue = cv2.subtract(current, opened)
        skel = cv2.bitwise_or(skel, residue)
        current = eroded
        if cv2.countNonZero(current) == 0:
            break
    return (skel > 0).astype(np.uint8)


def skeleton_endpoints(skel):
    if skel.sum() == 0:
        return np.zeros_like(skel, dtype=np.uint8)
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1],
        ],
        dtype=np.uint8,
    )
    neigh = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16U, kernel, borderType=cv2.BORDER_CONSTANT)
    return ((neigh == 11) & (skel > 0)).astype(np.uint8)


def connected_components(binary):
    count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return max(0, count - 1)


def topology_metrics_from_binary(pred, gt, valid, tolerance_px):
    kernel = None
    if tolerance_px > 0:
        size = 2 * tolerance_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    cc_ratios = []
    len_ratios = []
    bridge_rates = []
    for i in range(pred.shape[0]):
        p = (pred[i, 0] * valid[i, 0]).astype(np.uint8)
        g = (gt[i, 0] * valid[i, 0]).astype(np.uint8)

        pred_cc = connected_components(p)
        gt_cc = connected_components(g)
        if gt_cc == 0:
            cc_ratio = 1.0 if pred_cc == 0 else 0.0
        else:
            cc_ratio = pred_cc / gt_cc
        cc_ratios.append(float(cc_ratio))

        pred_skel = skeletonize_binary(p)
        gt_skel = skeletonize_binary(g)
        pred_len = float(pred_skel.sum())
        gt_len = float(gt_skel.sum())
        if gt_len <= 0.0:
            len_ratio = 1.0 if pred_len <= 0.0 else 0.0
        else:
            len_ratio = pred_len / gt_len
        len_ratios.append(float(len_ratio))

        gt_end = skeleton_endpoints(gt_skel)
        gt_end_count = float(gt_end.sum())
        if gt_end_count <= 0.0:
            bridge_rate = 1.0 if pred_len <= 0.0 else 0.0
        else:
            pred_match = cv2.dilate(pred_skel, kernel, iterations=1) if kernel is not None else pred_skel
            bridge_rate = float((gt_end & pred_match).sum()) / gt_end_count
        bridge_rates.append(float(bridge_rate))

    return {
        "cc_ratio": float(np.mean(cc_ratios)) if cc_ratios else 0.0,
        "len_ratio": float(np.mean(len_ratios)) if len_ratios else 0.0,
        "bridge_rate": float(np.mean(bridge_rates)) if bridge_rates else 0.0,
    }


def evaluate_with_threshold_sweep(
    model,
    loader,
    pos_weight,
    group_pos_weight_tensor=None,
    family_pos_weight_tensor=None,
    cldice_weight=0.0,
    skel_recall_weight=0.0,
):
    model.eval()
    total_loss = 0.0
    total_bce = 0.0
    total_tversky = 0.0
    total_cldice = 0.0
    total_ds = 0.0
    total_oob_neg = 0.0
    probs_all = []
    targets_all = []
    valid_all = []
    group_ids_all = []

    with torch.no_grad():
        for batch in loader:
            images, masks, valid, group_ids, family_ids = unpack_batch(batch)
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            masks = masks.to(DEVICE, non_blocking=PIN_MEMORY)
            valid = valid.to(DEVICE, non_blocking=PIN_MEMORY)
            if group_ids is not None:
                group_ids = group_ids.to(DEVICE, non_blocking=PIN_MEMORY)
            if family_ids is not None:
                family_ids = family_ids.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_pos_weight = resolve_batch_pos_weight(
                group_ids,
                family_ids,
                pos_weight,
                group_pos_weight_tensor,
                family_pos_weight_tensor,
            )
            if DEEP_SUPERVISION:
                logits, side_logits = model(images, return_side=True)
            else:
                logits = model(images)
                side_logits = []
            loss, bce, tversky, cldice, ds_loss, oob_neg, skel_rec = segmentation_loss(
                logits,
                masks,
                valid,
                batch_pos_weight,
                side_logits=side_logits,
                cldice_weight=cldice_weight,
                skel_recall_weight=skel_recall_weight,
            )
            total_loss += loss.item()
            total_bce += bce.item()
            total_tversky += tversky.item()
            total_cldice += cldice.item()
            total_ds += ds_loss.item()
            total_oob_neg += oob_neg.item()

            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(masks.cpu().numpy())
            valid_all.append(valid.cpu().numpy())
            if group_ids is None:
                group_ids_all.append(
                    np.full((images.shape[0],), GROUP_TO_INDEX["other"], dtype=np.int64)
                )
            else:
                group_ids_all.append(group_ids.detach().cpu().numpy().astype(np.int64))

    model.train()

    probs = np.concatenate(probs_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)
    valid = np.concatenate(valid_all, axis=0)
    sample_group_ids = np.concatenate(group_ids_all, axis=0)
    sample_groups = [group_name_from_index(g) for g in sample_group_ids]
    valid_bin = (valid >= 0.5).astype(np.uint8)
    gt_bin = (targets >= 0.5).astype(np.uint8)

    best_threshold = VAL_THRESHOLDS[0]
    best_tolerant_f1 = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_dice = 0.0
    best_pred_fg = 0.0
    best_oob_pixels = 0.0
    best_oob_ratio = 0.0
    best_oob_frac = 0.0
    best_pred_bin = (probs >= best_threshold).astype(np.uint8)

    for threshold in VAL_THRESHOLDS:
        pred_bin = (probs >= threshold).astype(np.uint8)
        pred_bin = apply_group_close(
            pred_bin,
            sample_groups,
            VAL_GROUP_CLOSE_GROUPS,
            VAL_GROUP_CLOSE_ITERS,
        )
        precision, recall, tolerant_f1 = tolerant_f1_from_binary(
            pred_bin,
            gt_bin,
            valid,
            VAL_TOLERANCE_PX,
            penalize_oob=VAL_PENALIZE_OOB,
        )
        dice = hard_dice_from_binary(
            pred_bin.astype(np.float32),
            targets,
            valid,
            penalize_oob=VAL_PENALIZE_OOB,
        )
        pred_fg = float(pred_bin.mean())
        oob_pixels, oob_ratio, oob_frac = oob_prediction_stats(pred_bin, valid_bin)

        if tolerant_f1 > best_tolerant_f1 or (
            abs(tolerant_f1 - best_tolerant_f1) < 1e-10 and dice > best_dice
        ):
            best_tolerant_f1 = tolerant_f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_dice = dice
            best_pred_fg = pred_fg
            best_oob_pixels = oob_pixels
            best_oob_ratio = oob_ratio
            best_oob_frac = oob_frac
            best_pred_bin = pred_bin

    combo_group_thresholds = None
    combo_tolerant_f1 = None
    combo_dice = None
    combo_oob_ratio = None
    combo_oob_fraction = None
    combo_oob_pixels = None
    combo_pred_fg = None
    combo_search_executed = False
    combo_search_skipped_reason = None
    if VAL_GROUP_COMBO_SEARCH:
        if len(VAL_THRESHOLDS) > 9:
            combo_search_skipped_reason = "threshold_count_gt_9"
        elif not sample_groups:
            combo_search_skipped_reason = "no_samples"
        else:
            combo_groups = [group for group in GROUP_NAMES if group in set(sample_groups)]
            combo_search_executed = True
            for values in itertools.product(VAL_THRESHOLDS, repeat=len(combo_groups)):
                group_thresholds = {
                    combo_groups[group_idx]: values[group_idx] for group_idx in range(len(combo_groups))
                }
                pred_bin = make_groupwise_pred(
                    probs,
                    sample_groups,
                    group_thresholds,
                    best_threshold,
                )
                pred_bin = apply_group_close(
                    pred_bin,
                    sample_groups,
                    VAL_GROUP_CLOSE_GROUPS,
                    VAL_GROUP_CLOSE_ITERS,
                )
                precision, recall, tolerant_f1 = tolerant_f1_from_binary(
                    pred_bin,
                    gt_bin,
                    valid,
                    VAL_TOLERANCE_PX,
                    penalize_oob=VAL_PENALIZE_OOB,
                )
                dice = hard_dice_from_binary(
                    pred_bin.astype(np.float32),
                    targets,
                    valid,
                    penalize_oob=VAL_PENALIZE_OOB,
                )
                oob_pixels, oob_ratio, oob_frac = oob_prediction_stats(pred_bin, valid_bin)
                if oob_ratio > VAL_GROUP_COMBO_OOB_GUARDRAIL:
                    continue
                if combo_tolerant_f1 is None or (
                    tolerant_f1 > combo_tolerant_f1
                    or (abs(tolerant_f1 - combo_tolerant_f1) < 1e-10 and dice > combo_dice)
                ):
                    combo_group_thresholds = group_thresholds
                    combo_tolerant_f1 = float(tolerant_f1)
                    combo_dice = float(dice)
                    combo_oob_pixels = float(oob_pixels)
                    combo_oob_ratio = float(oob_ratio)
                    combo_oob_fraction = float(oob_frac)
                    combo_pred_fg = float(pred_bin.mean())

            if combo_tolerant_f1 is None:
                combo_search_skipped_reason = "no_combo_within_guardrail"

    selection_tolerant_f1 = best_tolerant_f1
    selection_dice = best_dice
    selection_oob_ratio = best_oob_ratio
    selection_oob_fraction = best_oob_frac
    selection_oob_pixels = best_oob_pixels
    selection_pred_fg = best_pred_fg
    selection_source = "global_best"
    if BEST_MODEL_SELECTION == "combo_tolf1":
        if combo_tolerant_f1 is not None:
            selection_tolerant_f1 = combo_tolerant_f1
            selection_dice = combo_dice
            selection_oob_ratio = combo_oob_ratio
            selection_oob_fraction = combo_oob_fraction
            selection_oob_pixels = combo_oob_pixels
            selection_pred_fg = combo_pred_fg
            selection_source = "combo_search"
        else:
            selection_source = "global_fallback"

    soft_inter = float((probs * targets * valid).sum())
    soft_total = float((probs * valid).sum() + (targets * valid).sum())
    soft_dice = (2.0 * soft_inter + 1e-6) / (soft_total + 1e-6)
    topo = topology_metrics_from_binary(best_pred_bin, gt_bin, valid_bin, VAL_TOLERANCE_PX)

    return {
        "loss": total_loss / len(loader),
        "bce": total_bce / len(loader),
        "tversky": total_tversky / len(loader),
        "cldice": total_cldice / len(loader),
        "ds_loss": total_ds / len(loader),
        "oob_neg_loss": total_oob_neg / len(loader),
        "soft_dice": soft_dice,
        "best_dice": best_dice,
        "best_threshold": best_threshold,
        "best_tolerant_f1": best_tolerant_f1,
        "pred_fg_ratio": best_pred_fg,
        "best_oob_pred_pixels": best_oob_pixels,
        "best_oob_pred_ratio": best_oob_ratio,
        "best_oob_pred_fraction": best_oob_frac,
        "target_fg_ratio": float((targets * valid).sum() / max(valid.sum(), 1.0)),
        "mean_prob": float((probs * valid).sum() / max(valid.sum(), 1.0)),
        "best_precision": best_precision,
        "best_recall": best_recall,
        "cc_ratio": topo["cc_ratio"],
        "len_ratio": topo["len_ratio"],
        "bridge_rate": topo["bridge_rate"],
        "combo_search_executed": combo_search_executed,
        "combo_search_skipped_reason": combo_search_skipped_reason,
        "combo_group_thresholds": combo_group_thresholds,
        "combo_tolerant_f1": combo_tolerant_f1,
        "combo_dice": combo_dice,
        "combo_oob_pred_pixels": combo_oob_pixels,
        "combo_oob_pred_ratio": combo_oob_ratio,
        "combo_oob_pred_fraction": combo_oob_fraction,
        "combo_pred_fg_ratio": combo_pred_fg,
        "best_model_selection": BEST_MODEL_SELECTION,
        "selection_source": selection_source,
        "selection_tolerant_f1": float(selection_tolerant_f1),
        "selection_dice": float(selection_dice),
        "selection_oob_pred_pixels": float(selection_oob_pixels),
        "selection_oob_pred_ratio": float(selection_oob_ratio),
        "selection_oob_pred_fraction": float(selection_oob_fraction),
        "selection_pred_fg_ratio": float(selection_pred_fg),
    }


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_indices(pairs, split_path):
    if not split_path or not os.path.exists(split_path):
        return None

    try:
        with open(split_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        logging.warning("Failed to load split indices from %s", split_path)
        return None

    train_keys = data.get("train_keys", [])
    val_keys = data.get("val_keys", [])
    if not train_keys or not val_keys:
        return None

    key_to_idx = {}
    for i, pair in enumerate(pairs):
        key_to_idx[pair_key(pair)] = i

    missing = [k for k in train_keys + val_keys if k not in key_to_idx]
    if missing:
        print(f"Split file exists but has {len(missing)} missing keys. Regenerating split.")
        return None

    train_idx = [key_to_idx[k] for k in train_keys]
    val_idx = [key_to_idx[k] for k in val_keys]

    if not train_idx or not val_idx:
        return None

    return train_idx, val_idx


def stratified_split_indices(pairs, val_split, seed):
    if len(pairs) <= 1:
        return list(range(len(pairs))), []

    groups = defaultdict(list)
    for idx, pair in enumerate(pairs):
        key = pair_key(pair)
        groups[infer_group_from_key(key)].append(idx)

    rng = random.Random(seed)
    train_idx = []
    val_idx = []

    for group_name in sorted(groups.keys()):
        idxs = groups[group_name][:]
        rng.shuffle(idxs)
        n = len(idxs)

        if n < 2 or val_split <= 0.0:
            n_val = 0
        else:
            n_val = int(round(n * val_split))
            n_val = max(1, n_val)
            n_val = min(n_val, n - 1)

        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    # Safety fallback if split is somehow degenerate.
    if not val_idx:
        val_idx = [train_idx.pop()] if len(train_idx) > 1 else []
    if not train_idx and val_idx:
        train_idx = [val_idx.pop()]

    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


def save_split_file(pairs, train_idx, val_idx, split_path):
    if not split_path:
        return

    payload = {
        "seed": SEED,
        "val_split": VAL_SPLIT,
        "dataset_dir": DATASET_DIR,
        "train_keys": [pair_key(pairs[i]) for i in train_idx],
        "val_keys": [pair_key(pairs[i]) for i in val_idx],
    }
    write_json(split_path, payload)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_balanced_sampler(dataset):
    if len(dataset) == 0 or len(dataset.pairs) == 0:
        return None

    group_counts = defaultdict(int)
    family_counts = defaultdict(int)
    pair_groups = []
    pair_families = []
    for pair in dataset.pairs:
        key = pair_key(pair)
        grp = infer_group_from_key(key)
        fam = infer_family_from_key(key)
        group_counts[grp] += 1
        family_counts[fam] += 1
        pair_groups.append(grp)
        pair_families.append(fam)

    weights = []
    for idx in range(len(dataset)):
        pair_idx = idx % len(dataset.pairs)
        grp = pair_groups[pair_idx]
        fam = pair_families[pair_idx]
        weight = 1.0
        if BALANCE_GROUP_SAMPLING:
            weight *= 1.0 / max(group_counts[grp], 1)
        if BALANCE_FAMILY_SAMPLING:
            weight *= 1.0 / max(family_counts[fam], 1)
        weight *= FAMILY_SAMPLE_MULTIPLIER.get(fam, 1.0)
        weights.append(weight)

    weight_tensor = torch.tensor(weights, dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return WeightedRandomSampler(
        weights=weight_tensor,
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def make_loader(dataset, batch_size, shuffle, sampler=None):
    loader_kwargs = {
        "batch_size": min(batch_size, len(dataset)),
        "shuffle": shuffle if sampler is None else False,
        "drop_last": False,
        "worker_init_fn": seed_worker,
    }
    if sampler is not None:
        loader_kwargs["sampler"] = sampler

    if NUM_WORKERS > 0:
        loader_kwargs["num_workers"] = NUM_WORKERS
        loader_kwargs["pin_memory"] = PIN_MEMORY
        loader_kwargs["persistent_workers"] = PERSISTENT_WORKERS
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(dataset, **loader_kwargs)


def unpack_batch(batch):
    if len(batch) == 5:
        images, masks, valid, group_ids, family_ids = batch
        return images, masks, valid, group_ids, family_ids
    if len(batch) == 4:
        images, masks, valid, group_ids = batch
        return images, masks, valid, group_ids, None
    images, masks, valid = batch
    return images, masks, valid, None, None


# --- SMP encoder support ---
def build_smp_model():
    """Build smp.Unet with configurable encoder for grayscale fracture segmentation."""
    import segmentation_models_pytorch as smp

    encoder_name = ENCODER_NAME
    encoder_weights = ENCODER_WEIGHTS
    if encoder_weights.lower() == "none":
        encoder_weights = None

    class SmpGrayscaleUnet(nn.Module):
        def __init__(self, enc_name, enc_weights):
            super().__init__()
            self.net = smp.Unet(
                encoder_name=enc_name,
                encoder_weights=enc_weights,
                in_channels=3,
                classes=1,
                activation=None,
            )
            should_normalize = (
                IMAGENET_NORMALIZE == "1"
                or (IMAGENET_NORMALIZE == "auto"
                    and enc_weights is not None
                    and str(enc_weights).lower() == "imagenet")
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

        def forward(self, x, return_side=False):
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            if self.img_mean is not None:
                x = (x - self.img_mean) / self.img_std
            return self.net(x)

    m = SmpGrayscaleUnet(encoder_name, encoder_weights)
    n_p = sum(p.numel() for p in m.parameters())
    print(f"[SMP] {encoder_name}: {n_p/1e6:.1f}M params, weights={encoder_weights}")
    return m


# --- Training ---
def train():
    pair_dataset = FiberDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        augment=False,
        use_patches=False,
        patch_size=PATCH_SIZE,
        patches_per_image=1,
        positive_crop_prob=0.0,
        mask_dilation=0,
        ignore_nodata=IGNORE_NO_DATA,
        nodata_threshold=NO_DATA_THRESHOLD,
        cache_samples=False,
    )
    if len(pair_dataset) == 0:
        return

    pairs = pair_dataset.pairs
    if len(pairs) < MIN_TRAIN_PAIRS and not ALLOW_SMALL_DATASET:
        raise RuntimeError(
            f"Only {len(pairs)} training pairs found, below MIN_TRAIN_PAIRS={MIN_TRAIN_PAIRS}. "
            "Set DATASET_DIR correctly or set ALLOW_SMALL_DATASET=1 to override."
        )

    if VAL_SPLIT > 0.0 and len(pairs) > 1:
        split_indices = None
        if REUSE_SPLIT:
            split_indices = load_split_indices(pairs, SPLIT_FILE)
        if split_indices is None:
            train_idx, val_idx = stratified_split_indices(pairs, VAL_SPLIT, SEED)
            save_split_file(pairs, train_idx, val_idx, SPLIT_FILE)
        else:
            train_idx, val_idx = split_indices
    else:
        train_idx = list(range(len(pairs)))
        val_idx = []

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    family_names = sorted({infer_family_from_key(pair_key(pair)) for pair in pairs})
    family_to_index = {family: idx for idx, family in enumerate(family_names)}

    train_dataset = FiberDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        pairs=train_pairs,
        augment=True,
        use_patches=TRAIN_USE_PATCHES,
        patch_size=PATCH_SIZE,
        patches_per_image=TRAIN_PATCHES_PER_IMAGE,
        positive_crop_prob=POSITIVE_CROP_PROB,
        mask_dilation=TRAIN_MASK_DILATION,
        group_mask_dilation=TRAIN_MASK_DILATION_BY_GROUP,
        family_to_index=family_to_index,
        ignore_nodata=IGNORE_NO_DATA,
        nodata_threshold=NO_DATA_THRESHOLD,
        cache_samples=CACHE_SAMPLES,
    )
    use_weighted_sampler = BALANCE_GROUP_SAMPLING or BALANCE_FAMILY_SAMPLING or bool(FAMILY_SAMPLE_MULTIPLIER)
    train_sampler = build_balanced_sampler(train_dataset) if use_weighted_sampler else None
    train_loader = make_loader(train_dataset, BATCH_SIZE, shuffle=True, sampler=train_sampler)

    if val_pairs:
        val_dataset = FiberDataset(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            pairs=val_pairs,
            augment=False,
            use_patches=False,
            patch_size=PATCH_SIZE,
            patches_per_image=1,
            positive_crop_prob=0.0,
            mask_dilation=0,
            family_to_index=family_to_index,
            ignore_nodata=IGNORE_NO_DATA,
            nodata_threshold=NO_DATA_THRESHOLD,
            cache_samples=CACHE_SAMPLES,
        )
        val_loader = make_loader(val_dataset, BATCH_SIZE, shuffle=False)
    else:
        val_dataset = None
        val_loader = None

    raw_pos_weight_estimate, pos_weight_scalar_estimate, fg_ratio = estimate_pos_weight(train_pairs)
    if POS_WEIGHT_OVERRIDE is not None:
        raw_pos_weight = float(POS_WEIGHT_OVERRIDE)
        pos_weight_scalar = float(POS_WEIGHT_OVERRIDE)
    else:
        raw_pos_weight = float(raw_pos_weight_estimate)
        pos_weight_scalar = float(pos_weight_scalar_estimate)

    pos_weight = torch.tensor([pos_weight_scalar], device=DEVICE, dtype=torch.float32)
    effective_group_pos_weight = build_effective_pos_weight_by_group(pos_weight_scalar)
    effective_family_pos_weight = build_effective_pos_weight_by_family(pos_weight_scalar, family_names)
    has_group_pos_weight = bool(POS_WEIGHT_BY_GROUP)
    has_family_pos_weight = bool(POS_WEIGHT_BY_FAMILY)
    group_pos_weight_tensor = (
        make_group_pos_weight_tensor(effective_group_pos_weight, DEVICE) if has_group_pos_weight else None
    )
    family_pos_weight_tensor = (
        make_family_pos_weight_tensor(effective_family_pos_weight, family_names, DEVICE)
        if has_family_pos_weight
        else None
    )

    if ENCODER_NAME:
        model = build_smp_model().to(DEVICE)
    elif ARCHITECTURE == "cyclic":
        model = CyclicDualBottleneckUNet(
            in_channels=1,
            out_channels=1,
            base_channels=UNET_BASE_CHANNELS,
            depth=UNET_DEPTH,
            logit_bias=LOGIT_BIAS_INIT,
            decoder_dropout=UNET_DECODER_DROPOUT,
            norm=NORM,
            gn_groups=GN_GROUPS,
            upsample_mode=UPSAMPLE_MODE,
            n_iterations=CYCLIC_ITERATIONS,
        ).to(DEVICE)
    else:
        model = SimpleUNet(
            in_channels=1,
            out_channels=1,
            base_channels=UNET_BASE_CHANNELS,
            depth=UNET_DEPTH,
            logit_bias=LOGIT_BIAS_INIT,
            decoder_dropout=UNET_DECODER_DROPOUT,
            norm=NORM,
            gn_groups=GN_GROUPS,
            upsample_mode=UPSAMPLE_MODE,
            deep_supervision=DEEP_SUPERVISION,
        ).to(DEVICE)

    # --- Idea 3: Warm-start from checkpoint ---
    if ARCHITECTURE == "cyclic" and CYCLIC_WARMSTART_PATH:
        print(f"Loading warm-start checkpoint: {CYCLIC_WARMSTART_PATH}")
        ws_state = torch.load(CYCLIC_WARMSTART_PATH, map_location=DEVICE)
        model.load_state_dict(ws_state, strict=True)
        print(f"  cap_to_u weight norm after load: {model.cap_to_u.weight.data.norm().item():.6f}")
        if CYCLIC_WARMSTART_FREEZE_EPOCHS > 0:
            for name, p in model.named_parameters():
                if not name.startswith("cap_to_u"):
                    p.requires_grad = False
            print(f"  Froze all params except cap_to_u for {CYCLIC_WARMSTART_FREEZE_EPOCHS} epochs")

    # --- Idea 1: Separate param group for cap_to_u lr ---
    if ARCHITECTURE == "cyclic" and CYCLIC_FEEDBACK_LR_MULT != 1.0:
        cap_to_u_ids = {id(p) for p in model.cap_to_u.parameters()}
        base_params = [p for p in model.parameters() if id(p) not in cap_to_u_ids]
        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": LEARNING_RATE},
            {"params": list(model.cap_to_u.parameters()), "lr": LEARNING_RATE * CYCLIC_FEEDBACK_LR_MULT},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))
    use_amp = AMP and DEVICE == "cuda"
    if use_amp:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Starting training on {DEVICE}...")
    print(f"Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, image_size={IMAGE_SIZE}")
    if ARCHITECTURE == "cyclic":
        parts = [f"iterations={CYCLIC_ITERATIONS}", f"aux_weight={CYCLIC_AUX_WEIGHT:.2f}"]
        if CYCLIC_FEEDBACK_UNFREEZE_EPOCH > 0:
            parts.append(f"feedback_unfreeze_epoch={CYCLIC_FEEDBACK_UNFREEZE_EPOCH}")
        if CYCLIC_FEEDBACK_LR_MULT != 1.0:
            parts.append(f"feedback_lr_mult={CYCLIC_FEEDBACK_LR_MULT:.1f}")
        if CYCLIC_AUX_FINAL_ONLY:
            parts.append("aux_final_only=True")
        if CYCLIC_WARMSTART_PATH:
            parts.append(f"warmstart={os.path.basename(CYCLIC_WARMSTART_PATH)}")
        if CYCLIC_WARMSTART_FREEZE_EPOCHS > 0:
            parts.append(f"warmstart_freeze={CYCLIC_WARMSTART_FREEZE_EPOCHS}")
        print(f"Architecture: cyclic ({', '.join(parts)})")
    elif ENCODER_NAME:
        print(f"Architecture: smp ({ENCODER_NAME}, weights={ENCODER_WEIGHTS}, "
              f"imagenet_norm={IMAGENET_NORMALIZE})")
    else:
        print(f"Architecture: {ARCHITECTURE}")
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    print(
        f"Model: depth={UNET_DEPTH}, base_channels={UNET_BASE_CHANNELS}, "
        f"decoder_dropout={UNET_DECODER_DROPOUT:.3f}, norm={NORM}, "
        f"gn_groups={GN_GROUPS}, upsample={UPSAMPLE_MODE}, deep_supervision={DEEP_SUPERVISION}"
    )
    print(
        f"Deep supervision weights (1/2,1/4,1/8): "
        f"{DS_WEIGHTS[2]:.3f},{DS_WEIGHTS[4]:.3f},{DS_WEIGHTS[8]:.3f}"
    )
    print(
        f"clDice: enabled={LOSS_CLDICE}, max={CLDICE_MAX:.3f}, iters={CLDICE_ITERS}, "
        f"warmup={CLDICE_WARMUP_EPOCHS}, ramp={CLDICE_RAMP_EPOCHS}"
    )
    print(
        f"SkelRecall: enabled={LOSS_SKEL_RECALL}, max={SKEL_RECALL_MAX:.3f}, "
        f"tube_radius={SKEL_RECALL_TUBE_RADIUS}, "
        f"warmup={SKEL_RECALL_WARMUP_EPOCHS}, ramp={SKEL_RECALL_RAMP_EPOCHS}"
    )
    print(
        f"Patch mode: enabled={TRAIN_USE_PATCHES}, patch_size={PATCH_SIZE}, "
        f"patches_per_image={TRAIN_PATCHES_PER_IMAGE}, positive_crop_prob={POSITIVE_CROP_PROB:.2f}"
    )
    print(
        "Augmentations: "
        f"geometric={AUG_GEOMETRIC}, photometric={AUG_PHOTOMETRIC}, "
        f"brightness={AUG_BRIGHTNESS:.3f}, contrast={AUG_CONTRAST:.3f}, "
        f"gamma={AUG_GAMMA:.3f}, noise_std={AUG_NOISE_STD:.3f}, "
        f"noise_prob={AUG_NOISE_PROB:.2f}, blur_prob={AUG_BLUR_PROB:.2f}"
    )
    print(
        f"Contrast-mode branch: enabled={AUG_CONTRAST_MODE_BRANCH}, "
        f"mode_prob={AUG_CONTRAST_MODE_PROB:.2f}, invert_prob={AUG_INVERT_PROB:.2f}, "
        f"clahe_prob={AUG_CLAHE_PROB:.2f}, clahe_clip={AUG_CLAHE_CLIP:.2f}"
    )
    print(
        f"Sampler balancing: group={BALANCE_GROUP_SAMPLING}, family={BALANCE_FAMILY_SAMPLING}, "
        f"family_multipliers={FAMILY_SAMPLE_MULTIPLIER if FAMILY_SAMPLE_MULTIPLIER else '{}'}"
    )
    print(
        "Target dilation (train): "
        f"global={TRAIN_MASK_DILATION} "
        f"[ESP={TRAIN_MASK_DILATION_ESP}, tile_grid={TRAIN_MASK_DILATION_TILE_GRID}, "
        f"tile_numeric={TRAIN_MASK_DILATION_TILE_NUMERIC}, other={TRAIN_MASK_DILATION_OTHER}]"
    )
    print(f"OOB negative loss: enabled={LOSS_OOB_NEG}, weight={OOB_NEG_WEIGHT:.3f}")
    if CLDICE_MAX_LEGACY_OVERRIDE is not None and "CLDICE_MAX" not in os.environ:
        print(
            f"Legacy override: LOSS_CLDICE={CLDICE_MAX_LEGACY_OVERRIDE:g} interpreted as CLDICE_MAX."
        )
    if OOB_NEG_WEIGHT_LEGACY_OVERRIDE is not None and "OOB_NEG_WEIGHT" not in os.environ:
        print(
            f"Legacy override: LOSS_OOB_NEG={OOB_NEG_WEIGHT_LEGACY_OVERRIDE:g} interpreted as OOB_NEG_WEIGHT."
        )
    print(f"Ignore no-data in loss: {IGNORE_NO_DATA} (threshold={NO_DATA_THRESHOLD})")
    print(
        "No-data mask mode: "
        f"{'border-connected' if NO_DATA_BORDER_ONLY else 'all low-intensity'} "
        f"(pad={NO_DATA_BORDER_PAD})"
    )
    print(f"Validation penalizes out-of-bounds predictions: {VAL_PENALIZE_OOB}")
    print(
        f"Best-checkpoint selection: mode={BEST_MODEL_SELECTION}, "
        f"combo_search={VAL_GROUP_COMBO_SEARCH}, combo_guardrail={VAL_GROUP_COMBO_OOB_GUARDRAIL:.6f}, "
        f"close_iters={VAL_GROUP_CLOSE_ITERS}, close_groups={sorted(VAL_GROUP_CLOSE_GROUPS)}"
    )
    if VAL_GROUP_COMBO_SEARCH and len(VAL_THRESHOLDS) > 9:
        print(
            f"Combo search skipped during training checkpoint selection because "
            f"len(VAL_THRESHOLDS)={len(VAL_THRESHOLDS)} > 9."
        )
    print(
        "Pos-weight estimate uses dilated targets: "
        f"{POS_WEIGHT_ESTIMATE_USE_DILATED_TARGETS}"
    )
    if POS_WEIGHT_OVERRIDE is not None:
        print(f"Using POS_WEIGHT_OVERRIDE: {POS_WEIGHT_OVERRIDE:.6f}")
    print(f"Estimated train foreground ratio (after dilation): {fg_ratio * 100.0:.3f}%")
    print(f"Using raw pos_weight: {raw_pos_weight:.2f} | adjusted pos_weight: {pos_weight_scalar:.2f}")
    if has_group_pos_weight:
        print(f"Per-group BCE pos_weight overrides: {effective_group_pos_weight}")
    if has_family_pos_weight:
        print(
            "Per-family BCE pos_weight overrides: "
            f"{POS_WEIGHT_BY_FAMILY} (combine_mode={POS_WEIGHT_COMBINE_MODE})"
        )
    print(f"Train images: {TRAIN_IMG_DIR}")
    print(f"Train masks:  {TRAIN_MASK_DIR}")
    print(f"Split file:   {SPLIT_FILE}")
    if val_loader is not None:
        group_counts = defaultdict(lambda: {"train": 0, "val": 0})
        for pair in train_pairs:
            group_counts[infer_group_from_key(pair_key(pair))]["train"] += 1
        for pair in val_pairs:
            group_counts[infer_group_from_key(pair_key(pair))]["val"] += 1

        print(f"Split (pairs): train={len(train_pairs)} val={len(val_pairs)}")
        print(f"Split groups: {dict(group_counts)}")
    else:
        print("Split: train=all val=none")

    model.train()
    best_score = -1.0
    best_score_tie = -1.0
    best_epoch = -1
    best_model_path = os.path.splitext(MODEL_PATH)[0] + "_best.pth"
    best_val_metrics = None

    for epoch in range(EPOCHS):
        # --- Idea 1: Delayed feedback activation ---
        if ARCHITECTURE == "cyclic" and CYCLIC_FEEDBACK_UNFREEZE_EPOCH > 0:
            if epoch < CYCLIC_FEEDBACK_UNFREEZE_EPOCH:
                for p in model.cap_to_u.parameters():
                    p.requires_grad = False
            elif epoch == CYCLIC_FEEDBACK_UNFREEZE_EPOCH:
                for p in model.cap_to_u.parameters():
                    p.requires_grad = True
                print(f"Epoch {epoch+1}: Unfreezing cap_to_u feedback path")

        # --- Idea 3: Warm-start unfreeze ---
        if ARCHITECTURE == "cyclic" and CYCLIC_WARMSTART_PATH and CYCLIC_WARMSTART_FREEZE_EPOCHS > 0:
            if epoch == CYCLIC_WARMSTART_FREEZE_EPOCHS:
                for p in model.parameters():
                    p.requires_grad = True
                print(f"Epoch {epoch+1}: Unfreezing all parameters (warm-start phase complete)")

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_tversky = 0.0
        epoch_cldice = 0.0
        epoch_ds = 0.0
        epoch_oob_neg = 0.0
        epoch_skel_recall = 0.0
        current_cldice_weight = cldice_weight_for_epoch(epoch)
        current_skel_recall_weight = skel_recall_weight_for_epoch(epoch)

        for batch in train_loader:
            images, masks, valid, group_ids, family_ids = unpack_batch(batch)
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            masks = masks.to(DEVICE, non_blocking=PIN_MEMORY)
            valid = valid.to(DEVICE, non_blocking=PIN_MEMORY)
            if group_ids is not None:
                group_ids = group_ids.to(DEVICE, non_blocking=PIN_MEMORY)
            if family_ids is not None:
                family_ids = family_ids.to(DEVICE, non_blocking=PIN_MEMORY)
            batch_pos_weight = resolve_batch_pos_weight(
                group_ids,
                family_ids,
                pos_weight,
                group_pos_weight_tensor,
                family_pos_weight_tensor,
            )

            optimizer.zero_grad(set_to_none=True)

            if use_amp and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=True)
            elif use_amp:
                autocast_ctx = torch.cuda.amp.autocast(enabled=True)
            else:
                autocast_ctx = nullcontext()

            with autocast_ctx:
                if ARCHITECTURE == "cyclic":
                    output = model(images)
                    # output is dict with "main" and "aux" (single or list)
                    main_out = output["main"]
                    aux_out = output["aux"]
                    if isinstance(main_out, list):
                        K = len(main_out)
                        weights = [(i + 1) for i in range(K)]
                        total_w = sum(weights)
                        weights = [w / total_w for w in weights]
                        loss = images.new_tensor(0.0)
                        bce_acc = images.new_tensor(0.0)
                        tversky_acc = images.new_tensor(0.0)
                        cldice_acc = images.new_tensor(0.0)
                        for k_i in range(K):
                            m_loss, m_bce, m_tv, m_cl, _, _, _ = segmentation_loss(
                                main_out[k_i], masks, valid, batch_pos_weight,
                                cldice_weight=current_cldice_weight,
                                skel_recall_weight=current_skel_recall_weight)
                            if CYCLIC_AUX_FINAL_ONLY and k_i < K - 1:
                                a_loss = images.new_tensor(0.0)
                            else:
                                a_loss, _, _, _, _, _, _ = segmentation_loss(
                                    aux_out[k_i], masks, valid, batch_pos_weight,
                                    cldice_weight=current_cldice_weight,
                                skel_recall_weight=current_skel_recall_weight)
                            loss = loss + weights[k_i] * (
                                (1 - CYCLIC_AUX_WEIGHT) * m_loss + CYCLIC_AUX_WEIGHT * a_loss)
                            bce_acc = bce_acc + weights[k_i] * m_bce
                            tversky_acc = tversky_acc + weights[k_i] * m_tv
                            cldice_acc = cldice_acc + weights[k_i] * m_cl
                        bce, tversky, cldice = bce_acc, tversky_acc, cldice_acc
                    else:
                        m_loss, bce, tversky, cldice, _, _, _ = segmentation_loss(
                            main_out, masks, valid, batch_pos_weight,
                            cldice_weight=current_cldice_weight,
                                skel_recall_weight=current_skel_recall_weight)
                        a_loss, _, _, _, _, _, _ = segmentation_loss(
                            aux_out, masks, valid, batch_pos_weight,
                            cldice_weight=current_cldice_weight,
                                skel_recall_weight=current_skel_recall_weight)
                        loss = (1 - CYCLIC_AUX_WEIGHT) * m_loss + CYCLIC_AUX_WEIGHT * a_loss
                    logits = main_out[-1] if isinstance(main_out, list) else main_out
                    ds_loss = loss.new_tensor(0.0)
                    oob_neg = loss.new_tensor(0.0)
                else:
                    if DEEP_SUPERVISION:
                        logits, side_logits = model(images, return_side=True)
                    else:
                        logits = model(images)
                        side_logits = []
                    loss, bce, tversky, cldice, ds_loss, oob_neg, skel_rec = segmentation_loss(
                        logits,
                        masks,
                        valid,
                        batch_pos_weight,
                        side_logits=side_logits,
                        cldice_weight=current_cldice_weight,
                        skel_recall_weight=current_skel_recall_weight,
                    )

            if use_amp:
                scaler.scale(loss).backward()
                if GRAD_CLIP_NORM > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP_NORM > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            epoch_loss += loss.item()
            epoch_bce += bce.item()
            epoch_tversky += tversky.item()
            epoch_cldice += cldice.item()
            epoch_ds += ds_loss.item()
            epoch_oob_neg += oob_neg.item()
            epoch_skel_recall += skel_rec.item()

        scheduler.step()

        should_log = (epoch + 1) % max(1, EPOCHS // 20) == 0 or epoch == 0

        train_loss = epoch_loss / len(train_loader)
        train_bce = epoch_bce / len(train_loader)
        train_tversky = epoch_tversky / len(train_loader)
        train_cldice = epoch_cldice / len(train_loader)
        train_ds = epoch_ds / len(train_loader)
        train_oob_neg = epoch_oob_neg / len(train_loader)
        train_skel_recall = epoch_skel_recall / len(train_loader)

        if val_loader is None:
            if should_log:
                oob_train_str = f" | OOBneg: {train_oob_neg:.4f}" if LOSS_OOB_NEG else ""
                sr_str = f" | SkelRec: {train_skel_recall:.4f} (w={current_skel_recall_weight:.3f})" if LOSS_SKEL_RECALL else ""
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                    f"WBCE: {train_bce:.4f} | Tversky: {train_tversky:.4f} | "
                    f"clDice: {train_cldice:.4f} (w={current_cldice_weight:.3f}) | DS: {train_ds:.4f}"
                    f"{oob_train_str}{sr_str}"
                )
            continue

        val_metrics = evaluate_with_threshold_sweep(
            model,
            val_loader,
            pos_weight,
            group_pos_weight_tensor=group_pos_weight_tensor,
            family_pos_weight_tensor=family_pos_weight_tensor,
            cldice_weight=current_cldice_weight,
            skel_recall_weight=current_skel_recall_weight,
        )
        score = float(val_metrics.get("selection_tolerant_f1", val_metrics["best_tolerant_f1"]))
        score_tie = float(val_metrics.get("selection_dice", val_metrics["best_dice"]))

        if score > best_score or (
            abs(score - best_score) < 1e-10 and score_tie > best_score_tie
        ):
            best_score = score
            best_score_tie = score_tie
            best_epoch = epoch + 1
            best_val_metrics = val_metrics
            torch.save(model.state_dict(), best_model_path)

        if should_log:
            oob_train_str = f"OOBneg: {train_oob_neg:.4f} | " if LOSS_OOB_NEG else ""
            oob_val_str = f" | Val OOBneg: {val_metrics['oob_neg_loss']:.4f}" if LOSS_OOB_NEG else ""
            sr_train_str = f"SkelRec: {train_skel_recall:.4f} (w={current_skel_recall_weight:.3f}) | " if LOSS_SKEL_RECALL else ""
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                f"WBCE: {train_bce:.4f} | Tversky: {train_tversky:.4f} | "
                f"clDice: {train_cldice:.4f} (w={current_cldice_weight:.3f}) | DS: {train_ds:.4f} | "
                f"{sr_train_str}{oob_train_str}"
                f"Val Loss: {val_metrics['loss']:.4f} | Val Dice(best-thr): {val_metrics['best_dice']:.4f} | "
                f"Val tolF1@{VAL_TOLERANCE_PX}px: {val_metrics['best_tolerant_f1']:.4f} | "
                f"Thr*: {val_metrics['best_threshold']:.2f} | "
                f"CC ratio: {val_metrics['cc_ratio']:.3f} | "
                f"Len ratio: {val_metrics['len_ratio']:.3f} | "
                f"Bridge: {val_metrics['bridge_rate']:.3f} | "
                f"OOB+: {val_metrics['best_oob_pred_ratio']:.4f} | "
                f"Sel[{val_metrics['selection_source']}]: "
                f"{val_metrics['selection_tolerant_f1']:.4f}"
                f"{oob_val_str}"
            )

        # --- cap_to_u diagnostic ---
        if ARCHITECTURE == "cyclic" and should_log:
            cw = model.cap_to_u.weight.data
            cb = model.cap_to_u.bias.data
            cg = model.cap_to_u.weight.grad
            cg_norm = cg.norm().item() if cg is not None else 0.0
            print(
                f"  [cap_to_u] w_norm={cw.norm().item():.6f} "
                f"w_max={cw.abs().max().item():.6f} "
                f"b_norm={cb.norm().item():.6f} "
                f"grad_norm={cg_norm:.6f}"
            )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete. Model saved as '{MODEL_PATH}'")

    if val_loader is not None:
        print(f"Best validation model saved as '{best_model_path}'")
        if best_val_metrics is not None:
            print(
                f"Best epoch: {best_epoch} | best threshold: {best_val_metrics['best_threshold']:.2f} | "
                f"best tolF1: {best_val_metrics['best_tolerant_f1']:.4f} | "
                f"selection({best_val_metrics['selection_source']}): "
                f"{best_val_metrics['selection_tolerant_f1']:.4f}"
            )

            report_path = os.path.splitext(MODEL_PATH)[0] + "_metrics.json"
            report = {
                "model_path": MODEL_PATH,
                "best_model_path": best_model_path,
                "best_epoch": best_epoch,
                "best_threshold": best_val_metrics["best_threshold"],
                "best_tolerant_f1": best_val_metrics["best_tolerant_f1"],
                "best_dice": best_val_metrics["best_dice"],
                "best_soft_dice": best_val_metrics["soft_dice"],
                "best_precision": best_val_metrics["best_precision"],
                "best_recall": best_val_metrics["best_recall"],
                "best_cldice": best_val_metrics["cldice"],
                "best_ds_loss": best_val_metrics["ds_loss"],
                "best_oob_neg_loss": best_val_metrics["oob_neg_loss"],
                "best_cc_ratio": best_val_metrics["cc_ratio"],
                "best_len_ratio": best_val_metrics["len_ratio"],
                "best_bridge_rate": best_val_metrics["bridge_rate"],
                "best_oob_pred_pixels": best_val_metrics["best_oob_pred_pixels"],
                "best_oob_pred_ratio": best_val_metrics["best_oob_pred_ratio"],
                "best_oob_pred_fraction": best_val_metrics["best_oob_pred_fraction"],
                "target_fg_ratio": best_val_metrics["target_fg_ratio"],
                "pred_fg_ratio": best_val_metrics["pred_fg_ratio"],
                "mean_prob": best_val_metrics["mean_prob"],
                "combo_search_executed": best_val_metrics["combo_search_executed"],
                "combo_search_skipped_reason": best_val_metrics["combo_search_skipped_reason"],
                "combo_group_thresholds": best_val_metrics["combo_group_thresholds"],
                "combo_tolerant_f1": best_val_metrics["combo_tolerant_f1"],
                "combo_dice": best_val_metrics["combo_dice"],
                "combo_oob_pred_pixels": best_val_metrics["combo_oob_pred_pixels"],
                "combo_oob_pred_ratio": best_val_metrics["combo_oob_pred_ratio"],
                "combo_oob_pred_fraction": best_val_metrics["combo_oob_pred_fraction"],
                "combo_pred_fg_ratio": best_val_metrics["combo_pred_fg_ratio"],
                "best_model_selection": best_val_metrics["best_model_selection"],
                "selection_source": best_val_metrics["selection_source"],
                "selection_tolerant_f1": best_val_metrics["selection_tolerant_f1"],
                "selection_dice": best_val_metrics["selection_dice"],
                "selection_oob_pred_pixels": best_val_metrics["selection_oob_pred_pixels"],
                "selection_oob_pred_ratio": best_val_metrics["selection_oob_pred_ratio"],
                "selection_oob_pred_fraction": best_val_metrics["selection_oob_pred_fraction"],
                "selection_pred_fg_ratio": best_val_metrics["selection_pred_fg_ratio"],
                "val_thresholds": VAL_THRESHOLDS,
                "tolerance_px": VAL_TOLERANCE_PX,
                "split_file": SPLIT_FILE,
                "config": {
                    "architecture": ARCHITECTURE,
                    "cyclic_iterations": CYCLIC_ITERATIONS,
                    "cyclic_aux_weight": CYCLIC_AUX_WEIGHT,
                    "dataset_dir": DATASET_DIR,
                    "epochs": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "image_size": IMAGE_SIZE,
                    "unet_depth": UNET_DEPTH,
                    "encoder_name": ENCODER_NAME if ENCODER_NAME else None,
                    "encoder_weights": ENCODER_WEIGHTS if ENCODER_NAME else None,
                    "imagenet_normalize": IMAGENET_NORMALIZE if ENCODER_NAME else None,
                    "unet_base_channels": UNET_BASE_CHANNELS,
                    "unet_decoder_dropout": UNET_DECODER_DROPOUT,
                    "norm": NORM,
                    "gn_groups": GN_GROUPS,
                    "upsample_mode": UPSAMPLE_MODE,
                    "deep_supervision": DEEP_SUPERVISION,
                    "ds_weights": DS_WEIGHTS,
                    "patch_size": PATCH_SIZE,
                    "train_use_patches": TRAIN_USE_PATCHES,
                    "train_patches_per_image": TRAIN_PATCHES_PER_IMAGE,
                    "positive_crop_prob": POSITIVE_CROP_PROB,
                    "train_mask_dilation": TRAIN_MASK_DILATION,
                    "train_mask_dilation_esp": TRAIN_MASK_DILATION_ESP,
                    "train_mask_dilation_tile_grid": TRAIN_MASK_DILATION_TILE_GRID,
                    "train_mask_dilation_tile_numeric": TRAIN_MASK_DILATION_TILE_NUMERIC,
                    "train_mask_dilation_other": TRAIN_MASK_DILATION_OTHER,
                    "ignore_no_data": IGNORE_NO_DATA,
                    "no_data_threshold": NO_DATA_THRESHOLD,
                    "no_data_border_only": NO_DATA_BORDER_ONLY,
                    "no_data_border_pad": NO_DATA_BORDER_PAD,
                    "val_penalize_oob": VAL_PENALIZE_OOB,
                    "loss_bce_weight": LOSS_BCE_WEIGHT,
                    "loss_tversky_weight": LOSS_TVERSKY_WEIGHT,
                    "tversky_alpha": TVERSKY_ALPHA,
                    "tversky_beta": TVERSKY_BETA,
                    "loss_cldice": LOSS_CLDICE,
                    "loss_oob_neg": LOSS_OOB_NEG,
                    "oob_neg_weight": OOB_NEG_WEIGHT,
                    "cldice_iters": CLDICE_ITERS,
                    "cldice_max": CLDICE_MAX,
                    "cldice_warmup_epochs": CLDICE_WARMUP_EPOCHS,
                    "cldice_ramp_epochs": CLDICE_RAMP_EPOCHS,
                    "loss_skel_recall": LOSS_SKEL_RECALL,
                    "skel_recall_max": SKEL_RECALL_MAX,
                    "skel_recall_warmup_epochs": SKEL_RECALL_WARMUP_EPOCHS,
                    "skel_recall_ramp_epochs": SKEL_RECALL_RAMP_EPOCHS,
                    "skel_recall_tube_radius": SKEL_RECALL_TUBE_RADIUS,
                    "cldice_max_legacy_override": CLDICE_MAX_LEGACY_OVERRIDE,
                    "oob_neg_weight_legacy_override": OOB_NEG_WEIGHT_LEGACY_OVERRIDE,
                    "raw_pos_weight": raw_pos_weight,
                    "adjusted_pos_weight": pos_weight_scalar,
                    "pos_weight_multiplier": POS_WEIGHT_MULTIPLIER,
                    "pos_weight_override": POS_WEIGHT_OVERRIDE,
                    "pos_weight_estimate_use_dilated_targets": POS_WEIGHT_ESTIMATE_USE_DILATED_TARGETS,
                    "pos_weight_by_group": effective_group_pos_weight if has_group_pos_weight else None,
                    "pos_weight_by_family": effective_family_pos_weight if has_family_pos_weight else None,
                    "pos_weight_combine_mode": POS_WEIGHT_COMBINE_MODE,
                    "num_workers": NUM_WORKERS,
                    "pin_memory": PIN_MEMORY,
                    "persistent_workers": PERSISTENT_WORKERS,
                    "prefetch_factor": PREFETCH_FACTOR,
                    "cache_samples": CACHE_SAMPLES,
                    "amp": use_amp,
                    "grad_clip_norm": GRAD_CLIP_NORM,
                    "balance_group_sampling": BALANCE_GROUP_SAMPLING,
                    "balance_family_sampling": BALANCE_FAMILY_SAMPLING,
                    "family_sample_multiplier": FAMILY_SAMPLE_MULTIPLIER,
                    "aug_geometric": AUG_GEOMETRIC,
                    "aug_photometric": AUG_PHOTOMETRIC,
                    "aug_brightness": AUG_BRIGHTNESS,
                    "aug_contrast": AUG_CONTRAST,
                    "aug_gamma": AUG_GAMMA,
                    "aug_noise_std": AUG_NOISE_STD,
                    "aug_noise_prob": AUG_NOISE_PROB,
                    "aug_blur_prob": AUG_BLUR_PROB,
                    "aug_contrast_mode_branch": AUG_CONTRAST_MODE_BRANCH,
                    "aug_contrast_mode_prob": AUG_CONTRAST_MODE_PROB,
                    "aug_invert_prob": AUG_INVERT_PROB,
                    "aug_clahe_prob": AUG_CLAHE_PROB,
                    "aug_clahe_clip": AUG_CLAHE_CLIP,
                    "best_model_selection": BEST_MODEL_SELECTION,
                    "val_group_combo_search": VAL_GROUP_COMBO_SEARCH,
                    "val_group_combo_oob_guardrail": VAL_GROUP_COMBO_OOB_GUARDRAIL,
                    "val_group_close_iters": VAL_GROUP_CLOSE_ITERS,
                    "val_group_close_groups": sorted(VAL_GROUP_CLOSE_GROUPS),
                },
            }
            write_json(report_path, report)
            print(f"Metrics report saved as '{report_path}'")


if __name__ == "__main__":
    train()
