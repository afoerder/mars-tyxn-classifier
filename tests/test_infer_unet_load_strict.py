"""Regression test for B6 (REVIEW.md): U-Net load_state_dict must fail loudly
on prefix mismatch, not silently fall back to strict=False.
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mars_tyxn.infer_unet import _load_model_and_metadata


class _DifferentArch(nn.Module):
    """Has a `linear` submodule the SimpleUNet does not."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _write_metrics(tmp_path: Path) -> Path:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "best_threshold": 0.5,
        "config": {
            "image_size": 64,
            "encoder_type": "custom",
            "unet_depth": 2,
            "unet_base_channels": 4,
            "unet_decoder_dropout": 0.0,
            "unet_norm": "batch",
            "unet_gn_groups": 4,
            "unet_upsample_mode": "bilinear",
            "unet_deep_supervision": False,
        },
    }))
    return metrics_path


def test_load_state_dict_raises_on_prefix_mismatch(tmp_path: Path):
    """A state dict with wrong key prefixes must raise, not silently load zeros."""
    bad_arch = _DifferentArch()
    bad_state_path = tmp_path / "wrong_prefix.pth"
    torch.save(bad_arch.state_dict(), bad_state_path)

    metrics_path = _write_metrics(tmp_path)
    device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="load_state_dict"):
        _load_model_and_metadata(bad_state_path, metrics_path, device)
