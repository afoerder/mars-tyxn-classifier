"""B11: slicer must read tile windows directly from a rasterio dataset
(local or /vsis3/), not via opj_decompress subprocess only.

Includes a Tier-A parity test: a synthetic image processed via both the
existing full-decode path and the new COG path must produce byte-identical
output PNG files and identical tile_metadata.csv rows.
"""
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from mars_tyxn.tile_hirise_for_pipeline import (
    decode_cog_region,
    process_one_image,
)


def _write_synth_geotiff(path: Path, *, width=1024, height=1024, dtype=np.uint16):
    """Write a synthetic GeoTIFF (Mars-projected) to disk for the slicer."""
    arr = (np.arange(width * height, dtype=np.uint64) % 60000).astype(dtype).reshape(height, width)
    transform = from_origin(0, 0, 1.0, 1.0)
    with rasterio.open(
        path, "w", driver="GTiff",
        height=height, width=width, count=1, dtype=arr.dtype,
        crs=CRS.from_epsg(4326), transform=transform,
    ) as dst:
        dst.write(arr, 1)


def test_decode_cog_region_reads_window(tmp_path: Path):
    """Window read returns (h, w) shape with values matching the source array."""
    cog_path = tmp_path / "synth.tif"
    arr = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    with rasterio.open(
        cog_path, "w", driver="GTiff", height=256, width=256, count=1,
        dtype=arr.dtype, crs=CRS.from_epsg(4326),
        transform=from_origin(0, 0, 1.0, 1.0),
    ) as dst:
        dst.write(arr, 1)

    with rasterio.open(cog_path) as src:
        win = decode_cog_region(src, x0=10, y0=20, x1=42, y1=84)
    assert win.shape == (64, 32)
    assert win[0, 0] == arr[20, 10]
    assert win[-1, -1] == arr[83, 41]


def test_process_one_image_cog_path_produces_tiles(tmp_path: Path):
    """process_one_image with cog_dataset=... slices via the COG path."""
    cog_path = tmp_path / "synthproduct.tif"
    _write_synth_geotiff(cog_path, width=1024, height=1024)
    out_root = tmp_path / "out"

    with rasterio.open(cog_path) as cog:
        summary = process_one_image(
            image_path=cog_path,
            out_root=out_root,
            cog_dataset=cog,
            tile_size=512,
            overlap=0,
            min_valid_ratio=0.0,  # accept all tiles for synth data
        )
    assert summary["kept_tiles"] >= 1
    assert (out_root / cog_path.stem / "tile_metadata.csv").is_file()


def test_cog_and_full_paths_byte_identical(tmp_path: Path):
    """Tier-A parity: COG path output is byte-identical to the existing path.

    Uses a small synthetic GeoTIFF that fits in memory, processes it
    via process_one_image's existing default-mode path (no cog_dataset)
    AND via the new COG path (cog_dataset=...). The output PNG bytes
    and CSV rows must be byte-equal.
    """
    cog_path = tmp_path / "parity_input.tif"
    _write_synth_geotiff(cog_path, width=512, height=512)

    out_full = tmp_path / "out_full"
    out_cog = tmp_path / "out_cog"

    # Path 1: default decode (no cog_dataset; non-JP2 -> full-decode in-memory)
    summary_full = process_one_image(
        image_path=cog_path,
        out_root=out_full,
        tile_size=256,
        overlap=0,
        min_valid_ratio=0.0,
    )
    # Path 2: pass an open rasterio dataset; COG branch fires
    with rasterio.open(cog_path) as cog:
        summary_cog = process_one_image(
            image_path=cog_path,
            out_root=out_cog,
            cog_dataset=cog,
            tile_size=256,
            overlap=0,
            min_valid_ratio=0.0,
        )

    # Compare PNG files byte-by-byte
    full_tiles = sorted((out_full / cog_path.stem / "tiles").glob("*.png"))
    cog_tiles = sorted((out_cog / cog_path.stem / "tiles").glob("*.png"))
    assert len(full_tiles) == len(cog_tiles) > 0, (
        f"tile counts differ: full={len(full_tiles)} cog={len(cog_tiles)}"
    )
    for full_tile, cog_tile in zip(full_tiles, cog_tiles):
        assert full_tile.name == cog_tile.name, (
            f"tile name mismatch: {full_tile.name} vs {cog_tile.name}"
        )
        assert full_tile.read_bytes() == cog_tile.read_bytes(), (
            f"PNG bytes differ for {full_tile.name}"
        )

    # Compare tile_metadata.csv rows
    full_csv_text = (out_full / cog_path.stem / "tile_metadata.csv").read_text()
    cog_csv_text = (out_cog / cog_path.stem / "tile_metadata.csv").read_text()
    assert full_csv_text == cog_csv_text, "tile_metadata.csv differs between paths"


def test_existing_jp2_path_not_regressed(tmp_path: Path):
    """The existing decode_mode='full' path still works without cog_dataset arg."""
    img_path = tmp_path / "small.tif"
    _write_synth_geotiff(img_path, width=512, height=512)

    summary = process_one_image(
        image_path=img_path,
        out_root=tmp_path / "out",
        tile_size=256,
        overlap=0,
        min_valid_ratio=0.0,
    )
    assert summary["kept_tiles"] >= 1
