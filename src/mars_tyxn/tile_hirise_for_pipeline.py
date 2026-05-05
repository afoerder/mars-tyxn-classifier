#!/usr/bin/env python3
"""Tile HiRISE JP2 parent images into 768x768 PNG tiles with geolocation metadata.

Designed for the Utopia Planitia regional application: takes a directory of HiRISE
RED JP2 products, slices each into overlapping tiles suitable for the U-Net
segmentation pipeline, and records geographic coordinates for each tile.

Adapted from slice_hirise_parent_for_labeling.py. Core tiling logic preserved;
added batch processing, rasterio-based geolocation, and per-image summaries.

Usage:
    python tile_hirise_for_pipeline.py \
        --image-dir /path/to/jp2s \
        --out-root /path/to/output

    # Or single image:
    python tile_hirise_for_pipeline.py \
        --image /path/to/image.JP2 \
        --out-root /path/to/output
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 40))

import cv2
import numpy as np

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ---------------------------------------------------------------------------
# Geolocation
# ---------------------------------------------------------------------------

def mars_projected_to_latlon(
    easting: float, northing: float, crs_wkt: str
) -> Tuple[float, float]:
    """Convert Mars projected coordinates to latitude/longitude.

    Supports Polar Stereographic (used for polar HiRISE images) and
    Equirectangular (used for mid-latitude images). The projection type
    and parameters are parsed from the CRS WKT string embedded in the JP2.
    """
    wkt = str(crs_wkt)

    # Extract Mars spheroid radius
    R = 3376200.0
    m = re.search(r'SPHEROID\["[^"]*",(\d+\.?\d*)', wkt)
    if m:
        R = float(m.group(1))

    if "Polar_Stereographic" in wkt:
        rho = np.sqrt(easting ** 2 + northing ** 2)
        if rho == 0:
            return 90.0, 0.0
        lat = 90.0 - 2.0 * np.degrees(np.arctan(rho / (2.0 * R)))
        lon = np.degrees(np.arctan2(easting, -northing))
        return float(lat), float(lon)

    if "Equirectangular" in wkt or "Equidistant_Cylindrical" in wkt:
        lat_ts = 0.0
        m_lat = re.search(
            r"latitude_of_origin[^,]*,\s*([-\d.]+)", wkt, re.IGNORECASE
        )
        if not m_lat:
            m_lat = re.search(
                r"standard_parallel_1[^,]*,\s*([-\d.]+)", wkt, re.IGNORECASE
            )
        if m_lat:
            lat_ts = float(m_lat.group(1))

        lon0 = 0.0
        m_lon = re.search(
            r"central_meridian[^,]*,\s*([-\d.]+)", wkt, re.IGNORECASE
        )
        if m_lon:
            lon0 = float(m_lon.group(1))

        lat = np.degrees(northing / R)
        lon = lon0 + np.degrees(easting / (R * np.cos(np.radians(lat_ts))))
        return float(lat), float(lon)

    raise ValueError(f"Unsupported Mars projection: {wkt[:200]}")


def _geo_from_dataset(src) -> Optional[Dict]:
    """Extract geolocation dict from an open rasterio dataset."""
    if src.crs is None:
        return None
    return {
        "crs_wkt": src.crs.to_wkt(),
        "transform": src.transform,
        "width": src.width,
        "height": src.height,
    }


def read_jp2_geolocation(image_path: Path) -> Optional[Dict]:
    """Read CRS and affine transform from a JP2 via rasterio."""
    if not HAS_RASTERIO:
        return None
    try:
        with rasterio.open(str(image_path)) as src:
            return _geo_from_dataset(src)
    except Exception:
        return None


def tile_center_latlon(
    x0: int, y0: int, tile_size: int, geo: Dict
) -> Tuple[float, float, float, float]:
    """Compute geographic coordinates for the center of a tile.

    Returns (lat, lon, easting, northing).
    """
    center_col = x0 + tile_size // 2
    center_row = y0 + tile_size // 2
    easting, northing = rasterio.transform.xy(
        geo["transform"], center_row, center_col
    )
    lat, lon = mars_projected_to_latlon(easting, northing, geo["crs_wkt"])
    return lat, lon, float(easting), float(northing)


# ---------------------------------------------------------------------------
# Tiling (preserved from slice_hirise_parent_for_labeling.py)
# ---------------------------------------------------------------------------

def load_grayscale(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim != 2:
        raise ValueError(
            f"Expected single-channel image, got shape {arr.shape} for {path}"
        )
    return arr


def normalize_robust_uint8(
    raw: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x = raw.astype(np.float32, copy=False)
    valid = x > 3.0
    valid_count = int(valid.sum())
    total_count = int(x.size)
    stats: Dict[str, float] = {
        "raw_dtype": str(raw.dtype),
        "raw_min": float(x.min()) if total_count > 0 else 0.0,
        "raw_max": float(x.max()) if total_count > 0 else 0.0,
        "valid_pixels": valid_count,
        "total_pixels": total_count,
    }
    if valid_count == 0:
        stats.update({"p1": 0.0, "p99": 0.0, "scale_mode": "all_invalid"})
        return np.zeros_like(raw, dtype=np.uint8), stats

    vals = x[valid]
    p1 = float(np.percentile(vals, 1.0))
    p99 = float(np.percentile(vals, 99.0))
    if not np.isfinite(p1) or not np.isfinite(p99):
        p1, p99 = float(vals.min()), float(vals.max())
    if p99 <= p1:
        p99 = p1 + 1.0

    clipped = np.clip(x, p1, p99)
    scaled = ((clipped - p1) / (p99 - p1)) * 255.0
    scaled[~valid] = 0.0
    out = np.clip(np.round(scaled), 0.0, 255.0).astype(np.uint8)
    stats.update({"p1": p1, "p99": p99, "scale_mode": "robust_p1_p99"})
    return out, stats


def compute_anchors(length: int, tile_size: int, stride: int) -> List[int]:
    if length < tile_size:
        return []
    anchors = list(range(0, length - tile_size + 1, stride))
    edge_anchor = length - tile_size
    if not anchors or anchors[-1] != edge_anchor:
        anchors.append(edge_anchor)
    return anchors


def parse_jp2_dims_with_opj_dump(image_path: Path) -> Tuple[int, int]:
    opj_dump = shutil.which("opj_dump")
    if not opj_dump:
        raise RuntimeError("JP2 windowed mode requires 'opj_dump' in PATH.")
    result = subprocess.run(
        [opj_dump, "-i", str(image_path)],
        check=True, capture_output=True, text=True,
    )
    txt = result.stdout
    m0 = re.search(r"x0=(\d+),\s*y0=(\d+)", txt)
    m1 = re.search(r"x1=(\d+),\s*y1=(\d+)", txt)
    if not m0 or not m1:
        raise RuntimeError(
            f"Could not parse JP2 dimensions from opj_dump for {image_path}"
        )
    width = int(m1.group(1)) - int(m0.group(1))
    height = int(m1.group(2)) - int(m0.group(2))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid JP2 dims: {width}x{height}")
    return width, height


def decode_jp2_region(
    image_path: Path, x0: int, y0: int, x1: int, y1: int, tmp_dir: Path
) -> np.ndarray:
    opj_decompress = shutil.which("opj_decompress")
    if not opj_decompress:
        raise RuntimeError(
            "JP2 windowed mode requires 'opj_decompress' in PATH."
        )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=str(tmp_dir), suffix=".pgm", delete=False
    ) as f:
        tmp_path = Path(f.name)
    cmd = [
        opj_decompress, "-quiet",
        "-i", str(image_path),
        "-d", f"{x0},{y0},{x1},{y1}",
        "-o", str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        arr = load_grayscale(tmp_path)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return arr


def decode_cog_region(
    cog_dataset, x0: int, y0: int, x1: int, y1: int
) -> np.ndarray:
    """Read a window from an open rasterio dataset (local or /vsis3/).

    Parameters
    ----------
    cog_dataset
        An open ``rasterio.DatasetReader`` (e.g. from
        ``mars_tcp.pds.open_cog``). Either an s3-resident COG via
        ``/vsis3/`` or a local GeoTIFF.
    x0, y0, x1, y1
        Window bounds in pixel coordinates (inclusive of x0, y0;
        exclusive of x1, y1). Must lie inside ``[0, cog_dataset.width)``
        and ``[0, cog_dataset.height)``.

    Returns
    -------
    np.ndarray
        Single-band window of shape ``(y1 - y0, x1 - x0)``, native dtype.
        Caller normalizes to uint8 via ``normalize_robust_uint8``.
    """
    from rasterio.windows import Window
    width = x1 - x0
    height = y1 - y0
    window = Window(col_off=x0, row_off=y0, width=width, height=height)
    return cog_dataset.read(1, window=window)


# ---------------------------------------------------------------------------
# Single-image processing
# ---------------------------------------------------------------------------

def _process_anchor(
    raw_tile: np.ndarray,
    x0: int, y0: int,
    tile_size: int,
    parent_id: str,
    tiles_dir: Path,
    geo: Optional[Dict],
    min_valid_ratio: float,
) -> Tuple[Optional[Dict], bool]:
    """Process one tile anchor: gate, normalize, write PNG, build CSV row.

    Parameters
    ----------
    raw_tile
        The raw (unnormalized) tile array, shape ``(tile_size, tile_size)``.
    x0, y0
        Anchor pixel coordinates (top-left corner of tile).
    tile_size
        Tile side length in pixels.
    parent_id
        Stem of the parent image filename.
    tiles_dir
        Directory to write kept PNG tiles into.
    geo
        Geolocation dict (from ``read_jp2_geolocation`` or
        ``_geo_from_dataset``), or None.
    min_valid_ratio
        Fraction of pixels > 3 required to keep the tile.

    Returns
    -------
    row : dict or None
        The CSV metadata row dict, or None if shape is wrong.
    kept : bool
        Whether the tile passed the valid-ratio gate (and was written).
    """
    if raw_tile.shape != (tile_size, tile_size):
        return None, False

    x1 = x0 + tile_size
    y1 = y0 + tile_size
    key = f"hirise_{parent_id}__y{y0:05d}_x{x0:05d}_s{tile_size}"
    valid_ratio = float((raw_tile > 3).mean())
    keep = valid_ratio >= min_valid_ratio

    row: Dict = {
        "key": key,
        "parent_id": parent_id,
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "width": tile_size,
        "height": tile_size,
        "valid_ratio": round(valid_ratio, 4),
        "kept": keep,
    }

    if geo is not None:
        lat, lon, e, n = tile_center_latlon(x0, y0, tile_size, geo)
        row.update({
            "center_lat": round(lat, 6),
            "center_lon": round(lon, 6),
            "center_easting": round(e, 2),
            "center_northing": round(n, 2),
        })

    if keep:
        tile8, _ = normalize_robust_uint8(raw_tile)
        out_path = tiles_dir / f"{key}.png"
        if not cv2.imwrite(str(out_path), tile8):
            raise RuntimeError(f"Failed writing: {out_path}")

    return row, keep


def process_one_image(
    image_path: Path,
    out_root: Path,
    tile_size: int = 768,
    overlap: int = 128,
    min_valid_ratio: float = 0.60,
    tmp_dir: Optional[Path] = None,
    cog_dataset=None,
) -> Dict:
    """Tile one HiRISE image and return a summary dict.

    Three decode modes, selected automatically:

    1. **COG** (``cog_dataset`` is not None): reads windows directly from
       an open ``rasterio.DatasetReader`` (local GeoTIFF or ``/vsis3/``).
       Used by AWS Batch / GCP Vertex AI / K8s deployments.
    2. **windowed JP2** (``cog_dataset`` is None, ``.jp2``/``.j2k``/``.jpt``
       suffix): decodes strips via ``opj_decompress`` subprocess.
       Arizona.edu download fallback.
    3. **full-decode** (``cog_dataset`` is None, non-JP2 suffix): loads the
       entire image into memory via OpenCV.  Used for GeoTIFF inputs and
       during unit tests.

    The per-anchor normalization and PNG-write logic is shared across all
    three modes via ``_process_anchor``.
    """
    stride = tile_size - overlap
    parent_id = image_path.stem

    image_dir = out_root / parent_id
    tiles_dir = image_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    is_jp2 = image_path.suffix.lower() in (".jp2", ".j2k", ".jpt")

    if tmp_dir is None:
        tmp_dir = image_dir / ".tmp_jp2"

    # --- Determine decode mode and read dimensions / geo ---
    if cog_dataset is not None:
        decode_mode = "cog"
        w = cog_dataset.width
        h = cog_dataset.height
        geo = _geo_from_dataset(cog_dataset) if HAS_RASTERIO else None
    elif is_jp2:
        decode_mode = "windowed"
        w, h = parse_jp2_dims_with_opj_dump(image_path)
        geo = read_jp2_geolocation(image_path)
    else:
        decode_mode = "full"
        raw = load_grayscale(image_path)
        h, w = raw.shape
        geo = read_jp2_geolocation(image_path)

    has_geo = geo is not None
    if not has_geo:
        print(f"  WARNING: No geolocation for {parent_id}")

    y_anchors = compute_anchors(h, tile_size, stride)
    x_anchors = compute_anchors(w, tile_size, stride)
    if not y_anchors or not x_anchors:
        raise RuntimeError(
            f"Image too small: {w}x{h}, tile_size={tile_size}"
        )

    # Compute image center coordinates
    image_lat, image_lon = None, None
    if has_geo:
        try:
            image_lat, image_lon, _, _ = tile_center_latlon(
                w // 2 - tile_size // 2, h // 2 - tile_size // 2, tile_size, geo
            )
        except (ValueError, Exception):
            has_geo = False
            geo = None
            print(f"  WARNING: Unsupported projection for {parent_id}, disabling geolocation")

    rows: List[Dict] = []
    kept = 0
    dropped = 0

    # --- Strip-based decode modes (windowed JP2 and COG) ---
    if decode_mode in ("windowed", "cog"):
        for yi, y0 in enumerate(y_anchors):
            y1 = y0 + tile_size

            if decode_mode == "windowed":
                strip = decode_jp2_region(image_path, 0, y0, w, y1, tmp_dir)
                if strip.shape != (tile_size, w):
                    raise RuntimeError(
                        f"Strip shape mismatch at y0={y0}: {strip.shape}"
                    )
            else:  # cog
                strip = decode_cog_region(cog_dataset, 0, y0, w, y1)

            for x0 in x_anchors:
                raw_tile = strip[:, x0:x0 + tile_size]
                row, tile_kept = _process_anchor(
                    raw_tile, x0, y0, tile_size, parent_id,
                    tiles_dir, geo, min_valid_ratio,
                )
                if row is None:
                    continue
                rows.append(row)
                if tile_kept:
                    kept += 1
                else:
                    dropped += 1

            if (yi + 1) % 10 == 0 or (yi + 1) == len(y_anchors):
                print(
                    f"  Strip {yi + 1}/{len(y_anchors)}, "
                    f"kept {kept} tiles so far"
                )

    else:  # full-decode
        for y0 in y_anchors:
            for x0 in x_anchors:
                raw_tile = raw[y0:y0 + tile_size, x0:x0 + tile_size]
                row, tile_kept = _process_anchor(
                    raw_tile, x0, y0, tile_size, parent_id,
                    tiles_dir, geo, min_valid_ratio,
                )
                if row is None:
                    continue
                rows.append(row)
                if tile_kept:
                    kept += 1
                else:
                    dropped += 1

    # Write per-image tile metadata CSV
    if rows:
        fieldnames = list(rows[0].keys())
        csv_path = image_dir / "tile_metadata.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Clean up temp dir (windowed JP2 only)
    if decode_mode == "windowed":
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    summary = {
        "parent_id": parent_id,
        "image_path": str(image_path),
        "width": w,
        "height": h,
        "tile_size": tile_size,
        "overlap": overlap,
        "stride": stride,
        "total_windows": len(rows),
        "kept_tiles": kept,
        "dropped_tiles": dropped,
        "center_lat": image_lat,
        "center_lon": image_lon,
        "has_geolocation": has_geo,
        "tiles_dir": str(tiles_dir),
    }

    # Write per-image summary JSON
    summary_path = image_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Batch processing and CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile HiRISE JP2 parent images into 768x768 PNG tiles "
            "with geolocation metadata for the classification pipeline."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=Path, help="Path to a single JP2 image."
    )
    group.add_argument(
        "--image-dir", type=Path,
        help="Directory containing JP2 images (processes all).",
    )
    parser.add_argument(
        "--out-root", type=Path, required=True,
        help="Output root directory.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=768,
        help="Tile size in pixels (default: 768, matching U-Net training).",
    )
    parser.add_argument(
        "--overlap", type=int, default=128,
        help="Tile overlap in pixels (default: 128).",
    )
    parser.add_argument(
        "--min-valid-ratio", type=float, default=0.60,
        help="Drop tiles below this valid-pixel ratio (default: 0.60).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tile_size % 32 != 0:
        raise ValueError("--tile-size must be divisible by 32.")
    stride = args.tile_size - args.overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile-size.")

    if not HAS_RASTERIO:
        print(
            "WARNING: rasterio not installed. "
            "Tiles will lack geographic coordinates."
        )

    # Collect JP2 files
    if args.image:
        jp2_files = [args.image]
    else:
        jp2_files = sorted(
            p for p in args.image_dir.iterdir()
            if p.suffix.lower() in (".jp2", ".j2k", ".jpt")
        )
        if not jp2_files:
            raise FileNotFoundError(
                f"No JP2 files found in {args.image_dir}"
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    print(f"Processing {len(jp2_files)} image(s)")
    print(f"Tile size: {args.tile_size}, overlap: {args.overlap}, "
          f"stride: {stride}")
    print(f"Output: {args.out_root}")
    print()

    for i, jp2_path in enumerate(jp2_files):
        print(f"[{i + 1}/{len(jp2_files)}] {jp2_path.name}")
        summary = process_one_image(
            image_path=jp2_path,
            out_root=args.out_root,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_valid_ratio=args.min_valid_ratio,
        )
        summaries.append(summary)
        print(
            f"  {summary['kept_tiles']} tiles kept "
            f"({summary['dropped_tiles']} dropped), "
            f"lat={summary['center_lat']:.2f}"
            if summary["center_lat"] is not None
            else f"  {summary['kept_tiles']} tiles kept"
        )
        print()

    # Write master image summary CSV
    if summaries:
        summary_csv = args.out_root / "image_summary.csv"
        fieldnames = [
            "parent_id", "center_lat", "center_lon",
            "width", "height", "total_windows", "kept_tiles",
            "dropped_tiles", "has_geolocation", "tiles_dir",
        ]
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(summaries)
        print(f"Image summary: {summary_csv}")

    # Write master tile metadata CSV (all tiles across all images)
    master_rows = []
    for s in summaries:
        image_dir = args.out_root / s["parent_id"]
        csv_path = image_dir / "tile_metadata.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    master_rows.append(row)

    if master_rows:
        master_csv = args.out_root / "master_tile_metadata.csv"
        with master_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(master_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(master_rows)
        kept_total = sum(s["kept_tiles"] for s in summaries)
        print(f"Master metadata: {master_csv} ({len(master_rows)} tiles, "
              f"{kept_total} kept)")

    print()
    print("Done.")
    print(f"Total images: {len(summaries)}")
    print(f"Total tiles kept: {sum(s['kept_tiles'] for s in summaries)}")


if __name__ == "__main__":
    main()
