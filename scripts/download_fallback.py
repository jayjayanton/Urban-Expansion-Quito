"""Fallback CDSE STAC + S3 Sentinel-2 L2A annual tiled composites (no Sentinel Hub PUs).

Install: pip install requests boto3 rasterio numpy python-dotenv
Credentials: CDSE_S3_ACCESS_KEY, CDSE_S3_SECRET_KEY in .env
Run: python scripts/download_fallback.py --year 2025 --dry-run
Then: python scripts/download_fallback.py --start-year 2016 --end-year 2025
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np
import rasterio
import requests
from botocore.config import Config
from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

LOG = logging.getLogger("cdse_annual")
STAC_SEARCH = "https://stac.dataspace.copernicus.eu/v1/search"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
SCL_ALLOWED = (4, 5, 6, 7)
NODATA = -9999.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def args_parser():
    """Read CLI options. Paths default to the repository's data/ folder.

    --max-scenes limits input acquisitions for testing, not output tiles.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, help="Run a single year")
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--bbox", nargs=4, type=float, default=[-78.47, -0.248, -78.32, -0.11],
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--resolution", type=float, default=10)
    p.add_argument("--tile-pixels", type=int, default=512)
    p.add_argument("--max-cloud", type=float, default=100)
    p.add_argument("--max-scenes", type=int, default=None, help="Select up to N scenes distributed across the year; prefer lower cloud cover")
    p.add_argument("--output", type=Path, default=DATA_DIR)
    p.add_argument("--cache", type=Path, default=DATA_DIR / ".cdse_cache")

    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Catalogue only; no S3 credentials needed")
    return p.parse_args()


# STAGE 1: SEARCH THE COPERNICUS STAC CATALOGUE
def request_json(session, method, url, **kwargs):
    """Request catalogue metadata; retry transient HTTP failures."""
    for attempt in range(5):
        try:
            r = session.request(method, url, timeout=90, **kwargs)
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(90, 3 * (2 ** attempt))
                LOG.warning("STAC HTTP %s; waiting %ss", r.status_code, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(3 * (2 ** attempt))
    raise RuntimeError("STAC retries exhausted")


def search_items(session, year, bbox, max_cloud):
    """Find all L2A acquisitions intersecting our AOI in one year.

    STAC returns metadata and band URLs, not the actual raster pixels.
    Catalogue results are paginated, so follow 'next' links.
    """
    payload = {"collections": ["sentinel-2-l2a"],
               "bbox": list(bbox),
               "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
               "limit": 100,
               "query": {"eo:cloud_cover": {"lte": max_cloud}}}
    url, method, body = STAC_SEARCH, "POST", payload
    seen, items = set(), []
    while url:
        result = request_json(session, method, url, json=body if method == "POST" else None)
        for item in result.get("features", []):
            if item["id"] not in seen:
                items.append(item)
                seen.add(item["id"])
        nxt = next((l for l in result.get("links", []) if l.get("rel") == "next"), None)
        if nxt:
            url = nxt["href"]
            method = nxt.get("method", "GET").upper()
            body = nxt.get("body") if method == "POST" else None
            if method == "POST" and nxt.get("merge") and isinstance(body, dict):
                body = {**payload, **body}
        else:
            url = None
    items.sort(key=lambda x: (x.get("properties", {}).get("datetime", ""), x["id"]))
    return items



def select_distributed_scenes(items, year, limit):
    """Select N scenes distributed across the calendar year, not just January.

    Divide the year into N equal time windows. Within each window choose the
    acquisition with the lowest catalogue cloud-cover percentage. If a window
    has no acquisitions, choose the nearest unused scene to its center.

    Cloud cover describes the full source scene, not necessarily the Quito AOI.
    If no limit is given, preserve all catalogue results.
    """
    if limit is None or len(items) <= limit:
        return items
    if limit < 1:
        raise ValueError("--max-scenes must be a positive integer")

    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    duration = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) - start).total_seconds()

    def position(item):
        timestamp = item.get("properties", {}).get("datetime")
        if not timestamp:
            return None
        try:
            return (datetime.fromisoformat(timestamp.replace("Z", "+00:00")) - start).total_seconds() / duration
        except ValueError:
            return None

    def cloud(item):
        value = item.get("properties", {}).get("eo:cloud_cover")
        return float(value) if value is not None else float("inf")

    windows = [[] for _ in range(limit)]
    for item in items:
        pos = position(item)
        if pos is not None:
            index = min(limit - 1, max(0, int(pos * limit)))
            windows[index].append(item)

    chosen = []
    used = set()
    empty_centers = []
    for index, candidates in enumerate(windows):
        center = (index + 0.5) / limit
        if not candidates:
            empty_centers.append(center)
            continue
        best = min(candidates, key=lambda item: (
            cloud(item), abs(position(item) - center), item["id"]))
        chosen.append(best)
        used.add(best["id"])

    remaining = [item for item in items if item["id"] not in used]
    for center in empty_centers:
        if not remaining:
            break
        best = min(remaining, key=lambda item: (
            abs((position(item) if position(item) is not None else -10) - center),
            cloud(item), item["id"]))
        chosen.append(best)
        remaining.remove(best)

    return sorted(chosen, key=lambda item: (
        item.get("properties", {}).get("datetime") or "", item["id"]))


def asset_for(item, band):
    """Find a band URL in one scene's STAC metadata.

    Visible/NIR bands are 10 m; B11, B12 and SCL are 20 m.
    """
    assets = item.get("assets", {})
    preferred = (f"{band}_10m", band) if band in ("B02", "B03", "B04", "B08") else (f"{band}_20m", band)
    if band == "SCL":
        preferred = ("SCL_20m", "SCL", "SCL_60m")
    for key in preferred:
        if key in assets:
            return assets[key]["href"]
    candidates = [(k, v["href"]) for k, v in assets.items()
                  if re.match(rf"^{band}(?:_|$)", k, re.IGNORECASE) and v.get("href")]
    if candidates:
        return candidates[0][1]
    raise KeyError(f"Missing {band} in {item['id']}; available keys: {list(assets)}")


# STAGE 2: DOWNLOAD JP2 SOURCE BANDS THROUGH S3
def s3_client():
    """Authenticate to Copernicus object storage using .env credentials."""
    access = os.getenv("CDSE_S3_ACCESS_KEY")
    secret = os.getenv("CDSE_S3_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY in .env (S3 credentials, not Sentinel Hub OAuth).")
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=access,
                        aws_secret_access_key=secret,
                        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}))


def local_asset(s3, href, cache):
    """Download one JP2 band, or reuse it if already cached.

    The same scene can cover many output tiles. Caching prevents repeated
    network downloads. Hashing the full S3 URL gives a short, unique local
    filename (important for Windows path limits). A .partial file prevents
    an interrupted transfer being mistaken for a complete source image.
    """
    u = urlparse(href)
    if u.scheme != "s3":
        raise ValueError(f"Expected CDSE STAC S3 asset, received: {href}")
    digest = hashlib.sha256(href.encode("utf-8")).hexdigest()[:24]
    target = cache / f"{digest}_{Path(u.path).name}"

    if target.is_file() and target.stat().st_size > 0:
        LOG.info("CACHE HIT: %s", target.name)
        return target
    
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    LOG.info("Downloading %s", href)
    try:
        s3.download_file(u.netloc, u.path.lstrip("/"), str(partial))
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


# STAGE 3: DEFINE THE COMMON SPATIAL GRID
def grid_for_bbox(bbox, resolution):
    """Convert geographic AOI to UTM 17S and snap edges to 10 m pixels.

    All years must use the same grid for pixel-by-pixel change detection.
    """
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", "EPSG:32717", *bbox, densify_pts=21)
    xmin = math.floor(xmin / resolution) * resolution
    ymin = math.floor(ymin / resolution) * resolution
    xmax = math.ceil(xmax / resolution) * resolution
    ymax = math.ceil(ymax / resolution) * resolution
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))
    return from_origin(xmin, ymax, resolution, resolution), width, height


# STAGE 4: ALIGN BANDS, MASK CLOUDS, AND NORMALIZE REFLECTANCE
def scene_tile(s3, item, cache, tile_transform, width, height):
    """Process one acquisition for one output tile.

    Returns a NumPy array shaped (6 bands, tile rows, tile columns).
    SCL is resampled with nearest-neighbor because it contains classes;
    spectral bands are aligned to the common output grid. Disallowed SCL
    pixels become NaN, which the annual median calculation later ignores.

    NOTE: The baseline-dependent reflectance offset is a simplified
    correction and should be validated against source product metadata.
    """
    paths = {band: local_asset(s3, asset_for(item, band), cache) for band in (*BANDS, "SCL")}
    scl = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(paths["SCL"]) as src:
        reproject(source=rasterio.band(src, 1), destination=scl, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=tile_transform,
                  dst_crs="EPSG:32717", dst_nodata=0, resampling=Resampling.nearest)
    # SCL classes: 4 vegetation, 5 bare/nonvegetated, 6 water, 7 unclassified.
    # Exclude clouds and shadows from the temporal median.
    valid = np.isin(scl, SCL_ALLOWED)
    if not valid.any():
        return None
    arrays = []
    for band in BANDS:
        dest = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(paths[band]) as src:
            raw = np.zeros((height, width), dtype=np.float32)
            reproject(source=rasterio.band(src, 1), destination=raw,
                      src_transform=src.transform, src_crs=src.crs,
                      src_nodata=src.nodata or 0,
                      dst_transform=tile_transform, dst_crs="EPSG:32717",
                      dst_nodata=0, resampling=Resampling.nearest if band in ("B11", "B12") else Resampling.bilinear)
            match = re.search(r"_N(\d{4})_", item["id"])
            baseline = int(match.group(1)) if match else None
            if baseline is None:
                baseline_property = item.get("properties", {}).get("s2:processing_baseline")
                if baseline_property is not None:
                    baseline = int(round(float(baseline_property) * 100))
            if baseline is None:
                raise ValueError(f"Cannot determine processing baseline for {item['id']}")
            boa_offset = -1000 if baseline >= 400 else 0
            dest[:] = (raw + boa_offset) / 10000.0
            dest[(raw == 0) | ~valid] = np.nan
        arrays.append(dest)
    shared = valid & np.logical_and.reduce([np.isfinite(a) for a in arrays])
    if not shared.any():
        return None
    for a in arrays:
        a[~shared] = np.nan
    return np.stack(arrays, axis=0)


# STAGE 5: SAVE ANNUAL COMPOSITES AS GEOTIFFS
def write_tile(path, composite, counts, transform, year, scenes):
    """Write six median-reflectance bands plus valid_count as band seven.

    Write to a temporary TIFF before renaming, avoiding incomplete outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.tif")
    profile = dict(driver="GTiff", height=counts.shape[0], width=counts.shape[1],
                   count=7, dtype="float32", crs="EPSG:32717", transform=transform,
                   compress="deflate", predictor=3, tiled=True, nodata=NODATA)
    with rasterio.open(partial, "w", **profile) as dst:
        for i, band in enumerate(BANDS, start=1):
            dst.write(np.where(np.isfinite(composite[i - 1]), composite[i - 1], NODATA).astype("float32"), i)
            dst.set_band_description(i, band)
        dst.write(counts.astype("float32"), 7)
        dst.set_band_description(7, "valid_count")
        dst.update_tags(year=year, method="per-band median of cloud-masked L2A scenes",
                        scenes_considered=scenes, scl_allowed=",".join(map(str, SCL_ALLOWED)),
                        warning="valid_count=0 is a valid zero; nodata=-9999 applies to all bands")
    partial.replace(path)


# STAGE 6: ORCHESTRATE THE YEAR / ROW / COLUMN LOOPS
def main():
    """Run catalogue search, acquisition, tile compositing, and export."""
    load_dotenv()
    args = args_parser()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.tile_pixels < 32 or args.resolution <= 0:
        raise ValueError("Invalid tile size or resolution")
    years = [args.year] if args.year else range(args.start_year, args.end_year + 1)
    transform, width, height = grid_for_bbox(args.bbox, args.resolution)
    rows = math.ceil(height / args.tile_pixels)
    cols = math.ceil(width / args.tile_pixels)
    LOG.info("AOI EPSG:32717: %s x %s pixels, %s x %s tiles", width, height, rows, cols)
    session = requests.Session()
    s3 = None if args.dry_run else s3_client()
    for year in years:
        LOG.info("Searching Sentinel-2 L2A catalogue for %s", year)
        items = search_items(session, year, args.bbox, args.max_cloud)
        # Sample across the WHOLE year when a scene limit is supplied.
        # Existing test TIFFs must be deleted before changing scene selection:
        # the current resume check validates structure, not provenance.
        total_candidates = len(items)
        items = select_distributed_scenes(items, year, args.max_scenes)
        LOG.info("%s: selected %s of %s catalogue scenes", year, len(items), total_candidates)
        for item in items:
            LOG.info("  selected %s (cloud cover: %s%%)",
                     item.get("properties", {}).get("datetime"),
                     item.get("properties", {}).get("eo:cloud_cover"))
        if not items:
            LOG.warning("No L2A products found for %s; early 2016 L2A coverage may be limited", year)
            continue
        year_dir = args.output / str(year) / "tiles"
        metadata_dir = args.output / "metadata" / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "catalogue_manifest.json").write_text(json.dumps([
            {"id": x["id"], "datetime": x.get("properties", {}).get("datetime"),
             "cloud": x.get("properties", {}).get("eo:cloud_cover")} for x in items], indent=2))
        if args.dry_run:
            continue
        # Each (row, column) identifies one geographic output tile.
        # Source JP2 files are cached and reused for the remaining tiles.
        for r in range(rows):
            for c in range(cols):
                window = Window(c * args.tile_pixels, r * args.tile_pixels,
                                min(args.tile_pixels, width - c * args.tile_pixels),
                                min(args.tile_pixels, height - r * args.tile_pixels))
                tile_transform = rasterio.windows.transform(window, transform)
                th, tw = int(window.height), int(window.width)
                out = year_dir / f"sentinel2_{year}_r{r}_c{c}.tif"
                if out.exists():
                    try:
                        with rasterio.open(out) as ds:
                            if ds.count == 7 and ds.width == tw and ds.height == th and ds.crs.to_epsg() == 32717:
                                LOG.info("SKIP %s", out)
                                continue
                    except rasterio.RasterioError:
                        pass

                tile_number = r * cols + c + 1
                total_tiles = rows * cols

                LOG.info(
                    "START TILE %d/%d | Year %s | r%d c%d",
                    tile_number, total_tiles, year, r, c
                )

                observations = []
                failures = []

                for scene_idx, item in enumerate(items, start=1):
                    LOG.info(
                        "  TILE r%d c%d | Scene %d/%d | %s",
                        r, c,
                        scene_idx, len(items),
                        item.get("properties", {}).get("datetime")
                    )
                    try:
                        data = scene_tile(s3, item, args.cache, tile_transform, tw, th)
                        if data is not None:
                            observations.append(data)
                    except Exception as exc:
                        failures.append((item["id"], str(exc)))
                        LOG.error("Scene failed %s: %s", item["id"], exc)
                if failures:
                    (metadata_dir / f"failures_r{r}_c{c}.json").write_text(json.dumps(failures, indent=2))
                    LOG.error("Not writing incomplete tile %s (%s scene errors)", out, len(failures))
                    continue
                if not observations:
                    LOG.warning("No valid observations for tile %s", out)
                    continue
                # Shape: (scenes, six bands, tile height, tile width).
                # axis=0 is time; the median collapses acquisitions, not space.
                stack = np.stack(observations, axis=0)
                # Count valid observations for each pixel (output band 7).
                counts = np.isfinite(stack[:, 0]).sum(axis=0).astype(np.uint16)
                with np.errstate(all="ignore"):
                    composite = np.nanmedian(stack, axis=0)
                write_tile(out, composite, counts, tile_transform, year, len(items))
                LOG.info(
                    "FINISHED TILE %d/%d | r%d c%d | max valid_count=%d",
                    tile_number, total_tiles,
                    r, c, int(counts.max())
                )

                LOG.info("WROTE %s", out)

                del stack, composite, observations
        # Remove temporary source JP2s after the year unless --keep-cache.
        if not args.keep_cache:
            shutil.rmtree(args.cache, ignore_errors=True)
    LOG.info("Finished")


if __name__ == "__main__":
    main()
