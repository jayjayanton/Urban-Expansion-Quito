"""Shared Sentinel Hub acquisition engine for annual, quarterly, and monthly benchmarks."""
import hashlib
import os
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.transform import from_bounds
from sentinelhub import BBox, CRS, DataCollection, MimeType, SHConfig, SentinelHubRequest

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)
EVALSCRIPT_PATH = ROOT / "scripts" / "median_composite.js"
EVALSCRIPT = EVALSCRIPT_PATH.read_text(encoding="utf-8")
EVALSCRIPT_HASH = hashlib.sha256(EVALSCRIPT.encode("utf-8")).hexdigest()
BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "valid_count")
RESOLUTION = 10
SIDE_METERS = 5000
GRID_COLS = 2
GRID_ROWS = 2
# Same original test location near Tumbaco, centered in WGS84.
CENTER_LON_LAT = (-78.41, -0.22)
# Full historical run (uncomment after validating the 2016 benchmark):
# YEARS = range(2016, 2026)

# Test run: all three download scripts process 2016 only.
YEARS = [2016]
PAUSE_SECONDS = 5

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"].strip()
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"].strip()
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)
collection = DataCollection.SENTINEL2_L2A.define_from(
    name="sentinel2_l2a_cdse", service_url=config.sh_base_url
)

# Construct an exact 5 km square in projected meters, not longitude degrees.
center = BBox(
    (CENTER_LON_LAT[0], CENTER_LON_LAT[1],
     CENTER_LON_LAT[0], CENTER_LON_LAT[1]), crs=CRS.WGS84
).transform(CRS.UTM_17S)
x, y = center.min_x, center.min_y
half = SIDE_METERS / 2
origin_x, origin_y = x - half, y - half

def tile_bbox(col, row):
    left = origin_x + col * SIDE_METERS
    bottom = origin_y + row * SIDE_METERS
    return BBox((left, bottom, left + SIDE_METERS, bottom + SIDE_METERS), crs=CRS.UTM_17S)

bbox = tile_bbox(0, 0)
width = height = SIDE_METERS // RESOLUTION
transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, width, height)
crs = rasterio.crs.CRS.from_epsg(32717)


def periods(year, frequency):
    if frequency == "annual":
        yield "annual", date(year, 1, 1), date(year, 12, 31)
    elif frequency == "quarterly":
        for q in range(4):
            month = q * 3 + 1
            start = date(year, month, 1)
            end = date(year, month + 3, 1) - timedelta(days=1) if q < 3 else date(year, 12, 31)
            yield f"Q{q+1}", start, end
    elif frequency == "monthly":
        for month in range(1, 13):
            start = date(year, month, 1)
            end = (date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)) - timedelta(days=1)
            yield f"{month:02d}", start, end
    else:
        raise ValueError("frequency must be annual, quarterly, or monthly")


def valid_existing(path, start, end, tile):
    if not path.is_file():
        return False
    try:
        with rasterio.open(path) as src:
            tags = src.tags()
            return (
                src.count == 7 and src.width == width and src.height == height
                and src.crs == crs and src.transform.almost_equals(tile_transform(tile))
                and src.descriptions == BANDS
                and tags.get("evalscript_sha256") == EVALSCRIPT_HASH
                and tags.get("start_date") == start.isoformat()
                and tags.get("end_date") == end.isoformat()
                and np.all(np.isfinite(src.read(7)))
            )
    except (rasterio.errors.RasterioIOError, ValueError):
        return False


def tile_transform(tile):
    return from_bounds(tile.min_x, tile.min_y, tile.max_x, tile.max_y, width, height)


def download_period(year, frequency, period, start, end, col, row):
    out_dir = ROOT / "data" / "benchmarks" / frequency / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_id = f"tile_r{row:02d}_c{col:02d}"
    tile = tile_bbox(col, row)
    out = out_dir / f"sentinel2_{year}_{period}_{tile_id}.tif"
    if valid_existing(out, start, end, tile):
        print(f"SKIPPED {out.name}")
        return "skipped"

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=collection,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order="leastCC",
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=tile, size=(width, height), config=config,
    )
    print(f"REQUEST {frequency} {year} {period}: {start} to {end}", flush=True)
    started = time.monotonic()
    data = request.get_data()[0]
    elapsed = time.monotonic() - started
    if data.shape != (height, width, len(BANDS)):
        raise ValueError(f"Unexpected response shape: {data.shape}")
    count = data[:, :, 6]
    if not np.all(np.isfinite(count)) or np.any(count < 0):
        raise ValueError("Invalid observation count")
    coverage = float(np.mean(count > 0) * 100)
    print(f"Coverage: {coverage:.2f}%; API seconds: {elapsed:.1f}")
    if not np.any(count > 0):
        return "no_data"

    temp = out.with_suffix(".partial.tif")
    try:
        with rasterio.open(
            temp, "w", driver="GTiff", width=width, height=height,
            count=len(BANDS), dtype="float32", crs=crs,
            transform=tile_transform(tile), compress="deflate", nodata=-9999,
        ) as dst:
            for i, band in enumerate(BANDS):
                dst.write(data[:, :, i], i + 1)
                dst.set_band_description(i + 1, band)
            dst.update_tags(
                year=str(year), frequency=frequency, period=period, tile_id=tile_id,
                start_date=start.isoformat(), end_date=end.isoformat(),
                evalscript_sha256=EVALSCRIPT_HASH,
                valid_coverage_pct=f"{coverage:.4f}",
                api_seconds=f"{elapsed:.2f}",
                note="Band 7 is valid observation count; zero means no observations.",
            )
        if not valid_existing(temp, start, end, tile):
            raise ValueError("Written GeoTIFF failed validation")
        temp.replace(out)
    finally:
        temp.unlink(missing_ok=True)
    print(f"SAVED {out}")
    return "downloaded"


def run(frequency):
    results = {"downloaded": 0, "skipped": 0, "no_data": 0, "failed": 0}
    for year in YEARS:
        for period, start, end in periods(year, frequency):
          for row in range(GRID_ROWS):
           for col in range(GRID_COLS):
            try:
                status = download_period(year, frequency, period, start, end, col, row)
                results[status] += 1
                if status == "downloaded":
                    time.sleep(PAUSE_SECONDS)
            except Exception as exc:
                results["failed"] += 1
                print(f"FAILED {year} {period} tile r{row} c{col}: {type(exc).__name__}: {exc}")
    print(f"{frequency.upper()} summary: {results}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("frequency", choices=("annual", "quarterly", "monthly"))
    run(parser.parse_args().frequency)
