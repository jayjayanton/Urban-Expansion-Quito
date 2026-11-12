
import os
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import rasterio
from dotenv import load_dotenv
from rasterio.transform import from_bounds
from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions,
)

# ----------------------------------
# 1. PROJECT CONFIGURATION
# ----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=True)

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"].strip()
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"].strip()

# Copernicus Data Space endpoints
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

# Sentinel-2 L2A collection for Copernicus Data Space
SENTINEL2_CDSE = DataCollection.SENTINEL2_L2A.define_from(
    name="sentinel2_l2a_cdse",
    service_url=config.sh_base_url,
)

# Temporal configuration
YEARS = range(2016, 2026)
# MONTHS = range(1, 13)

# Spatial configuration
RESOLUTION = 10

# Small test area near Tumbaco
# Reduced study area: Nayón, Cumbayá, Tumbaco, Puembo
# West, South, East, North (WGS84)
AOI = (-78.47, -0.26, -78.32, -0.11)

# Original AOI near Tumbaco
# West, South, East, North (WGS84)
# AOI = (-78.42, -0.23, -78.40, -0.21)

BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B11",
    "B12",
    "valid_count",
]

# ----------------------------------
# 2. LOAD EVALSCRIPT
# ----------------------------------

evalscript_path = (
    PROJECT_ROOT / "scripts" / "median_composite.js"
)

if not evalscript_path.is_file():
    raise FileNotFoundError(
        f"Evalscript not found: {evalscript_path}"
    )

evalscript = evalscript_path.read_text(encoding="utf-8")

# ----------------------------------
# 3. DEFINE STUDY AREA
# ----------------------------------

bbox_wgs84 = BBox(AOI, crs=CRS.WGS84)
bbox = bbox_wgs84.transform(CRS.UTM_17S)

size = bbox_to_dimensions(
    bbox,
    resolution=RESOLUTION,
)

transform = from_bounds(
    bbox.min_x,
    bbox.min_y,
    bbox.max_x,
    bbox.max_y,
    size[0],
    size[1],
)

OUTPUT_CRS = rasterio.crs.CRS.from_epsg(32717)

print("Study area:", AOI)
print("Resolution:", RESOLUTION, "meters")
print("Raster dimensions:", size)

# ----------------------------------
# 4. VALIDATE EXISTING GEOTIFF
# ----------------------------------

def is_valid_geotiff(path):
    """
    Check whether an existing GeoTIFF matches
    the current spatial configuration.
    """

    if not path.is_file():
        return False

    try:
        with rasterio.open(path) as src:
            valid = (
                src.count == len(BANDS)
                and src.width == size[0]
                and src.height == size[1]
                and src.crs == OUTPUT_CRS
                and src.dtypes == ("float32",) * len(BANDS)
                and src.transform.almost_equals(transform)
                and tuple(src.descriptions) == tuple(BANDS)
            )

            if not valid:
                return False

            count_band = src.read(7)

            if not np.all(np.isfinite(count_band)):
                return False

            if np.any(count_band < 0):
                return False

            return True

    except (rasterio.errors.RasterioIOError, ValueError):
        return False

# ----------------------------------
# 5. DOWNLOAD ONE YEAR
# ----------------------------------

def download_year(year):

    output_dir = (
        PROJECT_ROOT / "data" /  str(year)
    )

    output_path = output_dir / (
        f"sentinel2_{year}_AOI.tif"
    )

    # Skip existing valid GeoTIFFs
    if is_valid_geotiff(output_path):
        print(
            f"SKIPPED: {year} "
            "(existing GeoTIFF is valid)"
        )
        return "skipped"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define monthly date interval
    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)


    print(f"\nProcessing: {year}")
    print(f"Date range: {start_date} to {end_date}")

    # Sentinel Hub Process API request
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=SENTINEL2_CDSE,
                time_interval=(
                    start_date.isoformat(),
                    end_date.isoformat(),
                ),
                mosaicking_order="leastCC",
            )
        ],
        responses=[
            SentinelHubRequest.output_response(
                "default",
                MimeType.TIFF,
            )
        ],
        bbox=bbox,
        size=size,
        config=config,
    )

    print("Processing URL:", request.download_list[0].url)

    # Download imagery
    print("Output dimensions:", size)
    print("Total pixels:", size[0] * size[1])
    print("Approximate area (km²):", size[0] * size[1] * 100 / 1_000_000)
    data = request.get_data()[0]

    # Verify response dimensions
    if (
        data.ndim != 3
        or data.shape[0] != size[1]
        or data.shape[1] != size[0]
        or data.shape[2] != len(BANDS)
    ):
        raise ValueError(
            f"Unexpected raster dimensions: {data.shape}"
        )

    # Assess valid observations
    valid_count = data[:, :, 6]

    if not np.all(np.isfinite(valid_count)):
        raise ValueError(
            "Observation-count band contains non-finite values."
        )

    valid_pixels = valid_count > 0

    print("Maximum valid observations:", np.max(valid_count))
    print("Pixels with valid observations:", np.sum(valid_pixels))
    print(
        "Valid coverage:",
        f"{100 * np.mean(valid_pixels):.2f}%"
    )

    # Do not save completely empty composites
    if not np.any(valid_pixels):
        print(
            f"NO DATA: {year}"
            "(no valid observations; file not saved)"
        )
        return "no_data"

    # Temporary file for safe export
    temp_path = output_path.with_suffix(".partial.tif")

    try:
        with rasterio.open(
            temp_path,
            "w",
            driver="GTiff",
            height=data.shape[0],
            width=data.shape[1],
            count=data.shape[2],
            dtype="float32",
            crs=OUTPUT_CRS,
            transform=transform,
            nodata=-9999.0,
            compress="deflate",
        ) as dst:

            for i, band_name in enumerate(BANDS):
                dst.write(data[:, :, i], i + 1)
                dst.set_band_description(i + 1, band_name)

            dst.update_tags(
                year=year,
                resolution_m=RESOLUTION,
                source="Sentinel-2 L2A CDSE",
                composite="annual median",
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )

        # Validate the completed export
        if not is_valid_geotiff(temp_path):
            raise ValueError(
                "Exported GeoTIFF failed validation."
            )

        temp_path.replace(output_path)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(f"SAVED: {output_path}")

    return "downloaded"

# ----------------------------------
# 6. PROCESS ALL YEARS
# ----------------------------------

results = {
    "downloaded": 0,
    "skipped": 0,
    "no_data": 0,
    "failed": 0,
}

for year in YEARS:
    try:
        status = download_year(year)
        results[status] += 1

    except Exception as exc:
        results["failed"] += 1
        print(f"\nFAILED: {year}")
        print(f"{type(exc).__name__}: {exc}")

# ----------------------------------
# 7. PROCESSING SUMMARY
# ----------------------------------

print("\n" + "=" * 40)
print("SENTINEL-2 PROCESSING SUMMARY")
print("=" * 40)

for status, count in results.items():
    print(f"{status.capitalize():12}: {count}")

print("=" * 40)