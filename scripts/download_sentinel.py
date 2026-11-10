
import os
from pathlib import Path

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
load_dotenv(PROJECT_ROOT / ".env")

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]

# Copernicus Data Space endpoints
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

YEAR = 2025 
MONTH = 1
RESOLUTION = 10

# Small test area near Tumbaco.
# Coordinates: west, south, east, north.
# Expand later to cover all five valleys.
AOI = (-78.42, -0.23, -78.40, -0.21)

BANDS = [
    "B02", "B03", "B04",
    "B08", "B11", "B12",
    "valid_count",
]

# ----------------------------------
# 2. MONTHLY MEDIAN EVALSCRIPT
# ----------------------------------

evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "B02", "B03", "B04",
        "B08", "B11", "B12",
        "SCL", "dataMask"
      ],
      units: "REFLECTANCE"
    }],
    mosaicking: "ORBIT",
    output: {
      bands: 7,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(samples) {
  let values = [[], [], [], [], [], []];

  for (let s of samples) {
    // Valid scene classification classes:
    // 4 = vegetation
    // 5 = bare / non-vegetated
    // 6 = water
    // 7 = unclassified
    if (s.dataMask !== 1 ||
        ![4, 5, 6, 7].includes(s.SCL)) {
      continue;
    }

    let bands = [
      s.B02, s.B03, s.B04,
      s.B08, s.B11, s.B12
    ];

    for (let i = 0; i < 6; i++) {
      values[i].push(bands[i]);
    }
  }

  function median(arr) {
    if (arr.length === 0) return NaN;
    arr.sort((a, b) => a - b);
    let mid = Math.floor(arr.length / 2);

    return arr.length % 2
      ? arr[mid]
      : (arr[mid - 1] + arr[mid]) / 2;
  }

  let count = values[0].length;

  return [
    median(values[0]),
    median(values[1]),
    median(values[2]),
    median(values[3]),
    median(values[4]),
    median(values[5]),
    count
  ];
}
"""

# ----------------------------------
# 3. REQUEST SENTINEL-2 DATA
# ----------------------------------

bbox_wgs84 = BBox(AOI, crs=CRS.WGS84)
bbox = bbox_wgs84.transform(CRS.UTM_17S)

size = bbox_to_dimensions(
    bbox,
    resolution=RESOLUTION
)

start_date = f"{YEAR}-{MONTH:02d}-01"

if MONTH == 12:
    end_date = f"{YEAR + 1}-01-01"
else:
    end_date = f"{YEAR}-{MONTH + 1:02d}-01"

print(f"Requesting imagery: {start_date} to {end_date}")
print(f"Output dimensions: {size}")

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[
        SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start_date, end_date),
            mosaicking_order="leastCC",
        )
    ],
    responses=[
        SentinelHubRequest.output_response(
            "default", MimeType.TIFF
        )
    ],
    bbox=bbox,
    size=size,
    config=config,
)

data = request.get_data()[0]

# ----------------------------------
# 4. SAVE GEOTIFF
# ----------------------------------

output_dir = (
    PROJECT_ROOT / "data" / "processed" / str(YEAR)
)
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
    f"sentinel2_{YEAR}_{MONTH:02d}.tif"
)

transform = from_bounds(
    bbox.min_x, bbox.min_y,
    bbox.max_x, bbox.max_y,
    size[0], size[1]
)

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=data.shape[0],
    width=data.shape[1],
    count=data.shape[2],
    dtype="float32",
    crs="EPSG:32717",
    transform=transform,
    compress="deflate",
) as dst:

    for i, band_name in enumerate(BANDS):
        dst.write(data[:, :, i], i + 1)
        dst.set_band_description(i + 1, band_name)

print(f"Saved: {output_path}")
print("Valid observations:", np.nanmax(data[:, :, 6]))
