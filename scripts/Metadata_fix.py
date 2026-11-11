from pathlib import Path
import rasterio

data_dir = Path("data")

for tif in data_dir.rglob("sentinel2_*.tif"):
    with rasterio.open(tif, "r+") as src:
        src.nodata = -9999
    print("Updated:", tif)