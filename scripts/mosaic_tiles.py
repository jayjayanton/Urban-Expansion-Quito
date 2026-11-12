"""Spatially mosaic already-downloaded, aligned Sentinel-2 tiles.

Example: python scripts/mosaic_tiles.py --frequency monthly --year 2016 --period 01
"""
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[1]

def mosaic(frequency, year, period):
    source_dir = ROOT / "data" / "benchmarks" / frequency / str(year)
    paths = sorted(source_dir.glob(f"sentinel2_{year}_{period}_tile*.tif"))
    if len(paths) < 2:
        raise ValueError(f"Need at least two spatial tiles, found {len(paths)} in {source_dir}")
    sources = [rasterio.open(p) for p in paths]
    try:
        reference = sources[0]
        for src in sources[1:]:
            if src.crs != reference.crs or src.count != reference.count or src.res != reference.res:
                raise ValueError("Tiles differ in CRS, bands, or resolution")
            if src.descriptions != reference.descriptions:
                raise ValueError("Band descriptions differ")
        data, transform = merge(sources, method="first", nodata=-9999)
        output_dir = ROOT / "data" / "mosaics" / frequency / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"sentinel2_{year}_{period}_mosaic.tif"
        profile = reference.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform,
                       count=data.shape[0], compress="deflate", nodata=-9999)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data)
            for i, name in enumerate(reference.descriptions, 1):
                dst.set_band_description(i, name)
            dst.update_tags(year=str(year), period=period, frequency=frequency,
                            source_tiles=len(paths))
        print(f"Saved {output} from {len(paths)} tiles")
    finally:
        for src in sources:
            src.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", required=True, choices=("annual", "quarterly", "monthly"))
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--period", required=True, help="annual, Q1..Q4, or 01..12")
    args = parser.parse_args()
    mosaic(args.frequency, args.year, args.period)
