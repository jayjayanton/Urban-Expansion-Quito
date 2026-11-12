
"""Mosaic Sentinel-2 spatial tiles into a single GeoTIFF.

Annual workflow:
    python scripts/mosaicing/mosaic_tiles.py --frequency annual --year 2025

Legacy monthly workflow:
    python scripts/mosaicing/mosaic_tiles.py --frequency monthly --year 2016 --period 01

Legacy quarterly workflow:
    python scripts/mosaicing/mosaic_tiles.py --frequency quarterly --year 2016 --period Q1
"""

import argparse
import logging
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("sentinel_mosaic")
NODATA = -9999.0

EXPECTED_BANDS = (
    "B02", "B03", "B04", "B08",
    "B11", "B12", "valid_count"
)


def mosaic(frequency, year, period=None, expected_tiles=None):

    # --------------------------------------------------
    # 1. Determine source directory and filename pattern
    # --------------------------------------------------

    if frequency == "annual":
        source_dir = ROOT / "data" / str(year) / "tiles"
        pattern = f"sentinel2_{year}_r*_c*.tif"

        output_dir = ROOT / "data" / "mosaics" / "annual"
        output_name = f"sentinel2_{year}_annual.tif"

    else:
        if period is None:
            raise ValueError(
                "--period is required for monthly/quarterly mosaics"
            )

        source_dir = (
            ROOT / "data" / "benchmarks" /
            frequency / str(year)
        )

        pattern = f"sentinel2_{year}_{period}_tile*.tif"

        output_dir = (
            ROOT / "data" / "mosaics" /
            frequency / str(year)
        )

        output_name = f"sentinel2_{year}_{period}_mosaic.tif"

    paths = sorted(source_dir.glob(pattern))

    if len(paths) < 2:
        raise ValueError(
            f"Need at least two tiles; found {len(paths)} "
            f"in {source_dir}"
        )

    LOG.info("Found %d tiles in %s", len(paths), source_dir)

    # --------------------------------------------------
    # 2. Validate expected annual tile coverage
    # --------------------------------------------------

    if frequency == "annual":

        tile_pattern = re.compile(
            rf"sentinel2_{year}_r(\d+)_c(\d+)\.tif$"
        )

        coordinates = set()

        for path in paths:
            match = tile_pattern.fullmatch(path.name)

            if match is None:
                raise ValueError(f"Invalid tile name: {path.name}")

            coordinates.add(
                (int(match.group(1)), int(match.group(2)))
            )

        if expected_tiles is not None:
            if len(coordinates) != expected_tiles:
                raise ValueError(
                    f"Expected {expected_tiles} tiles, "
                    f"found {len(coordinates)}"
                )

        max_row = max(r for r, c in coordinates)
        max_col = max(c for r, c in coordinates)

        expected_coordinates = {
            (r, c)
            for r in range(max_row + 1)
            for c in range(max_col + 1)
        }

        missing = expected_coordinates - coordinates

        if missing:
            raise ValueError(
                f"Missing spatial tiles: {sorted(missing)}"
            )

        if expected_tiles is not None:
            if len(expected_coordinates) != expected_tiles:
                raise ValueError(
                    "Tile grid dimensions do not match "
                    f"--expected-tiles={expected_tiles}"
                )

    # --------------------------------------------------
    # 3. Open and validate rasters
    # --------------------------------------------------

    sources = [rasterio.open(p) for p in paths]

    try:
        reference = sources[0]

        for src in sources:

            if src.crs != reference.crs:
                raise ValueError("CRS mismatch")

            if src.count != reference.count:
                raise ValueError("Band count mismatch")

            if src.res != reference.res:
                raise ValueError("Resolution mismatch")

            if src.descriptions != reference.descriptions:
                raise ValueError("Band descriptions mismatch")

            if src.nodata != reference.nodata:
                raise ValueError("Nodata mismatch")

        if frequency == "annual":

            if reference.crs.to_epsg() != 32717:
                raise ValueError("Expected EPSG:32717")

            if reference.count != 7:
                raise ValueError(
                    "Annual imagery must contain 7 bands"
                )

            if reference.descriptions != EXPECTED_BANDS:
                raise ValueError(
                    "Unexpected annual band order: "
                    f"{reference.descriptions}"
                )

        # --------------------------------------------------
        # 4. Mosaic tiles
        # --------------------------------------------------

        LOG.info("Mosaicking %d tiles...", len(sources))

        data, transform = merge(
            sources,
            method="first",
            nodata=NODATA
        )

        # --------------------------------------------------
        # 5. Save final mosaic
        # --------------------------------------------------

        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / output_name
        temporary = output.with_name(
            output.stem + ".partial.tif"
        )

        profile = reference.profile.copy()

        # Source strips or block sizes may be incompatible with tiled GeoTIFFs.
        # Explicit 256x256 blocks satisfy GDAL's multiple-of-16 requirement.
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("rows_per_strip", None)

        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            count=data.shape[0],
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            nodata=NODATA
        )

        with rasterio.open(temporary, "w", **profile) as dst:

            dst.write(data)

            for i, name in enumerate(reference.descriptions, 1):
                dst.set_band_description(i, name)

            dst.update_tags(
                year=str(year),
                period=period or "annual",
                frequency=frequency,
                source_tiles=len(paths)
            )

        temporary.replace(output)

        LOG.info("Saved mosaic: %s", output)
        LOG.info(
            "Mosaic dimensions: %d x %d",
            data.shape[2],
            data.shape[1]
        )

        # --------------------------------------------------
        # 6. Report valid observation coverage
        # --------------------------------------------------

        if frequency == "annual":

            counts = data[6]

            valid_domain = counts != NODATA
            valid_values = counts[valid_domain]

            if valid_values.size:

                zero_pct = (
                    np.count_nonzero(valid_values == 0)
                    / valid_values.size * 100
                )

                three_pct = (
                    np.count_nonzero(valid_values >= 3)
                    / valid_values.size * 100
                )

                LOG.info(
                    "Pixels with zero observations: %.2f%%",
                    zero_pct
                )

                LOG.info(
                    "Pixels with >=3 observations: %.2f%%",
                    three_pct
                )

                LOG.info(
                    "Mean valid observations: %.2f",
                    valid_values.mean()
                )

                LOG.info(
                    "Maximum valid observations: %.0f",
                    valid_values.max()
                )

        return output

    finally:
        for src in sources:
            src.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frequency",
        required=True,
        choices=("annual", "quarterly", "monthly")
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025
    )
    parser.add_argument(
        "--period",
        default=None
    )
    parser.add_argument(
        "--expected-tiles",
        type=int,
        default=None
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    mosaic(
        frequency=args.frequency,
        year=args.year,
        period=args.period,
        expected_tiles=args.expected_tiles
    )
