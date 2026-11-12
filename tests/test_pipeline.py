"""Offline regression tests: no Copernicus requests or real credentials."""
import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "data_acquisition"))
sys.path.insert(0, str(ROOT / "scripts" / "mosaicing"))
os.environ.setdefault("SH_CLIENT_ID", "offline-test-client")
os.environ.setdefault("SH_CLIENT_SECRET", "offline-test-secret")

import acquisition_common as ac
import mosaic_tiles as mt


class AcquisitionTests(unittest.TestCase):
    def test_date_intervals_2016(self):
        annual = list(ac.periods(2016, "annual"))
        quarterly = list(ac.periods(2016, "quarterly"))
        monthly = list(ac.periods(2016, "monthly"))
        self.assertEqual(annual, [("annual", date(2016, 1, 1), date(2016, 12, 31))])
        self.assertEqual(len(quarterly), 4)
        self.assertEqual(quarterly[0], ("Q1", date(2016, 1, 1), date(2016, 3, 31)))
        self.assertEqual(quarterly[-1], ("Q4", date(2016, 10, 1), date(2016, 12, 31)))
        self.assertEqual(len(monthly), 12)
        self.assertEqual(monthly[1], ("02", date(2016, 2, 1), date(2016, 2, 29)))
        self.assertEqual(monthly[-1], ("12", date(2016, 12, 1), date(2016, 12, 31)))

    def test_four_tiles_are_adjacent_and_500_pixels(self):
        self.assertEqual((ac.width, ac.height), (500, 500))
        self.assertEqual(ac.GRID_ROWS * ac.GRID_COLS, 4)
        a = ac.tile_bbox(0, 0)
        b = ac.tile_bbox(1, 0)
        c = ac.tile_bbox(0, 1)
        self.assertAlmostEqual(a.max_x, b.min_x)
        self.assertAlmostEqual(a.max_y, c.min_y)
        self.assertAlmostEqual(a.max_x - a.min_x, 5000)
        self.assertAlmostEqual(a.max_y - a.min_y, 5000)
        self.assertEqual(ac.tile_transform(a).a, 10)

    def test_mosaic_four_synthetic_quarterly_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "data" / "benchmarks" / "quarterly" / "2016"
            source.mkdir(parents=True)
            for row in range(2):
                for col in range(2):
                    tile = ac.tile_bbox(col, row)
                    value = float(1 + row * 2 + col)
                    path = source / f"sentinel2_2016_Q1_tile_r{row:02d}_c{col:02d}.tif"
                    with rasterio.open(
                        path, "w", driver="GTiff", height=500, width=500,
                        count=7, dtype="float32", crs="EPSG:32717",
                        transform=ac.tile_transform(tile), nodata=-9999
                    ) as dst:
                        for band in range(1, 8):
                            dst.write(np.full((500, 500), value, dtype="float32"), band)
                            dst.set_band_description(band, ac.BANDS[band - 1])
            with patch.object(mt, "ROOT", root):
                mt.mosaic("quarterly", 2016, "Q1")
            result = root / "data" / "mosaics" / "quarterly" / "2016" / "sentinel2_2016_Q1_mosaic.tif"
            with rasterio.open(result) as ds:
                self.assertEqual((ds.width, ds.height, ds.count), (1000, 1000, 7))
                self.assertEqual(ds.crs.to_epsg(), 32717)
                self.assertEqual(ds.read(1, window=((0, 1), (0, 1)))[0, 0], 3)
                self.assertEqual(ds.read(1, window=((999, 1000), (999, 1000)))[0, 0], 2)

    def test_fallback_annual_mosaic_twelve_tiles(self):
        """Verify the current fallback r{row}_c{col} convention."""
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "data" / "2025" / "tiles"
            source.mkdir(parents=True)

            for row in range(3):
                for col in range(4):
                    value = float(1 + row * 4 + col)
                    path = source / f"sentinel2_2025_r{row}_c{col}.tif"
                    transform = from_origin(
                        500000 + col * 320,
                        9980000 - row * 320,
                        10, 10
                    )
                    with rasterio.open(
                        path, "w", driver="GTiff", height=32, width=32,
                        count=7, dtype="float32", crs="EPSG:32717",
                        transform=transform, nodata=-9999
                    ) as dst:
                        for band in range(1, 8):
                            dst.write(np.full((32, 32), value, dtype="float32"), band)
                            dst.set_band_description(band, ac.BANDS[band - 1])

            with patch.object(mt, "ROOT", root):
                mt.mosaic("annual", 2025, expected_tiles=12)

            output = root / "data" / "mosaics" / "annual" / "sentinel2_2025_annual.tif"
            with rasterio.open(output) as ds:
                self.assertEqual((ds.width, ds.height, ds.count), (128, 96, 7))
                self.assertEqual(ds.crs.to_epsg(), 32717)
                self.assertEqual(ds.read(1)[0, 0], 1)
                self.assertEqual(ds.read(1)[-1, -1], 12)

    def test_invalid_frequency_rejected(self):
        with self.assertRaises(ValueError):
            list(ac.periods(2016, "weekly"))

    def test_non_leap_february(self):
        monthly = list(ac.periods(2017, "monthly"))
        self.assertEqual(monthly[1][2], date(2017, 2, 28))

    def test_tile_transforms_do_not_overlap(self):
        a = ac.tile_bbox(0, 0)
        b = ac.tile_bbox(1, 0)
        self.assertAlmostEqual(ac.tile_transform(a).c + 500 * 10,
                               ac.tile_transform(b).c)
        self.assertAlmostEqual(ac.tile_transform(a).e, -10)

    def test_missing_file_is_not_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(ac.valid_existing(
                Path(folder) / "missing.tif",
                date(2016, 1, 1), date(2016, 12, 31),
                ac.tile_bbox(0, 0)))

    def test_mosaic_rejects_insufficient_tiles(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(mt, "ROOT", Path(folder)):
                with self.assertRaises(ValueError):
                    mt.mosaic("quarterly", 2016, "Q1")


if __name__ == "__main__":
    unittest.main()
