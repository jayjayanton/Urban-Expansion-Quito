"""Run a complete, credential-free experiment on invented imagery and known labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

from .data import BANDS, IGNORE
from .workflow import evaluate, fit, predict


def make_fixture(folder):
    """Small, deliberately learnable transition with no relation to Quito land cover."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    h, w = 81, 163
    y, x = np.indices((h, w))
    gain = ((x % 25) < 10) & ((y % 27) < 15)
    stable_built = ((x % 25) >= 14) & ((y % 21) < 8)
    rng = np.random.default_rng(7)
    vegetation = np.array([0.04, 0.08, 0.06, 0.36, 0.16, 0.09], "float32")[
        :, None, None
    ]
    built = np.array([0.15, 0.17, 0.20, 0.20, 0.27, 0.24], "float32")[:, None, None]
    profile = {
        "driver": "GTiff",
        "width": w,
        "height": h,
        "crs": "EPSG:32717",
        "transform": from_origin(500000, 9900000, 10, 10),
    }
    for year, occupied in ((2016, stable_built), (2025, stable_built | gain)):
        image = np.where(occupied, built, vegetation) + rng.normal(0, 0.015, (6, h, w))
        count = np.full((1, h, w), 8, "float32")
        count[0, : 3 if year == 2016 else 6, :] = 0
        data = np.concatenate((image, count)).astype("float32")
        data[:6, count[0] == 0] = -9999
        with rasterio.open(
            folder / f"{year}.tif",
            "w",
            **profile,
            dtype="float32",
            count=7,
            nodata=-9999,
        ) as dst:
            dst.write(data)
            dst.descriptions = BANDS
            dst.update_tags(year=str(year), source="synthetic fixture")
    labels = gain.astype("uint8")
    labels[12:15, :] = IGNORE
    with rasterio.open(
        folder / "labels.tif", "w", **profile, dtype="uint8", count=1, nodata=IGNORE
    ) as dst:
        dst.write(labels, 1)
    return folder / "2016.tif", folder / "2025.tif", folder / "labels.tif"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out", type=Path, help="New directory for the synthetic experiment"
    )
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    args.out.mkdir(parents=True, exist_ok=False)
    before, after, labels = make_fixture(args.out / "fixture")
    fit(before, after, labels, args.out / "run", epochs=8, size=32, buffer=4)
    checkpoint = args.out / "run" / "best.pt"
    report = evaluate(before, after, labels, checkpoint, "cpu")
    report["source"] = "synthetic fixture"
    (args.out / "test.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predict(before, after, checkpoint, args.out / "maps", "cpu")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
