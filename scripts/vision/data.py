"""Read a pair of project composites and sample georeferenced, spatially isolated chips."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "valid_count")
IGNORE = 255
# Two pooling stages require a multiple of four. The halo covers the network's
# receptive field; predictions near interior chip edges can see adjacent pixels.
HALO = 20


@dataclass
class Pair:
    image: np.ndarray  # chronological order, 12 x H x W
    valid: np.ndarray  # H x W, both dates usable
    profile: dict
    years: list
    counts: list


def check_grid(src, profile):
    if (
        src.crs != profile["crs"]
        or src.width != profile["width"]
        or src.height != profile["height"]
        or not src.transform.almost_equals(profile["transform"], precision=1e-7)
    ):
        raise ValueError(
            f"{src.name} must match the CRS, transform and shape of the first image"
        )


def read_pair(before, after, min_count=1):
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if Path(before).samefile(after):
        raise ValueError("Choose distinct before and after composites")
    images, masks, years, counts = [], [], [], []
    for path in (before, after):
        with rasterio.open(path) as src:
            if src.descriptions != BANDS:
                raise ValueError(f"{path} must have the named bands in order {BANDS}")
            if (
                src.crs is None
                or not src.crs.is_projected
                or src.transform.b != 0
                or src.transform.d != 0
                or src.transform.a <= 0
                or src.transform.e >= 0
            ):
                raise ValueError(
                    "Use a north-up projected composite with a defined CRS"
                )
            if not images:
                profile = src.profile.copy()
            else:
                check_grid(src, profile)
            data = src.read(out_dtype="float32", masked=True)
            raw = data.data
            good = (
                ~np.ma.getmaskarray(data).any(axis=0)
                & np.isfinite(raw).all(axis=0)
                & (raw != -9999).all(axis=0)
                & (raw[6] >= min_count)
            )
            images.append(raw[:6])
            masks.append(good)
            years.append(src.tags().get("year"))
            counts.append(
                [float(raw[6][good].min()), float(raw[6][good].max())]
                if good.any()
                else None
            )
    if all(years) and int(years[0]) >= int(years[1]):
        raise ValueError("The before year must precede the after year")
    return Pair(np.concatenate(images), masks[0] & masks[1], profile, years, counts)


def read_labels(path, pair):
    with rasterio.open(path) as src:
        check_grid(src, pair.profile)
        if src.count != 1 or src.nodata in (0, 1):
            raise ValueError("Labels need one band; both 0 and 1 must be valid classes")
        labels = src.read(1, masked=True).astype("float64").filled(IGNORE)
    if not np.isin(labels, [0, 1, IGNORE]).all():
        raise ValueError("Use labels 1=gain, 0=reviewed no gain, 255=ignore")
    return np.where(pair.valid, labels, IGNORE).astype("uint8")


def regions(width, buffer):
    """Return half-open column ranges; buffer pixels on each side of both cuts."""
    if buffer < 0:
        raise ValueError("buffer must be nonnegative")
    a, b = 3 * width // 5, 4 * width // 5
    spans = {
        "train": (0, a - buffer),
        "val": (a + buffer, b - buffer),
        "test": (b + buffer, width),
    }
    if any(right <= left for left, right in spans.values()):
        raise ValueError("Image is too narrow for this split buffer")
    return spans


def normalization(pair, train):
    left, right = train
    valid = pair.valid[:, left:right]
    if not valid.any():
        raise ValueError("No valid paired observations in the training region")
    # Pool both dates per physical band, using only the training geography.
    pixels = pair.image[:, :, left:right][:, valid].reshape(2, 6, -1)
    return (
        pixels.mean(axis=(0, 2), dtype="float64").tolist(),
        pixels.std(axis=(0, 2), dtype="float64").clip(1e-6).tolist(),
    )


def normalize(pair, stats):
    mean, std = (
        np.tile(np.asarray(v, dtype="float32"), 2)[:, None, None] for v in stats
    )
    # Fill jointly invalid observations at the training mean before arithmetic.
    return (np.where(pair.valid, pair.image, mean) - mean) / std


def class_counts(labels):
    return [int(np.count_nonzero(labels == i)) for i in (0, 1)]


class Chips(Dataset):
    """Sample a core plus context strictly inside a single geographic split."""

    def __init__(self, image, labels, span, size, labeled_only=True):
        if size < 4 or size % 4:
            raise ValueError("Chip size must be a positive multiple of four")
        left, right = span
        self.image = image[:, :, left:right]
        self.labels = labels[:, left:right]
        self.size = size
        h, w = self.labels.shape
        self.locations = [
            (y, x)
            for y in range(0, h, size)
            for x in range(0, w, size)
            if not labeled_only
            or (self.labels[y : y + size, x : x + size] != IGNORE).any()
        ]

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        y, x = self.locations[index]
        h, w = self.labels.shape
        size = self.size
        y0, x0 = max(y - HALO, 0), max(x - HALO, 0)
        y1, x1 = min(y + size + HALO, h), min(x + size + HALO, w)
        iy, ix = y0 - y + HALO, x0 - x + HALO
        patch = np.zeros((12, size + 2 * HALO, size + 2 * HALO), dtype="float32")
        patch[:, iy : iy + y1 - y0, ix : ix + x1 - x0] = self.image[:, y0:y1, x0:x1]
        core = self.labels[y : y + size, x : x + size]
        target = np.full((size, size), IGNORE, dtype="uint8")
        target[: core.shape[0], : core.shape[1]] = core
        return torch.from_numpy(patch), torch.from_numpy(target), y, x
