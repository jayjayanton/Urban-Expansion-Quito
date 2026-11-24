"""Fit, select, evaluate and apply the pilot change detector."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    BANDS,
    HALO,
    IGNORE,
    Chips,
    class_counts,
    normalization,
    normalize,
    read_labels,
    read_pair,
    regions,
)
from .model import ChangeUNet


def fingerprints(paths):
    result = []
    for path in paths:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result.append(digest.hexdigest())
    return result


def core_logits(model, inputs):
    return model(inputs)[:, HALO:-HALO, HALO:-HALO]


def metrics(counts):
    tn, fp, fn, tp = map(int, counts)

    def ratio(n, d):
        return n / d if d else None

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "reviewed": tn + fp + fn + tp,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
    }


@torch.inference_mode()
def score(model, loader, thresholds, device):
    model.eval()
    counts = np.zeros((len(thresholds), 4), dtype="int64")
    for inputs, labels, _, _ in loader:
        keep = labels != IGNORE
        probs = core_logits(model, inputs.to(device)).sigmoid().cpu()[keep].numpy()
        truth = labels[keep].numpy()
        for i, threshold in enumerate(thresholds):
            # Bincount order: true negative, false positive, false negative, true positive.
            counts[i] += np.bincount(2 * truth + (probs >= threshold), minlength=4)
    return [dict(threshold=t, **metrics(row)) for t, row in zip(thresholds, counts)]


def fit(
    before,
    after,
    labels_path,
    out,
    *,
    epochs=12,
    size=32,
    buffer=8,
    min_count=1,
    batch_size=4,
    seed=7,
    device="cpu",
):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    pair = read_pair(before, after, min_count)
    labels = read_labels(labels_path, pair)
    spans = regions(labels.shape[1], buffer)
    stats = normalization(pair, spans["train"])
    image = normalize(pair, stats)
    torch.manual_seed(seed)
    loaders = {}
    class_totals = {}
    for role in ("train", "val"):
        chips = Chips(image, labels, spans[role], size)
        class_totals[role] = class_counts(chips.labels)
        if min(class_totals[role]) == 0:
            raise ValueError(
                f"{role} needs reviewed gain and no-gain pixels on valid imagery"
            )
        loaders[role] = DataLoader(
            chips, batch_size=batch_size, shuffle=role == "train"
        )

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model = ChangeUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    thresholds = [i / 20 for i in range(1, 20)]
    transform = pair.profile["transform"]
    metadata = {
        "format_version": 1,
        "bands": list(BANDS),
        "stats": stats,
        "chip_size": size,
        "min_count": min_count,
        "seed": seed,
        "spans": spans,
        "buffer": buffer,
        "years": pair.years,
        "resolution": [transform.a, -transform.e],
        "crs": pair.profile["crs"].to_string(),
        "input_sha256": fingerprints((before, after, labels_path)),
        "class_counts": class_totals,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    best_f1 = -1
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total, n = 0.0, 0
        for inputs, target, _, _ in loaders["train"]:
            target = target.to(device)
            keep = target != IGNORE
            logits = core_logits(model, inputs.to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits[keep], target[keep].float()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pixels = int(keep.sum().item())
            loss_total += loss.item() * pixels
            n += pixels
        candidates = score(model, loaders["val"], thresholds, device)
        chosen = max(
            candidates, key=lambda row: (row["f1"], -abs(row["threshold"] - 0.5))
        )
        history.append(
            {"epoch": epoch, "train_bce": loss_total / n, "validation": chosen}
        )
        if chosen["f1"] > best_f1:
            best_f1 = chosen["f1"]
            best = dict(**metadata, epoch=epoch, threshold=chosen["threshold"])
            temp = out / "best.pt.partial"
            torch.save(dict(**best, weights=model.state_dict()), temp)
            temp.replace(out / "best.pt")
    report = dict(**best, history=history)
    (out / "training.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def restore(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format_version"] != 1 or state["bands"] != list(BANDS):
        raise ValueError("Unsupported checkpoint or input bands")
    model = ChangeUNet().to(device)
    model.load_state_dict(state["weights"])
    model.eval()
    return model, state


def evaluate(before, after, labels_path, checkpoint, device):
    model, state = restore(checkpoint, device)
    if fingerprints((before, after, labels_path)) != state["input_sha256"]:
        raise ValueError(
            "Held-out evaluation requires the image pair and label raster used at fit"
        )
    pair = read_pair(before, after, state["min_count"])
    chips = Chips(
        normalize(pair, state["stats"]),
        read_labels(labels_path, pair),
        state["spans"]["test"],
        state["chip_size"],
    )
    if not len(chips):
        raise ValueError("No reviewed valid test pixels")
    (result,) = score(
        model, DataLoader(chips, batch_size=4), [state["threshold"]], device
    )
    return dict(
        split="test", columns=state["spans"]["test"], epoch=state["epoch"], **result
    )


def write_layer(path, pair, array, nodata, name, **tags):
    profile = pair.profile
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=profile["height"],
        width=profile["width"],
        transform=profile["transform"],
        crs=profile["crs"],
        count=1,
        dtype=array.dtype,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array, 1)
        dst.set_band_description(1, name)
        dst.update_tags(**tags)


@torch.inference_mode()
def predict(before, after, checkpoint, out, device):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    model, state = restore(checkpoint, device)
    pair = read_pair(before, after, state["min_count"])
    transform = pair.profile["transform"]
    if pair.profile["crs"].to_string() != state["crs"] or not np.allclose(
        [transform.a, -transform.e], state["resolution"], rtol=0.01
    ):
        raise ValueError(
            "Inference needs the training CRS and pixel spacing (within 1%)"
        )
    h, w = pair.valid.shape
    mask = np.full((h, w), IGNORE, dtype="uint8")
    chips = Chips(
        normalize(pair, state["stats"]),
        mask,
        (0, w),
        state["chip_size"],
        labeled_only=False,
    )
    scores = np.full((h, w), -9999, dtype="float32")
    for inputs, _, ys, xs in DataLoader(chips, batch_size=4):
        probs = core_logits(model, inputs.to(device)).sigmoid().cpu().numpy()
        for patch, y, x in zip(probs, ys.tolist(), xs.tolist()):
            target = scores[y : y + chips.size, x : x + chips.size]
            target[:] = patch[: target.shape[0], : target.shape[1]]
    scores[~pair.valid] = -9999
    mask[pair.valid] = scores[pair.valid] >= state["threshold"]
    tags = {
        "before": Path(before).name,
        "after": Path(after).name,
        "min_count": state["min_count"],
        "threshold": state["threshold"],
        "checkpoint_sha256": fingerprints([checkpoint])[0],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as staging:
        maps = Path(staging) / "maps"
        maps.mkdir()
        write_layer(
            maps / "gain_score.tif", pair, scores, -9999, "built_up_gain_score", **tags
        )
        write_layer(maps / "gain_mask.tif", pair, mask, IGNORE, "built_up_gain", **tags)
        maps.rename(out)
