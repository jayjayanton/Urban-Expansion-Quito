"""Train or apply the Quito two-date change detector. Run from the repository root."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import BANDS, IGNORE, class_counts, read_labels, read_pair, regions
from .workflow import evaluate, fit, predict, write_layer

DEFAULT_BEFORE = "data/2016/sentinel2_2016.tif"
DEFAULT_AFTER = "data/2025/sentinel2_2025.tif"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "init-labels", "train", "evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--before", default=DEFAULT_BEFORE)
        sub.add_argument("--after", default=DEFAULT_AFTER)
        if name in ("inspect", "train"):
            sub.add_argument("--min-count", type=int, default=1)
        if name == "inspect":
            sub.add_argument("--labels", help="Optional reviewed mask for split counts")
            sub.add_argument("--buffer", type=int, default=8)
        if name in ("init-labels", "train", "evaluate"):
            sub.add_argument("--labels", required=True)
        if name in ("evaluate", "predict"):
            sub.add_argument("--checkpoint", required=True)
        if name in ("train", "predict"):
            sub.add_argument("--out", required=True, help="New output directory")
        if name in ("train", "evaluate", "predict"):
            sub.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
        if name == "train":
            sub.add_argument("--epochs", type=int, default=12)
            sub.add_argument("--chip-size", type=int, default=32)
            sub.add_argument(
                "--buffer",
                type=int,
                default=8,
                help="Excluded columns on each side of both spatial split lines",
            )
            sub.add_argument("--batch-size", type=int, default=4)
            sub.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.command in ("inspect", "init-labels"):
        pair = read_pair(args.before, args.after, getattr(args, "min_count", 1))
        if args.command == "init-labels":
            path = Path(args.labels)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects existing annotations.
            with path.open("xb"):
                pass
            try:
                write_layer(
                    path,
                    pair,
                    np.full(pair.valid.shape, IGNORE, "uint8"),
                    IGNORE,
                    "reviewed_built_up_gain",
                )
            except Exception:
                path.unlink(missing_ok=True)
                raise
            print(f"Created {path} with all pixels marked unreviewed")
            return
        report = {
            "years": pair.years,
            "shape": list(pair.valid.shape),
            "crs": pair.profile["crs"].to_string(),
            "bands": BANDS,
            "pixel_size": [pair.profile["transform"].a, -pair.profile["transform"].e],
            "paired_valid": int(pair.valid.sum()),
            "total": int(pair.valid.size),
            "min_count": args.min_count,
            "usable_count_ranges": pair.counts,
        }
        if args.labels:
            labels = read_labels(args.labels, pair)
            report["regions"] = {}
            for role, (left, right) in regions(labels.shape[1], args.buffer).items():
                negative, positive = class_counts(labels[:, left:right])
                report["regions"][role] = {
                    "columns": [left, right],
                    "no_gain": negative,
                    "gain": positive,
                }
        print(json.dumps(report, indent=2))
    elif args.command == "train":
        run = fit(
            args.before,
            args.after,
            args.labels,
            args.out,
            epochs=args.epochs,
            size=args.chip_size,
            buffer=args.buffer,
            min_count=args.min_count,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )
        print(
            f"Selected epoch {run['epoch']} and threshold {run['threshold']} on validation"
        )
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate(
                    args.before, args.after, args.labels, args.checkpoint, args.device
                ),
                indent=2,
            )
        )
    else:
        predict(args.before, args.after, args.checkpoint, args.out, args.device)
        print(f"Wrote gain_score.tif and gain_mask.tif to {args.out}")


if __name__ == "__main__":
    main()
