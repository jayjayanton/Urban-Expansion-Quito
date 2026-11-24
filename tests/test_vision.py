"""Offline regression checks for the optional PyTorch workflow."""

import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio

if find_spec("torch"):
    import torch

    from scripts.vision.data import (
        HALO,
        IGNORE,
        Chips,
        normalization,
        normalize,
        read_labels,
        read_pair,
        regions,
    )
    from scripts.vision.demo import make_fixture
    from scripts.vision.model import ChangeUNet
    from scripts.vision.workflow import evaluate, fit, metrics, predict, write_layer


@unittest.skipUnless(
    find_spec("torch"), "Install requirements-vision.txt for model tests"
)
class VisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.before, self.after, self.labels = make_fixture(self.root / "fixture")

    def test_geography_dates_and_band_order(self):
        with self.assertRaisesRegex(ValueError, "precede"):
            read_pair(self.after, self.before)
        with rasterio.open(self.after, "r+") as dst:
            t = dst.transform
            dst.transform = rasterio.Affine(t.a, t.b, t.c + t.a / 2, t.d, t.e, t.f)
        with self.assertRaisesRegex(ValueError, "transform"):
            read_pair(self.before, self.after)
        with rasterio.open(self.before, "r+") as dst:
            dst.set_band_description(1, "B04")
        with self.assertRaisesRegex(ValueError, "named bands"):
            read_pair(self.before, self.after)

    def test_label_contract_and_valid_observations(self):
        pair = read_pair(self.before, self.after)
        labels = read_labels(self.labels, pair)
        self.assertTrue((labels[:6] == IGNORE).all())
        self.assertTrue((labels[12:15] == IGNORE).all())
        self.assertFalse(read_pair(self.before, self.after, min_count=9).valid.any())
        with rasterio.open(self.labels, "r+") as dst:
            values = dst.read(1)
            values[22, 23] = 2
            dst.write(values, 1)
        with self.assertRaisesRegex(ValueError, "Use labels"):
            read_labels(self.labels, pair)
        with rasterio.open(self.labels, "r+") as dst:
            dst.nodata = 0
        with self.assertRaisesRegex(ValueError, "valid classes"):
            read_labels(self.labels, pair)

    def test_split_excludes_held_out_context_and_normalization(self):
        pair = read_pair(self.before, self.after)
        spans = regions(pair.valid.shape[1], 4)
        stats = normalization(pair, spans["train"])
        right = spans["train"][1]
        pair.image[:, :, right:] = 12345
        self.assertEqual(normalization(pair, spans["train"]), stats)
        chips = Chips(
            normalize(pair, stats), read_labels(self.labels, pair), spans["train"], 32
        )
        # Last core has partial width. Even its halo cannot use withheld geography.
        patch, target, _y, x = chips[len(chips) - 1]
        self.assertTrue(torch.all(patch[:, :, HALO + right - x :] == 0))
        self.assertTrue(torch.all(target[:, right - x :] == IGNORE))
        self.assertEqual(tuple(ChangeUNet()(patch[None]).shape), (1, 72, 72))

    def test_held_out_labels_never_select_weights(self):
        pair = read_pair(self.before, self.after)
        values = read_labels(self.labels, pair)
        left, right = regions(values.shape[1], 4)["test"]
        values[:, left:right] = IGNORE
        masked = self.root / "masked.tif"
        write_layer(masked, pair, values, IGNORE, "gain")
        for tag, label in (("a", self.labels), ("b", masked)):
            fit(self.before, self.after, label, self.root / tag, epochs=1, buffer=4)
        a, b = [
            torch.load(self.root / tag / "best.pt", weights_only=True)
            for tag in ("a", "b")
        ]
        self.assertEqual(a["threshold"], b["threshold"])
        self.assertTrue(
            all(torch.equal(w, b["weights"][k]) for k, w in a["weights"].items())
        )
        with self.assertRaisesRegex(ValueError, "No reviewed valid test"):
            evaluate(
                self.before, self.after, masked, self.root / "b" / "best.pt", "cpu"
            )

    def test_train_evaluate_and_predict_partial_edges(self):
        pair = read_pair(self.before, self.after)
        run = self.root / "run"
        report = fit(
            self.before, self.after, self.labels, run, epochs=2, size=16, buffer=4
        )
        checkpoint = run / "best.pt"
        result = evaluate(self.before, self.after, self.labels, checkpoint, "cpu")
        self.assertEqual(result["threshold"], report["threshold"])
        self.assertEqual(result["reviewed"], 2088)
        self.assertEqual(result["tp"] + result["fn"], 396)
        predict(self.before, self.after, checkpoint, self.root / "maps", "cpu")
        with rasterio.open(self.root / "maps" / "gain_score.tif") as ds:
            self.assertEqual(ds.transform, pair.profile["transform"])
            self.assertEqual(ds.crs, pair.profile["crs"])
            scores = ds.read(1)
            np.testing.assert_array_equal(scores == ds.nodata, ~pair.valid)
            self.assertTrue(
                ((scores[pair.valid] >= 0) & (scores[pair.valid] <= 1)).all()
            )
        with rasterio.open(self.root / "maps" / "gain_mask.tif") as ds:
            mask = ds.read(1)
            np.testing.assert_array_equal(mask == IGNORE, ~pair.valid)
            np.testing.assert_array_equal(
                mask[pair.valid], scores[pair.valid] >= report["threshold"]
            )
        with self.assertRaisesRegex(ValueError, "used at fit"):
            evaluate(self.after, self.before, self.labels, checkpoint, "cpu")

    def test_confusion_metrics(self):
        counts = metrics([9, 2, 1, 3])
        self.assertEqual(counts["precision"], 3 / 5)
        self.assertEqual(counts["recall"], 3 / 4)
        self.assertEqual(counts["iou"], 0.5)
        self.assertIsNone(metrics([8, 0, 0, 0])["f1"])


if __name__ == "__main__":
    unittest.main()
