"""Synthetic software tests only; no observations, fitted study values or IDs."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import assignment, atlas, features, reconstruction

ROOT = Path(__file__).resolve().parents[1]


def toy_features():
    rng = np.random.default_rng(7)
    blocks = [rng.normal(i*6, 0.4, (40+i*4, 8)) for i in range(3)]
    return pd.DataFrame(np.vstack(blocks), columns=features.FEATURES)


class ReconstructionTests(unittest.TestCase):
    def test_boundary_splits_touching_foreground(self):
        foreground = np.zeros((48, 64), np.uint8)
        foreground[8:40, 8:56] = 1
        boundary = np.zeros_like(foreground)
        boundary[:, 31:33] = 1
        split = reconstruction.reconstruct(foreground, boundary, 0, 0)
        merged = reconstruction.reconstruct(foreground, boundary*0, 0, 0)
        self.assertEqual(len(np.unique(split[split > 0])), 2)
        self.assertEqual(len(np.unique(merged[merged > 0])), 1)
        np.testing.assert_array_equal(split > 0, foreground > 0)

    def test_size_thresholds_and_edge_policy(self):
        foreground = np.zeros((48, 64), np.uint8)
        foreground[:8, :8] = 1
        foreground[20:24, 20:24] = 1
        labels = reconstruction.reconstruct(foreground, foreground*0, 0, 0)
        self.assertEqual(np.count_nonzero(labels), 64)
        self.assertGreater(labels[0, 0], 0)
        self.assertTrue(features.extract_features(labels).empty)

    def test_empty_and_no_seed(self):
        empty = np.zeros((16, 16), np.uint8)
        self.assertFalse(reconstruction.reconstruct(empty, empty).any())
        self.assertFalse(reconstruction.reconstruct(empty+1, empty+1).any())

    def test_probability_masks_fail_explicitly(self):
        with self.assertRaises(ValueError):
            reconstruction.reconstruct(np.full((8, 8), 0.5), np.zeros((8, 8)))
        with self.assertRaises(ValueError):
            reconstruction.reconstruct(np.zeros((8, 8)), np.zeros((9, 8)))


class FeatureTests(unittest.TestCase):
    def test_schema_geometry_translation(self):
        y, x = np.mgrid[:80, :100]
        mask = ((y-40)/12)**2 + ((x-50)/22)**2 < 1
        labels = mask.astype(int)*4
        frame = features.extract_features(labels)
        row = frame.iloc[0]
        self.assertEqual(set(frame.columns), set(features.FEATURES + features.LOCATION))
        self.assertEqual(len(features.FEATURES), 8)
        self.assertEqual(row.area_px, mask.sum())
        self.assertEqual(row.instance_id, 4)
        self.assertAlmostEqual(row.centroid_col, 50)
        shifted = features.extract_features(np.roll(labels, 2, axis=0))
        np.testing.assert_allclose(frame.loc[:, features.FEATURES], shifted.loc[:, features.FEATURES], equal_nan=True)

    def test_empty_and_invalid_labels(self):
        self.assertEqual(len(features.extract_features(np.zeros((8, 8)))), 0)
        with self.assertRaises(ValueError):
            features.extract_features(np.full((8, 8), 1.5))

    def test_filter_reasons_and_mad_dimension_rule(self):
        rng = np.random.default_rng(4)
        frame = pd.DataFrame(rng.normal(100, 1, (50, 8)), columns=features.FEATURES)
        frame["touches_border"] = False
        frame.loc[0, "touches_border"] = True
        frame.loc[1, "area_px"] = 2
        frame.loc[2, "extent"] = np.nan
        frame.loc[3, list(features.FEATURES)[:7]] = 10000
        frame.loc[4, "extent"] = 10000
        keep, reasons = features.quality_filter(frame)
        self.assertEqual(reasons[:4], ["edge_instance", "small_area", "nonfinite_feature", "mad_outlier"])
        self.assertTrue(keep[4])


class AssignmentTests(unittest.TestCase):
    def test_votes_are_not_one_nearest_neighbor(self):
        centers = np.array([[1.0], [1.1], [1.2]])
        result = assignment.assign(np.array([[0.0]]), centers, np.ones((3, 1, 1)), np.array([0, 1, 1]))
        row = result.iloc[0]
        self.assertEqual(row.source_label, 1)
        self.assertEqual(row.prototype_id, 1)
        self.assertAlmostEqual(row.prototype_distance, 1.1)
        expected = (1/(1.1+1e-8) + 1/(1.2+1e-8))/(1/(1+1e-8)+1/(1.1+1e-8)+1/(1.2+1e-8))
        self.assertAlmostEqual(row.prototype_vote_share, expected)

    def test_mahalanobis_and_zero_distance(self):
        result = assignment.assign(np.array([[0.0], [1.0]]), np.array([[1.0], [2.0]]),
                                   np.array([[[100.0]], [[0.01]]]), np.array([0, 1]), neighbors=1)
        self.assertEqual(result.source_label.tolist(), [1, 0])
        np.testing.assert_allclose(result.prototype_vote_share, 1)

    def test_chunking_and_external_mapping(self):
        rng = np.random.default_rng(11)
        values, centers = rng.normal(size=(17, 2)), rng.normal(size=(6, 2))
        args = (values, centers, np.repeat(np.eye(2)[None], 6, axis=0), np.arange(6))
        pd.testing.assert_frame_equal(assignment.assign(*args, chunk_size=1), assignment.assign(*args))
        self.assertEqual(assignment.regroup([0, 1, 2], {0: 2, 1: 1, 2: 1}).tolist(), [2, 1, 1])
        with self.assertRaises(ValueError):
            assignment.regroup([0, 1], {0: 1})

    def test_empty_input(self):
        result = assignment.assign(np.empty((0, 2)), np.zeros((1, 2)), np.eye(2)[None], np.array([0]))
        self.assertTrue(result.empty)


class AtlasTests(unittest.TestCase):
    def test_fit_roundtrip_and_frozen_transform(self):
        frame = toy_features()
        model, ref = atlas.fit(frame, clusters=3, runs=2, n_init=2)
        self.assertEqual(len(ref), len(frame))
        self.assertEqual(model["centers"].shape, (9, 8))
        self.assertEqual(ref.source_label.value_counts().index[0], 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "atlas.npz"
            atlas.save(model, path)
            loaded = atlas.load(path)
        first = frame.iloc[:3].copy()
        first.iloc[0, 0] = np.nan
        expected = (model["median"][0]-model["mean"][0])/model["scale"][0]
        transformed = atlas.transform(first, loaded)
        self.assertAlmostEqual(transformed[0, 0], expected)
        np.testing.assert_allclose(transformed, atlas.transform(first, model))

    def test_reproducible_seed_and_feature_order_validation(self):
        frame = toy_features()
        first, labels1 = atlas.fit(frame, clusters=3, runs=2, n_init=2)
        second, labels2 = atlas.fit(frame, clusters=3, runs=2, n_init=2)
        np.testing.assert_allclose(first["centers"], second["centers"])
        pd.testing.assert_frame_equal(labels1, labels2)
        first["features"] = first["features"][::-1]
        with self.assertRaises(ValueError):
            atlas.transform(frame, first)

    def test_small_population_covariance(self):
        np.testing.assert_array_equal(atlas.precision_matrix(np.zeros((1, 8))), np.eye(8))
        values = np.zeros((4, 8))
        centers, precision, labels = atlas.prototypes(values, np.zeros(4, int))
        self.assertEqual(centers.shape, (1, 8))
        np.testing.assert_array_equal(precision[0], np.eye(8))

    def test_missing_feature_and_invalid_population(self):
        frame = toy_features()
        frame.iloc[:, 0] = np.nan
        with self.assertRaises(ValueError):
            atlas.fit(frame, clusters=3, runs=1)


class CommandTests(unittest.TestCase):
    def call(self, *args):
        result = subprocess.run([sys.executable, str(ROOT / "core.py"), *map(str, args)],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_external_input_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "inputs"
            source.mkdir()
            yy, xx = np.mgrid[:64, :64]
            mask = ((yy-32)/12)**2+((xx-32)/20)**2 < 1
            np.save(source / "foreground.npy", mask)
            np.save(source / "boundary.npy", mask*0)
            self.call("instances", "--foreground", source / "foreground.npy", "--boundary", source / "boundary.npy",
                      "--output-dir", root / "instances")
            self.call("features", "--labels", root / "instances/instances.npy", "--output-dir", root / "features")
            self.assertEqual(len(pd.read_csv(root / "features/features.csv")), 1)
            toy_features().to_csv(source / "reference.csv", index=False)
            self.call("fit", "--input-csv", source / "reference.csv", "--clusters", 3, "--runs", 2,
                      "--n-init", 2, "--output-dir", root / "model")
            self.call("assign", "--input-csv", source / "reference.csv", "--atlas", root / "model/atlas.npz",
                      "--output-dir", root / "assigned")
            self.assertEqual(len(pd.read_csv(root / "assigned/assignments.csv")), len(toy_features()))

    def test_reject_output_in_checkout(self):
        result = subprocess.run([sys.executable, str(ROOT / "core.py"), "instances", "--foreground", "unused.npy",
                                 "--boundary", "unused.npy", "--output-dir", str(ROOT / "results")],
                                capture_output=True, text=True, timeout=90)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside this code-only checkout", result.stderr)


if __name__ == "__main__":
    unittest.main()
