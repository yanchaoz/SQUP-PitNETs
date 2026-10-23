"""Purely synthetic software tests; no study observations or identifiers."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import sampling, sg_audit


def synthetic_spatial(folder):
    """Generate a uniform toy grid, not a biological validation dataset."""
    cases = ["reference_synthetic", "sample_synthetic"]
    domains, density, composition = [], [], []
    for case in cases:
        values = [1, 1, 1, 1] if case == cases[0] else [4, 3, 1, 1]
        for org, count in [("SG", sum(values)), ("Mito", 1)]:
            domains.append(dict(case_id=case, organelle=org, n_profiles=count*100,
                                valid_area_um2=10000, full_density_per_um2=count/100,
                                height_um=100, width_um=100))
            for i in range(10):
                for j in range(10):
                    density.append(dict(case_id=case, organelle=org, grid_i=i, grid_j=j,
                                        count=count, valid_area_um2=100, candidate_cell=1,
                                        y0_um=10*i, y1_um=10*(i+1), x0_um=10*j, x1_um=10*(j+1)))
        for i in range(10):
            for j in range(10):
                composition.append(dict(case_id=case, grid_i=i, grid_j=j, valid_area_um2=100,
                                        **dict(zip(sampling.CLASSES, values))))
    folder.mkdir()
    (folder / "case_order.json").write_text(json.dumps(cases), encoding="utf-8")
    pd.DataFrame(domains).to_csv(folder / "case_domains.csv", index=False)
    pd.DataFrame(density).to_csv(folder / "density_grid.csv.gz", index=False)
    pd.DataFrame(composition).to_csv(folder / "composition_grid.csv.gz", index=False)


def synthetic_sg():
    records = []
    for side in sg_audit.SIDES:
        for class_id, count in enumerate([4, 3, 2, 1, 1], start=1):
            extra = int(side == "prediction" and class_id == 1)
            records.append(dict(crop_id="field_synthetic", case_id="sample_synthetic", side=side,
                                display_class=class_id, n_profiles=count+extra,
                                full_profile_area_px=(count+extra)*100,
                                matched_both_included_n=count, unmatched_n=extra, counterpart_outside_inclusion_n=0))
    return pd.DataFrame(records)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.old = sampling.REPEATS, sampling.AREA_MODE, sampling.OUT
        sampling.REPEATS = 7
        sampling.AREA_MODE = "corrected"

    def tearDown(self):
        sampling.REPEATS, sampling.AREA_MODE, sampling.OUT = self.old

    def test_field_changes_do_not_cancel(self):
        _, a = sg_audit.compare([9, 1, 0, 0], [1, 9, 0, 0])
        _, b = sg_audit.compare([1, 9, 0, 0], [9, 1, 0, 0])
        _, pooled = sg_audit.compare([10, 10, 0, 0], [10, 10, 0, 0])
        self.assertAlmostEqual(a, .8)
        self.assertAlmostEqual(b, .8)
        self.assertEqual(pooled, 0)

    def test_empty_composition_rejected(self):
        with self.assertRaises(ValueError):
            sg_audit.fractions([0, 0, 0, 0])

    def test_seed_determinism(self):
        np.testing.assert_array_equal(sampling.rng_for(7, "synthetic", 1).integers(100, size=20),
                                      sampling.rng_for(7, "synthetic", 1).integers(100, size=20))

    def test_nonreplacement_and_feasibility(self):
        groups = [(np.array([1., 2., 4.]), np.ones(3))]
        c, a = sampling.draw(groups, 3, np.random.default_rng(2))
        np.testing.assert_array_equal(c, np.full(7, 7.))
        np.testing.assert_array_equal(a, np.full(7, 3.))
        self.assertIsNone(sampling.draw(groups, 4, np.random.default_rng(2)))

    def test_exact_half_area_threshold(self):
        groups = sampling.density_windows(np.ones((2, 2)), np.full((2, 2), 50.))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][1][0], 200.)
        self.assertFalse(sampling.density_windows(np.ones((2, 2)), np.full((2, 2), 49.999999)))

    def test_empty_sample_is_unsuccessful(self):
        sampling.REPEATS = 2
        with tempfile.TemporaryDirectory() as folder:
            sampling.OUT = Path(folder)
            row = sampling.sample_stats("synthetic", 50, 1, np.zeros((2, 4)), np.ones(2),
                                        np.array([.6, .3, .05, .05]), np.full(4, .25))
        self.assertEqual(row["empty_sample_probability"], 1.)
        self.assertEqual(row["tvd_le_005_probability"], 0.)
        self.assertTrue(np.isnan(row["tvd_median"]))

    def test_all_profiles_and_matching_closure(self):
        data = synthetic_sg()
        with tempfile.TemporaryDirectory() as folder:
            fields, cases, summary = sg_audit.analyse(data, folder)
        self.assertGreater(fields.count_tvd.iloc[0], 0)
        self.assertEqual(len(cases), 1)
        self.assertEqual(set(summary.unit), {"review_field", "case_median"})
        data.loc[0, "unmatched_n"] = 1
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            sg_audit.analyse(data, folder)

    def test_cli_synthetic_both_area_modes(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "input"
            synthetic_spatial(source)
            for mode in ("corrected", "legacy"):
                output = base / mode
                proc = subprocess.run([sys.executable, str(ROOT / "run.py"), "spatial", "--input-dir", str(source),
                                       "--reference-id", "reference_synthetic", "--area-mode", mode,
                                       "--repeats", "7", "--output-dir", str(output)], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["stable_at_one_field"], 1)
                self.assertEqual(summary["median_required_nominal_area_um2"], 2500)
                errors = pd.read_csv(output / "figure4D_cohort.csv")
                np.testing.assert_allclose(errors["median"], 0, atol=1e-12)

    def test_cli_sg(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "input"
            source.mkdir()
            synthetic_sg().to_csv(source / "composition.csv", index=False)
            proc = subprocess.run([sys.executable, str(ROOT / "run.py"), "sg", "--input-csv", str(source / "composition.csv"),
                                   "--output-dir", str(base / "output")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((base / "output" / (sg_audit.PREFIX + "review_composition_summary.csv")).exists())

    def test_physical_area_bound_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input"
            synthetic_spatial(source)
            grid = pd.read_csv(source / "density_grid.csv.gz")
            grid.loc[0, "valid_area_um2"] += 1
            grid.loc[1, "valid_area_um2"] -= 1
            grid.to_csv(source / "density_grid.csv.gz", index=False)
            with self.assertRaisesRegex(ValueError, "physical"):
                sampling.load_inputs(source, "reference_synthetic")

    def test_unknown_reference_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input"
            synthetic_spatial(source)
            with self.assertRaisesRegex(ValueError, "Reference"):
                sampling.load_inputs(source, "unknown_synthetic")


if __name__ == "__main__":
    unittest.main()
