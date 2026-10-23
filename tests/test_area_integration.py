import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from area_integration import exact_grid


def tile(name, y, x, h, w):
    return {'tile': name, 'tile_y0_um': y, 'tile_x0_um': x,
            'tile_y1_um': y+h, 'tile_x1_um': x+w}


class AreaTests(unittest.TestCase):
    def test_full_aligned(self):
        t = pd.DataFrame([tile('a', 0, 0, 20, 20)])
        a, _, direct = exact_grid(t, {'a': np.zeros((250, 250))}, np.array([0, 10, 20]), np.array([0, 10, 20]), (20, 20))
        np.testing.assert_allclose(a, 100)
        self.assertAlmostEqual(direct, 400)

    def test_fractional_tile_boundary_and_padding(self):
        t = pd.DataFrame([tile('a', 0, 0, 10, 7.5), tile('b', 0, 7.5, 10, 12.5)])
        masks = {'a': np.zeros((125, 94)), 'b': np.zeros((125, 157))}
        a, _, direct = exact_grid(t, masks, np.array([0, 10]), np.array([0, 10, 20]), (10, 20))
        np.testing.assert_allclose(a, 100)
        self.assertAlmostEqual(direct, 200)

    def test_roi_fractional_terminal(self):
        t = pd.DataFrame([tile('a', 0, 0, 20, 20)])
        a, _, direct = exact_grid(t, {'a': np.zeros((250, 250))}, np.array([0, 10, 20]), np.array([0, 10, 20]), (15, 15))
        np.testing.assert_allclose(a, [[100, 50], [50, 25]])
        self.assertAlmostEqual(direct, 225)

    def test_half_tissue(self):
        mask = np.ones((250, 250))
        mask[:, :125] = 0
        t = pd.DataFrame([tile('a', 0, 0, 20, 20)])
        a, _, direct = exact_grid(t, {'a': mask}, np.array([0, 10, 20]), np.array([0, 10, 20]), (20, 20))
        np.testing.assert_allclose(a, [[100, 0], [100, 0]])
        self.assertAlmostEqual(direct, 200)

    def test_missing_tile_no_imputation(self):
        t = pd.DataFrame([tile('a', 0, 0, 10, 10)])
        a, _, direct = exact_grid(t, {'a': np.zeros((125, 125))}, np.array([0, 10, 20]), np.array([0, 10, 20]), (20, 20))
        np.testing.assert_allclose(a, [[100, 0], [0, 0]])
        self.assertAlmostEqual(direct, 100)


if __name__ == '__main__':
    unittest.main()
