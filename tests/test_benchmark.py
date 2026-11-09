import tempfile
import unittest
from pathlib import Path

import numpy as np

from ppg_bench import FS, SECONDS, extract, run, simulate, split_groups


class BenchmarkTests(unittest.TestCase):
    def test_known_frequency_in_baseline(self):
        t = np.arange(FS * SECONDS) / FS
        features = extract(np.sin(2*np.pi*1.37*t) + .2*np.sin(2*np.pi*.2*t))
        self.assertAlmostEqual(features["spectral_bpm"], 82.2, delta=1)
        self.assertAlmostEqual(features["peak_bpm"], 82.2, delta=2)

    def test_flatline_and_missing(self):
        x = np.ones(FS * SECONDS)
        self.assertTrue(np.isnan(extract(x)["spectral_bpm"]))
        x[:len(x)//2] = np.nan
        self.assertAlmostEqual(extract(x)["missing_fraction"], .5)
        with self.assertRaises(ValueError):
            extract([1., 2.])

    def test_missing_and_clipped_waveforms(self):
        t = np.arange(FS * SECONDS) / FS
        x = np.minimum(np.sin(2*np.pi*1.17*t), .7)
        x[:FS] = np.nan
        result = extract(x)
        self.assertEqual(result["missing_fraction"], 1/SECONDS)
        self.assertGreater(result["rail_fraction"], .1)
        self.assertAlmostEqual(result["spectral_bpm"], 70.2, delta=2)

    def test_no_group_leakage_and_reproducible(self):
        ids = np.repeat(np.arange(40), 5)
        parts = split_groups(ids, 23)
        for a,b in zip(parts, split_groups(ids, 23)):
            np.testing.assert_array_equal(a, b)
        groups = [set(ids[p]) for p in parts]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        self.assertEqual(set.union(*groups), set(ids))

    def test_simulator_repeats(self):
        a = list(simulate(23, 1, 5))
        b = list(simulate(23, 1, 5))
        self.assertEqual({r[2] for r in a}, {"clean","drift","motion","dropout","mixed"})
        for l,r in zip(a,b):
            self.assertEqual(l[:4], r[:4])
            np.testing.assert_array_equal(l[4],r[4])

    def test_end_to_end_exports(self):
        with tempfile.TemporaryDirectory() as f:
            path = Path(f)
            m = run(23, 24, 5, path)
            self.assertEqual(m["data"], "synthetic only")
            self.assertEqual(m["test_metrics"]["gradient_boosting"]["n"], 30)
            self.assertTrue((path/"metrics.json").is_file())
            self.assertEqual(len((path/"heldout_predictions.csv").read_text().splitlines()), 31)
            self.assertEqual(m["test_metrics"]["welch"]["coverage"], 1.)
            self.assertLessEqual(m["test_metrics"]["gated_gradient_boosting"]["coverage"], 1.)


if __name__ == "__main__":
    unittest.main()
