from __future__ import annotations

import unittest

import numpy as np

from paper_results.bootstrap_paer_v3_vs_mixed_at import bootstrap_reduction


class AESPairedBootstrapTests(unittest.TestCase):
    def test_positive_reduction_means_paer_has_lower_asr(self):
        baseline = np.asarray([1.0, 1.0, 1.0, 0.0])
        paer = np.asarray([0.0, 0.0, 1.0, 0.0])

        result = bootstrap_reduction(
            baseline,
            paer,
            n_bootstrap=2000,
            seed=42,
        )

        self.assertAlmostEqual(result["baseline_asr"], 0.75)
        self.assertAlmostEqual(result["paer_asr"], 0.25)
        self.assertAlmostEqual(result["absolute_asr_reduction"], 0.5)
        self.assertGreater(
            result["bootstrap_probability_reduction_gt_0"], 0.8
        )
        self.assertGreater(result["two_sided_bootstrap_p"], 0.0)

    def test_rejects_unaligned_shapes(self):
        with self.assertRaises(ValueError):
            bootstrap_reduction(
                np.asarray([1.0, 0.0]),
                np.asarray([1.0]),
                n_bootstrap=10,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
