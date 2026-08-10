from __future__ import annotations

import unittest

from paer.analyze_aes_paer_routing_contribution import compute_routing_metrics


class RoutingDiagnosticTests(unittest.TestCase):
    def test_reports_fixed_set_asr_and_correction_contribution(self):
        records = [
            {
                "original_text": "clean a",
                "perturbed_text": "attack a",
                "original_score": 1.0,
                "perturbed_score": 1.2,
                "true_score": 2.0,
            },
            {
                "original_text": "clean b",
                "perturbed_text": "attack b",
                "original_score": 2.0,
                "perturbed_score": 2.05,
                "true_score": 3.0,
            },
        ]
        # Tuple order: routed score, base-branch score, correction.
        scored = {
            "clean a": (1.0, 1.01, 0.01),
            "attack a": (1.2, 1.31, 0.11),
            "clean b": (2.0, 2.01, 0.01),
            "attack b": (2.05, 2.11, 0.06),
        }

        metrics = compute_routing_metrics(
            records,
            scored,
            success_threshold=0.1,
        )

        self.assertEqual(metrics["routed_fixed_set_asr"], 0.5)
        self.assertEqual(metrics["base_branch_fixed_set_asr"], 1.0)
        self.assertAlmostEqual(
            metrics["fixed_set_asr_reduction_from_routing"], 0.5
        )
        self.assertAlmostEqual(
            metrics["mean_adversarial_minus_original_correction"], 0.075
        )
        self.assertAlmostEqual(metrics["mean_delta_reduction_from_routing"], 0.075)


if __name__ == "__main__":
    unittest.main()
