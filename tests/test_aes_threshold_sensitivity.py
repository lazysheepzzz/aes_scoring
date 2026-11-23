import unittest

from paer.evaluate_aes_rhi_threshold_sensitivity import counts, rhi_macro


class Thresholds(unittest.TestCase):
    def setUp(self):
        self.ids = list("abcd")
        self.rows = [
            dict(essay_id=i, perturbed_text="retained", delta=d, original_score=0.,
                 perturbed_score=d, success=d >= .3)
            for i, d in zip(self.ids, [.1999999, .2, .2999999, .3])
        ]
        self.payload = dict(summary=dict(attack="hotflip", success_threshold=.3, asr=.25),
                            details=self.rows)

    def test_inclusive_cutoffs_without_rounding(self):
        self.assertEqual(counts(self.payload, self.ids, "hotflip"), [3, 1])

    def test_reject_mixed_protocols_or_unpaired_samples(self):
        with self.assertRaises(ValueError):
            counts(self.payload, self.ids[::-1], "hotflip")
        self.payload["summary"]["success_threshold"] = .1
        with self.assertRaises(ValueError):
            counts(self.payload, self.ids, "hotflip")

    def test_attack_failure_not_counted_as_unsuccessful(self):
        self.rows[0]["perturbed_text"] = None
        with self.assertRaises(ValueError):
            counts(self.payload, self.ids, "hotflip")

    def test_macro_gives_injection_one_family_weight(self):
        self.assertAlmostEqual(rhi_macro(1., 0., 1., 0.), .5)
