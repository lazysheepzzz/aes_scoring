import copy
import unittest
from paer.run_aes_paer_rhi_ablation_training import ABLATIONS, ablated_config


class AblationConfigTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(training_mode="paer_rh_v3", experiment_name="paer_rhi_v3",
                              correction_scale=1.0, localization_loss_weight=0.25,
                              attention_alignment_loss_weight=0.05, correction_calibration_weight=1.0,
                              route_lift_loss_weight=4.0, clean_correction_weight=1.0,
                              learning_rate=1e-5, seed=42, num_epochs=3, max_length=1024,
                              gradient_accumulation_steps=8, trace_jsonl="same.jsonl", output_dir="full")

    def test_only_declared_changes(self):
        original = copy.deepcopy(self.reference)
        for variant, changes in ABLATIONS.items():
            result = ablated_config(self.reference, variant, "new")
            changed = {k for k in result if result[k] != self.reference.get(k)}
            self.assertEqual(changed, set(changes) | {"output_dir", "experiment_name"})
        self.assertEqual(original, self.reference)

    def test_localization_keeps_score_level_supervision(self):
        result = ablated_config(self.reference, "without_token_localization", "new")
        self.assertEqual(result["route_lift_loss_weight"], 4.0)
        self.assertEqual(result["correction_scale"], 1.0)

    def test_route_off_removes_dependent_objectives(self):
        result = ablated_config(self.reference, "without_routing", "new")
        for key in ABLATIONS["without_routing"]:
            self.assertEqual(result[key], 0.0)
        self.assertEqual(result["localization_loss_weight"], 0.25)

    def test_smoke_is_explicit(self):
        result = ablated_config(self.reference, "without_routing", "new", smoke=True)
        self.assertEqual((result["max_train_samples"], result["max_valid_samples"], result["num_epochs"]), (160, 32, 1))

    def test_reject_already_ablated(self):
        self.reference["correction_scale"] = 0.0
        with self.assertRaises(ValueError):
            ablated_config(self.reference, "without_routing", "new")


if __name__ == "__main__":
    unittest.main()
