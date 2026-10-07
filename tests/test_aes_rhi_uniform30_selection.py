import unittest
from pathlib import Path

from paer.select_aes_rhi_uniform30_checkpoints import _selection_cli


class UniformThirtySelectionTests(unittest.TestCase):
    def test_only_selection_budget_and_output_bindings_are_replaced(self):
        reference = {
            "attack": "hotflip",
            "seed": 42,
            "selection_steps": 10,
            "rudimentary_selection_steps": 30,
            "injection_selection_steps": 30,
            "batch_size": 4,
            "defense_output_dir": "historical_training",
            "selection_output_dir": "historical_selection",
            "subset_ids_path": "historical_ids",
            "optional": None,
            "flag": False,
        }
        result = _selection_cli(
            reference,
            training_dir=Path("new training"),
            selection_dir=Path("new selection"),
            training_inputs=Path("new training/provenance.json"),
        )
        parsed = dict(zip(result[::2], result[1::2]))
        self.assertEqual(parsed["--selection-steps"], "30")
        self.assertEqual(parsed["--rudimentary-selection-steps"], "30")
        self.assertEqual(parsed["--injection-selection-steps"], "30")
        self.assertEqual(parsed["--defense-output-dir"], "new training")
        self.assertEqual(parsed["--selection-output-dir"], "new selection")
        self.assertNotIn("--attack", result)
        self.assertNotIn("--optional", result)
        self.assertEqual(reference["selection_steps"], 10)


if __name__ == "__main__":
    unittest.main()
