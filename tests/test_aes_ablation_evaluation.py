import unittest
from pathlib import Path
from paer.evaluate_aes_paer_rhi_ablations import selection_arguments


class SelectionArgumentsTests(unittest.TestCase):
    def test_preserves_reference_budget(self):
        original = dict(seed=42, subset_seed=42, subset_size=256, selection_steps=10,
                        rudimentary_selection_steps=30, injection_selection_steps=30,
                        max_checkpoint_step=1400, qwk_tolerance=0.02, dtype="float32",
                        batch_size=4, defense_output_dir="old", selection_output_dir="old_selection",
                        subset_ids_path="old_ids", optional=None, flag=False)
        command = selection_arguments(original, Path("new_train"), Path("new_selection"))
        result = dict(zip(command[::2], command[1::2]))
        self.assertEqual(result["--selection-steps"], "10")
        self.assertEqual(result["--qwk-tolerance"], "0.02")
        self.assertEqual(result["--defense-output-dir"], "new_train")
        self.assertIn("reference_rhi_training_inputs.json", result["--training-inputs"])
        self.assertNotIn("--optional", result)
        self.assertEqual(original["defense_output_dir"], "old")

    def test_paths_with_spaces_are_single_arguments(self):
        args = selection_arguments({}, Path("train dir"), Path("select dir"))
        self.assertIn("train dir", args)
        self.assertIn("select dir", args)


if __name__ == "__main__":
    unittest.main()
