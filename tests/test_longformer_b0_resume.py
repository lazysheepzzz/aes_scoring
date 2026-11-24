import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paer.train_longformer_aes_baseline import checked_resume


class ResumeTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        self.cp = self.out/"intermediate/checkpoint-576"
        self.cp.mkdir(parents=True)
        self.expected = {"training": {"seed": 42}, "train_sha256": "a"}
        (self.out/"run_config.json").write_text(json.dumps(self.expected))
        self.best = self.cp.with_name("checkpoint-504")
        self.best.mkdir()
        for path in (self.best/"model.safetensors", *(self.cp/k for k in (
                "model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth"))):
            path.write_bytes(b"fixture")
        (self.cp/"trainer_state.json").write_text(json.dumps({
            "global_step": 576, "max_steps": 3030, "best_model_checkpoint": str(self.best)}))

    def test_latest_with_older_best(self):
        self.assertEqual(checked_resume(self.out, self.cp, self.expected), self.cp.resolve())

    def test_no_rewind(self):
        with self.assertRaisesRegex(ValueError, "rewind"):
            checked_resume(self.out, self.best, self.expected)

    def test_changed_config(self):
        with self.assertRaisesRegex(ValueError, "fold changed"):
            checked_resume(self.out, self.cp, dict(self.expected, train_sha256="b"))

    def test_empty_state(self):
        (self.cp/"optimizer.pt").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "optimizer.pt"):
            checked_resume(self.out, self.cp, self.expected)

    def test_completed_run(self):
        (self.out/"best").mkdir()
        with self.assertRaises(FileExistsError):
            checked_resume(self.out, self.cp, self.expected)
