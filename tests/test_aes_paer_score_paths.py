"""CPU-only accounting, provenance and orchestration tests (no essay results)."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paer import analyze_aes_paer_score_paths as diagnostic
from paer.rhi_experiment_utils import checkpoint_identity, read_json, save_json, sha256


def components(global_score, positive=0.25, negative=0.125, correction=0.0625,
               fingerprint="visible"):
    base = global_score + positive - negative
    return {"global": global_score, "positive": positive, "negative": negative,
            "correction": correction, "base": base, "routed": base - correction,
            "visible_input_sha256": fingerprint}


def fixture(root):
    source, checkpoint = root / "evaluation", root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test marker, NOT model weights")
    save_json(checkpoint / "paer_config.json", {"model_type": "paer_aes_v3"})
    data = root / "valid.csv"
    data.write_text("essay_id,text,score\n0,clean,3\n", encoding="utf-8")
    before, after = components(1.0), components(1.125)
    for attack, relative in diagnostic.ATTACK_FILES.items():
        save_json(source / "paer" / relative, {
            "summary": {"attack": attack, "success_threshold": 0.125},
            "details": [{"essay_id": "0", "original_text": "clean",
                         "perturbed_text": "edited", "original_score": before["routed"],
                         "perturbed_score": after["routed"]}],
        })
    for family in ("rudimentary", "hotflip", "injection_family"):
        directory = source / "paer" / family
        for name in ("run_manifest.json", "clean_qwk.json", "asr_summary.json"):
            save_json(directory / name, {})
        if family == "injection_family":
            save_json(directory / "injection_family_summary.json", {})
        save_json(directory / "completed_result_hashes.json",
                  {path.name: sha256(path) for path in directory.glob("*.json")})
    reference = {
        "protocol": "frozen_checkpoint_evaluation_v1", "suite": "rhi",
        "data": str(data), "data_sha256": sha256(data), "n_essays": 1,
        "checkpoints": {"paer": {"path": str(checkpoint), "files": checkpoint_identity(checkpoint)}},
        "dtype": "float32", "max_length": 1024, "success_threshold": 0.125,
        "evaluation_role": "development", "seed": 42,
        "code_hashes": {"old_attack_file.py": "old-code-is-not-reexecuted"},
    }
    save_json(source / "rhi_run_binding.json", reference)
    return source, checkpoint


class ScorePathAccountingTests(unittest.TestCase):
    def analyze(self, before, after, threshold=0.125, saved_after=None):
        records = [{"essay_id": "example", "original_text": "a", "perturbed_text": "b",
                    "original_score": before["routed"],
                    "perturbed_score": after["routed"] if saved_after is None else saved_after}]
        return diagnostic.decompose_pairs(records, {diagnostic.text_id("a"): before,
                                                    diagnostic.text_id("b"): after}, threshold)

    def test_signed_terms_sum_to_final_score_change(self):
        summary, pairs = self.analyze(components(1.0), components(1.25, 0.5, 0.25, 0.125))
        row = pairs[0]
        self.assertEqual(row["global_delta"], 0.25)
        self.assertEqual(row["positive_evidence_delta"], 0.25)
        self.assertEqual(row["negative_evidence_contribution"], -0.125)
        self.assertEqual(row["correction_contribution"], -0.0625)
        self.assertEqual(row["routed_delta"], 0.3125)
        self.assertEqual(summary["identity_max_abs_error"], 0)

    def test_reduced_correction_can_be_the_entire_gain(self):
        summary, pairs = self.analyze(components(1.0, correction=0.25),
                                     components(1.0, correction=0.0625))
        row = pairs[0]
        self.assertEqual(row["global_delta"], 0)
        self.assertEqual(row["correction_contribution"], 0.1875)
        self.assertEqual(summary["saved_attack_successes"]["largest_positive_component"],
                         "correction_contribution")

    def test_positive_mass_share_is_bounded_despite_cancellation(self):
        summary, _ = self.analyze(components(1.0, positive=0.5, negative=0.0, correction=0.0),
                                 components(2.0, positive=0.0, negative=0.49, correction=0.0))
        s = summary["all_pairs"]
        self.assertAlmostEqual(s["mean_routed_delta"], 0.01)
        self.assertEqual(s["positive_component_mass_shares"]["global_delta"], 1.0)
        self.assertGreater(s["mean_global_delta"] / s["mean_routed_delta"], 90)

    def test_exact_threshold_and_empty_success_cohort(self):
        summary, pairs = self.analyze(components(1.0), components(1.125))
        self.assertTrue(pairs[0]["saved_success"])
        self.assertTrue(pairs[0]["replay_success"])
        summary, _ = self.analyze(components(1.0), components(1.0625))
        self.assertEqual(summary["saved_attack_successes"], {"n_pairs": 0})

    def test_numerical_threshold_flip_is_disclosed_not_redefined(self):
        before, after = components(1.0), components(1.125 - 1e-6)
        summary, pairs = self.analyze(before, after, saved_after=before["routed"] + 0.125)
        self.assertTrue(pairs[0]["saved_success"])
        self.assertFalse(pairs[0]["replay_success"])
        self.assertEqual(summary["saved_vs_replay_success_disagreements"], 1)
        self.assertEqual(summary["near_threshold_disagreements"], 1)

    def test_identical_visible_inputs_are_reported(self):
        summary, _ = self.analyze(components(1.0), components(1.0))
        self.assertEqual(summary["all_pairs"]["identical_model_input_pairs"], 1)

    def test_nan_and_broken_component_identity(self):
        with self.assertRaises(ValueError):
            self.analyze(components(float("nan")), components(1.0))
        after = components(1.125)
        after["global"] += 0.2
        summary, _ = self.analyze(components(1.0), after)
        self.assertGreater(summary["identity_max_abs_error"], diagnostic.REPLAY_TOLERANCE)

    def test_rhi_macro_does_not_give_injection_double_weight(self):
        rows = []
        for attack, delta in zip(diagnostic.ATTACK_FILES, (0.9, 0.6, 0.2, 0.4)):
            summary, _ = self.analyze(components(1.0), components(1.0 + delta))
            rows.append({"attack": attack, **summary})
        self.assertAlmostEqual(diagnostic.rhi_macro(rows)["mean_routed_delta"], 0.6)
        with self.assertRaises(ValueError):
            diagnostic.rhi_macro(rows[:-1])


class ScorePathWorkflowTests(unittest.TestCase):
    def test_dry_run_no_torch_no_output_and_code_drift_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            out = root / "new-diagnostic"
            with patch.dict("sys.modules", {"torch": None}):
                self.assertEqual(diagnostic.main(["--evaluation-dir", str(source),
                                                 "--output-dir", str(out), "--dry-run"]), 0)
            self.assertFalse(out.exists())
            *_, drift = diagnostic.load_reference(source)
            self.assertIn("old_attack_file.py", drift)

    def test_frozen_weights_and_details_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source, checkpoint = fixture(Path(directory))
            (checkpoint / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                diagnostic.load_reference(source)
        with tempfile.TemporaryDirectory() as directory:
            source, _ = fixture(Path(directory))
            path = source / "paer" / diagnostic.ATTACK_FILES["hotflip"]
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evaluation changed"):
                diagnostic.load_reference(source)

    def test_output_guards_and_cpu_cache_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, checkpoint = fixture(root)
            out = root / "new-diagnostic"
            argv = ["--evaluation-dir", str(source), "--output-dir", str(out), "--device", "cpu"]
            before_hashes = {path: sha256(path) for path in source.rglob("*.json")}
            calls = []

            def replay(scorer, texts, *args):
                calls.append(texts)
                return {diagnostic.text_id(text): components(1.125 if text == "edited" else 1.0)
                        for text in texts}

            fake_scorer = SimpleNamespace(AESScorer=lambda *args, **kwargs: object())
            with patch.object(diagnostic, "replay_chunk", side_effect=replay), patch.dict("sys.modules", {
                    "torch": SimpleNamespace(float32="float32"),
                    "text_scoring_adv_training.evaluation.aes.scorer": fake_scorer}):
                self.assertEqual(diagnostic.main(argv), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(set(calls[0]), {"clean", "edited"})  # shared originals replayed once
            with patch.dict("sys.modules", {"torch": None}):
                self.assertEqual(diagnostic.main(argv + ["--cpu-summary-only"]), 0)
            self.assertEqual(before_hashes, {path: sha256(path) for path in before_hashes})
            report = read_json(out / "score_path_summary.json")
            self.assertEqual(report["quality_check"]["status"], "PASS")
            self.assertEqual(len(report["attacks"]), 4)
            for protected in (source, source / "child", root, checkpoint):
                with self.assertRaises(ValueError):
                    diagnostic.guard_output(protected, source, checkpoint, {})
            with self.assertRaises(ValueError):
                diagnostic.main(argv + ["--batch-size", "1", "--dry-run"])
            cache = next((out / "replay_cache").glob("texts_*.json"))
            cache.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                diagnostic.main(argv + ["--cpu-summary-only"])

    def test_unbound_existing_directory_and_incomplete_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            out = root / "new-diagnostic"
            argv = ["--evaluation-dir", str(source), "--output-dir", str(out)]
            with self.assertRaises(FileNotFoundError):
                diagnostic.main(argv + ["--cpu-summary-only"])
            self.assertFalse(out.exists())
            out.mkdir()
            (out / "old-result.txt").write_text("preserve me", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                diagnostic.main(argv + ["--dry-run"])

    def test_bad_replay_report_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = fixture(root)
            out = root / "new-diagnostic"
            argv = ["--evaluation-dir", str(source), "--output-dir", str(out), "--device", "cpu"]
            fake_scorer = SimpleNamespace(AESScorer=lambda *a, **kw: object())
            with patch.object(diagnostic, "replay_chunk", return_value={
                    diagnostic.text_id("clean"): components(2.0),
                    diagnostic.text_id("edited"): components(2.125)}), patch.dict("sys.modules", {
                        "torch": SimpleNamespace(float32="float32"),
                        "text_scoring_adv_training.evaluation.aes.scorer": fake_scorer}):
                with self.assertRaisesRegex(ValueError, "Replay/QC mismatch"):
                    diagnostic.main(argv)
            self.assertEqual(read_json(out / "score_path_summary.json")["quality_check"]["status"],
                             "FAIL_DO_NOT_INTERPRET")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tqdm")),
                     "Optional real PAER forward test requires Torch/Transformers/tqdm")
class RealScorePathForwardTests(unittest.TestCase):
    def test_extracts_components_from_actual_paer_v3_forward(self):
        import torch
        from torch import nn
        from paer.modeling_paer_v3 import PAERV3ForEssayScoring

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(12, 4)

            def forward(self, input_ids, attention_mask, **kwargs):
                h = self.embedding(input_ids)
                mask = attention_mask.unsqueeze(-1)
                context = (h * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
                return SimpleNamespace(last_hidden_state=h + context)

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.deberta = Encoder()
                self.pooler = lambda x: x[:, 0]
                self.dropout = nn.Identity()
                self.classifier = nn.Linear(4, 1)

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                ids = [[1, 3, 4, 2] if text != "edited" else [1, 6, 2] for text in texts]
                length = max(map(len, ids))
                return {"input_ids": torch.tensor([x + [0] * (length - len(x)) for x in ids]),
                        "attention_mask": torch.tensor([[1] * len(x) + [0] * (length - len(x)) for x in ids])}

        torch.manual_seed(7)
        model = PAERV3ForEssayScoring(Base(), risk_bias_init=0.0).eval()
        with torch.no_grad():
            model.token_evidence_head.weight.fill_(0.3)
        scorer = SimpleNamespace(tokenizer=Tokenizer(), model=model, device="cpu")
        texts = ["clean", "edited", "different text, identical visible tokens"]
        scores = diagnostic.replay_chunk(scorer, texts, 2, 1024, False)
        for value in scores.values():
            self.assertLess(diagnostic.validate_score(value), 1e-6)
            self.assertLessEqual(value["correction"], value["positive"] + 1e-6)
        before, after = scores[diagnostic.text_id("clean")], scores[diagnostic.text_id("edited")]
        self.assertNotEqual(before["visible_input_sha256"], after["visible_input_sha256"])
        self.assertEqual(before["visible_input_sha256"], scores[diagnostic.text_id(texts[2])]["visible_input_sha256"])
        summary, _ = diagnostic.decompose_pairs([
            {"essay_id": "0", "original_text": "clean", "perturbed_text": "edited",
             "original_score": before["routed"], "perturbed_score": after["routed"]}], scores, 0.1)
        self.assertLess(summary["identity_max_abs_error"], 1e-6)
        self.assertEqual(summary["replay_max_abs_error"], 0)


if __name__ == "__main__":
    unittest.main()
