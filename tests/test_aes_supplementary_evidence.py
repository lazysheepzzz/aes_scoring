"""Synthetic artifact tests; these are not project measurements."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paer import audit_aes_supplementary_evidence as audit
from paer import analyze_aes_paer_score_paths as replay
from paer.rhi_experiment_utils import read_json, save_json, sha256
from paper_results.bootstrap_paer_v3_vs_mixed_at import paired_statistics
from test_aes_paer_score_paths import components, fixture


class SupplementaryEvidenceTests(unittest.TestCase):
    def test_joint_macro_weights_and_scores(self):
        delta = [0.0, 0.125]
        zero = [0.0, 0.0]
        attacks = {"rudimentary": [delta, delta], "hotflip": [zero, zero],
                   "injection_external": [delta, zero], "injection_self_dup": [delta, zero]}
        stats = paired_statistics([1, 3], [[1, 3], [1, 3]], attacks,
                                  threshold=0.1, n_bootstrap=300, seed=42)
        macro = stats["contrasts"]["rhi_macro.asr"]
        self.assertAlmostEqual(macro["mixed"], 1 / 3)
        self.assertAlmostEqual(macro["paer"], 1 / 6)
        self.assertAlmostEqual(stats["contrasts"]["clean.qwk"]["mixed"], 1)
        self.assertEqual([x["true_score"] for x in stats["score_strata"]], [2, 4])

    def test_inventory_uses_victim_not_directory_name_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data/valid_fold0.csv"
            data.parent.mkdir()
            data.write_text("essay_id,text,score\na,one,3\nb,two,4\n", encoding="utf-8")
            outputs = root / "outputs"
            key = "without_routing"
            train_pattern, selection_pattern = audit.TARGETS[key]
            selection = outputs / selection_pattern.format(seed=42)
            checkpoint = outputs / train_pattern.format(seed=42) / "gstep1000"
            save_json(selection / "best_checkpoint.json", {
                "checkpoint_path": str(checkpoint), "checkpoint_name": "gstep1000",
                "clean_qwk": 0.8, "selection_csv_sha256": sha256(data), "mlm_used_for_selection": False,
            })
            save_json(selection / "rhi_run_binding.json", {
                "protocol": "rhi_selection_v1", "valid_sha256": sha256(data),
                "mlm_used_for_selection": False, "candidate_weights": {"gstep1000": {"test": "fixture"}},
                "arguments": {"selection_steps": 30, "rudimentary_selection_steps": 30,
                              "injection_selection_steps": 30, "subset_size": 2,
                              "defense_output_dir": str(checkpoint.parent)},
            })
            params = {"n_steps": 30, "beam_size": 1, "max_candidates_per_step": 16, "max_token_edit_rate": 0.1}
            summary = {"attack": "rudimentary", "n_essays": 2, "asr": 0.5,
                       "avg_delta": 0.0625, "success_threshold": 0.1}
            details = []
            for essay, body, label, score, gain in (("a", "one", 3, 2, 0.125), ("b", "two", 4, 3, 0)):
                details.append({"essay_id": essay, "original_text": body, "perturbed_text": body + " edited",
                                "true_score": label, "original_score": score, "perturbed_score": score + gain,
                                "delta": gain, "success": gain >= 0.1})
            directory = outputs / "legacy_label_gstep400"
            manifest = {"victim": str(checkpoint), "data": str(data), "seed": 42, "n_essays": 2,
                        "attack": "rudimentary", "success_threshold": 0.1, "attack_parameters": params}
            save_json(directory / "run_manifest.json", manifest)
            save_json(directory / "rudimentary_details.json", {"summary": summary, "details": details[::-1]})
            save_json(directory / "clean_qwk.json", {"n": 2, "qwk": 1, "mae": 0, "rmse": 0})
            save_json(directory / "asr_summary.json", [summary])
            save_json(outputs / "actually_gstep400/run_manifest.json",
                      {**manifest, "victim": str(checkpoint.with_name("gstep400"))})
            report = root / "report.json"
            cli = ["--project-root", str(root), "--inventory-only", "--report", str(report)]
            with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, {"torch": None}):
                self.assertEqual(audit.main(cli), 0)
                with self.assertRaises(FileExistsError):
                    audit.main(cli)
            parsed = read_json(report)
            row = parsed["coverage"][key]["rudimentary"]
            self.assertEqual(row["status"], "LEGACY_NO_FULL_BINDING")
            self.assertEqual(len(row["candidates"]), 1)
            self.assertEqual(row["candidates"][0]["metrics"]["rudimentary"]["asr"], 0.5)
            self.assertEqual(parsed["coverage"][key]["hotflip"]["status"], "NOT_FOUND_OR_NOT_VERIFIED")
            details[0]["true_score"] = 4
            save_json(directory / "rudimentary_details.json", {"summary": summary, "details": details})
            with contextlib.redirect_stdout(io.StringIO()):
                audit.main(cli[:-1] + [str(root / "rejected.json")])
            self.assertIn("changed essay", read_json(root / "rejected.json")["rejected"][0]["reason"])

    def test_nonzero_absolute_negative_even_when_negative_delta_is_zero(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"torch": None}):
            root = Path(temporary)
            source, checkpoint = fixture(root)
            reference, _, _, _, hashes, _ = replay.load_reference(source)
            destination = root / "diagnostic"
            binding = {"diagnostic": "paer_v3_fixed_set_score_paths_v1", "reference": reference,
                       "saved_inputs_sha256": hashes, "reference_binding_sha256": sha256(source / "rhi_run_binding.json"),
                       "n_unique_texts": 2, "cache_shard_size": 128}
            save_json(destination / "rhi_run_binding.json", binding)
            save_json(destination / "score_path_summary.json",
                      {"protocol": binding, "quality_check": {"status": "PASS"}})
            path = destination / "replay_cache/texts_000000.json"
            save_json(path, {audit.text_hash("clean"): components(1.0),
                             audit.text_hash("edited"): components(1.125)})
            save_json(path.with_suffix(".sha256.json"), {"sha256": sha256(path)})
            selected = {"checkpoint": str(checkpoint), "files": reference["checkpoints"]["paer"]["files"]}
            result = audit.summarize_replay(destination, source, selected, reference["data_sha256"])
            self.assertEqual(result["cohorts"]["clean_unique_texts"]["negative"]["mean"], 0.125)
            self.assertEqual(result["cohorts"]["hotflip"]["perturbed"]["negative"]["positive_n_gt_1e_6"], 1)
            save_json(path, {})
            with self.assertRaisesRegex(ValueError, "shard changed"):
                audit.summarize_replay(destination, source, selected, reference["data_sha256"])


if __name__ == "__main__":
    unittest.main()
