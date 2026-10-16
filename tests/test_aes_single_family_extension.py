"""CPU orchestration tests: no downloaded model or GPU training is used."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paer import run_aes_single_family_extension as study
from paer.aes_rh_training_launcher import build_parser, build_config, PAER_RH_V3
from paer.rhi_experiment_utils import (
    RHI_ATTACKS, checkpoint_identity, read_json, save_json, sha256, training_families,
)


class SingleFamilyExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.train = self.root / "source/train.csv"
        self.valid = self.root / "source/valid.csv"
        self.trace = self.root / "pool/traces.jsonl"
        self.bank = self.root / "bank.txt"
        self.b0 = self.root / "b0"
        self.c0 = self.root / "c0"
        self.output = self.root / "new_study"
        self.train.parent.mkdir()
        self.trace.parent.mkdir()
        self.train.write_text("essay_id,score,full_text\n" + "".join(f"{i},3,Essay {i}\n" for i in range(60)))
        self.valid.write_text("essay_id,score,full_text\n" + "".join(f"v{i},3,Validation {i}\n" for i in range(8)))
        self.bank.write_text("An external sentence.\n")
        for path in (self.b0, self.c0):
            path.mkdir()
            (path / "model.safetensors").write_bytes(path.name.encode())
        self.records = []
        for attack, indices in (("rudimentary", range(6, 14)), ("hotflip", range(20, 28)),
                                ("injection_external", range(30, 34)), ("injection_self_dup", range(34, 38))):
            for i in indices:
                self.records.append({"record_id": f"{attack}:{i}", "row_index": i, "attack": attack,
                                     "original_text": f"Essay {i}", "before_text": f"Essay {i}",
                                     "adversarial_text": f"Essay {i} edited", "label_score_space": 3.0,
                                     "step_gain": 0.05, "cumulative_delta": -0.03 if attack == "hotflip" else 0.05,
                                     "accepted_edit_order": 1})
        self.trace.write_text("".join(json.dumps(r) + "\n" for r in self.records))
        config = build_config(build_parser(PAER_RH_V3).parse_args([]), PAER_RH_V3)
        config.update(checkpoint_path=str(self.b0), train_csv=str(self.train), valid_csv=str(self.valid),
                      trace_jsonl=str(self.trace), output_dir=str(self.reference), training_attacks=list(RHI_ATTACKS),
                      experiment_name="paer_rhi_v3")
        save_json(self.reference / "launcher_config.json", config)
        inputs = {"train_sha256": sha256(self.train), "valid_sha256": sha256(self.valid),
                  "trace_sha256": sha256(self.trace), "checkpoint": checkpoint_identity(self.b0),
                  "essay_counts_by_attack": dict(zip(RHI_ATTACKS, (8, 8, 4, 4)))}
        save_json(self.reference / "rhi_training_inputs.json", inputs)
        save_json(self.trace.with_suffix(".manifest.json"), {
            "trace_sha256": sha256(self.trace), "mlm_excluded": True,
            "arguments": {"max_length": 1024}, "source_hashes": {"bank": sha256(self.bank)}})

    def argv(self, *extra):
        return ["--reference-training-dir", str(self.reference), "--output-dir", str(self.output),
                "--sentence-bank", str(self.bank), "--c0-checkpoint", str(self.c0), *extra]

    def args(self, *extra):
        return study.build_parser().parse_args(self.argv(*extra))

    def test_exposure_and_legacy_sets(self):
        for family, attacks in study.FAMILIES.items():
            expected = "injection_family" if family == "injection" else family
            self.assertEqual(training_families(attacks), [expected])
        self.assertEqual(training_families(("rudimentary", "hotflip")), ["rudimentary", "hotflip"])
        self.assertEqual(training_families(RHI_ATTACKS), list(study.EVALUATION_ATTACKS))
        for attacks in (("mlm_guided",), ("injection_external",), ("rudimentary", "rudimentary"),
                        ("hotflip", "injection_self_dup")):
            with self.assertRaises(ValueError):
                training_families(attacks)

    def test_full_pool_isolated_and_paired_without_rewriting_source(self):
        protected = {p: sha256(p) for p in (self.train, self.valid, self.trace, self.bank)}
        args = self.args()
        plans, protocol, valid = study.prepare(args)
        self.assertEqual(valid, self.valid)
        for family, allowed in study.FAMILIES.items():
            a, b = plans[(family, "at")], plans[(family, "paer")]
            differences = {k for k in a["config"] if a["config"][k] != b["config"][k]}
            self.assertEqual(differences, {"training_mode", "experiment_name", "output_dir"})
            self.assertEqual(a["inputs"], b["inputs"])
            records = [json.loads(line) for line in Path(a["config"]["trace_jsonl"]).read_text().splitlines()]
            self.assertEqual({r["attack"] for r in records}, set(allowed))
            self.assertEqual(a["inputs"]["n_attacked_essays"], 8)
            self.assertEqual(a["inputs"]["n_clean_essays"], 60)
            if family == "hotflip":
                self.assertTrue(all(r["cumulative_delta"] == -0.03 for r in records))
        self.assertFalse(protocol["mlm_used_for_training_or_selection"])
        self.assertEqual([protocol["selection"][k] for k in ("steps", "subset_size", "max_checkpoint_step")], [30, 256, 1400])
        study.prepare(args)  # Identical preparation is safe to repeat.
        self.assertEqual(protected, {p: sha256(p) for p in protected})

    def test_smoke_contains_all_families_and_remaps_exact_rows(self):
        args = self.args("--smoke", "--smoke-train-samples", "18", "--smoke-valid-samples", "6")
        plans, _, _ = study.prepare(args)
        selected = read_json(self.output / "data/source_row_indices.json")
        self.assertEqual(len(selected), 18)
        for family in study.FAMILIES:
            plan = plans[(family, "paer")]
            self.assertEqual(plan["inputs"]["n_attacked_essays"], 6)
            records = [json.loads(line) for line in Path(plan["config"]["trace_jsonl"]).read_text().splitlines()]
            for record in records:
                self.assertEqual(record["source_row_index"], selected[record["row_index"]])
                self.assertEqual(record["original_text"], f"Essay {selected[record['row_index']]}")
            if family == "injection":
                self.assertEqual(plan["inputs"]["essay_counts_by_attack"], {"injection_external": 3, "injection_self_dup": 3})
        study.prepare(args)  # CSV CRLF on Windows must not break safe reuse.

    def test_windows_newlines_are_preserved_on_reuse(self):
        path = self.root / "artifact.txt"
        study.immutable_text(path, "first\r\nsecond\r\n")
        study.immutable_text(path, "first\r\nsecond\r\n")
        self.assertEqual(path.read_bytes(), b"first\r\nsecond\r\n")

    def test_generated_configs_are_accepted_by_actual_trainer(self):
        from paer.aes_rh_trainer import RHTrainingConfig
        plans, _, _ = study.prepare(self.args())
        for plan in plans.values():
            config = RHTrainingConfig(**plan["config"])
            self.assertEqual(tuple(config.training_attacks), tuple(plan["inputs"]["training_attacks"]))

    def test_nonempty_foreign_output_and_input_tampering_rejected(self):
        self.output.mkdir()
        sentinel = self.output / "historical.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            study.prepare(self.args())
        self.assertEqual(sentinel.read_text(), "preserve")
        args = self.args()
        args.output_dir = self.root / "fresh"
        study.prepare(args)
        trace = args.output_dir / "data/hotflip_training_traces.jsonl"
        trace.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Saved input differs"):
            study.prepare(args)

    def test_source_hash_and_dry_run_guards(self):
        args = self.args("--dry-run")
        self.assertIsNone(study.prepare(args))
        self.assertFalse(self.output.exists())
        self.train.write_text(self.train.read_text() + "99,3,Extra\n")
        with self.assertRaisesRegex(ValueError, "Reference source changed"):
            study.prepare(self.args())

    def test_partial_training_is_preserved(self):
        args = self.args()
        plans, _, _ = study.prepare(args)
        plan = plans[("hotflip", "paer")]
        path = plan["training_dir"] / "training_diagnostics.jsonl"
        path.write_text("partial run\n")
        with patch.object(study, "run_child") as child:
            with self.assertRaisesRegex(RuntimeError, "Partial training preserved"):
                study.train_model(plan, args)
            child.assert_not_called()
        self.assertEqual(path.read_text(), "partial run\n")

    def fake_child(self, command, args):
        command = [str(v) for v in command]
        self.calls.append(command)
        script = Path(command[1]).name
        value = lambda flag: command[command.index(flag) + 1]
        if script == "aes_rh_trainer.py":
            config = read_json(value("--config"))
            directory = Path(config["output_dir"])
            for tag in ("best", "final", "gstep200"):
                (directory / tag).mkdir()
                (directory / tag / "model.safetensors").write_bytes(f"{config['experiment_name']}:{tag}".encode())
            save_json(directory / "training_config.json", config)
            save_json(directory / "final_clean_metrics.json", {"qwk": 0.83})
        elif script == "select_aes_rhi_checkpoint.py":
            self.assertNotIn("mlm_guided", command)
            for flag in ("--selection-steps", "--rudimentary-selection-steps", "--injection-selection-steps"):
                self.assertEqual(value(flag), "30")
            directory = Path(value("--selection-output-dir"))
            checkpoint = Path(value("--defense-output-dir")) / "gstep200"
            save_json(directory / "best_checkpoint.json", {"checkpoint_path": str(checkpoint), "checkpoint_name": "gstep200"})
            save_json(directory / "rhi_run_binding.json", {"candidate_weights": {"gstep200": checkpoint_identity(checkpoint)}})
            (directory / "selection_subset_256.csv").write_text("same subset\n")
        elif script == "evaluate_aes_checkpoint.py":
            directory = Path(value("--out"))
            attack = value("--attack")
            metric = {"attack": attack, "asr": {"rudimentary": 0.9, "hotflip": 0.6, "injection_family": 0.3, "mlm_guided": 0.5}[attack],
                      "avg_delta": 0.1, "n_essays": int(value("--n-essays"))}
            save_json(directory / "clean_qwk.json", {"qwk": 0.83, "n": metric["n_essays"]})
            save_json(directory / "asr_summary.json", [metric])
            save_json(directory / "run_manifest.json", {"victim": value("--checkpoint"), "seed": int(value("--seed")),
                      "n_essays": metric["n_essays"], "attack_parameters": {"n_steps": int(value("--n-steps"))}})
            for name in study.result_files(attack):
                if not (directory / name).exists():
                    save_json(directory / name, metric if name == "injection_family_summary.json" else {})
        else:
            self.fail(f"Unexpected child: {command}")

    def test_end_to_end_six_models_resume_and_mlm_is_evaluation_only(self):
        self.calls = []
        argv = ["extension", *self.argv("--stage", "all", "--include-mlm")]
        with patch("sys.argv", argv), patch.object(study, "run_child", self.fake_child):
            self.assertEqual(study.main(), 0)
            training_calls = [c for c in self.calls if Path(c[1]).name == "aes_rh_trainer.py"]
            eval_calls = [c for c in self.calls if Path(c[1]).name == "evaluate_aes_checkpoint.py"]
            self.assertEqual((len(training_calls), len(eval_calls)), (6, 24))
            self.assertEqual(study.main(), 0)
            self.assertEqual(len([c for c in self.calls if Path(c[1]).name == "aes_rh_trainer.py"]), 6)
            self.assertEqual(len([c for c in self.calls if Path(c[1]).name == "evaluate_aes_checkpoint.py"]), 24)
        summary = read_json(self.output / "single_family_results.json")
        self.assertEqual(len(summary["results"]), 6)
        for row in summary["results"]:
            self.assertAlmostEqual(row["rhi_macro_asr"], 0.6)
        self.assertFalse(summary["smoke"])
        # Completed results cannot be silently reused after modification.
        (self.output / "hotflip/paer/evaluation/hotflip/asr_summary.json").write_text("[]")
        with patch("sys.argv", argv), patch.object(study, "run_child", self.fake_child):
            with self.assertRaisesRegex(ValueError, "(Saved evaluation|Completed summary source) changed"):
                study.main()


if __name__ == "__main__":
    unittest.main()
