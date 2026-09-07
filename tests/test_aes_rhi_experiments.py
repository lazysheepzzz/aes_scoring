"""CPU tests for balancing, input guards and orchestration; no downloaded models."""
import json
import tempfile
import unittest
import subprocess
import pandas as pd
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from paer.prepare_aes_rhi_training_traces import allocate_rhi, balance_successful_groups
from paer.rhi_experiment_utils import bind_directory, read_json, save_json, validate_trace_groups, rhi_macro, sha256
from paer.aes_rhi_training_launcher import build_parser, main as launch_training
from paer import select_aes_rhi_checkpoint as selector
from paer import evaluate_aes_rhi_experiments as evaluator
from paer import prepare_aes_rhi_training_traces as preparation
from paer.finalize_aes_rhi_training_pool import finalize
from paer.rhi_experiment_utils import checkpoint_identity


def record(i, attack, order=1):
    return {"record_id": f"{i}:{attack}:{order}", "row_index": i, "attack": attack,
            "original_text": f"Essay {i}", "before_text": f"Essay {i}",
            "adversarial_text": f"Essay {i} edited {order}", "label_score_space": 3.0,
            "step_gain": 0.05, "cumulative_delta": order * 0.05,
            "accepted_edit_order": order}


class RHIProtocolTests(unittest.TestCase):
    def test_local_recovery_can_have_signed_cumulative_gain(self):
        rows = [{"text": f"Essay {i}", "score": 3.0} for i in range(6)]
        records = [record(i, a) for i, a in enumerate(
            ("rudimentary", "rudimentary", "hotflip", "hotflip", "injection_external", "injection_self_dup"))]
        records[2]["cumulative_delta"] = -0.03
        records[3]["cumulative_delta"] = 0.0
        validate_trace_groups(records, rows)
        self.assertEqual(records[2]["cumulative_delta"], -0.03)
        for field, values in (("step_gain", (0, -0.1, float("nan"), float("inf"))),
                              ("cumulative_delta", (float("nan"), float("inf"), -float("inf")))):
            for value in values:
                with self.subTest(field=field, value=value):
                    bad = [dict(r) for r in records]
                    bad[2][field] = value
                    with self.assertRaisesRegex(ValueError, "record_id=.*hotflip"):
                        validate_trace_groups(bad, rows)

    def test_cpu_finalizer_reuses_legacy_shards_and_guards_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train, rh, bank, ckpt, out = (root / n for n in ("train.csv", "rh.jsonl", "bank.txt", "b0", "pool"))
            train.write_text("essay_id,score,full_text\n" + "".join(f"{i},3,Essay {i}\n" for i in range(60)))
            bank.write_text("External sentence.\n")
            ckpt.mkdir()
            (ckpt / "model.safetensors").write_bytes(b"frozen")
            source = [record(i, "hotflip" if i % 2 else "rudimentary") for i in range(50)]
            for r in source:
                if r["attack"] == "hotflip":
                    r["cumulative_delta"] = -0.03
            rh.write_text("".join(json.dumps(r) + "\n" for r in source))
            save_json(rh.with_suffix(".manifest.json"), {"n_steps": 3})
            rows = preparation.load_rows(train)
            groups, jobs = allocate_rhi(rows, source, 0.5, 42)
            files = ("paer/prepare_aes_rhi_training_traces.py", "paer/rhi_experiment_utils.py",
                     "text_scoring_adv_training/evaluation/aes/scorer.py",
                     "text_scoring_adv_training/evaluation/aes/attacks/injection.py")
            protocol = {
                "protocol": "balanced_rhi_positive_trajectories_v1", "mlm_excluded": True,
                "arguments": {"train_csv": str(train), "rh_traces": str(rh), "sentence_bank": str(bank),
                              "checkpoint": str(ckpt), "attack_fraction": 0.5, "seed": 42},
                "source_hashes": {k: sha256(p) for k, p in
                                  {"train": train, "rh": rh, "rh_manifest": rh.with_suffix(".manifest.json"), "bank": bank}.items()},
                "checkpoint": checkpoint_identity(ckpt),
                "generation_code": {p: sha256(preparation.ROOT / p) for p in files},
                "rh_rows": sorted(groups), "injection_jobs": {str(i): a for i, a in jobs.items()},
            }
            # A completed run bound before the validator repair.
            protocol["generation_code"]["paer/rhi_experiment_utils.py"] = "legacy-validator-hash"
            bind_directory(out, protocol)
            with self.assertRaisesRegex(FileNotFoundError, "missing shard"):
                finalize(out)
            for i, a in jobs.items():
                save_json(out / "injection_essay_progress" / f"row_{i}.json", {"records": [record(i, a)]})
            preserved = {p: sha256(p) for p in out.rglob("*.json")}
            preserved[rh] = sha256(rh)
            bank.write_text("Changed bank")
            with self.assertRaisesRegex(ValueError, "input changed: bank"):
                finalize(out)
            bank.write_text("External sentence.\n")
            with patch("paer.finalize_aes_rhi_training_pool.ROOT", root):
                with self.assertRaises(FileNotFoundError):
                    finalize(out)  # Search source cannot be silently exempted.
            with patch.dict("sys.modules", {"torch": None, "transformers": None}):
                self.assertEqual(finalize(out), 0)
                self.assertEqual(finalize(out), 0)
            self.assertEqual(preserved, {p: sha256(p) for p in preserved})
            manifest = read_json(out / "rhi_counterfactual_training_traces.manifest.json")
            self.assertEqual(list(manifest["essay_counts_by_attack"].values()), [10, 10, 5, 5])
            self.assertEqual(manifest["finalization"]["nonpositive_cumulative_trace_counts_by_attack"], {"hotflip": 10})
            target = out / "rhi_counterfactual_training_traces.jsonl"
            target.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "pool was modified"):
                finalize(out)

    def test_macro_does_not_double_weight_injection(self):
        self.assertAlmostEqual(rhi_macro(0.9, 0.6, 0.2, 0.4), 0.6)
        self.assertNotAlmostEqual(rhi_macro(0.9, 0.6, 0.2, 0.4), (0.9 + 0.6 + 0.2 + 0.4) / 4)

    def test_assignments_are_balanced_disjoint_and_reproducible(self):
        rows = [{"text": f"Essay {i}", "score": 3.0} for i in range(120)]
        rh = [record(i, "hotflip" if i % 2 else "rudimentary") for i in range(90)]
        groups, jobs = allocate_rhi(rows, rh, 0.5, 42)
        self.assertEqual(len(groups), 40)
        self.assertEqual(len(jobs), 20)
        self.assertFalse(set(groups) & set(jobs))
        self.assertEqual((groups, jobs), allocate_rhi(rows, rh, 0.5, 42))
        injected = {i: [record(i, a)] for i, a in jobs.items()}
        balanced = balance_successful_groups(groups, injected, 42)
        _, counts = validate_trace_groups(balanced, rows)
        self.assertEqual(counts, {"rudimentary": 20, "hotflip": 20,
                                  "injection_external": 10, "injection_self_dup": 10})

    def test_edit_count_and_failed_searches_do_not_skew_exposure(self):
        rows = [{"text": f"Essay {i}", "score": 3.0} for i in range(20)]
        rh = {i: [record(i, "rudimentary" if i < 4 else "hotflip", order)
                  for order in range(1, 4 if i < 4 else 2)] for i in range(8)}
        injections = {8: [record(8, "injection_external")], 9: [],
                      10: [record(10, "injection_self_dup")],
                      11: [record(11, "injection_self_dup")]}
        _, counts = validate_trace_groups(balance_successful_groups(rh, injections, 42), rows)
        self.assertEqual(list(counts.values()), [2, 2, 1, 1])

    def test_mlm_is_forbidden_and_mismatched_text_fails(self):
        rows = [{"text": "Essay 0", "score": 3.0}]
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            validate_trace_groups([record(0, "mlm_guided")], rows)
        with self.assertRaisesRegex(ValueError, "differs"):
            allocate_rhi([{"text": "Different", "score": 3.0}], [record(0, "hotflip")], 1, 42)

    def test_old_outputs_rejected_and_binding_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "old"
            old.mkdir()
            (old / "model.safetensors").write_bytes(b"historical")
            with self.assertRaises(FileExistsError):
                bind_directory(old, {"seed": 42})
            self.assertEqual((old / "model.safetensors").read_bytes(), b"historical")
            new = Path(d) / "new"
            bind_directory(new, {"seed": 42})
            bind_directory(new, {"seed": 42})
            with self.assertRaises(ValueError):
                bind_directory(new, {"seed": 43})

    def test_training_guard_runs_before_loading_data(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "historical.txt").write_text("retain")
            with patch("sys.argv", ["train", "--output-dir", d]):
                with self.assertRaises(FileExistsError):
                    launch_training("paer_rhi_v3")

    def test_generation_recovers_after_interruption_without_duplicate_records(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            train, rh, bank, ckpt, out = (root / n for n in ("train.csv", "rh.jsonl", "bank.txt", "b0", "pool"))
            train.write_text("essay_id,score,full_text\n" + "".join(f"{i},3,Essay {i}\n" for i in range(60)))
            bank.write_text("External sentence.\n")
            ckpt.mkdir()
            (ckpt / "model.safetensors").write_bytes(b"frozen")
            original = "".join(json.dumps(record(i, "hotflip" if i % 2 else "rudimentary")) + "\n" for i in range(50))
            rh.write_text(original)
            save_json(rh.with_suffix(".manifest.json"), {"n_steps": 3, "max_candidates_per_step": 16,
                       "success_threshold": 0.1, "checkpoint": str(ckpt)})
            state = {"calls": 0, "interrupt": True}

            class FakeAttack:
                def __init__(self, scorer, **kwargs):
                    pass

                def attack(self, text):
                    state["calls"] += 1
                    if state["interrupt"] and state["calls"] == 3:
                        raise RuntimeError("simulated interruption")
                    return text + " added", [{"before_text": text, "after_text": text + " added",
                                              "step_gain": 0.05, "delta": 0.05, "score": 2.05, "step": 0}]

            mock_modules = {
                "tqdm.auto": SimpleNamespace(tqdm=lambda items, **kwargs: items),
                "torch": SimpleNamespace(float32="float32", manual_seed=lambda seed: None,
                                         cuda=SimpleNamespace(is_available=lambda: False)),
                "text_scoring_adv_training.evaluation.aes.scorer": SimpleNamespace(AESScorer=lambda *a, **kw: object()),
                "text_scoring_adv_training.evaluation.aes.attacks.injection": SimpleNamespace(IterativeInjectionAttack=FakeAttack),
            }
            argv = ["prepare", "--train-csv", str(train), "--rh-traces", str(rh), "--checkpoint", str(ckpt),
                    "--sentence-bank", str(bank), "--output-dir", str(out), "--no-progress"]
            with patch("sys.argv", argv), patch.dict("sys.modules", mock_modules):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    preparation.main()
                self.assertEqual(len(list((out / "injection_essay_progress").glob("*.json"))), 2)
                state["interrupt"] = False
                self.assertEqual(preparation.main(), 0)
                self.assertEqual(state["calls"], 11)  # ten jobs plus one failed call
                self.assertEqual(preparation.main(), 0)
                self.assertEqual(state["calls"], 11)
            target = out / "rhi_counterfactual_training_traces.jsonl"
            records = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual(len({r["record_id"] for r in records}), len(records))
            self.assertEqual(rh.read_text(), original)
            _, counts = validate_trace_groups(records, [{"text": f"Essay {i}", "score": 3.0} for i in range(60)])
            self.assertEqual(list(counts.values()), [10, 10, 5, 5])

    def test_paired_defaults_match_and_smoke_flags_exist(self):
        mixed, _ = build_parser("mixed_at_rhi")
        paer, _ = build_parser("paer_rhi_v3")
        a = mixed.parse_args([])
        b = paer.parse_args(["--max-train-samples", "256", "--max-valid-samples", "32"])
        for key in ("learning_rate", "num_epochs", "gradient_accumulation_steps", "per_device_train_batch_size", "trace_jsonl"):
            self.assertEqual(getattr(a, key), getattr(b, key))
        self.assertEqual(b.per_device_train_batch_size * b.gradient_accumulation_steps, 32)
        self.assertEqual(b.max_valid_samples, 32)

    def test_rhi_selection_uses_all_three_families_and_resumes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            defense, c0, output = root / "model", root / "c0", root / "selection"
            csv = root / "dev.csv"
            csv.write_text("essay_id,prompt_name,score,full_text\nx,p,3,example\n")
            c0.mkdir()
            (c0 / "model.safetensors").write_bytes(b"c0")
            for name in ("gstep200", "gstep400"):
                (defense / name).mkdir(parents=True)
                (defense / name / "model.safetensors").write_bytes(name.encode())
            save_json(defense / "rhi_training_inputs.json", {"valid_sha256": sha256(csv)})
            calls = []

            def clean(**kw):
                save_json(kw["out_dir"] / "clean_qwk.json", {"qwk": 0.83})

            def attack(**kw):
                name = kw["args"].attack
                calls.append(name)
                save_json(kw["out_dir"] / "run_manifest.json", {})
                # Candidate 200 wins on RH but loses once Injection participates.
                asr = (0.1 if kw["checkpoint"].name == "gstep200" else 0.3)
                if name == "injection_family":
                    asr = 1.0 if kw["checkpoint"].name == "gstep200" else 0.0
                    save_json(kw["out_dir"] / "asr_summary.json", [
                        {"attack": a, "asr": asr, "avg_delta": 0.1, "n_essays": 1}
                        for a in ("injection_external", "injection_self_dup")])
                    save_json(kw["out_dir"] / "injection_family_summary.json", {"asr": asr, "avg_delta": 0.1})
                    for a in ("injection_external", "injection_self_dup"):
                        save_json(kw["out_dir"] / f"{a}_details.json", {})
                else:
                    save_json(kw["out_dir"] / "asr_summary.json", [{"attack": name, "asr": asr, "avg_delta": 0.1}])
                    save_json(kw["out_dir"] / f"{name}_details.json", {})

            argv = ["select", "--defense-output-dir", str(defense), "--c0-checkpoint", str(c0),
                    "--valid-csv", str(csv), "--selection-output-dir", str(output), "--subset-size", "1"]
            with patch("sys.argv", argv), patch.object(selector, "_evaluate_clean_if_needed", clean), \
                    patch.object(selector, "_evaluate_subset_if_needed", attack), \
                    patch.object(selector, "create_or_load_debug_subset"):
                self.assertEqual(selector.main(), 0)
                self.assertEqual(read_json(output / "best_checkpoint.json")["checkpoint_name"], "gstep400")
                self.assertEqual(calls.count("injection_family"), 2)
                self.assertNotIn("mlm_guided", calls)
                self.assertEqual(selector.main(), 0)
                self.assertEqual(len(calls), 6)  # completed selection is immutable/reused

    def test_specialist_mlm_runner_uses_selected_weights_and_all_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "dev.csv"
            data.write_text("essay_id,score,full_text\nx,3,Essay x\ny,4,Essay y\n")
            argv = ["evaluate", "--suite", "specialists-mlm", "--data", str(data),
                    "--output-dir", str(root / "results")]
            for model in ("hotflip", "rudimentary", "injection"):
                ckpt = root / model / "gstep400"
                ckpt.mkdir(parents=True)
                (ckpt / "model.safetensors").write_bytes(model.encode())
                selection = root / f"{model}.json"
                save_json(selection, {"checkpoint_path": str(ckpt)})
                argv += [f"--{model}-selection", str(selection)]
            calls = []

            def run(command, **kwargs):
                calls.append(command)
                out = Path(command[command.index("--out") + 1])
                self.assertEqual(command[command.index("--n-essays") + 1], "2")
                self.assertEqual(command[command.index("--attack") + 1], "mlm_guided")
                self.assertTrue(command[command.index("--checkpoint") + 1].endswith("gstep400"))
                save_json(out / "clean_qwk.json", {"qwk": 0.8})
                save_json(out / "asr_summary.json", [{"attack": "mlm_guided", "asr": 0.5, "avg_delta": 0.1, "n_essays": 2}])
                save_json(out / "mlm_guided_details.json", {"details": []})
                save_json(out / "run_manifest.json", {})

            with patch("sys.argv", argv), patch.object(evaluator, "subprocess", SimpleNamespace(
                    run=run, list2cmdline=subprocess.list2cmdline)):
                self.assertEqual(evaluator.main(), 0)
                self.assertEqual(evaluator.main(), 0)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(read_json(root / "results/experiment_results.json")["results"]), 3)


if __name__ == "__main__":
    unittest.main()
