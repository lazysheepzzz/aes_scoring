"""Figure-data tests use synthetic temporary fixtures, not experimental results."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aes_paper_figures", ROOT / "paper_results/figures/build_aes_paper_figures.py")
figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figures)


def payload(attack="rudimentary", hits=(0, 4, None, None)):
    details = []
    for i, hit in enumerate(hits):
        before, gain = 1.0, 0.2 if hit is not None else 0.0
        details.append({"essay_id": str(i), "original_text": f"synthetic fixture {i}",
                        "original_score": before, "perturbed_score": before + gain,
                        "success": hit is not None,
                        "history": [] if hit is None else [{"step": hit, "score": before + gain, "delta": gain}]})
    return {"summary": {"attack": attack, "success_threshold": 0.1,
                        "asr": sum(x is not None for x in hits)/len(hits)}, "details": details}


def make_sources(root, data):
    data["n_essays"] = 4
    configs = {"rudimentary": (0, 4, None, None), "hotflip": (2, None, None, None),
               "injection_external": (1, None, None, None), "injection_self_dup": (0, 9, None, None)}
    paths = {}
    for model in data["models"]:
        if model["id"] not in ("mixed", "paer"):
            continue
        model["asr"].update(rudimentary=.5, hotflip=.25, injection_family=.375)
        model["rhi_macro_asr"] = .375
        for family in figures.RHI:
            directory = root/model["id"]/family
            directory.mkdir(parents=True)
            manifest = {"victim": "E:/fixture/" + model["checkpoint_suffix"], "data": "valid_fold0.csv",
                        "attack": family, "n_essays": 4, "seed": 42, "success_threshold": .1,
                        "attack_parameters": {"n_steps": 30, "beam_size": 1,
                            "max_candidates_per_step": 16, "success_threshold": .1,
                            "max_token_edit_rate": .1, "n_sample_pos": 8, "top_k_per_pos": 2,
                            "sentence_bank_sha256": data["attack_settings"]["sentence_bank_sha256"]}}
            figures.save_json(directory/"run_manifest.json", manifest)
            figures.save_json(directory/"clean_qwk.json", {"qwk": model["clean_qwk"], "n": 4})
            for attack in figures.SUBTYPES if family == "injection_family" else (family,):
                path = directory/f"{attack}_details.json"
                figures.save_json(path, payload(attack, configs[attack]))
                paths[model["id"], attack] = path
    return paths


class DataTests(unittest.TestCase):
    def setUp(self):
        self.data = figures.read_json(figures.DEFAULT_DATA)

    def test_reported_snapshot_macro_and_version(self):
        figures.validate_data(self.data)
        by_id = {row["id"]: row for row in self.data["models"]}
        self.assertEqual(by_id["mixed"]["clean_qwk"], .8300)
        self.assertEqual(by_id["paer"]["clean_qwk"], .8366)

    def test_injection_is_one_family(self):
        values = figures.family_values({"rudimentary": .9, "hotflip": .6,
                                       "injection_external": .2, "injection_self_dup": .4})
        for actual, expected in zip(values, (.9, .6, .3)):
            self.assertAlmostEqual(actual, expected)

    def test_snapshot_cannot_claim_raw_verification(self):
        self.data["source_status"] = "RAW_VERIFIED"
        with self.assertRaises(ValueError):
            figures.validate_data(self.data)

    def test_first_hit_rounds_and_early_stop(self):
        curve, _ = figures.first_hit_curve(payload(), n=4, threshold=.1, max_rounds=30)
        self.assertEqual(curve[0], 0)
        self.assertEqual(curve[1], .25)
        self.assertEqual(curve[4], .25)
        self.assertEqual(curve[5], .5)
        self.assertEqual(curve[30], .5)

    def test_failed_pairs_never_become_success_at_ceiling(self):
        curve, _ = figures.first_hit_curve(payload(hits=(None, None)), n=2, threshold=.1, max_rounds=30)
        self.assertEqual(curve, [0]*31)

    def test_success_without_recorded_history_is_rejected(self):
        value = payload()
        value["details"][0]["history"] = []
        with self.assertRaises(ValueError):
            figures.first_hit_curve(value, n=4, threshold=.1, max_rounds=30)

    def test_duplicate_essay_ids_are_rejected(self):
        value = payload()
        value["details"][1]["essay_id"] = "0"
        with self.assertRaises(ValueError):
            figures.first_hit_curve(value, n=4, threshold=.1, max_rounds=30)

    def test_invalid_step_or_score_is_rejected(self):
        for key, item in (("step", 30), ("step", True), ("score", float("nan")), ("delta", .3)):
            value = payload()
            value["details"][0]["history"][0][key] = item
            with self.subTest(key=key, item=item), self.assertRaises(ValueError):
                figures.first_hit_curve(value, n=4, threshold=.1, max_rounds=30)

    def test_false_stored_success_is_rejected(self):
        value = payload()
        value["details"][0]["success"] = False
        with self.assertRaises(ValueError):
            figures.first_hit_curve(value, n=4, threshold=.1, max_rounds=30)

    def test_history_discovery_and_injection_average(self):
        with tempfile.TemporaryDirectory() as directory:
            make_sources(Path(directory), self.data)
            result = figures.collect_histories(directory, self.data)
            self.assertTrue(result["ready"])
            self.assertEqual(result["curves"]["mixed"]["injection_family"][-1], .375)
            self.assertEqual(len(result["sources"]), 8)

    def test_wrong_checkpoint_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            make_sources(Path(directory), self.data)
            path = Path(directory)/"mixed/hotflip/run_manifest.json"
            manifest = figures.read_json(path)
            manifest["victim"] = manifest["victim"].replace("gstep1000", "gstep400")
            figures.save_json(path, manifest)
            self.assertFalse(figures.collect_histories(directory, self.data)["ready"])

    def test_incompatible_search_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            make_sources(Path(directory), self.data)
            path = Path(directory)/"mixed/hotflip/run_manifest.json"
            manifest = figures.read_json(path)
            manifest["attack_parameters"]["n_steps"] = 10
            figures.save_json(path, manifest)
            result = figures.collect_histories(directory, self.data)
            self.assertFalse(result["ready"])
            self.assertTrue(result["rejected"])

    def test_different_valid_duplicates_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sources(root, self.data)
            duplicate = root/"duplicate"
            duplicate.mkdir()
            source = root/"mixed/rudimentary"
            for filename in ("run_manifest.json", "clean_qwk.json"):
                figures.save_json(duplicate/filename, figures.read_json(source/filename))
            value = figures.read_json(source/"rudimentary_details.json")
            value["details"][0]["history"][0]["step"] = 1
            figures.save_json(duplicate/"rudimentary_details.json", value)
            result = figures.collect_histories(root, self.data)
            self.assertFalse(result["ready"])
            self.assertTrue(any("ambiguous" in row["reason"] for row in result["pending"]))

    def test_nonempty_outputs_are_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path/"existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                figures.guard_output(path, figures.DEFAULT_DATA)
            self.assertEqual((path/"existing.txt").read_text(), "keep")

    def test_dry_run_creates_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"new"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(figures.main(["--output-dir", str(output), "--dry-run"]), 0)
            self.assertFalse(output.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
    def test_all_four_renderers_export_vector_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_sources(root/"raw", self.data)
            histories = figures.collect_histories(root/"raw", self.data)
            plt = figures.plotting()
            files = figures.draw_framework(plt, root) + figures.draw_heatmap(plt, root, self.data)
            files += figures.draw_routing(plt, root, self.data) + figures.draw_budget(plt, root, self.data, histories)
            self.assertEqual(len(files), 12)
            self.assertTrue(all((root/name).stat().st_size > 500 for name in files))


if __name__ == "__main__":
    unittest.main()
