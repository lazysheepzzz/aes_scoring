"""CPU orchestration tests; optional real Torch score/gradient regression test."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paer import analyze_aes_rhi_threshold_flips as flips
from paer import evaluate_aes_paer_route_off as adaptive
from paer import routing_experiment_utils as utils
from paer.rhi_experiment_utils import read_json, save_json, sha256


def fixture(root, n=3):
    source, checkpoint = root / "source", root / "checkpoint"
    checkpoint.mkdir()
    data = root / "data.csv"
    data.write_text("essay_id,full_text,score\n" + "".join(f"{i},Essay {i},3\n" for i in range(n)))
    reference = {"n_essays": n, "data": str(data), "seed": 42, "steps": 30,
                 "dtype": "float32", "device": "cuda", "batch_size": 4,
                 "max_length": 1024, "success_threshold": 0.125, "evaluation_role": "development"}
    for family in ("rudimentary", "hotflip", "injection_family"):
        out = source / "paer" / family
        save_json(out / "run_manifest.json", {"victim": str(checkpoint), "seed": 42, "n_essays": n,
                  "attack_parameters": {"n_steps": 30, "success_threshold": 0.125}})
        metrics = {"asr": 0.5, "avg_delta": 0.1, "n_essays": n}
        save_json(out / "clean_qwk.json", {"qwk": 0.83})
        save_json(out / "asr_summary.json", [metrics])
        if family == "injection_family":
            save_json(out / "injection_family_summary.json", metrics)
    for attack, name in utils.ATTACK_FILES.items():
        records = [{"essay_id": str(i), "original_text": f"Essay {i}",
                    "perturbed_text": f"Essay {i} edited", "original_score": 0.875,
                    "perturbed_score": 1.0} for i in range(n)]
        save_json(source / "paer" / name, {"summary": {"attack": attack, "success_threshold": 0.125}, "details": records})
    return source, checkpoint, reference


class RoutingTests(unittest.TestCase):
    def test_flip_directions_exact_threshold_and_identity(self):
        scores = {"clean": (0.875, 1.0, 0.125),
                  "harm": (1.0, 1.0625, 0.0625),
                  "help": (0.9375, 1.25, 0.3125),
                  "fail": (0.875, 1.0, 0.125),
                  "success": (1.125, 1.25, 0.125)}
        records = [{"original_text": "clean", "perturbed_text": text,
                    "original_score": 0.875, "perturbed_score": scores[text][0]} for text in ("harm", "help", "fail", "success")]
        summary, pairs = flips.paired_flips(records, scores, threshold=0.125)
        self.assertEqual(set(summary["transition_counts"].values()), {1})
        self.assertEqual(summary["identity_max_abs_error"], 0)
        self.assertEqual(summary["fixed_set_asr_reduction_from_routing"], 0)
        self.assertTrue(pairs[0]["routed_success"])  # equality uses >=
        self.assertTrue(pairs[0]["near_threshold"])
        self.assertGreater(summary["mean_correction_lift"], 0)

    def test_family_macro_not_four_way_mean(self):
        rows = [{"attack": a, "n_pairs": 10, "value": v} for a, v in
                zip(utils.ATTACK_FILES, (0.9, 0.6, 0.2, 0.4))]
        self.assertAlmostEqual(utils.equal_family_average(rows, ("value",))["value"], 0.6)
        with self.assertRaises(ValueError):
            utils.equal_family_average(rows[:-1], ("value",))
        rows[0]["n_pairs"] = 2
        with self.assertRaises(ValueError):
            utils.equal_family_average(rows, ("value",))

    def test_positive_mean_lift_can_still_increase_asr(self):
        scores = {"clean": (0.875, 1.0, 0.125), "harm": (1.0, 1.0625, 0.0625),
                  "help": (0.9375, 1.25, 0.3125)}
        records = [{"original_text": "clean", "perturbed_text": t,
                    "original_score": 0.875, "perturbed_score": scores[t][0]} for t in ("harm", "harm", "help")]
        summary, _ = flips.paired_flips(records, scores, threshold=0.125)
        self.assertGreater(summary["mean_correction_lift"], 0)
        self.assertAlmostEqual(summary["fixed_set_asr_reduction_from_routing"], -1 / 3)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            flips.paired_flips([{"original_text": "a", "perturbed_text": "a"}], {"a": [float("nan"), 1, 0]})

    def test_cached_replay_cpu_only_and_tamper_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, checkpoint, reference = fixture(root)
            out = root / "flips"
            argv = ["flips", "--evaluation-dir", str(source), "--output-dir", str(out)]
            calls = []

            def score(scorer, texts, **kwargs):
                calls.append(texts)
                return {t: (1.0, 1.125, 0.125) if t.endswith("edited") else (0.875, 1.0, 0.125) for t in texts}

            fake = SimpleNamespace(AESScorer=lambda *a, **kw: object(), _score_unique_texts=score)
            with patch.object(flips, "frozen_reference", return_value=(reference, checkpoint)):
                with patch("sys.argv", argv + ["--cpu-summary-only"]):
                    with self.assertRaises(FileNotFoundError):
                        flips.main()
                with patch("sys.argv", argv), patch.dict("sys.modules", {
                        "torch": SimpleNamespace(float32="float32"),
                        "paer.analyze_aes_paer_routing_contribution": fake}):
                    self.assertEqual(flips.main(), 0)
                with patch("sys.argv", argv + ["--cpu-summary-only"]), patch.dict("sys.modules", {"torch": None}):
                    self.assertEqual(flips.main(), 0)
                self.assertEqual(len(calls), 1)
                cache = next((out / "replay_cache").glob("texts_*.json"))
                cache.write_text("tampered")
                with patch("sys.argv", argv + ["--cpu-summary-only"]):
                    with self.assertRaises(ValueError):
                        flips.main()

    def test_adaptive_failure_resume_smoke_and_output_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source, checkpoint, reference = fixture(root)
            out = root / "adaptive"
            argv = ["adaptive", "--evaluation-dir", str(source), "--output-dir", str(out), "--n-essays", "2"]
            calls, state = [], {"fail": True, "expected_n": 2}

            def run(command, **kwargs):
                config = read_json(command[-1])
                family = config["attack"]
                calls.append(family)
                self.assertEqual(config["n_steps"], 30)
                self.assertEqual(config["n_essays"], state["expected_n"])
                if family == "hotflip" and state["fail"]:
                    state["fail"] = False
                    raise RuntimeError("simulated interruption")
                target = Path(config["out"])
                metrics = {"asr": 0.5, "avg_delta": 0.1, "n_essays": config["n_essays"]}
                save_json(target / "clean_qwk.json", {"qwk": 0.8})
                save_json(target / "asr_summary.json", [metrics])
                save_json(target / "run_manifest.json", {"scoring_mode": "route_off", "n_essays": config["n_essays"]})
                if family == "injection_family":
                    save_json(target / "injection_family_summary.json", metrics)
                for attack, name in utils.ATTACK_FILES.items():
                    if Path(name).parent.name == family:
                        save_json(target / Path(name).name, {"details": []})

            originals = {p: sha256(p) for p in source.rglob("*.json")}
            with patch.object(adaptive, "frozen_reference", return_value=(reference, checkpoint)), patch.object(adaptive.subprocess, "run", side_effect=run):
                with patch("sys.argv", argv + ["--dry-run"]):
                    self.assertEqual(adaptive.main(), 0)
                self.assertFalse(out.exists())
                with patch("sys.argv", argv):
                    with self.assertRaisesRegex(RuntimeError, "simulated"):
                        adaptive.main()
                    self.assertEqual(adaptive.main(), 0)
                    self.assertEqual(adaptive.main(), 0)
                self.assertEqual(calls, ["rudimentary", "hotflip", "hotflip", "injection_family"])
                result = read_json(out / "adaptive_route_off_summary.json")
                self.assertTrue(all(s["routed_reference"] is None for s in result["attacks"]))
                self.assertNotIn("mlm", calls)
                with patch("sys.argv", argv[:-2]):
                    with self.assertRaises(ValueError):
                        adaptive.main()  # Cannot mix full and smoke outputs.
                state["expected_n"] = 3
                full_out = root / "full"
                with patch("sys.argv", ["adaptive", "--evaluation-dir", str(source), "--output-dir", str(full_out)]):
                    self.assertEqual(adaptive.main(), 0)
                full_result = read_json(full_out / "adaptive_route_off_summary.json")
                self.assertTrue(all(s["routed_reference"] is not None for s in full_result["attacks"]))
                self.assertEqual(full_result["attacks"][0]["routed_reference_clean"]["qwk"], 0.83)
            self.assertEqual(originals, {p: sha256(p) for p in originals})

    def test_frozen_reference_rejects_mutated_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"new weights")
            save_json(root / "rhi_run_binding.json", {"protocol": "frozen_checkpoint_evaluation_v1", "suite": "rhi",
                      "checkpoints": {"paer": {"path": str(checkpoint), "files": {"model.safetensors": "old hash"}}}})
            with self.assertRaisesRegex(ValueError, "weights/config"):
                utils.frozen_reference(root)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "Torch/Transformers unavailable locally; run this CPU test on training server")
class RealTorchRoutingTests(unittest.TestCase):
    def test_route_off_score_and_embedding_gradient_keep_signed_evidence(self):
        import torch
        from torch import nn
        from paer.modeling_paer_v3 import PAERV3ForEssayScoring

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(20, 4)

            def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids) if inputs_embeds is None else inputs_embeds)

        class Base(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(hidden_size=4)
                self.deberta = Encoder()
                self.pooler = lambda x: x[:, 0]
                self.dropout = nn.Identity()
                self.classifier = nn.Linear(4, 1)

            def get_input_embeddings(self):
                return self.deberta.embedding

        torch.manual_seed(42)
        model = PAERV3ForEssayScoring(Base(), risk_bias_init=0.0).eval()
        with torch.no_grad():
            model.token_evidence_head.weight.fill_(0.1)
            model.token_evidence_head.bias.fill_(0.5)
        x = torch.full((1, 5, 4), 0.25, requires_grad=True)
        mask = torch.ones((1, 5), dtype=torch.long)
        original = model(inputs_embeds=x, attention_mask=mask)
        expected_grad = torch.autograd.grad(original.base_logits.sum(), x, retain_graph=True)[0]
        routed_grad = torch.autograd.grad(original.logits.sum(), x)[0]
        expected_score = original.base_logits.detach()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        adaptive.disable_routing(model)
        output = model(inputs_embeds=x, attention_mask=mask)
        actual_grad = torch.autograd.grad(output.logits.sum(), x)[0]
        torch.testing.assert_close(output.logits, expected_score)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertFalse(torch.allclose(actual_grad, routed_grad))
        self.assertFalse(torch.allclose(output.logits, output.global_logits))
        self.assertEqual(output.correction.count_nonzero().item(), 0)
        for k, v in state.items():
            torch.testing.assert_close(v, model.state_dict()[k])


if __name__ == "__main__":
    unittest.main()
