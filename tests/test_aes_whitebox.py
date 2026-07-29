from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from text_scoring_adv_training.evaluation.aes.evaluate import (
    AttackResult,
    evaluate_attack,
)
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.training.aes_trainer import (
    AESAdversarialConfig,
    build_adamw_parameter_groups,
    one_sided_score_inflation_loss,
    tensor_to_float_numpy,
)
from whitebox.aes_stage2_training_launcher import (
    CLEAN_CONTINUATION,
    HOTFLIP_DEFENSE,
    build_config,
    build_parser,
)
from whitebox.select_aes_hotflip_defense_checkpoint import (
    create_or_load_debug_subset,
    discover_checkpoint_candidates,
    select_best_checkpoint,
)


class _FakeScorer:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score_batch(self, texts, batch_size=32):
        return [self.scores[text] for text in texts]


class AESWhiteboxMetricsTest(unittest.TestCase):
    def test_asr_uses_delta_threshold_and_only_upward_band_crossings(self):
        result = AttackResult("hotflip", success_threshold=0.1)
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

        result.add_essay(1.40, [1.45], thresholds)
        result.add_essay(1.40, [1.60], thresholds)
        result.add_essay(2.00, [1.00], thresholds)

        summary = result.summary()
        self.assertEqual(summary["asr"], 0.3333)
        self.assertEqual(summary["upward_band_asr"], 0.3333)

    def test_score_space_thresholds_are_converted_to_label_space(self):
        result = AttackResult("hotflip", success_threshold=0.1)
        thresholds = {
            "thresholds_score_space": [1.5, 2.5],
            "label_offset": 1,
        }
        result.add_essay(0.4, [0.6], thresholds)
        self.assertEqual(result.original_band, [0])
        self.assertEqual(result.perturbed_band, [1])

    def test_evaluator_saves_text_metadata_and_history(self):
        scorer = _FakeScorer({"original": 1.0, "adversarial": 1.2})

        def attack_fn(_text):
            return "adversarial", [{"step": 0, "pos": 1}]

        result = evaluate_attack(
            attack_name="hotflip",
            attack_fn=attack_fn,
            scorer=scorer,
            essays=[
                {
                    "essay_id": "essay-1",
                    "text": "original",
                    "score": 2.0,
                    "prompt": "prompt-a",
                }
            ],
            success_threshold=0.1,
        )

        self.assertEqual(result.summary()["asr"], 1.0)
        self.assertEqual(result.details[0]["essay_id"], "essay-1")
        self.assertEqual(result.details[0]["perturbed_text"], "adversarial")
        self.assertEqual(result.details[0]["history"][0]["step"], 0)


class AESWhiteboxTrainingTest(unittest.TestCase):
    def test_one_sided_loss_does_not_push_clean_score_up(self):
        clean = torch.tensor([1.0], requires_grad=True)
        adversarial = torch.tensor([2.0], requires_grad=True)
        target = torch.tensor([1.0])

        loss = one_sided_score_inflation_loss(
            clean,
            adversarial,
            target,
            tolerance=0.05,
        )
        loss.backward()

        self.assertIsNone(clean.grad)
        self.assertGreater(float(adversarial.grad.item()), 0.0)

    def test_legacy_margin_config_maps_to_tolerance(self):
        config = AESAdversarialConfig(hotflip_margin=0.1)
        self.assertEqual(config.hotflip_tolerance, 0.1)

    def test_c0_and_hotflip_share_all_optimization_parameters(self):
        clean_args = build_parser(CLEAN_CONTINUATION).parse_args([])
        hotflip_args = build_parser(HOTFLIP_DEFENSE).parse_args([])
        clean_config = build_config(clean_args, CLEAN_CONTINUATION)
        hotflip_config = build_config(hotflip_args, HOTFLIP_DEFENSE)

        intentionally_different = {
            "training_mode",
            "output_dir",
            "use_hotflip_swaps",
        }
        shared_keys = set(clean_config) - intentionally_different
        self.assertEqual(shared_keys, set(hotflip_config) - intentionally_different)
        for key in shared_keys:
            self.assertEqual(clean_config[key], hotflip_config[key], key)

        self.assertFalse(clean_config["use_hotflip_swaps"])
        self.assertTrue(hotflip_config["use_hotflip_swaps"])
        self.assertEqual(clean_config["precision"], "bfloat16")

    def test_adamw_excludes_bias_and_normalization_vectors_from_decay(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.LayerNorm(3),
        )
        groups = build_adamw_parameter_groups(model, weight_decay=0.01)

        decay_ids = {id(parameter) for parameter in groups[0]["params"]}
        no_decay_ids = {id(parameter) for parameter in groups[1]["params"]}

        self.assertIn(id(model[0].weight), decay_ids)
        self.assertIn(id(model[0].bias), no_decay_ids)
        self.assertIn(id(model[1].weight), no_decay_ids)
        self.assertIn(id(model[1].bias), no_decay_ids)
        self.assertEqual(groups[0]["weight_decay"], 0.01)
        self.assertEqual(groups[1]["weight_decay"], 0.0)

    def test_bfloat16_evaluation_output_is_numpy_compatible(self):
        values = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)

        converted = tensor_to_float_numpy(values)

        self.assertEqual(converted.dtype.name, "float32")
        self.assertEqual(converted.tolist(), [0.25, 0.75])


class AESCheckpointSelectionTest(unittest.TestCase):
    def test_discovers_saved_training_checkpoints_in_step_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            for name in ("final", "gstep400", "best", "gstep200", "notes"):
                candidate = output_dir / name
                candidate.mkdir()
                if name != "notes":
                    (candidate / "model.safetensors").write_bytes(name.encode())

            candidates = discover_checkpoint_candidates(output_dir)

            self.assertEqual(
                [candidate.name for candidate in candidates],
                ["gstep200", "gstep400", "best", "final"],
            )

    def test_selects_lowest_asr_after_clean_qwk_gate(self):
        rows = [
            {
                "checkpoint_name": "gstep200",
                "clean_qwk": 0.82,
                "subset_asr": 0.70,
            },
            {
                "checkpoint_name": "gstep400",
                "clean_qwk": 0.84,
                "subset_asr": 0.70,
            },
            {
                "checkpoint_name": "final",
                "clean_qwk": 0.80,
                "subset_asr": 0.10,
            },
        ]

        selected = select_best_checkpoint(rows, minimum_qwk=0.815)

        self.assertEqual(selected["checkpoint_name"], "gstep400")

    def test_debug_subset_is_fixed_and_stratified(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid_csv = root / "valid.csv"
            ids_path = root / "debug_subset_ids.json"
            subset_csv = root / "debug_subset.csv"
            rows = []
            for prompt in ("p1", "p2"):
                for score in (1, 2):
                    for index in range(10):
                        rows.append(
                            {
                                "essay_id": f"{prompt}-{score}-{index}",
                                "prompt_name": prompt,
                                "score": score,
                                "full_text": "essay",
                            }
                        )
            pd.DataFrame(rows).to_csv(valid_csv, index=False)

            first = create_or_load_debug_subset(
                valid_csv,
                ids_path,
                subset_csv,
                subset_size=20,
                seed=42,
            )
            second = create_or_load_debug_subset(
                valid_csv,
                ids_path,
                subset_csv,
                subset_size=20,
                seed=999,
            )
            subset = pd.read_csv(subset_csv)

            self.assertEqual(first["essay_ids"], second["essay_ids"])
            self.assertEqual(
                subset.groupby(["prompt_name", "score"]).size().tolist(),
                [5, 5, 5, 5],
            )


@unittest.skipUnless(
    os.environ.get("RUN_AES_INTEGRATION_TESTS") == "1"
    and Path("deberta_checkpoints/fold0_best").is_dir(),
    "set RUN_AES_INTEGRATION_TESTS=1 with the local checkpoint to run",
)
class AESPaddingIntegrationTest(unittest.TestCase):
    def test_single_and_mixed_length_batch_scores_match(self):
        scorer = AESScorer(
            "deberta_checkpoints/fold0_best",
            device="cpu",
            dtype=torch.float32,
        )
        short = "This essay has a clear claim and one supporting reason."
        long = (
            "Students should use technology responsibly because it affects "
            "school, work, and communication. "
        ) * 25

        single_score = scorer.score_single(short)
        batch_score = scorer.score_batch([short, long], batch_size=2)[0]

        self.assertEqual(scorer.tokenizer.padding_side, "right")
        self.assertLess(abs(single_score - batch_score), 1e-5)


if __name__ == "__main__":
    unittest.main()
