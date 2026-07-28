from __future__ import annotations

import os
import unittest
from pathlib import Path

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
)
from whitebox.aes_stage2_training_launcher import (
    CLEAN_CONTINUATION,
    HOTFLIP_DEFENSE,
    build_config,
    build_parser,
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
