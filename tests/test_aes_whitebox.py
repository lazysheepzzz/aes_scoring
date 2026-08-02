from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from text_scoring_adv_training.evaluation.aes.attacks.rudimentary import (
    IterativeRudimentaryAttack,
)
from text_scoring_adv_training.evaluation.aes.evaluate import (
    AttackResult,
    evaluate_attack,
)
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.training.aes_trainer import (
    AESAdversarialConfig,
    build_adamw_parameter_groups,
    one_sided_score_inflation_loss,
    run_rudimentary_one_row,
    tensor_to_float_numpy,
)
from whitebox.aes_stage2_training_launcher import (
    CLEAN_CONTINUATION,
    HOTFLIP_DEFENSE,
    RUDIMENTARY_DEFENSE,
    build_config,
    build_parser,
)
from whitebox.eval_hotflip_defended import (
    build_parser as build_evaluation_parser,
)
from whitebox.select_aes_hotflip_defense_checkpoint import (
    build_parser as build_selection_parser,
    create_or_load_debug_subset,
    discover_checkpoint_candidates,
    select_best_checkpoint,
    stratified_sample_indices,
)


class _FakeScorer:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score_batch(self, texts, batch_size=32):
        return [self.scores[text] for text in texts]


class _FakeTokenizer:
    all_special_ids = []

    def __init__(self, token_count: int):
        self.token_count = token_count

    def __call__(self, text, **_kwargs):
        token_offset = sum(ord(character) for character in text) % 1000
        return {
            "input_ids": [
                token_offset + index
                for index in range(self.token_count)
            ]
        }


class _FakeAttackScorer(_FakeScorer):
    def __init__(self, scores: dict[str, float], token_count: int):
        super().__init__(scores)
        self.tokenizer = _FakeTokenizer(token_count)

    def score_single(self, text):
        return self.scores[text]


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


class AESRudimentaryAttackTest(unittest.TestCase):
    def test_iterative_search_uses_true_scores_and_stops_at_threshold(self):
        scorer = _FakeAttackScorer(
            {
                "original": 1.0,
                "edit-one": 1.06,
                "edit-two": 1.12,
            },
            token_count=20,
        )
        variants = {
            "original": ["edit-one"],
            "edit-one": ["edit-two"],
        }
        attack = IterativeRudimentaryAttack(
            scorer,
            n_steps=30,
            candidates_per_step=16,
            threshold=0.1,
            max_token_edit_rate=0.1,
        )

        with patch(
            "text_scoring_adv_training.evaluation.aes.attacks."
            "rudimentary.sample_variants",
            side_effect=lambda text, _count: variants.get(text, []),
        ):
            perturbed, history = attack.attack("original")

        self.assertEqual(perturbed, "edit-two")
        self.assertEqual(len(history), 2)
        self.assertAlmostEqual(history[-1]["delta"], 0.12)
        self.assertEqual(history[-1]["accepted_edit_count"], 2)
        self.assertEqual(history[-1]["max_edits"], 2)

    def test_edit_budget_limits_accepted_operations(self):
        scorer = _FakeAttackScorer(
            {
                "original": 1.0,
                "edit-one": 1.05,
                "edit-two": 1.20,
            },
            token_count=10,
        )
        variants = {
            "original": ["edit-one"],
            "edit-one": ["edit-two"],
        }
        attack = IterativeRudimentaryAttack(
            scorer,
            n_steps=30,
            candidates_per_step=16,
            threshold=0.5,
            max_token_edit_rate=0.1,
        )

        with patch(
            "text_scoring_adv_training.evaluation.aes.attacks."
            "rudimentary.sample_variants",
            side_effect=lambda text, _count: variants.get(text, []),
        ):
            perturbed, history = attack.attack("original")

        self.assertEqual(perturbed, "edit-one")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["max_edits"], 1)
        self.assertTrue(history[0]["changes"])

    def test_invalid_rudimentary_parameters_are_rejected(self):
        scorer = _FakeAttackScorer({"original": 1.0}, token_count=10)

        with self.assertRaises(ValueError):
            IterativeRudimentaryAttack(scorer, beam_size=2)
        with self.assertRaises(ValueError):
            IterativeRudimentaryAttack(
                scorer,
                max_token_edit_rate=0,
            )
        with self.assertRaises(ValueError):
            IterativeRudimentaryAttack(
                scorer,
                improvement_tolerance=-1,
            )

    def test_token_equivalent_candidate_is_not_counted_as_an_edit(self):
        class _EquivalentTokenizer(_FakeTokenizer):
            def __call__(self, _text, **_kwargs):
                return {"input_ids": list(range(self.token_count))}

        scorer = _FakeAttackScorer(
            {"original": 1.0, "trailing-space ": 1.0000001},
            token_count=20,
        )
        scorer.tokenizer = _EquivalentTokenizer(20)
        attack = IterativeRudimentaryAttack(
            scorer,
            n_steps=1,
            candidates_per_step=1,
        )

        with patch(
            "text_scoring_adv_training.evaluation.aes.attacks."
            "rudimentary.sample_variants",
            return_value=["trailing-space "],
        ):
            perturbed, history = attack.attack("original")

        self.assertEqual(perturbed, "original")
        self.assertEqual(history, [])


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

    def test_linear_relative_loss_preserves_small_rudimentary_signal(self):
        clean = torch.tensor([1.0], requires_grad=True)
        adversarial = torch.tensor([1.003], requires_grad=True)
        target = torch.tensor([2.0])

        loss = one_sided_score_inflation_loss(
            clean,
            adversarial,
            target,
            tolerance=0.0,
            relative_loss_power=1.0,
        )
        loss.backward()

        self.assertAlmostEqual(float(loss.item()), 0.0015, places=6)
        self.assertIsNone(clean.grad)
        self.assertAlmostEqual(float(adversarial.grad.item()), 0.5)

    def test_legacy_margin_config_maps_to_tolerance(self):
        config = AESAdversarialConfig(hotflip_margin=0.1)
        self.assertEqual(config.hotflip_tolerance, 0.1)

    def test_c0_and_defenses_share_all_optimization_parameters(self):
        clean_args = build_parser(CLEAN_CONTINUATION).parse_args([])
        hotflip_args = build_parser(HOTFLIP_DEFENSE).parse_args([])
        rudimentary_args = build_parser(RUDIMENTARY_DEFENSE).parse_args([])
        clean_config = build_config(clean_args, CLEAN_CONTINUATION)
        hotflip_config = build_config(hotflip_args, HOTFLIP_DEFENSE)
        rudimentary_config = build_config(
            rudimentary_args,
            RUDIMENTARY_DEFENSE,
        )

        intentionally_different = {
            "training_mode",
            "output_dir",
            "use_hotflip_swaps",
            "use_rudimentary_edits",
        }
        shared_keys = set(clean_config) - intentionally_different
        self.assertEqual(shared_keys, set(hotflip_config) - intentionally_different)
        self.assertEqual(
            shared_keys,
            set(rudimentary_config) - intentionally_different,
        )
        for key in shared_keys:
            self.assertEqual(clean_config[key], hotflip_config[key], key)
            self.assertEqual(
                clean_config[key],
                rudimentary_config[key],
                key,
            )

        self.assertFalse(clean_config["use_hotflip_swaps"])
        self.assertFalse(clean_config["use_rudimentary_edits"])
        self.assertTrue(hotflip_config["use_hotflip_swaps"])
        self.assertFalse(hotflip_config["use_rudimentary_edits"])
        self.assertFalse(rudimentary_config["use_hotflip_swaps"])
        self.assertTrue(rudimentary_config["use_rudimentary_edits"])
        self.assertEqual(clean_config["precision"], "bfloat16")
        self.assertEqual(rudimentary_config["rudimentary_tolerance"], 0.0)
        self.assertEqual(
            rudimentary_config["rudimentary_relative_loss_power"],
            1.0,
        )
        self.assertEqual(
            rudimentary_config["rudimentary_edits_per_candidate"],
            3,
        )

    def test_rudimentary_training_selects_true_best_effective_edit(self):
        class _TrainingTokenizer:
            token_ids = {
                "original": [1, 2],
                "weaker": [1, 3],
                "stronger": [1, 4],
            }

            def __call__(self, texts, return_tensors=None, padding=False, **_kwargs):
                if isinstance(texts, str):
                    return {"input_ids": list(self.token_ids[texts])}
                rows = [self.token_ids[text] for text in texts]
                return {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.ones(
                        (len(rows), len(rows[0])),
                        dtype=torch.long,
                    ),
                }

        class _ScoreModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                return SimpleNamespace(
                    logits=input_ids.float().sum(dim=1, keepdim=True)
                )

        scorer = SimpleNamespace(
            tokenizer=_TrainingTokenizer(),
            model=_ScoreModel(),
        )
        scorer.model.train()

        with patch(
            "text_scoring_adv_training.training.aes_trainer."
            "sample_variants",
            return_value=["weaker", "stronger"],
        ):
            selected = run_rudimentary_one_row(
                "original",
                scorer,
                max_candidates=2,
                edits_per_candidate=1,
                max_length=1024,
                device="cpu",
            )

        self.assertEqual(selected, "stronger")
        self.assertTrue(scorer.model.training)

    def test_rudimentary_v2_composes_three_original_edits(self):
        class _TrainingTokenizer:
            token_ids = {
                "original": [1, 1],
                "a1": [1, 2],
                "a2": [1, 3],
                "a3": [1, 8],
                "b1": [1, 4],
                "b2": [1, 5],
                "b3": [1, 6],
            }

            def __call__(self, texts, return_tensors=None, **_kwargs):
                if isinstance(texts, str):
                    return {"input_ids": list(self.token_ids[texts])}
                rows = [self.token_ids[text] for text in texts]
                return {
                    "input_ids": torch.tensor(rows, dtype=torch.long),
                    "attention_mask": torch.ones(
                        (len(rows), len(rows[0])),
                        dtype=torch.long,
                    ),
                }

        class _ScoreModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                del attention_mask
                return SimpleNamespace(
                    logits=input_ids.float().sum(dim=1, keepdim=True)
                )

        chains = {
            "original": ["a1", "b1"],
            "a1": ["a2"],
            "a2": ["a3"],
            "b1": ["b2"],
            "b2": ["b3"],
        }
        scorer = SimpleNamespace(
            tokenizer=_TrainingTokenizer(),
            model=_ScoreModel(),
        )

        with patch(
            "text_scoring_adv_training.training.aes_trainer."
            "sample_variants",
            side_effect=lambda text, _count: chains.get(text, []),
        ):
            selected = run_rudimentary_one_row(
                "original",
                scorer,
                max_candidates=2,
                edits_per_candidate=3,
                max_length=1024,
                device="cpu",
            )

        self.assertEqual(selected, "a3")

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
    def test_rudimentary_evaluator_has_rudimentary_default(self):
        args = build_evaluation_parser("rudimentary").parse_args([])

        self.assertEqual(args.attack, "rudimentary")

    def test_rudimentary_selector_uses_separate_readable_output_paths(self):
        args = build_selection_parser("rudimentary").parse_args([])

        self.assertEqual(args.attack, "rudimentary")
        self.assertEqual(
            args.defense_output_dir.name,
            "aes_rudimentary_defense_v2_seed42",
        )
        self.assertEqual(
            args.selection_output_dir.name,
            "aes_rudimentary_defense_v2_checkpoint_selection_seed42",
        )
        self.assertEqual(args.selection_steps, 30)

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

    def test_stratified_subset_keeps_singleton_strata(self):
        dataframe = pd.DataFrame(
            [
                {
                    "essay_id": "rare",
                    "prompt_name": "The Face on Mars",
                    "score": 6,
                },
                *[
                    {
                        "essay_id": f"common-{index}",
                        "prompt_name": "Common prompt",
                        "score": 3,
                    }
                    for index in range(19)
                ],
            ]
        )

        selected_indices = stratified_sample_indices(
            dataframe,
            stratify_columns=["prompt_name", "score"],
            sample_size=10,
            seed=42,
        )

        self.assertEqual(len(selected_indices), 10)
        self.assertIn(0, selected_indices)


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
