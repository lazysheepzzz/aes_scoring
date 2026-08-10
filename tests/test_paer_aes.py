from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn

from paer.aes_rh_trainer import (
    PairedEssayTrainingDataset,
    RHTraceCollator,
    changed_token_uplift_targets,
)
from paer.modeling_paer import PAERForEssayScoring
from paer.select_aes_rh_checkpoint import restrict_candidates_to_common_budget
from text_scoring_adv_training.evaluation.aes.run_attacks import build_attack


class _FakeEncoder(nn.Module):
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        **kwargs,
    ):
        hidden = inputs_embeds
        if hidden is None:
            hidden = torch.nn.functional.one_hot(
                input_ids % 4, num_classes=4
            ).float()
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.deberta = _FakeEncoder()
        self.pooler = lambda hidden: hidden[:, 0]
        self.dropout = nn.Identity()
        self.classifier = nn.Linear(4, 1, bias=False)
        self.embedding = nn.Embedding(16, 4)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value


class PAERModelTests(unittest.TestCase):
    def test_directional_correction_can_only_reduce_positive_score(self):
        model = PAERForEssayScoring(_FakeBaseModel(), correction_scale=1.0)
        with torch.no_grad():
            model.risk_head.bias.fill_(5.0)
            model.positive_evidence_head.bias.fill_(1.0)
        output = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
        self.assertGreaterEqual(float(output.correction.item()), 0.0)
        self.assertLessEqual(
            float(output.logits.item()),
            float(output.base_logits.item()),
        )

    def test_paer_supports_inputs_embeds_for_adaptive_hotflip(self):
        model = PAERForEssayScoring(_FakeBaseModel())
        embeddings = torch.randn(1, 4, 4, requires_grad=True)
        output = model(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
        output.logits.sum().backward()
        self.assertIsNotNone(embeddings.grad)


class CounterfactualTargetTests(unittest.TestCase):
    def test_replacement_marks_only_after_text_change(self):
        targets = changed_token_uplift_targets(
            [1, 10, 11, 2],
            [1, 10, 12, 2],
            special_ids={1, 2},
            step_gain=0.05,
            gain_scale=0.1,
        )
        self.assertEqual(targets, [0.0, 0.0, 0.5, 0.0])

    def test_deletion_marks_nearest_remaining_content_token(self):
        targets = changed_token_uplift_targets(
            [1, 10, 11, 12, 2],
            [1, 10, 12, 2],
            special_ids={1, 2},
            step_gain=0.2,
            gain_scale=0.1,
        )
        self.assertEqual(sum(value > 0 for value in targets), 1)
        self.assertEqual(max(targets), 1.0)


class PairedTrainingDataTests(unittest.TestCase):
    def test_every_clean_essay_is_kept_and_only_selected_rows_are_attacked(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "train.csv"
            pd.DataFrame(
                {
                    "full_text": ["clean zero", "clean one", "clean two"],
                    "score": [1, 2, 3],
                }
            ).to_csv(csv_path, index=False)
            traces = SimpleNamespace(
                items=[
                    {
                        "row_index": 1,
                        "original_text": "clean one",
                        "before_text": "clean one",
                        "adversarial_text": "changed one",
                        "step_gain": 0.1,
                        "attack": "hotflip",
                    }
                ]
            )
            dataset = PairedEssayTrainingDataset(
                csv_path,
                traces,
                label_offset=1.0,
            )

            self.assertEqual(len(dataset), 3)
            self.assertFalse(dataset[0]["has_adversarial"])
            self.assertTrue(dataset[1]["has_adversarial"])
            self.assertFalse(dataset[2]["has_adversarial"])
            self.assertEqual(dataset[2]["label"], 2.0)

    def test_collator_encodes_only_selected_adversarial_rows(self):
        class _TinyTokenizer:
            all_special_ids = [0, 2]

            def __call__(self, texts, **_kwargs):
                vocabulary = {"alpha": 10, "beta": 11, "gamma": 12}
                rows = [
                    [0, *[vocabulary.get(word, 9) for word in text.split()], 2]
                    for text in texts
                ]
                width = max(len(row) for row in rows)
                input_ids = torch.tensor(
                    [row + [0] * (width - len(row)) for row in rows]
                )
                attention_mask = (input_ids != 0).long()
                attention_mask[:, 0] = 1
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

        collator = RHTraceCollator(
            _TinyTokenizer(),
            max_length=16,
            label_offset=1.0,
            gain_scale=0.1,
        )
        batch = collator(
            [
                {
                    "original_text": "alpha beta",
                    "before_text": "alpha beta",
                    "adversarial_text": "alpha beta",
                    "label_score_space": 2.0,
                    "step_gain": 0.0,
                    "attack": "clean_only",
                    "has_adversarial": False,
                },
                {
                    "original_text": "alpha beta",
                    "before_text": "alpha beta",
                    "adversarial_text": "alpha gamma",
                    "label_score_space": 3.0,
                    "step_gain": 0.1,
                    "attack": "hotflip",
                    "has_adversarial": True,
                },
            ]
        )

        self.assertEqual(batch["clean_input_ids"].shape[0], 2)
        self.assertEqual(batch["adversarial_input_ids"].shape[0], 1)
        self.assertEqual(batch["adversarial_indices"].tolist(), [1])
        self.assertGreater(float(batch["uplift_targets"].sum()), 0.0)


class CheckpointBudgetTests(unittest.TestCase):
    def test_common_budget_keeps_only_explicit_steps_at_or_below_cap(self):
        candidates = [
            Path("gstep200"),
            Path("gstep1400"),
            Path("gstep1600"),
            Path("best"),
            Path("final"),
        ]
        eligible, excluded = restrict_candidates_to_common_budget(candidates, 1400)

        self.assertEqual([path.name for path in eligible], ["gstep200", "gstep1400"])
        self.assertEqual(excluded, ["gstep1600", "best", "final"])

    def test_zero_budget_cap_preserves_all_candidates(self):
        candidates = [Path("gstep200"), Path("best"), Path("final")]
        eligible, excluded = restrict_candidates_to_common_budget(candidates, 0)

        self.assertEqual(eligible, candidates)
        self.assertEqual(excluded, [])

    def test_evaluation_batch_size_reaches_attack_candidate_scorer(self):
        scorer = SimpleNamespace(
            tokenizer=SimpleNamespace(all_special_ids=[0, 1, 2])
        )
        attack = build_attack("hotflip", scorer, batch_size=4)

        self.assertEqual(attack.batch_size, 4)


if __name__ == "__main__":
    unittest.main()
