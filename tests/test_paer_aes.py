from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from paer.aes_rh_trainer import changed_token_uplift_targets
from paer.modeling_paer import PAERForEssayScoring


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


if __name__ == "__main__":
    unittest.main()
