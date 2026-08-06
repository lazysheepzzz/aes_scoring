"""Direction-aware score-inflation correction for DeBERTa AES models.

The original AES regressor remains the global scoring branch.  PAER adds two
small token heads: one estimates the probability that a token participates in
victim-induced score inflation, and the other estimates the token's positive
score evidence.  Only the intersection of those two positive signals is
subtracted from the global score; negative quality evidence is never masked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.utils import ModelOutput


PAER_CONFIG_NAME = "paer_config.json"
PAER_HEADS_NAME = "paer_heads.pt"


@dataclass
class PAEROutput(ModelOutput):
    """Outputs needed by scoring, white-box attacks, and PAER supervision."""

    logits: torch.Tensor | None = None
    base_logits: torch.Tensor | None = None
    risk_logits: torch.Tensor | None = None
    positive_evidence: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    content_mask: torch.Tensor | None = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class PAERForEssayScoring(nn.Module):
    """DeBERTa AES scorer with fixed-form directional evidence routing."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        correction_scale: float = 1.0,
        risk_bias_init: float = -4.0,
        evidence_bias_init: float = -2.0,
    ):
        super().__init__()
        if correction_scale < 0:
            raise ValueError("correction_scale must be non-negative")
        hidden_size = int(base_model.config.hidden_size)
        self.base_model = base_model
        self.risk_head = nn.Linear(hidden_size, 1)
        self.positive_evidence_head = nn.Linear(hidden_size, 1)
        self.correction_scale = float(correction_scale)

        nn.init.zeros_(self.risk_head.weight)
        nn.init.constant_(self.risk_head.bias, risk_bias_init)
        nn.init.normal_(self.positive_evidence_head.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.positive_evidence_head.bias, evidence_bias_init)

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.base_model.set_input_embeddings(value)

    def _encode_and_score(
        self,
        *,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run DeBERTa once and reuse its final token states for both branches."""
        encoder = getattr(self.base_model, "deberta", None)
        if encoder is None:
            raise TypeError(
                "PAER currently requires a DeBERTa sequence-classification model"
            )
        outputs = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        pooled_output = self.base_model.pooler(sequence_output)
        pooled_output = self.base_model.dropout(pooled_output)
        base_logits = self.base_model.classifier(pooled_output)
        return sequence_output, base_logits

    @staticmethod
    def _content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Exclude padding plus the leading CLS and trailing SEP positions."""
        mask = attention_mask.to(dtype=torch.float32).clone()
        if mask.shape[1] == 0:
            return mask
        mask[:, 0] = 0.0
        lengths = attention_mask.long().sum(dim=1)
        rows = torch.arange(mask.shape[0], device=mask.device)
        last_positions = torch.clamp(lengths - 1, min=0)
        mask[rows, last_positions] = 0.0
        return mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **_: object,
    ) -> PAEROutput | tuple[torch.Tensor, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")
        if attention_mask is None:
            shape = input_ids.shape[:2] if input_ids is not None else inputs_embeds.shape[:2]
            attention_mask = torch.ones(shape, device=(
                input_ids.device if input_ids is not None else inputs_embeds.device
            ), dtype=torch.long)

        token_states, base_logits = self._encode_and_score(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        risk_logits = self.risk_head(token_states).squeeze(-1)
        positive_evidence = F.softplus(
            self.positive_evidence_head(token_states).squeeze(-1)
        )
        content_mask = self._content_mask(attention_mask)
        routed_positive_evidence = (
            torch.sigmoid(risk_logits) * positive_evidence * content_mask
        )
        denominator = content_mask.sum(dim=1).clamp_min(1.0)
        correction = (
            routed_positive_evidence.sum(dim=1) / denominator
        ) * self.correction_scale
        logits = base_logits - correction.unsqueeze(-1).to(base_logits.dtype)

        hidden_states = (token_states,) if output_hidden_states else None
        if not return_dict:
            return logits, base_logits, risk_logits, positive_evidence, correction
        return PAEROutput(
            logits=logits,
            base_logits=base_logits,
            risk_logits=risk_logits,
            positive_evidence=positive_evidence,
            correction=correction,
            content_mask=content_mask,
            hidden_states=hidden_states,
        )

    @classmethod
    def from_base_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
        correction_scale: float = 1.0,
    ) -> "PAERForEssayScoring":
        checkpoint_path = Path(checkpoint_path)
        config = AutoConfig.from_pretrained(
            str(checkpoint_path), trust_remote_code=True
        )
        config.num_labels = 1
        base_model = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path),
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        return cls(base_model, correction_scale=correction_scale)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> "PAERForEssayScoring":
        checkpoint_path = Path(checkpoint_path)
        paer_config = json.loads(
            (checkpoint_path / PAER_CONFIG_NAME).read_text(encoding="utf-8")
        )
        model = cls.from_base_checkpoint(
            checkpoint_path,
            dtype=dtype,
            correction_scale=float(paer_config["correction_scale"]),
        )
        try:
            state = torch.load(
                checkpoint_path / PAER_HEADS_NAME,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(
                checkpoint_path / PAER_HEADS_NAME,
                map_location="cpu",
            )
        model.risk_head.load_state_dict(state["risk_head"])
        model.positive_evidence_head.load_state_dict(
            state["positive_evidence_head"]
        )
        return model

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output_dir)
        torch.save(
            {
                "risk_head": self.risk_head.state_dict(),
                "positive_evidence_head": self.positive_evidence_head.state_dict(),
            },
            output_dir / PAER_HEADS_NAME,
        )
        (output_dir / PAER_CONFIG_NAME).write_text(
            json.dumps(
                {
                    "model_type": "paer_aes",
                    "version": 1,
                    "correction_scale": self.correction_scale,
                    "directional_routing": (
                        "subtract predicted suspicious positive evidence only"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
