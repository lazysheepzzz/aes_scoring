"""PAER-v3: direction-aware token evidence aggregation for AES.

The pretrained DeBERTa regressor supplies a stable global score prior.  A
separate token branch learns signed local score evidence and decomposes it
into positive and negative parts.  Risk routing is applied *inside* this
aggregation: it can suppress suspicious positive evidence, but it cannot
erase negative evidence that may represent genuine writing-quality defects.
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

from paer.modeling_paer import PAER_CONFIG_NAME, PAER_HEADS_NAME


@dataclass
class PAERV3Output(ModelOutput):
    """Scores and token-level quantities used by training and diagnostics."""

    logits: torch.Tensor | None = None
    # Route-off counterfactual.  This includes signed token evidence but does
    # not suppress any positive evidence.
    base_logits: torch.Tensor | None = None
    global_logits: torch.Tensor | None = None
    risk_logits: torch.Tensor | None = None
    token_evidence: torch.Tensor | None = None
    positive_evidence: torch.Tensor | None = None
    negative_evidence: torch.Tensor | None = None
    attention_weights: torch.Tensor | None = None
    correction: torch.Tensor | None = None
    content_mask: torch.Tensor | None = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


class PAERV3ForEssayScoring(nn.Module):
    """DeBERTa scorer with directional routing during token aggregation."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        correction_scale: float = 1.0,
        risk_bias_init: float = -5.0,
        max_token_evidence: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= correction_scale <= 1.0:
            raise ValueError("correction_scale must be in [0, 1] for v3")
        if max_token_evidence <= 0:
            raise ValueError("max_token_evidence must be greater than zero")

        hidden_size = int(base_model.config.hidden_size)
        self.base_model = base_model
        self.risk_head = nn.Linear(hidden_size, 1)
        self.token_evidence_head = nn.Linear(hidden_size, 1)
        self.attention_head = nn.Linear(hidden_size, 1)
        self.correction_scale = float(correction_scale)
        self.max_token_evidence = float(max_token_evidence)

        # Start very close to the pretrained scorer.  The token residual is
        # initially tiny, while the trace losses can still train all heads.
        nn.init.zeros_(self.risk_head.weight)
        nn.init.constant_(self.risk_head.bias, risk_bias_init)
        nn.init.normal_(self.token_evidence_head.weight, mean=0.0, std=1e-3)
        # A very small positive start keeps the route-calibration gradient
        # alive; the signed residual remains only about 0.01 score points.
        nn.init.constant_(self.token_evidence_head.bias, 0.02)
        nn.init.zeros_(self.attention_head.weight)
        nn.init.zeros_(self.attention_head.bias)

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
        attention_mask: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = getattr(self.base_model, "deberta", None)
        if encoder is None:
            raise TypeError(
                "PAER-v3 currently requires a DeBERTa sequence-classification model"
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
        token_states = outputs.last_hidden_state
        pooled = self.base_model.pooler(token_states)
        pooled = self.base_model.dropout(pooled)
        return token_states, self.base_model.classifier(pooled)

    @staticmethod
    def _content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.float32).clone()
        if mask.shape[1] == 0:
            return mask
        mask[:, 0] = 0.0
        lengths = attention_mask.long().sum(dim=1)
        rows = torch.arange(mask.shape[0], device=mask.device)
        mask[rows, torch.clamp(lengths - 1, min=0)] = 0.0
        return mask

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_logits = logits.float().masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(masked_logits, dim=1) * mask.float()
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

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
    ) -> PAERV3Output | tuple[torch.Tensor, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds must be provided")
        if attention_mask is None:
            source = input_ids if input_ids is not None else inputs_embeds
            attention_mask = torch.ones(
                source.shape[:2], device=source.device, dtype=torch.long
            )

        token_states, global_logits = self._encode_and_score(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        content_mask = self._content_mask(attention_mask)
        risk_logits = self.risk_head(token_states).squeeze(-1)
        raw_evidence = self.token_evidence_head(token_states).squeeze(-1)
        token_evidence = self.max_token_evidence * torch.tanh(raw_evidence.float())
        positive_evidence = F.relu(token_evidence)
        negative_evidence = F.relu(-token_evidence)
        attention_weights = self._masked_softmax(
            self.attention_head(token_states).squeeze(-1),
            content_mask,
        )

        # The route-off score is the exact architectural counterfactual used
        # by the causal diagnostic.  Routing alters only the positive summand.
        positive_score = (attention_weights * positive_evidence).sum(dim=1)
        negative_score = (attention_weights * negative_evidence).sum(dim=1)
        route_off_logits = global_logits + (positive_score - negative_score).unsqueeze(-1)
        correction = self.correction_scale * (
            attention_weights
            * torch.sigmoid(risk_logits.float())
            * positive_evidence
        ).sum(dim=1)
        logits = route_off_logits - correction.unsqueeze(-1).to(route_off_logits.dtype)

        hidden_states = (token_states,) if output_hidden_states else None
        if not return_dict:
            return (
                logits,
                route_off_logits,
                risk_logits,
                positive_evidence,
                correction,
            )
        return PAERV3Output(
            logits=logits,
            base_logits=route_off_logits,
            global_logits=global_logits,
            risk_logits=risk_logits,
            token_evidence=token_evidence,
            positive_evidence=positive_evidence,
            negative_evidence=negative_evidence,
            attention_weights=attention_weights,
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
        risk_bias_init: float = -5.0,
        max_token_evidence: float = 0.5,
    ) -> "PAERV3ForEssayScoring":
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
        return cls(
            base_model,
            correction_scale=correction_scale,
            risk_bias_init=risk_bias_init,
            max_token_evidence=max_token_evidence,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> "PAERV3ForEssayScoring":
        checkpoint_path = Path(checkpoint_path)
        config = json.loads(
            (checkpoint_path / PAER_CONFIG_NAME).read_text(encoding="utf-8")
        )
        if config.get("model_type") != "paer_aes_v3":
            raise ValueError(f"Not a PAER-v3 checkpoint: {checkpoint_path}")
        model = cls.from_base_checkpoint(
            checkpoint_path,
            dtype=dtype,
            correction_scale=float(config["correction_scale"]),
            max_token_evidence=float(config.get("max_token_evidence", 0.5)),
        )
        try:
            state = torch.load(
                checkpoint_path / PAER_HEADS_NAME,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(checkpoint_path / PAER_HEADS_NAME, map_location="cpu")
        model.risk_head.load_state_dict(state["risk_head"])
        model.token_evidence_head.load_state_dict(state["token_evidence_head"])
        model.attention_head.load_state_dict(state["attention_head"])
        return model

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.base_model.save_pretrained(output_dir)
        torch.save(
            {
                "risk_head": self.risk_head.state_dict(),
                "token_evidence_head": self.token_evidence_head.state_dict(),
                "attention_head": self.attention_head.state_dict(),
            },
            output_dir / PAER_HEADS_NAME,
        )
        (output_dir / PAER_CONFIG_NAME).write_text(
            json.dumps(
                {
                    "model_type": "paer_aes_v3",
                    "version": 3,
                    "correction_scale": self.correction_scale,
                    "max_token_evidence": self.max_token_evidence,
                    "aggregation": "attention_weighted_signed_token_evidence",
                    "directional_routing": (
                        "risk suppresses positive token evidence only; "
                        "negative evidence is preserved"
                    ),
                    "base_logits_semantics": (
                        "route-off score with signed token aggregation enabled"
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
