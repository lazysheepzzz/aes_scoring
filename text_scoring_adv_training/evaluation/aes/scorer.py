"""
AESScorer: DeBERTa-v3-base AES victim wrapper.
Loads fold0_best checkpoint and provides score_essay() / score_batch().
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

__all__ = ["AESScorer"]


class AESScorer(nn.Module):
    """
    DeBERTa-v3-base regression scorer for AES.
    Expects checkpoint dir with:
        config.json          (has num_labels=1, pad_token_id)
        model.safetensors    (or pytorch_model.bin)
        tokenizer files
    Optionally loads best_thresholds.json for band-based eval.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        thresholds_path: str | Path | None = None,
        *,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path), trust_remote_code=True
        )
        # DeBERTa's sequence-classification pooler reads the first token.  Keep
        # CLS at position zero by padding encoder inputs on the right.
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        # Load either a standard AES regressor or a PAER directional-routing
        # checkpoint.  The original checkpoint format remains unchanged.
        if (self.checkpoint_path / "paer_config.json").is_file():
            paer_config = json.loads(
                (self.checkpoint_path / "paer_config.json").read_text(
                    encoding="utf-8"
                )
            )
            if paer_config.get("model_type") == "paer_aes_v3":
                from paer.modeling_paer_v3 import PAERV3ForEssayScoring

                paer_class = PAERV3ForEssayScoring
            else:
                from paer.modeling_paer import PAERForEssayScoring

                paer_class = PAERForEssayScoring
            self.model = paer_class.from_pretrained(
                self.checkpoint_path,
                dtype=dtype,
            ).to(device)
        else:
            config = AutoConfig.from_pretrained(
                str(self.checkpoint_path), trust_remote_code=True
            )
            config.num_labels = 1          # regression head
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(self.checkpoint_path),
                config=config,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(device)
        self.model.eval()

        self.device = device

        # Load thresholds for band-based eval
        self.thresholds: Optional[List[float]] = None
        if thresholds_path:
            with open(thresholds_path) as f:
                self.thresholds = json.load(f)
        elif (self.checkpoint_path.parent / "best_thresholds.json").exists():
            with open(self.checkpoint_path.parent / "best_thresholds.json") as f:
                self.thresholds = json.load(f)

    @torch.no_grad()
    def score_single(self, text: str) -> float:
        """Return the raw regression score for a single essay."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return float(logits.squeeze(-1).item())

    @torch.no_grad()
    def score_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Score multiple essays in batches."""
        scores = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            scores.extend(logits.squeeze(-1).tolist())
        return scores

    def get_embeddings(self) -> nn.Embedding:
        """Return the model's token embedding layer for HotFlip."""
        return self.model.get_input_embeddings()

    def embed_inputs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get input embeddings for the given input IDs."""
        return self.model.get_input_embeddings()(input_ids)
