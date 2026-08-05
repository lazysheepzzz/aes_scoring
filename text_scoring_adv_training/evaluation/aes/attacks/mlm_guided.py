"""Formal MLM-guided attack for the AES experiments.

The paper's generic MLM helpers remain in ``robustness_tests/common/mlm.py``.
This AES wrapper keeps ModernBERT and DeBERTa token IDs strictly separated:
ModernBERT proposes and decodes complete candidate texts, then the victim
tokenizer independently encodes those texts for scoring.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from text_scoring_adv_training.evaluation.robustness_tests.common.mlm import (
    build_replacement_map,
    load_mlm,
)

__all__ = [
    "MLMGuidedAttack",
    "MLMGuidedCandidateGenerator",
    "SemanticSimilarityFilter",
]


DEFAULT_MLM_MODEL = "answerdotai/ModernBERT-large"
DEFAULT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        return torch.device(getattr(model, "device", "cpu"))


class SemanticSimilarityFilter:
    """Batched cosine-similarity filter using a frozen encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIMILARITY_MODEL,
        *,
        device: str = "cuda",
        minimum_similarity: float = 0.90,
        tokenizer=None,
        model=None,
        max_length: int = 256,
    ):
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be in [-1, 1]")
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        self.model_name = model_name
        self.device = torch.device(device)
        self.minimum_similarity = minimum_similarity
        self.max_length = max_length
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = model or AutoModel.from_pretrained(model_name)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        outputs = self.model(**encoded)
        token_embeddings = outputs.last_hidden_state.float()
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embeddings * attention_mask).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, p=2, dim=1)

    def filter(
        self,
        reference_text: str,
        candidates: Sequence[str],
    ) -> List[Tuple[str, float]]:
        if not candidates:
            return []
        embeddings = self.encode([reference_text, *candidates])
        similarities = embeddings[1:] @ embeddings[0]
        return [
            (candidate, float(similarity.item()))
            for candidate, similarity in zip(candidates, similarities)
            if float(similarity.item()) >= self.minimum_similarity
        ]


class MLMGuidedCandidateGenerator:
    """Generate semantically filtered candidate texts in MLM token space."""

    def __init__(
        self,
        *,
        mlm_model_name: str = DEFAULT_MLM_MODEL,
        similarity_model_name: str = DEFAULT_SIMILARITY_MODEL,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        n_sample_pos: int = 8,
        top_k_per_pos: int = 2,
        max_candidates: int = 16,
        minimum_similarity: float = 0.90,
        mlm_max_length: int = 8192,
        mlm_batch_size: int = 32,
        mlm_tokenizer=None,
        mlm_model=None,
        mask_token_id: Optional[int] = None,
        mlm_special_ids: Optional[set[int]] = None,
        semantic_filter: Optional[SemanticSimilarityFilter] = None,
    ):
        positive = {
            "n_sample_pos": n_sample_pos,
            "top_k_per_pos": top_k_per_pos,
            "max_candidates": max_candidates,
            "mlm_max_length": mlm_max_length,
            "mlm_batch_size": mlm_batch_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not -1.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be in [-1, 1]")

        if mlm_tokenizer is None or mlm_model is None:
            load_dtype = dtype
            if not str(device).startswith("cuda") and dtype == torch.bfloat16:
                load_dtype = torch.float32
            loaded = load_mlm(
                mlm_model_name,
                device=device,
                dtype=load_dtype,
            )
            mlm_tokenizer, mlm_model, loaded_mask, loaded_specials = loaded
            mask_token_id = loaded_mask
            mlm_special_ids = loaded_specials
        if mask_token_id is None:
            mask_token_id = mlm_tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("The MLM tokenizer has no mask token")

        self.mlm_model_name = mlm_model_name
        self.similarity_model_name = similarity_model_name
        self.mlm_tokenizer = mlm_tokenizer
        self.mlm_model = mlm_model.eval()
        self.mask_token_id = int(mask_token_id)
        self.mlm_special_ids = set(
            mlm_special_ids
            if mlm_special_ids is not None
            else mlm_tokenizer.all_special_ids
        )
        self.n_sample_pos = n_sample_pos
        self.top_k_per_pos = top_k_per_pos
        self.max_candidates = max_candidates
        self.minimum_similarity = minimum_similarity
        self.mlm_max_length = mlm_max_length
        self.mlm_batch_size = mlm_batch_size
        self.semantic_filter = semantic_filter or SemanticSimilarityFilter(
            similarity_model_name,
            device=device,
            minimum_similarity=minimum_similarity,
        )

    def _encode(self, text: str) -> torch.Tensor:
        encoded = self.mlm_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.mlm_max_length,
            padding=False,
        )
        return encoded["input_ids"].squeeze(0)

    def generate(
        self,
        text: str,
        *,
        reference_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return candidate texts; IDs never leave the ModernBERT boundary."""
        ids = self._encode(text)
        positions = [
            index
            for index, token_id in enumerate(ids.tolist())
            if int(token_id) not in self.mlm_special_ids
        ]
        if not positions:
            return []
        sampled_positions = random.sample(
            positions,
            min(self.n_sample_pos, len(positions)),
        )
        ids_device = ids.to(_model_device(self.mlm_model))
        replacements = build_replacement_map(
            ids_device,
            self.mask_token_id,
            self.mlm_special_ids,
            self.mlm_model,
            batch_size=self.mlm_batch_size,
            top_k=self.top_k_per_pos,
            prob_min=0.0,
            positions=sampled_positions,
        )

        raw: List[Tuple[str, int, int, int]] = []
        seen = {text}
        for position in sampled_positions:
            old_id = int(ids_device[position].item())
            for new_id in replacements.get(position, []):
                candidate_ids = ids_device.clone()
                candidate_ids[position] = int(new_id)
                candidate_text = self.mlm_tokenizer.decode(
                    candidate_ids.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                if not candidate_text or candidate_text in seen:
                    continue
                seen.add(candidate_text)
                raw.append((candidate_text, position, old_id, int(new_id)))

        if len(raw) > self.max_candidates:
            raw = random.sample(raw, self.max_candidates)
        if not raw:
            return []

        similarity_by_text = dict(
            self.semantic_filter.filter(
                reference_text if reference_text is not None else text,
                [item[0] for item in raw],
            )
        )
        return [
            {
                "text": candidate_text,
                "mlm_position": position,
                "mlm_old_id": old_id,
                "mlm_new_id": new_id,
                "cosine_similarity": similarity_by_text[candidate_text],
            }
            for candidate_text, position, old_id, new_id in raw
            if candidate_text in similarity_by_text
        ]


class MLMGuidedAttack:
    """Greedy, budgeted MLM-guided score-inflation attack."""

    def __init__(
        self,
        scorer,
        *,
        n_steps: int = 30,
        beam_size: int = 1,
        n_sample_pos: int = 8,
        top_k_per_pos: int = 2,
        max_candidates_per_step: int = 16,
        batch_size: int = 32,
        threshold: float = 0.1,
        max_token_edit_rate: float = 0.05,
        minimum_similarity: float = 0.90,
        mlm_model_name: str = DEFAULT_MLM_MODEL,
        similarity_model_name: str = DEFAULT_SIMILARITY_MODEL,
        mlm_max_length: int = 8192,
        dtype: torch.dtype = torch.bfloat16,
        candidate_generator: Optional[MLMGuidedCandidateGenerator] = None,
        improvement_tolerance: float = 1e-6,
    ):
        if n_steps <= 0 or batch_size <= 0:
            raise ValueError("n_steps and batch_size must be greater than zero")
        if beam_size != 1:
            raise ValueError("MLMGuidedAttack currently supports beam_size=1")
        if threshold < 0 or improvement_tolerance < 0:
            raise ValueError("thresholds must be non-negative")
        if not 0 < max_token_edit_rate <= 1:
            raise ValueError("max_token_edit_rate must be in (0, 1]")

        self.scorer = scorer
        self.n_steps = n_steps
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_token_edit_rate = max_token_edit_rate
        self.improvement_tolerance = improvement_tolerance
        self.generator = candidate_generator or MLMGuidedCandidateGenerator(
            mlm_model_name=mlm_model_name,
            similarity_model_name=similarity_model_name,
            device=scorer.device,
            dtype=dtype,
            n_sample_pos=n_sample_pos,
            top_k_per_pos=top_k_per_pos,
            max_candidates=max_candidates_per_step,
            minimum_similarity=minimum_similarity,
            mlm_max_length=mlm_max_length,
        )

    def _victim_token_ids(self, text: str) -> Tuple[int, ...]:
        ids = self.scorer.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=1024,
        )["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return tuple(int(token_id) for token_id in ids)

    def attack(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        original_score = float(self.scorer.score_single(text))
        original_token_count = len(self._victim_token_ids(text))
        edit_budget = min(
            self.n_steps,
            int(original_token_count * self.max_token_edit_rate),
        )
        if edit_budget <= 0:
            return text, []

        best_text = text
        best_score = original_score
        best_token_ids = self._victim_token_ids(text)
        history: List[Dict[str, Any]] = []
        accepted_edits = 0
        visited = {text}

        for step in range(self.n_steps):
            if accepted_edits >= edit_budget:
                break
            proposed = self.generator.generate(
                best_text,
                reference_text=text,
            )
            candidates: List[Dict[str, Any]] = []
            candidate_token_ids: List[Tuple[int, ...]] = []
            for candidate in proposed:
                candidate_text = candidate["text"]
                if candidate_text in visited:
                    continue
                visited.add(candidate_text)
                token_ids = self._victim_token_ids(candidate_text)
                if token_ids == best_token_ids:
                    continue
                candidates.append(candidate)
                candidate_token_ids.append(token_ids)
            if not candidates:
                continue

            scores = self.scorer.score_batch(
                [candidate["text"] for candidate in candidates],
                batch_size=self.batch_size,
            )
            best_index = max(
                range(len(candidates)),
                key=lambda index: float(scores[index]),
            )
            candidate_score = float(scores[best_index])
            if candidate_score > best_score + self.improvement_tolerance:
                previous_score = best_score
                selected = candidates[best_index]
                best_text = selected["text"]
                best_token_ids = candidate_token_ids[best_index]
                best_score = candidate_score
                accepted_edits += 1
                history.append(
                    {
                        "step": step,
                        "score": best_score,
                        "step_gain": best_score - previous_score,
                        "delta": best_score - original_score,
                        "accepted_edit_count": accepted_edits,
                        "max_edits": edit_budget,
                        "mlm_position": selected["mlm_position"],
                        "mlm_old_id": selected["mlm_old_id"],
                        "mlm_new_id": selected["mlm_new_id"],
                        "cosine_similarity": selected["cosine_similarity"],
                    }
                )
            if best_score - original_score >= self.threshold:
                break

        return best_text, history

    def attack_batch(self, texts: List[str]):
        return [self.attack(text) for text in texts]
