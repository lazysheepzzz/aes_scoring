"""
HotFlipAttack: white-box gradient-guided token replacement (greedy).
Paper-level greedy HotFlip: for each step, pick the best single-token replacement.
"""
from __future__ import annotations

import heapq
import random
import sys as _sys
import os as _os
from typing import Any, Dict, List, Tuple

import torch

_project_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import (
    _sample_positions,
    _topk_per_position,
)

__all__ = ["HotFlipAttack"]


class HotFlipAttack:
    """
    White-box gradient-guided token replacement (greedy beam).

    At each step: samples positions, computes gradient w.r.t. embeddings,
    finds top-k token replacements, evaluates them all, keeps the best.
    Repeats for max_steps. Returns best text found.
    """

    def __init__(
        self,
        scorer: "AESScorer",               # noqa: F821
        *,
        n_steps: int = 20,
        beam_size: int = 4,
        n_sample_pos: int = 16,
        top_k_per_pos: int = 8,
        specials: set | None = None,
    ):
        self.scorer = scorer
        self.n_steps = n_steps
        self.beam_size = beam_size
        self.n_sample_pos = n_sample_pos
        self.top_k_per_pos = top_k_per_pos
        self.specials = specials or set()

    def attack(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        device = self.scorer.device
        inputs = self.scorer.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        span = (0, len(input_ids))

        best_ids = input_ids.clone()
        with torch.no_grad():
            best_score = float(
                self.scorer.model(
                    input_ids=input_ids.unsqueeze(0).to(device),
                    attention_mask=attention_mask.unsqueeze(0).to(device),
                ).logits.squeeze(-1).item()
            )

        swap_history: List[Dict[str, Any]] = []

        for step in range(self.n_steps):
            # ---- compute gradient ----
            emb_layer = self.scorer.get_embeddings()
            E = emb_layer.weight.detach().float()
            ids_dev = input_ids.unsqueeze(0).to(device)
            att_dev = attention_mask.unsqueeze(0).to(device)

            with torch.enable_grad():
                emb_in = emb_layer(ids_dev).detach().clone().requires_grad_(True)
                logits = self.scorer.model(
                    inputs_embeds=emb_in, attention_mask=att_dev
                ).logits
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                (grad,) = torch.autograd.grad(logits.sum(), emb_in, retain_graph=False)

            g = grad.squeeze(0).float()
            ids_list = ids_dev.squeeze(0).tolist()

            # ---- sample positions ----
            positions = _sample_positions(input_ids, span, self.specials)
            if not positions:
                break
            kpos = min(self.n_sample_pos, len(positions))
            sampled = random.sample(positions, kpos)

            # ---- collect candidates ----
            pooled: List[Dict[str, Any]] = []
            for pos in sampled:
                old_id = int(ids_list[pos])
                cand_ids, cand_gains = _topk_per_position(
                    E, -g[pos], old_id, self.specials, self.top_k_per_pos
                )
                for nid, gain in zip(cand_ids, cand_gains):
                    if gain == float("-inf"):
                        continue
                    pooled.append({"pos": pos, "old_id": old_id, "new_id": int(nid), "gain": float(gain)})

            if not pooled:
                break

            top_candidates = heapq.nlargest(
                min(self.beam_size, len(pooled)), pooled, key=lambda x: x["gain"]
            )

            # ---- evaluate candidates ----
            improved = False
            with torch.no_grad():
                for cand in top_candidates:
                    new_ids = input_ids.clone()
                    new_ids[cand["pos"]] = cand["new_id"]
                    score = float(
                        self.scorer.model(
                            input_ids=new_ids.unsqueeze(0).to(device),
                            attention_mask=attention_mask.unsqueeze(0).to(device),
                        ).logits.squeeze(-1).item()
                    )
                    if score > best_score:
                        best_score = score
                        best_ids = new_ids
                        improved = True
                        swap_history.append({
                            "step": step,
                            "pos": cand["pos"],
                            "old_id": cand["old_id"],
                            "new_id": cand["new_id"],
                            "score": score,
                            "gain": score - float(
                                self.scorer.model(
                                    input_ids=input_ids.unsqueeze(0).to(device),
                                    attention_mask=attention_mask.unsqueeze(0).to(device),
                                ).logits.squeeze(-1).item()
                            ),
                        })

            if improved:
                input_ids = best_ids
            else:
                break

        best_text = self.scorer.tokenizer.decode(best_ids, skip_special_tokens=True)
        return best_text, swap_history

    def attack_batch(self, texts: List[str]) -> List[Tuple[str, List[Dict[str, Any]]]]:
        return [self.attack(t) for t in texts]
