"""
HotFlipAttack: white-box gradient-guided token replacement.

改进：
1. Batch scoring：每步候选分批评分，GPU 利用率高且不 OOM
2. 限制候选数量：默认每步最多评 16 个候选
3. 候选先 decode 为文本，再由 victim tokenizer 重新编码并真实评分
"""
from __future__ import annotations

import random
import sys as _sys
import os as _os
from typing import Dict, List, Tuple

import numpy as np
import torch

_d = _os.path.dirname
_repo_root = _d(_d(_d(_d(_d(_os.path.abspath(__file__))))))
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import (
    _sample_positions,
    _topk_per_position,
)

__all__ = ["HotFlipAttack"]


class HotFlipAttack:
    """
    White-box gradient-guided token replacement.

    Batch scoring 实现：
    - 每步收集候选后，分批评分（batch_size=32）
    - 默认每步最多评 16 个候选
    - 候选先还原成文本，再由 AES tokenizer 编码并评分
    """

    def __init__(
        self,
        scorer,
        *,
        n_steps: int = 30,
        beam_size: int = 1,                 # 贪心（beam=1）
        n_sample_pos: int = 8,
        top_k_per_pos: int = 2,             # 每位置 top_k，凑够 max_candidates
        max_candidates_per_step: int = 16,  # 每步最多评 16 个候选
        batch_size: int = 32,               # 评分批次大小
        threshold: float = 0.1,             # 成功阈值（pert - orig >= threshold）
        max_token_edit_rate: float | None = 0.1,
        specials=None,
        record_intermediate_texts: bool = False,
    ):
        if n_steps <= 0:
            raise ValueError("n_steps must be greater than zero")
        if beam_size <= 0:
            raise ValueError("beam_size must be greater than zero")
        if n_sample_pos <= 0 or top_k_per_pos <= 0:
            raise ValueError("candidate sampling parameters must be greater than zero")
        if max_candidates_per_step <= 0:
            raise ValueError("max_candidates_per_step must be greater than zero")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if max_token_edit_rate is not None and not 0 < max_token_edit_rate <= 1:
            raise ValueError("max_token_edit_rate must be in (0, 1]")
        self.scorer = scorer
        self.n_steps = n_steps
        self.beam_size = beam_size
        self.n_sample_pos = n_sample_pos
        self.top_k_per_pos = top_k_per_pos
        self.max_candidates_per_step = max_candidates_per_step
        self.batch_size = batch_size
        self.threshold = threshold
        self.max_token_edit_rate = max_token_edit_rate
        self.record_intermediate_texts = record_intermediate_texts
        tok = scorer.tokenizer
        self.specials = specials if specials is not None else set(tok.all_special_ids)

    def _encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.scorer.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        return (
            inputs["input_ids"].squeeze(0),
            inputs["attention_mask"].squeeze(0),
        )

    def attack(self, text: str) -> Tuple[str, List[Dict]]:
        device = self.scorer.device
        model = self.scorer.model
        tokenizer = self.scorer.tokenizer

        input_ids, _ = self._encode(text)
        original_score = self.scorer.score_single(text)
        editable_positions = _sample_positions(
            input_ids,
            (0, input_ids.size(0)),
            self.specials,
        )
        if not editable_positions:
            return text, []

        max_steps = self.n_steps
        if self.max_token_edit_rate is not None:
            edit_budget = int(
                len(editable_positions) * self.max_token_edit_rate
            )
            if edit_budget == 0:
                return text, []
            max_steps = min(max_steps, edit_budget)

        beams: List[Tuple[str, float, List[Dict]]] = [
            (text, original_score, [])
        ]
        best_text = text
        best_score = original_score
        best_history: List[Dict] = []
        visited = {text}

        for step in range(max_steps):
            emb_layer = model.get_input_embeddings()
            E = emb_layer.weight.detach().float()

            candidates: List[Dict] = []

            for beam_text, beam_score, beam_history in beams:
                beam_ids, attention_mask = self._encode(beam_text)
                ids_list = beam_ids.tolist()
                span = (0, beam_ids.size(0))

                ids_dev = beam_ids.unsqueeze(0).to(device)
                att_dev = attention_mask.unsqueeze(0).to(device)

                with torch.enable_grad():
                    emb_in = emb_layer(ids_dev).detach().clone().requires_grad_(True)
                    logits = model(inputs_embeds=emb_in, attention_mask=att_dev).logits
                    if logits.ndim > 1:
                        logits = logits.squeeze(-1)
                    (grad,) = torch.autograd.grad(logits.sum(), emb_in, retain_graph=False)

                g = grad.squeeze(0).float()

                positions = _sample_positions(beam_ids, span, self.specials)
                if not positions:
                    continue

                # 梯度幅度加权采样
                grad_mag = np.array([float(g[p].norm().item()) for p in positions])
                grad_mag_sum = grad_mag.sum()
                if grad_mag_sum > 0:
                    probs = grad_mag / grad_mag_sum
                    kpos = min(self.n_sample_pos, len(positions))
                    try:
                        sampled = list(np.random.choice(
                            positions, size=kpos, replace=False, p=probs
                        ))
                    except Exception:
                        sampled = random.sample(positions, kpos)
                else:
                    sampled = random.sample(positions, min(self.n_sample_pos, len(positions)))

                for pos in sampled:
                    old_id = int(ids_list[pos])
                    cand_ids, approximate_gains = _topk_per_position(
                        E, -g[pos], old_id, self.specials, self.top_k_per_pos
                    )
                    for cand_id, approximate_gain in zip(
                        cand_ids,
                        approximate_gains,
                    ):
                        if cand_id == old_id or cand_id in self.specials:
                            continue
                        new_ids = beam_ids.clone()
                        new_ids[pos] = cand_id
                        candidate_text = tokenizer.decode(
                            new_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip()
                        if not candidate_text or candidate_text in visited:
                            continue
                        visited.add(candidate_text)
                        candidates.append(
                            {
                                "text": candidate_text,
                                "old_id": old_id,
                                "new_id": cand_id,
                                "pos": int(pos),
                                "approximate_gain": float(approximate_gain),
                                "beam_score": beam_score,
                                "history": beam_history,
                            }
                        )

            if not candidates:
                break

            candidates.sort(
                key=lambda item: item["approximate_gain"],
                reverse=True,
            )
            candidates = candidates[: self.max_candidates_per_step]
            scores = self.scorer.score_batch(
                [candidate["text"] for candidate in candidates],
                batch_size=self.batch_size,
            )

            all_candidates: List[Dict] = []
            for candidate, score in zip(candidates, scores):
                score = float(score)
                history_entry = {
                    "step": step,
                    "pos": candidate["pos"],
                    "old_id": candidate["old_id"],
                    "new_id": candidate["new_id"],
                    "score": score,
                    "step_gain": score - candidate["beam_score"],
                    "delta": score - original_score,
                }
                if self.record_intermediate_texts:
                    history_entry["before_text"] = (
                        candidate["history"][-1]["after_text"]
                        if candidate["history"]
                        else text
                    )
                    history_entry["after_text"] = candidate["text"]
                new_history = candidate["history"] + [history_entry]
                all_candidates.append({
                    "text": candidate["text"],
                    "score": score,
                    "history": new_history,
                })

            # beam search：统一排序，保留 top beam_size
            all_candidates.sort(key=lambda x: -x["score"])
            top_candidates = all_candidates[:self.beam_size]

            new_beams: List[Tuple[str, float, List[Dict]]] = []
            for cand in top_candidates:
                new_beams.append((cand["text"], cand["score"], cand["history"]))
                if cand["score"] > best_score:
                    best_score = cand["score"]
                    best_text = cand["text"]
                    best_history = cand["history"]

            beams = new_beams

            if best_score - original_score >= self.threshold:
                break

        return best_text, best_history

    def attack_batch(self, texts: List[str]) -> List[Tuple[str, List[Dict]]]:
        return [self.attack(t) for t in texts]
