"""
HotFlipAttack: white-box gradient-guided token replacement.

改进：
1. Batch scoring：每步候选分批评分，GPU 利用率高且不 OOM
2. 限制候选数量：每步最多评 64 个候选
3. 不用 padding：直接过 model，避免显存浪费
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
    - 每步最多评 64 个候选（2批）
    - 直接用 input_ids list 过 model，不 padding
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
        threshold: float = 0.0,             # 成功阈值（pert - orig >= threshold）
        specials=None,
    ):
        self.scorer = scorer
        self.n_steps = n_steps
        self.beam_size = beam_size
        self.n_sample_pos = n_sample_pos
        self.top_k_per_pos = top_k_per_pos
        self.max_candidates_per_step = max_candidates_per_step
        self.batch_size = batch_size
        self.threshold = threshold
        tok = scorer.tokenizer
        self.specials = specials if specials is not None else set(tok.all_special_ids)

    def _score_batch(self, ids_list: List[torch.Tensor], attention_mask_list: List[torch.Tensor]) -> List[float]:
        """Batch score 多个候选，返回分数列表。"""
        device = self.scorer.device
        model = self.scorer.model
        n = len(ids_list)
        scores = []

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_ids = torch.nn.utils.rnn.pad_sequence(
                [ids_list[i] for i in range(start, end)],
                batch_first=True, padding_value=self.scorer.tokenizer.pad_token_id or 0
            ).to(device)
            batch_mask = (batch_ids != self.scorer.tokenizer.pad_token_id).long().to(device)

            with torch.no_grad():
                logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                scores.extend(logits.tolist())

        return scores

    def attack(self, text: str) -> Tuple[str, List[Dict]]:
        device = self.scorer.device
        model = self.scorer.model
        tokenizer = self.scorer.tokenizer

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        span = (0, input_ids.size(0))

        with torch.no_grad():
            original_score = float(
                model(
                    input_ids=input_ids.unsqueeze(0).to(device),
                    attention_mask=attention_mask.unsqueeze(0).to(device),
                ).logits.squeeze(-1).item()
            )

        beams: List[Tuple[torch.Tensor, float, List[Dict]]] = [
            (input_ids.clone(), original_score, [])
        ]
        best_ids = input_ids.clone()
        best_score = original_score

        for step in range(self.n_steps):
            emb_layer = model.get_input_embeddings()
            E = emb_layer.weight.detach().float()

            candidate_ids: List[Tuple[torch.Tensor, int, int, int, List[Dict]]] = []

            for beam_ids, beam_score, beam_history in beams:
                ids_list = beam_ids.tolist()

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
                    cand_ids, _ = _topk_per_position(
                        E, -g[pos], old_id, self.specials, self.top_k_per_pos
                    )
                    for cand_id in cand_ids:
                        if cand_id == old_id or cand_id in self.specials:
                            continue
                        new_ids = beam_ids.clone()
                        new_ids[pos] = cand_id
                        candidate_ids.append((new_ids, old_id, cand_id, pos, beam_history))

            if not candidate_ids:
                break

            # 限制候选数量
            if len(candidate_ids) > self.max_candidates_per_step:
                # 按梯度幅度排序，优先保留梯度大的位置的候选
                cand_with_grad = []
                for new_ids, old_id, cand_id, pos, history in candidate_ids:
                    grad_mag = float(g[pos].norm().item()) if pos < len(g) else 0.0
                    cand_with_grad.append((grad_mag, new_ids, old_id, cand_id, pos, history))
                cand_with_grad.sort(key=lambda x: -x[0])
                candidate_ids = cand_with_grad[:self.max_candidates_per_step]
                candidate_ids = [(c[1], c[2], c[3], c[4], c[5]) for c in candidate_ids]

            # Batch score 所有候选
            ids_batch = [c[0] for c in candidate_ids]
            att_batch = [
                torch.ones(c.size(0), dtype=torch.long) for c in ids_batch
            ]
            scores = self._score_batch(ids_batch, att_batch)

            all_candidates: List[Dict] = []
            for i, (new_ids, old_id, cand_id, pos, history) in enumerate(candidate_ids):
                sc = float(scores[i])
                new_history = history + [{
                    "step": step, "pos": pos,
                    "old_id": old_id, "new_id": cand_id,
                    "score": sc, "gain": sc - original_score,
                }]
                all_candidates.append({
                    "ids": new_ids,
                    "score": sc,
                    "history": new_history,
                })

            # beam search：统一排序，保留 top beam_size
            all_candidates.sort(key=lambda x: -x["score"])
            top_candidates = all_candidates[:self.beam_size]

            new_beams: List[Tuple[torch.Tensor, float, List[Dict]]] = []
            for cand in top_candidates:
                new_beams.append((cand["ids"], cand["score"], cand["history"]))
                if cand["score"] > best_score:
                    best_score = cand["score"]
                    best_ids = cand["ids"].clone()

            beams = new_beams

            if best_score - original_score >= self.threshold:
                break

        best_text = tokenizer.decode(best_ids, skip_special_tokens=True)
        best_beam_history = beams[0][2] if beams else []
        return best_text, best_beam_history

    def attack_batch(self, texts: List[str]) -> List[Tuple[str, List[Dict]]]:
        return [self.attack(t) for t in texts]
