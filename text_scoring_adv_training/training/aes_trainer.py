#!/usr/bin/env python3
"""
AES Stage-Two Training — C0 clean continuation and adversarial defenses.

Defense modes use the current model to select one-step score-inflating
perturbations and optimize:
  Loss = MSE(clean_score, label) + attack_weight * one_sided_inflation_loss

C0 mode 只计算 clean MSE。通用论文代码保持只读；AES 专用候选筛选和损失
在本文件中实现。
"""
from __future__ import annotations

import json, math, os, random, sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import (
    _sample_positions,
    _topk_per_position,
)
from text_scoring_adv_training.evaluation.robustness_tests.common.rudimentary_edits import (
    sample_variants,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AESAdversarialConfig:
    training_mode: str = "hotflip_defense"
    checkpoint_path: str = "/root/autodl-tmp/victim/fold0_best"
    train_csv: str = "/root/autodl-tmp/data/train_fold0.csv"
    valid_csv: str = "/root/autodl-tmp/data/valid_fold0.csv"
    output_dir: str = "/root/autodl-tmp/aes_adv_training"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4   # 减小避免OOM
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 1024
    seed: int = 42
    eval_every: int = 200
    save_every: int = 200
    precision: str = "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    clean_loss_weight: float = 1.0
    show_progress: bool = True
    use_hotflip_swaps: bool = True
    hotflip_weight: float = 1.0
    hotflip_n_sample_pos: int = 8
    hotflip_top_k_per_pos: int = 2
    hotflip_max_candidates: int = 16
    hotflip_fraction: float = 0.5
    hotflip_tolerance: float = 0.05
    # Backward compatibility for old launcher_config.json files.  The old
    # "margin" is interpreted as the new one-sided tolerance.
    hotflip_margin: Optional[float] = None
    use_rudimentary_edits: bool = False
    rudimentary_weight: float = 1.0
    rudimentary_candidates: int = 16
    rudimentary_fraction: float = 0.5
    rudimentary_tolerance: float = 0.05
    rudimentary_improvement_tolerance: float = 1e-6

    def __post_init__(self):
        if self.hotflip_margin is not None:
            self.hotflip_tolerance = float(self.hotflip_margin)
        if self.training_mode not in (
            "clean_continuation",
            "hotflip_defense",
            "rudimentary_defense",
        ):
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError(f"Unsupported precision: {self.precision}")
        expected_hotflip = self.training_mode == "hotflip_defense"
        expected_rudimentary = self.training_mode == "rudimentary_defense"
        if self.use_hotflip_swaps != expected_hotflip:
            raise ValueError(
                "training_mode and use_hotflip_swaps disagree: "
                f"{self.training_mode=}, {self.use_hotflip_swaps=}"
            )
        if self.use_rudimentary_edits != expected_rudimentary:
            raise ValueError(
                "training_mode and use_rudimentary_edits disagree: "
                f"{self.training_mode=}, {self.use_rudimentary_edits=}"
            )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class KaggleEssayDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: PreTrainedTokenizer,
                 max_length: int = 1024, label_offset: int = 1,
                 max_samples: int | None = None):
        df = pd.read_csv(csv_path)
        text_col = "full_text" if "full_text" in df.columns else "text"
        texts = df[text_col].tolist()
        scores = df["score"].tolist()
        if max_samples:
            texts, scores = texts[:max_samples], scores[:max_samples]
        self.items = [{"text": str(t), "label": float(s) - label_offset}
                      for t, s in zip(texts, scores)]

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


class AESCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]):
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)
        enc = self.tokenizer(texts, return_tensors="pt",
                             truncation=True, max_length=self.max_length, padding=True)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "texts": texts,
        }


# ---------------------------------------------------------------------------
# QWK
# ---------------------------------------------------------------------------

def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    true_bins = np.clip(np.round(y_true).astype(int), 0, 5)
    pred_bins = np.clip(np.round(y_pred).astype(int), 0, 5)
    return cohen_kappa_score(true_bins, pred_bins, weights="quadratic")


# ---------------------------------------------------------------------------
# AES-specific adversarial objective
# ---------------------------------------------------------------------------

def one_sided_score_inflation_loss(
    clean_scores: torch.Tensor,
    adversarial_scores: torch.Tensor,
    target_scores: torch.Tensor,
    tolerance: float = 0.05,
    relative_weight: float = 0.5,
) -> torch.Tensor:
    """Penalize score inflation without pushing the clean prediction upward."""
    gold_excess = torch.relu(
        adversarial_scores - target_scores - tolerance
    ).pow(2)
    relative_excess = torch.relu(
        adversarial_scores - clean_scores.detach() - tolerance
    ).pow(2)
    return gold_excess.mean() + relative_weight * relative_excess.mean()


def build_adamw_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """Match Hugging Face Trainer's decay/no-decay parameter separation."""
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]


def tensor_to_float_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert CUDA/autocast outputs to a NumPy-compatible float array."""
    return tensor.detach().float().cpu().numpy()


# ---------------------------------------------------------------------------
# HotFlip: one row
# ---------------------------------------------------------------------------

def run_hotflip_one_row(
    ids: torch.Tensor,   # [seq_len]
    att: torch.Tensor,   # [seq_len]
    scorer: AESScorer,
    specials: set,
    n_sample_pos: int,
    top_k_per_pos: int,
    max_candidates: int,
    device: str,
    autocast_dtype: Optional[torch.dtype] = None,
) -> Optional[torch.Tensor]:
    nonpad_positions = torch.nonzero(att, as_tuple=False).flatten()
    if nonpad_positions.numel() == 0:
        return None
    span = (
        int(nonpad_positions[0].item()),
        int(nonpad_positions[-1].item()) + 1,
    )

    model = scorer.model
    token_embeddings = model.get_input_embeddings()
    ids_dev = ids.to(device)
    att_dev = att.to(device)
    was_training = model.training

    try:
        # Generate candidates deterministically with dropout disabled.
        model.eval()
        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            )
            if autocast_dtype is not None and str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.enable_grad(), autocast_context:
            emb_in = (
                token_embeddings(ids_dev.unsqueeze(0))
                .detach()
                .clone()
                .requires_grad_(True)
            )
            logits = model(
                inputs_embeds=emb_in,
                attention_mask=att_dev.unsqueeze(0),
            ).logits.squeeze(-1)
            (grad,) = torch.autograd.grad(logits.sum(), emb_in)

        gradient = grad.squeeze(0).float()
        embedding_matrix = token_embeddings.weight.detach().float()
        positions = _sample_positions(ids_dev, span, specials)
        if not positions:
            return None

        sampled = random.sample(
            positions,
            min(max(1, int(n_sample_pos)), len(positions)),
        )
        pooled: List[Dict[str, Any]] = []
        for pos in sampled:
            old_id = int(ids_dev[pos].item())
            candidate_ids, candidate_gains = _topk_per_position(
                embedding_matrix,
                -gradient[pos],
                old_id,
                specials,
                max(1, int(top_k_per_pos)),
            )
            for new_id, gain in zip(candidate_ids, candidate_gains):
                pooled.append(
                    {
                        "pos": pos,
                        "old_id": old_id,
                        "new_id": int(new_id),
                        "gain": float(gain),
                    }
                )

        if not pooled:
            return None
        pooled.sort(key=lambda item: item["gain"], reverse=True)
        pooled = pooled[: max(1, int(max_candidates))]

        candidate_tensors: List[torch.Tensor] = []
        for candidate in pooled:
            swapped = ids_dev.clone()
            swapped[candidate["pos"]] = candidate["new_id"]
            candidate_tensors.append(swapped)

        # Gradient gains are only a proposal mechanism.  Select the actual
        # highest-scoring candidate with a real model forward pass.
        candidate_batch = torch.stack(candidate_tensors)
        candidate_attention = att_dev.unsqueeze(0).expand(
            candidate_batch.size(0),
            -1,
        )
        autocast_context = (
            torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
            )
            if autocast_dtype is not None and str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            scores = model(
                input_ids=candidate_batch,
                attention_mask=candidate_attention,
            ).logits.squeeze(-1)
        best_index = int(torch.argmax(scores).item())
        return candidate_tensors[best_index].detach()
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Rudimentary: one row
# ---------------------------------------------------------------------------

def run_rudimentary_one_row(
    text: str,
    scorer: AESScorer,
    *,
    max_candidates: int,
    max_length: int,
    device: str,
    autocast_dtype: Optional[torch.dtype] = None,
    improvement_tolerance: float = 1e-6,
) -> Optional[str]:
    """Select one effective paper-style edit using true current-model scores."""
    if max_candidates <= 0:
        raise ValueError("max_candidates must be greater than zero")
    if improvement_tolerance < 0:
        raise ValueError("improvement_tolerance must be non-negative")

    tokenizer = scorer.tokenizer
    original_ids = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    if hasattr(original_ids, "tolist"):
        original_ids = original_ids.tolist()
    original_ids = tuple(int(token_id) for token_id in original_ids)

    candidates: List[str] = []
    seen_token_sequences = {original_ids}
    for candidate in sample_variants(text, max_candidates):
        if not candidate.strip():
            continue
        candidate_ids = tokenizer(
            candidate,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if hasattr(candidate_ids, "tolist"):
            candidate_ids = candidate_ids.tolist()
        token_sequence = tuple(int(token_id) for token_id in candidate_ids)
        if token_sequence in seen_token_sequences:
            continue
        seen_token_sequences.add(token_sequence)
        candidates.append(candidate)

    if not candidates:
        return None

    # Score the original and candidates in one padded batch.  This prevents
    # dynamic-padding roundoff from being mistaken for an improvement.
    scored_texts = [text, *candidates]
    encoded = tokenizer(
        scored_texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    model = scorer.model
    was_training = model.training
    try:
        model.eval()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None and str(device).startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_context:
            scores = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits.squeeze(-1)
        best_candidate_index = int(torch.argmax(scores[1:]).item()) + 1
        if (
            float(scores[best_candidate_index].item())
            <= float(scores[0].item()) + improvement_tolerance
        ):
            return None
        return candidates[best_candidate_index - 1]
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AESAdversarialTrainer:
    def __init__(self, config: AESAdversarialConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _set_seed(config.seed)
        self.autocast_dtype: Optional[torch.dtype] = None
        if config.precision == "bfloat16":
            if self.device == "cuda":
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError(
                        "bfloat16 was requested but this CUDA device does not "
                        "report bf16 support"
                    )
                self.autocast_dtype = torch.bfloat16
            else:
                print(
                    "[WARN] bfloat16 requested without CUDA; using float32.",
                    flush=True,
                )

        print(f"Loading scorer from {config.checkpoint_path}...", flush=True)
        self.scorer = AESScorer(config.checkpoint_path, device=self.device, dtype=torch.float32)
        self.tokenizer = self.scorer.tokenizer
        self.scorer.model.train()

        specials = set(self.tokenizer.all_special_ids)
        if self.tokenizer.pad_token_id is not None:
            specials.add(self.tokenizer.pad_token_id)
        self.specials = specials

        self.train_dataset = KaggleEssayDataset(config.train_csv, self.tokenizer, config.max_length)
        self.valid_dataset = KaggleEssayDataset(config.valid_csv, self.tokenizer, config.max_length)
        print(f"Train: {len(self.train_dataset)}, Valid: {len(self.valid_dataset)}", flush=True)

        self.collator = AESCollator(self.tokenizer, config.max_length)

        optimizer_groups = build_adamw_parameter_groups(
            self.scorer.model,
            config.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
        total_steps = self._total_steps()
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = self._make_scheduler(total_steps, warmup_steps)

        os.makedirs(config.output_dir, exist_ok=True)
        resolved_config = asdict(config)
        resolved_config["device"] = self.device
        resolved_config["effective_batch_size"] = (
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
        )
        resolved_config["autocast_enabled"] = self.autocast_dtype is not None
        Path(config.output_dir, "training_config.json").write_text(
            json.dumps(resolved_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"Mode={config.training_mode} precision={config.precision} "
            f"effective_batch_size={resolved_config['effective_batch_size']}",
            flush=True,
        )

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=self.autocast_dtype,
        )

    def _total_steps(self) -> int:
        cfg = self.config
        steps_per_epoch = math.ceil(len(self.train_dataset) / cfg.per_device_train_batch_size)
        total_micro_batches = steps_per_epoch * cfg.num_epochs
        return math.ceil(total_micro_batches / cfg.gradient_accumulation_steps)

    def _make_scheduler(self, total_steps: int, warmup_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _adversarial_mask(self, batch_size: int) -> List[bool]:
        cfg = self.config
        if cfg.use_hotflip_swaps:
            fraction = cfg.hotflip_fraction
        elif cfg.use_rudimentary_edits:
            fraction = cfg.rudimentary_fraction
        else:
            return [False] * batch_size
        n = int(batch_size * fraction)
        indices = random.sample(range(batch_size), min(n, batch_size))
        return [i in indices for i in range(batch_size)]

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.scorer.model.eval()
        loader = DataLoader(self.valid_dataset,
                           batch_size=self.config.per_device_train_batch_size * 4,
                           shuffle=False, collate_fn=self.collator)
        preds, labels_list = [], []
        for batch in tqdm(
            loader,
            desc="Validation",
            unit="batch",
            dynamic_ncols=True,
            disable=True if not self.config.show_progress else None,
        ):
            b = {
                key: value.to(self.device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            with self._autocast_context():
                logits = self.scorer.model(
                    input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"],
                ).logits.squeeze(-1)
            preds.extend(tensor_to_float_numpy(logits))
            labels_list.extend(tensor_to_float_numpy(b["labels"]))
        preds, labels_list = np.array(preds), np.array(labels_list)
        metrics = {"qwk": compute_qwk(labels_list, preds),
                    "mae": float(np.mean(np.abs(preds - labels_list))),
                    "rmse": float(np.sqrt(np.mean((preds - labels_list) ** 2)))}
        self.scorer.model.train()
        return metrics

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        cfg = self.config
        b = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = b["labels"]

        # --- Clean forward ---
        with self._autocast_context():
            clean_logits = self.scorer.model(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
            ).logits.squeeze(-1)

        # --- One-step adversarial candidate generation ---
        adversarial_mask = self._adversarial_mask(b["input_ids"].size(0))
        adversarial_indices = [
            index
            for index, selected in enumerate(adversarial_mask)
            if selected
        ]
        loss_indices = list(adversarial_indices)

        if cfg.use_hotflip_swaps and adversarial_indices:
            # Run hotflip per-row
            adv_ids_list = []
            for i in range(b["input_ids"].size(0)):
                if adversarial_mask[i]:
                    swap = run_hotflip_one_row(
                        b["input_ids"][i], b["attention_mask"][i],
                        self.scorer, self.specials,
                        cfg.hotflip_n_sample_pos, cfg.hotflip_top_k_per_pos,
                        cfg.hotflip_max_candidates, self.device,
                        self.autocast_dtype,
                    )
                    adv_ids_list.append(swap if swap is not None else b["input_ids"][i])
                else:
                    adv_ids_list.append(b["input_ids"][i])

            hf_ids = torch.stack(adv_ids_list)
            with self._autocast_context():
                adv_logits = self.scorer.model(
                    input_ids=hf_ids,
                    attention_mask=b["attention_mask"],
                ).logits.squeeze(-1)
        elif cfg.use_rudimentary_edits and adversarial_indices:
            adversarial_texts = list(b["texts"])
            changed_indices: List[int] = []
            for index in adversarial_indices:
                perturbed_text = run_rudimentary_one_row(
                    b["texts"][index],
                    self.scorer,
                    max_candidates=cfg.rudimentary_candidates,
                    max_length=cfg.max_length,
                    device=self.device,
                    autocast_dtype=self.autocast_dtype,
                    improvement_tolerance=(
                        cfg.rudimentary_improvement_tolerance
                    ),
                )
                if perturbed_text is not None:
                    adversarial_texts[index] = perturbed_text
                    changed_indices.append(index)

            loss_indices = changed_indices
            if changed_indices:
                adversarial_batch = self.tokenizer(
                    adversarial_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=cfg.max_length,
                    padding=True,
                )
                with self._autocast_context():
                    adv_logits = self.scorer.model(
                        input_ids=adversarial_batch["input_ids"].to(
                            self.device
                        ),
                        attention_mask=adversarial_batch[
                            "attention_mask"
                        ].to(self.device),
                    ).logits.squeeze(-1)
            else:
                adv_logits = clean_logits.detach()
        else:
            adv_logits = clean_logits.detach()

        # --- Losses ---
        base_loss = F.mse_loss(clean_logits.float(), labels.float())

        if loss_indices:
            if cfg.use_hotflip_swaps:
                attack_weight = cfg.hotflip_weight
                attack_tolerance = cfg.hotflip_tolerance
            else:
                attack_weight = cfg.rudimentary_weight
                attack_tolerance = cfg.rudimentary_tolerance
            adversarial_loss = one_sided_score_inflation_loss(
                clean_logits[loss_indices].float(),
                adv_logits[loss_indices].float(),
                labels[loss_indices].float(),
                tolerance=attack_tolerance,
            )
            total_loss = (
                cfg.clean_loss_weight * base_loss
                + attack_weight * adversarial_loss
            )
        else:
            adversarial_loss = torch.tensor(0.0, device=self.device)
            total_loss = cfg.clean_loss_weight * base_loss

        # --- Backward with gradient accumulation ---
        (total_loss / cfg.gradient_accumulation_steps).backward()

        metrics = {
            "base_loss": base_loss.item(),
            "hotflip_loss": adversarial_loss.item(),
            "adversarial_loss": adversarial_loss.item(),
            "total_loss": total_loss.item(),
            "n_adversarial": len(loss_indices),
        }
        return metrics

    def _optimizer_step(self, accumulated_micro_batches: int) -> None:
        cfg = self.config
        if accumulated_micro_batches <= 0:
            return
        if accumulated_micro_batches < cfg.gradient_accumulation_steps:
            correction = cfg.gradient_accumulation_steps / accumulated_micro_batches
            for parameter in self.scorer.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(self.scorer.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def train(self):
        cfg = self.config
        global_step = 0
        best_qwk = -1.0
        accumulated_micro_batches = 0

        data_generator = torch.Generator()
        data_generator.manual_seed(cfg.seed)
        train_loader = DataLoader(
            self.train_dataset, batch_size=cfg.per_device_train_batch_size,
            shuffle=True, collate_fn=self.collator, num_workers=0, drop_last=False,
            generator=data_generator,
        )
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(cfg.num_epochs):
            self.scorer.model.train()
            epoch_loss, epoch_steps = 0.0, 0

            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.num_epochs}",
                unit="batch",
                dynamic_ncols=True,
                disable=True if not cfg.show_progress else None,
            )
            for batch in train_progress:
                metrics = self.train_step(batch)
                epoch_loss += metrics["total_loss"]
                epoch_steps += 1
                accumulated_micro_batches += 1
                did_optimizer_step = False

                if accumulated_micro_batches == cfg.gradient_accumulation_steps:
                    self._optimizer_step(accumulated_micro_batches)
                    accumulated_micro_batches = 0
                    global_step += 1
                    did_optimizer_step = True

                if epoch_steps % 20 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    train_progress.set_postfix(
                        gstep=global_step,
                        loss=f"{metrics['total_loss']:.4f}",
                        base=f"{metrics['base_loss']:.4f}",
                        adv=f"{metrics['adversarial_loss']:.4f}",
                        lr=f"{lr:.2e}",
                    )

                if did_optimizer_step and global_step % cfg.eval_every == 0:
                    eval_metrics = self.evaluate()
                    print(f"[eval gstep={global_step}] "
                          f"QWK={eval_metrics['qwk']:.4f} "
                          f"MAE={eval_metrics['mae']:.4f}", flush=True)
                    if eval_metrics["qwk"] > best_qwk:
                        best_qwk = eval_metrics["qwk"]
                        self.save("best")
                        print(f"  → New best QWK: {best_qwk:.4f}", flush=True)
                    self.scorer.model.train()

                if did_optimizer_step and global_step % cfg.save_every == 0:
                    self.save(f"gstep{global_step}")

            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"Epoch {epoch+1}/{cfg.num_epochs} done. Avg loss: {avg_loss:.4f}", flush=True)

        if accumulated_micro_batches:
            self._optimizer_step(accumulated_micro_batches)
            global_step += 1

        final_metrics = self.evaluate()
        print(f"Final — QWK={final_metrics['qwk']:.4f}", flush=True)
        if final_metrics["qwk"] > best_qwk:
            best_qwk = final_metrics["qwk"]
            self.save("best")
        self.save("final")
        return best_qwk

    def save(self, tag: str):
        out_dir = os.path.join(self.config.output_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        self.scorer.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"Saved: {out_dir}", flush=True)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    cfg = AESAdversarialConfig(**json.load(open(args.config))) if args.config else AESAdversarialConfig()
    trainer = AESAdversarialTrainer(cfg)
    trainer.train()
