#!/usr/bin/env python3
"""
AES HotFlip Adversarial Training — 在线对抗训练（原论文方案B）

每步用当前模型生成梯度引导的token替换，计算：
  Loss = MSE(clean_score, label) + hotflip_weight * hinge(clean_score, adv_score)

复用：
- AESScorer (evaluation/aes/scorer.py)
- hotflip_pointwise (evaluation/robustness_tests/common/hotflip.py)
- hinge_loss (training/losses.py)
"""
from __future__ import annotations

import json, math, os, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.robustness_tests.common.hotflip import hotflip_pointwise
from text_scoring_adv_training.training.losses import hinge_loss


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AESAdversarialConfig:
    checkpoint_path: str = "/root/autodl-tmp/victim/fold0_best"
    train_csv: str = "/root/autodl-tmp/data/train_fold0.csv"
    valid_csv: str = "/root/autodl-tmp/data/valid_fold0.csv"
    output_dir: str = "/root/autodl-tmp/aes_adv_training"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4   # 减小避免OOM
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    max_length: int = 1024
    seed: int = 42
    eval_every: int = 200
    save_every: int = 1000
    use_hotflip_swaps: bool = True
    hotflip_weight: float = 1.0   # 中等权重
    hotflip_n_sample_pos: int = 8
    hotflip_top_k_per_pos: int = 2
    hotflip_max_candidates: int = 16
    hotflip_fraction: float = 1.0  # 全部batch都做hotflip
    hotflip_margin: float = 0.05  # hinge margin > 0，让训练更稳定


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
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


# ---------------------------------------------------------------------------
# QWK
# ---------------------------------------------------------------------------

def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score
    true_bins = np.clip(np.round(y_true).astype(int), 0, 5)
    pred_bins = np.clip(np.round(y_pred).astype(int), 0, 5)
    return cohen_kappa_score(true_bins, pred_bins, weights="quadratic")


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
) -> Optional[torch.Tensor]:
    span = (0, int(att.sum().item()))
    swapped_ids_list, _, _ = hotflip_pointwise(
        ids=ids, att=att, span=span, scorer=scorer.model,
        specials=specials, n_sample_pos=n_sample_pos,
        top_k_overall=top_k_per_pos,
        token_embeddings=scorer.model.get_input_embeddings(),
        device=device,
    )
    if not swapped_ids_list:
        return None
    return swapped_ids_list[0].to(device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AESAdversarialTrainer:
    def __init__(self, config: AESAdversarialConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _set_seed(config.seed)

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

        self.optimizer = torch.optim.AdamW(
            self.scorer.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        total_steps = self._total_steps()
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = self._make_scheduler(total_steps, warmup_steps)

        os.makedirs(config.output_dir, exist_ok=True)

    def _total_steps(self) -> int:
        cfg = self.config
        steps_per_epoch = math.ceil(len(self.train_dataset) / cfg.per_device_train_batch_size)
        return steps_per_epoch * cfg.num_epochs // cfg.gradient_accumulation_steps

    def _make_scheduler(self, total_steps: int, warmup_steps: int):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _hotflip_mask(self, batch_size: int) -> List[bool]:
        cfg = self.config
        if not cfg.use_hotflip_swaps:
            return [False] * batch_size
        n = int(batch_size * cfg.hotflip_fraction)
        indices = random.sample(range(batch_size), min(n, batch_size))
        return [i in indices for i in range(batch_size)]

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.scorer.model.eval()
        loader = DataLoader(self.valid_dataset,
                           batch_size=self.config.per_device_train_batch_size * 4,
                           shuffle=False, collate_fn=self.collator)
        preds, labels_list = [], []
        for batch in loader:
            b = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.scorer.model(input_ids=b["input_ids"],
                                       attention_mask=b["attention_mask"]).logits.squeeze(-1)
            preds.extend(logits.cpu().numpy())
            labels_list.extend(b["labels"].cpu().numpy())
        preds, labels_list = np.array(preds), np.array(labels_list)
        metrics = {"qwk": compute_qwk(labels_list, preds),
                    "mae": float(np.mean(np.abs(preds - labels_list))),
                    "rmse": float(np.sqrt(np.mean((preds - labels_list) ** 2)))}
        self.scorer.model.train()
        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.config
        b = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = b["labels"]

        # --- Clean forward ---
        clean_logits = self.scorer.model(
            input_ids=b["input_ids"], attention_mask=b["attention_mask"]
        ).logits.squeeze(-1)

        # --- HotFlip ---
        hot_mask = self._hotflip_mask(b["input_ids"].size(0))
        n_hot = sum(hot_mask)

        if cfg.use_hotflip_swaps and n_hot > 0:
            hot_indices = [i for i, m in enumerate(hot_mask) if m]
            # Run hotflip per-row
            adv_ids_list = []
            for i in range(b["input_ids"].size(0)):
                if hot_mask[i]:
                    swap = run_hotflip_one_row(
                        b["input_ids"][i], b["attention_mask"][i],
                        self.scorer, self.specials,
                        cfg.hotflip_n_sample_pos, cfg.hotflip_top_k_per_pos,
                        cfg.hotflip_max_candidates, self.device,
                    )
                    adv_ids_list.append(swap if swap is not None else b["input_ids"][i])
                else:
                    adv_ids_list.append(b["input_ids"][i])

            hf_ids = torch.stack(adv_ids_list)
            adv_logits = self.scorer.model(input_ids=hf_ids,
                                          attention_mask=b["attention_mask"]).logits.squeeze(-1)
        else:
            adv_logits = clean_logits.detach()

        # --- Losses ---
        base_loss = F.mse_loss(clean_logits, labels)

        if cfg.use_hotflip_swaps and n_hot > 0:
            hot_indices = [i for i, m in enumerate(hot_mask) if m]
            if hot_indices:
                hl = hinge_loss(clean_logits[hot_indices], adv_logits[hot_indices],
                               margin=cfg.hotflip_margin, squared=True)
                total_loss = base_loss + cfg.hotflip_weight * hl
            else:
                hl = torch.tensor(0.0)
                total_loss = base_loss
        else:
            hl = torch.tensor(0.0)
            total_loss = base_loss

        # --- Backward with gradient accumulation ---
        (total_loss / cfg.gradient_accumulation_steps).backward()

        metrics = {
            "base_loss": base_loss.item(),
            "hotflip_loss": hl.item(),
            "total_loss": total_loss.item(),
            "n_hot": n_hot,
        }
        return metrics

    def train(self):
        cfg = self.config
        global_step = 0
        best_qwk = -1.0

        train_loader = DataLoader(
            self.train_dataset, batch_size=cfg.per_device_train_batch_size,
            shuffle=True, collate_fn=self.collator, num_workers=0, drop_last=True,
        )

        for epoch in range(cfg.num_epochs):
            self.scorer.model.train()
            epoch_loss, epoch_steps, accum_steps = 0.0, 0, 0

            for batch in train_loader:
                metrics = self.train_step(batch)
                epoch_loss += metrics["total_loss"]
                epoch_steps += 1
                accum_steps += 1

                if accum_steps % cfg.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.scorer.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                if epoch_steps % 20 == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"[epoch={epoch+1} step={epoch_steps} gstep={global_step}] "
                          f"loss={metrics['total_loss']:.4f} "
                          f"(base={metrics['base_loss']:.4f} hf={metrics['hotflip_loss']:.4f}) "
                          f"lr={lr:.2e}", flush=True)

                if global_step > 0 and global_step % cfg.eval_every == 0:
                    eval_metrics = self.evaluate()
                    print(f"[eval gstep={global_step}] "
                          f"QWK={eval_metrics['qwk']:.4f} "
                          f"MAE={eval_metrics['mae']:.4f}", flush=True)
                    if eval_metrics["qwk"] > best_qwk:
                        best_qwk = eval_metrics["qwk"]
                        self.save(f"best_gstep{global_step}")
                        print(f"  → New best QWK: {best_qwk:.4f}", flush=True)
                    self.scorer.model.train()

                if global_step > 0 and global_step % cfg.save_every == 0:
                    self.save(f"gstep{global_step}")

            avg_loss = epoch_loss / max(1, epoch_steps)
            print(f"Epoch {epoch+1}/{cfg.num_epochs} done. Avg loss: {avg_loss:.4f}", flush=True)

        final_metrics = self.evaluate()
        print(f"Final — QWK={final_metrics['qwk']:.4f}", flush=True)
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
