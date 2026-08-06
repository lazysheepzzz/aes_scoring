#!/usr/bin/env python3
"""Shared offline trainer for Mixed-AT-RH and PAER-RH.

Both modes consume exactly the same accepted Rudimentary/HotFlip edit rows.
The only intentional difference is that PAER adds counterfactual localization
supervision and directional positive-evidence correction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paer.modeling_paer import PAERForEssayScoring
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.training.aes_trainer import (
    build_adamw_parameter_groups,
    compute_qwk,
    one_sided_score_inflation_loss,
    tensor_to_float_numpy,
)


MIXED_AT_RH = "mixed_at_rh"
PAER_RH = "paer_rh"
TRAINING_MODES = (MIXED_AT_RH, PAER_RH)


@dataclass
class RHTrainingConfig:
    training_mode: str
    checkpoint_path: str
    train_csv: str
    trace_jsonl: str
    valid_csv: str
    output_dir: str
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
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
    label_offset: float = 1.0
    clean_loss_weight: float = 1.0
    adversarial_loss_weight: float = 1.0
    inflation_tolerance: float = 0.0
    relative_loss_power: float = 1.0
    localization_loss_weight: float = 0.25
    clean_false_suppression_weight: float = 0.10
    localization_positive_weight: float = 8.0
    attribution_gain_scale: float = 0.1
    correction_scale: float = 1.0
    show_progress: bool = True
    max_trace_records: int | None = None
    max_train_samples: int | None = None

    def __post_init__(self) -> None:
        if self.training_mode not in TRAINING_MODES:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
        if self.precision not in ("bfloat16", "float32"):
            raise ValueError(f"Unsupported precision: {self.precision}")
        for name in (
            "num_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "eval_every",
            "save_every",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in (
            "learning_rate",
            "clean_loss_weight",
            "adversarial_loss_weight",
            "relative_loss_power",
            "attribution_gain_scale",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        for name in (
            "weight_decay",
            "inflation_tolerance",
            "localization_loss_weight",
            "clean_false_suppression_weight",
            "correction_scale",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


class CounterfactualTraceDataset(Dataset):
    def __init__(self, path: str | Path, max_records: int | None = None):
        self.path = Path(path)
        self.items: list[dict[str, Any]] = []
        seen_record_ids: set[str] = set()
        with self.path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                required = {
                    "original_text",
                    "before_text",
                    "adversarial_text",
                    "label_score_space",
                    "step_gain",
                    "attack",
                }
                missing = required - set(item)
                if missing:
                    raise ValueError(
                        f"Trace line {line_number} is missing: {sorted(missing)}"
                    )
                if float(item["step_gain"]) <= 0:
                    continue
                if item["attack"] not in ("rudimentary", "hotflip"):
                    raise ValueError(
                        f"Unexpected training attack at line {line_number}: "
                        f"{item['attack']!r}; MLM must remain held out"
                    )
                record_id = str(item.get("record_id", f"line:{line_number}"))
                if record_id in seen_record_ids:
                    continue
                seen_record_ids.add(record_id)
                self.items.append(item)
                if max_records is not None and len(self.items) >= max_records:
                    break
        if not self.items:
            raise ValueError(f"No usable positive-gain traces found in {self.path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class PairedEssayTrainingDataset(Dataset):
    """All clean essays, with one rotating offline trace for selected rows."""

    def __init__(
        self,
        csv_path: str | Path,
        traces: CounterfactualTraceDataset,
        *,
        label_offset: float,
        max_samples: int | None = None,
    ):
        frame = pd.read_csv(csv_path)
        text_col = "full_text" if "full_text" in frame.columns else "text"
        if text_col not in frame.columns or "score" not in frame.columns:
            raise ValueError("Training CSV must contain full_text/text and score")
        if max_samples is not None:
            if max_samples <= 0:
                raise ValueError("max_train_samples must be greater than zero")
            frame = frame.iloc[:max_samples]
        self.items = [
            {
                "row_index": int(index),
                "original_text": str(row[text_col]),
                "label_score_space": float(row["score"]),
                "label": float(row["score"]) - label_offset,
            }
            for index, row in frame.iterrows()
        ]
        self.traces_by_row: dict[int, list[dict[str, Any]]] = {}
        clean_by_row = {item["row_index"]: item for item in self.items}
        valid_row_indices = set(clean_by_row)
        for trace in traces.items:
            row_index = int(trace["row_index"])
            if row_index not in valid_row_indices:
                continue
            clean_item = clean_by_row[row_index]
            if str(trace["original_text"]) != clean_item["original_text"]:
                raise ValueError(
                    f"Trace original_text does not match train CSV row {row_index}"
                )
            self.traces_by_row.setdefault(row_index, []).append(trace)
        if not self.traces_by_row:
            raise ValueError("No trace records match the selected training essays")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clean_item = dict(self.items[index])
        traces = self.traces_by_row.get(clean_item["row_index"], [])
        if not traces:
            clean_item.update(
                {
                    "before_text": clean_item["original_text"],
                    "adversarial_text": clean_item["original_text"],
                    "step_gain": 0.0,
                    "attack": "clean_only",
                    "has_adversarial": False,
                }
            )
            return clean_item
        trace = traces[self.epoch % len(traces)]
        clean_item.update(trace)
        clean_item["has_adversarial"] = True
        return clean_item


class CleanValidationDataset(Dataset):
    def __init__(self, csv_path: str | Path, label_offset: float):
        frame = pd.read_csv(csv_path)
        text_col = "full_text" if "full_text" in frame.columns else "text"
        self.items = [
            {
                "text": str(row[text_col]),
                "label": float(row["score"]) - label_offset,
            }
            for _, row in frame.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def changed_token_uplift_targets(
    before_ids: list[int],
    after_ids: list[int],
    *,
    special_ids: set[int],
    step_gain: float,
    gain_scale: float,
) -> list[float]:
    """Map one accepted edit to soft token-level positive-gain targets.

    Replacements and insertions supervise the corresponding after-text tokens.
    Pure deletions supervise the nearest remaining content token because the
    deleted token has no representation in the after-text input.
    """
    targets = [0.0] * len(after_ids)
    severity = min(max(float(step_gain) / gain_scale, 0.0), 1.0)
    if severity == 0.0:
        return targets
    matcher = SequenceMatcher(a=before_ids, b=after_ids, autojunk=False)
    changed_positions: set[int] = set()
    for tag, _a_start, _a_end, b_start, b_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if b_end > b_start:
            changed_positions.update(range(b_start, b_end))
        else:
            candidates = [b_start, b_start - 1]
            nearest = next(
                (
                    position
                    for position in candidates
                    if 0 <= position < len(after_ids)
                    and after_ids[position] not in special_ids
                ),
                None,
            )
            if nearest is not None:
                changed_positions.add(nearest)
    for position in changed_positions:
        if after_ids[position] not in special_ids:
            targets[position] = severity
    return targets


class RHTraceCollator:
    def __init__(self, tokenizer, max_length: int, label_offset: float, gain_scale: float):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_offset = label_offset
        self.gain_scale = gain_scale
        self.special_ids = set(tokenizer.all_special_ids)

    def _encode(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        clean = self._encode([str(item["original_text"]) for item in batch])
        before = self._encode([str(item["before_text"]) for item in batch])
        adversarial = self._encode([str(item["adversarial_text"]) for item in batch])
        token_targets = torch.zeros_like(
            adversarial["attention_mask"], dtype=torch.float32
        )
        localization_mask = adversarial["attention_mask"].to(torch.float32)
        adversarial_mask = torch.tensor(
            [bool(item["has_adversarial"]) for item in batch],
            dtype=torch.bool,
        )
        for row_index, item in enumerate(batch):
            before_length = int(before["attention_mask"][row_index].sum())
            after_length = int(adversarial["attention_mask"][row_index].sum())
            before_ids = before["input_ids"][row_index, :before_length].tolist()
            after_ids = adversarial["input_ids"][row_index, :after_length].tolist()
            targets = changed_token_uplift_targets(
                before_ids,
                after_ids,
                special_ids=self.special_ids,
                step_gain=float(item["step_gain"]),
                gain_scale=self.gain_scale,
            )
            token_targets[row_index, :after_length] = torch.tensor(targets)
            for position, token_id in enumerate(after_ids):
                if token_id in self.special_ids:
                    localization_mask[row_index, position] = 0.0
            if not bool(item["has_adversarial"]):
                localization_mask[row_index].zero_()
        return {
            "clean_input_ids": clean["input_ids"],
            "clean_attention_mask": clean["attention_mask"],
            "adversarial_input_ids": adversarial["input_ids"],
            "adversarial_attention_mask": adversarial["attention_mask"],
            "labels": torch.tensor(
                [
                    float(item["label_score_space"]) - self.label_offset
                    for item in batch
                ],
                dtype=torch.float32,
            ),
            "uplift_targets": token_targets,
            "localization_mask": localization_mask,
            "adversarial_mask": adversarial_mask,
            "step_gains": torch.tensor(
                [float(item["step_gain"]) for item in batch],
                dtype=torch.float32,
            ),
            "attacks": [str(item["attack"]) for item in batch],
        }


class CleanCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )
        encoded["labels"] = torch.tensor(
            [item["label"] for item in batch], dtype=torch.float32
        )
        return encoded


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class RHTrainer:
    def __init__(self, config: RHTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._set_seed(config.seed)
        self.autocast_dtype: Optional[torch.dtype] = None
        if config.precision == "bfloat16" and self.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This CUDA device does not support bfloat16")
            self.autocast_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.checkpoint_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        print(f"Loading {config.training_mode} from {config.checkpoint_path}...", flush=True)
        if config.training_mode == PAER_RH:
            self.model = PAERForEssayScoring.from_base_checkpoint(
                config.checkpoint_path,
                dtype=torch.float32,
                correction_scale=config.correction_scale,
            ).to(self.device)
        else:
            self.model = AESScorer(
                config.checkpoint_path,
                device=self.device,
                dtype=torch.float32,
            ).model
        self.model.train()

        self.trace_dataset = CounterfactualTraceDataset(
            config.trace_jsonl,
            max_records=config.max_trace_records,
        )
        self.train_dataset = PairedEssayTrainingDataset(
            config.train_csv,
            self.trace_dataset,
            label_offset=config.label_offset,
            max_samples=config.max_train_samples,
        )
        self.valid_dataset = CleanValidationDataset(
            config.valid_csv,
            config.label_offset,
        )
        self.trace_collator = RHTraceCollator(
            self.tokenizer,
            config.max_length,
            config.label_offset,
            config.attribution_gain_scale,
        )
        self.clean_collator = CleanCollator(self.tokenizer, config.max_length)
        attack_counts: dict[str, int] = {}
        for item in self.trace_dataset.items:
            attack_counts[item["attack"]] = attack_counts.get(item["attack"], 0) + 1
        missing_attacks = {"rudimentary", "hotflip"} - set(attack_counts)
        if missing_attacks:
            raise ValueError(
                "The shared RH trace dataset must contain both training "
                f"attacks; missing {sorted(missing_attacks)}"
            )
        print(
            f"Train essays: {len(self.train_dataset)}; "
            f"attacked essays: {len(self.train_dataset.traces_by_row)}; "
            f"trace records: {len(self.trace_dataset)} {attack_counts}; "
            f"Valid essays: {len(self.valid_dataset)}",
            flush=True,
        )

        self.optimizer = torch.optim.AdamW(
            build_adamw_parameter_groups(self.model, config.weight_decay),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )
        total_steps = self._total_steps()
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_lambda(total_steps, warmup_steps),
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        resolved = asdict(config)
        resolved.update(
            {
                "device": self.device,
                "autocast_enabled": self.autocast_dtype is not None,
                "effective_batch_size": (
                    config.per_device_train_batch_size
                    * config.gradient_accumulation_steps
                ),
                "training_attacks": ["rudimentary", "hotflip"],
                "held_out_attack": "mlm_guided",
                "shared_trace_dataset_for_mixed_at_and_paer": True,
            }
        )
        Path(config.output_dir, "training_config.json").write_text(
            json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _autocast_context(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _total_steps(self) -> int:
        micro_batches = math.ceil(
            len(self.train_dataset) / self.config.per_device_train_batch_size
        ) * self.config.num_epochs
        return math.ceil(micro_batches / self.config.gradient_accumulation_steps)

    @staticmethod
    def _lr_lambda(total_steps: int, warmup_steps: int):
        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            return max(
                0.0,
                float(total_steps - step) / max(1, total_steps - warmup_steps),
            )

        return schedule

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        cfg = self.config
        batch = self._to_device(batch)
        with self._autocast_context():
            clean_output = self.model(
                input_ids=batch["clean_input_ids"],
                attention_mask=batch["clean_attention_mask"],
            )
            adversarial_output = self.model(
                input_ids=batch["adversarial_input_ids"],
                attention_mask=batch["adversarial_attention_mask"],
            )
            clean_scores = clean_output.logits.squeeze(-1).float()
            adversarial_scores = adversarial_output.logits.squeeze(-1).float()

        labels = batch["labels"].float()
        clean_loss = F.mse_loss(clean_scores, labels)
        adversarial_rows = batch["adversarial_mask"].bool()
        if bool(adversarial_rows.any()):
            adversarial_loss = one_sided_score_inflation_loss(
                clean_scores[adversarial_rows],
                adversarial_scores[adversarial_rows],
                labels[adversarial_rows],
                tolerance=cfg.inflation_tolerance,
                relative_loss_power=cfg.relative_loss_power,
            )
        else:
            adversarial_loss = torch.zeros((), device=self.device)
        localization_loss = torch.zeros((), device=self.device)
        false_suppression = torch.zeros((), device=self.device)
        mean_correction = torch.zeros((), device=self.device)

        if cfg.training_mode == PAER_RH:
            targets = batch["uplift_targets"].float()
            mask = batch["localization_mask"].float()
            adv_risk_logits = adversarial_output.risk_logits.float()
            clean_risk_logits = clean_output.risk_logits.float()
            positive_multiplier = 1.0 + (
                cfg.localization_positive_weight - 1.0
            ) * (targets > 0).float()
            adv_localization = F.binary_cross_entropy_with_logits(
                adv_risk_logits,
                targets,
                reduction="none",
            )
            adv_localization = masked_mean(
                adv_localization * positive_multiplier,
                mask,
            )
            clean_content_mask = clean_output.content_mask.float()
            clean_localization = masked_mean(
                F.binary_cross_entropy_with_logits(
                    clean_risk_logits,
                    torch.zeros_like(clean_risk_logits),
                    reduction="none",
                ),
                clean_content_mask,
            )
            localization_loss = 0.5 * (adv_localization + clean_localization)
            false_suppression = masked_mean(
                torch.sigmoid(clean_risk_logits),
                clean_content_mask,
            )
            if bool(adversarial_rows.any()):
                mean_correction = adversarial_output.correction.float()[
                    adversarial_rows
                ].mean()

        total_loss = (
            cfg.clean_loss_weight * clean_loss
            + cfg.adversarial_loss_weight * adversarial_loss
            + cfg.localization_loss_weight * localization_loss
            + cfg.clean_false_suppression_weight * false_suppression
        )
        (total_loss / cfg.gradient_accumulation_steps).backward()
        return {
            "total_loss": float(total_loss.item()),
            "clean_loss": float(clean_loss.item()),
            "adversarial_loss": float(adversarial_loss.item()),
            "localization_loss": float(localization_loss.item()),
            "false_suppression": float(false_suppression.item()),
            "mean_correction": float(mean_correction.item()),
            "n_adversarial": int(adversarial_rows.sum().item()),
        }

    def _optimizer_step(self, accumulated_micro_batches: int) -> None:
        cfg = self.config
        if accumulated_micro_batches < cfg.gradient_accumulation_steps:
            correction = cfg.gradient_accumulation_steps / accumulated_micro_batches
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.per_device_train_batch_size * 4,
            shuffle=False,
            collate_fn=self.clean_collator,
        )
        predictions: list[float] = []
        labels: list[float] = []
        corrections: list[float] = []
        for batch in tqdm(
            loader,
            desc="Clean validation",
            unit="batch",
            dynamic_ncols=True,
            disable=True if not self.config.show_progress else None,
        ):
            batch = self._to_device(batch)
            with self._autocast_context():
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
            predictions.extend(tensor_to_float_numpy(output.logits.squeeze(-1)))
            labels.extend(tensor_to_float_numpy(batch["labels"]))
            if self.config.training_mode == PAER_RH:
                corrections.extend(tensor_to_float_numpy(output.correction))
        prediction_array = np.asarray(predictions)
        label_array = np.asarray(labels)
        metrics = {
            "qwk": compute_qwk(label_array, prediction_array),
            "mae": float(np.mean(np.abs(prediction_array - label_array))),
            "rmse": float(np.sqrt(np.mean((prediction_array - label_array) ** 2))),
            "mean_clean_correction": float(np.mean(corrections)) if corrections else 0.0,
        }
        self.model.train()
        return metrics

    def save(self, tag: str) -> None:
        output_dir = Path(self.config.output_dir) / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Saved: {output_dir}", flush=True)

    def train(self) -> float:
        cfg = self.config
        generator = torch.Generator().manual_seed(cfg.seed)
        loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.trace_collator,
            num_workers=0,
            drop_last=False,
            generator=generator,
        )
        global_step = 0
        best_qwk = -1.0
        accumulated = 0
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(cfg.num_epochs):
            self.train_dataset.set_epoch(epoch)
            progress = tqdm(
                loader,
                desc=f"{cfg.training_mode} epoch {epoch + 1}/{cfg.num_epochs}",
                unit="batch",
                dynamic_ncols=True,
                disable=True if not cfg.show_progress else None,
            )
            for batch_index, batch in enumerate(progress, start=1):
                metrics = self.train_step(batch)
                accumulated += 1
                did_step = False
                if accumulated == cfg.gradient_accumulation_steps:
                    self._optimizer_step(accumulated)
                    accumulated = 0
                    global_step += 1
                    did_step = True
                if batch_index % 20 == 0:
                    progress.set_postfix(
                        gstep=global_step,
                        loss=f"{metrics['total_loss']:.4f}",
                        adv=f"{metrics['adversarial_loss']:.4f}",
                        loc=f"{metrics['localization_loss']:.4f}",
                        corr=f"{metrics['mean_correction']:.4f}",
                        n_adv=metrics["n_adversarial"],
                    )
                if did_step and global_step % cfg.eval_every == 0:
                    evaluation = self.evaluate()
                    print(
                        f"[eval gstep={global_step}] QWK={evaluation['qwk']:.4f} "
                        f"MAE={evaluation['mae']:.4f} "
                        f"clean_corr={evaluation['mean_clean_correction']:.4f}",
                        flush=True,
                    )
                    if evaluation["qwk"] > best_qwk:
                        best_qwk = evaluation["qwk"]
                        self.save("best")
                    self.model.train()
                if did_step and global_step % cfg.save_every == 0:
                    self.save(f"gstep{global_step}")

        if accumulated:
            self._optimizer_step(accumulated)
            global_step += 1
        final_metrics = self.evaluate()
        print(
            f"Final clean QWK={final_metrics['qwk']:.4f} "
            f"MAE={final_metrics['mae']:.4f}",
            flush=True,
        )
        if final_metrics["qwk"] > best_qwk:
            best_qwk = final_metrics["qwk"]
            self.save("best")
        self.save("final")
        Path(cfg.output_dir, "final_clean_metrics.json").write_text(
            json.dumps(final_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        return best_qwk


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    trainer = RHTrainer(RHTrainingConfig(**payload))
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
