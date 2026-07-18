"""
Train fold 0 with HuggingFace Trainer — Kaggle yekenot notebook alignment.

Differs from src/train_fold0.py:
  - labels = score - 1 (0-5)
  - AddedToken("\\n") and AddedToken("  ")
  - resize_token_embeddings
  - dropout = 0.0
  - step-based eval/save/logging
  - metric_for_best_model = qwk
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import inspect
import transformers
from transformers import Trainer, TrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import add_config_argument, resolve_config_path
from src.kaggle_dataset import KaggleEssayDataset
from src.kaggle_metrics import compute_kaggle_eval_metrics
from src.kaggle_model_utils import (
    load_kaggle_regression_model,
    load_kaggle_tokenizer,
    save_kaggle_checkpoint,
)
from src.metrics import mae, rmse


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_compute_metrics(label_min: int, label_max: int):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.squeeze(predictions)
        labels = np.squeeze(labels)
        metrics = compute_kaggle_eval_metrics(
            predictions,
            labels,
            label_min=label_min,
            label_max=label_max,
        )
        metrics["eval_mae"] = mae(labels, predictions)
        metrics["eval_rmse"] = rmse(labels, predictions)
        return metrics

    return compute_metrics


def write_report(
    report_path: Path,
    config: dict,
    trainer_metrics: dict,
    checkpoint_saved: bool,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    train_epochs = config.get("train_epochs", config.get("epochs", 3))
    lines = [
        "# Kaggle-Aligned DeBERTa-v3-base AES Regression Report (Fold 0)",
        "",
        "> Pipeline: `train_fold0_kaggle_trainer.py` (HuggingFace Trainer)",
        "",
        "## Kaggle Alignment",
        "- labels = score - 1 (0-5 training space)",
        "- AddedToken `\\n` and `  ` (two spaces)",
        "- resize_token_embeddings",
        "- attention_probs_dropout_prob = 0.0",
        "- hidden_dropout_prob = 0.0",
        "- metric_for_best_model = qwk",
        "- threshold optimization in 0-5 space (+1 for 1-6 output)",
        "",
        "## Training Parameters",
        f"- train_epochs: {train_epochs}",
        f"- learning_rate: {config['learning_rate']}",
        f"- train_batch_size: {config['train_batch_size']}",
        f"- valid_batch_size: {config.get('valid_batch_size', config.get('eval_batch_size'))}",
        f"- gradient_accumulation_steps: {config['gradient_accumulation_steps']}",
        f"- effective batch size: "
        f"{config['train_batch_size'] * config['gradient_accumulation_steps']}",
        f"- warmup_ratio: {config['warmup_ratio']}",
        f"- eval_steps: {config.get('eval_steps')}",
        f"- logging_steps: {config.get('logging_steps')}",
        f"- save_steps: {config.get('save_steps')}",
        f"- fp16: {config['fp16']}",
        "",
        "## Trainer Metrics",
    ]
    for key, value in sorted(trainer_metrics.items()):
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Checkpoint",
            f"- Saved to fold0_best: **{'Yes' if checkpoint_saved else 'No'}**",
            "",
            "## Next Steps",
            "1. `predict_valid_kaggle_trainer.py --mode raw`",
            "2. `optimize_thresholds_kaggle.py`",
            "3. `predict_valid_kaggle_trainer.py --mode final`",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle-aligned fold 0 training via HuggingFace Trainer."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing trainer output directory.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, PROJECT_ROOT)
    config = load_config(config_path)
    set_seed(config["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle-aligned training requires CUDA. "
            "Run via run_fold0_kaggle_6gb_1024.bat after verify_cuda_env.py."
        )

    train_path = PROJECT_ROOT / config["train_csv_path"]
    valid_path = PROJECT_ROOT / config["valid_csv_path"]
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Processed data not found. Run prepare_data.py first.\n"
            f"  Expected: {train_path}\n"
            f"  Expected: {valid_path}"
        )

    label_offset = config.get("label_offset", 1)
    label_min = config.get("label_min", 0)
    label_max = config.get("label_max", 5)
    train_epochs = config.get("train_epochs", config.get("epochs", 3))
    eval_batch_size = config.get("valid_batch_size", config.get("eval_batch_size", 2))

    print(f"Loading Kaggle-aligned model: {config['model_name']}")
    tokenizer = load_kaggle_tokenizer(config["model_name"])
    dropout_config = {
        key: config.get(key, 0.0)
        for key in (
            "attention_probs_dropout_prob",
            "hidden_dropout_prob",
            "classifier_dropout",
            "hidden_act_dropout_prob",
            "pooler_dropout",
        )
    }
    model = load_kaggle_regression_model(
        config["model_name"],
        tokenizer,
        dropout_config=dropout_config,
    )
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        print("gradient_checkpointing: enabled")
    use_bf16 = bool(config.get("bf16", False))
    use_fp16 = bool(config.get("fp16", False)) if not use_bf16 else False
    if not use_bf16 and "fp16" not in config:
        use_fp16 = True
    if use_fp16 and use_bf16:
        raise ValueError("Config cannot enable both fp16 and bf16.")
    # Keep master weights FP32 under AMP (avoids FP16 unscale errors on 6GB fp16 runs)
    if use_fp16 or use_bf16:
        model = model.float()
    print(f"AMP: fp16={use_fp16}, bf16={use_bf16}")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)

    max_train = config.get("max_train_samples")
    max_valid = config.get("max_valid_samples")
    if max_train is not None:
        train_df = train_df.sample(n=min(max_train, len(train_df)), random_state=config["seed"])
    if max_valid is not None:
        valid_df = valid_df.sample(n=min(max_valid, len(valid_df)), random_state=config["seed"])
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    if max_train is not None or max_valid is not None:
        print(f"[smoke/subset] train={len(train_df)}, valid={len(valid_df)}")

    train_dataset = KaggleEssayDataset(
        train_df,
        tokenizer,
        max_length=config["max_length"],
        label_offset=label_offset,
    )
    valid_dataset = KaggleEssayDataset(
        valid_df,
        tokenizer,
        max_length=config["max_length"],
        label_offset=label_offset,
    )

    checkpoint_root = PROJECT_ROOT / config["checkpoint_dir"]
    trainer_output_dir = checkpoint_root / "trainer_output"
    best_checkpoint_dir = checkpoint_root / "fold0_best"
    log_path = PROJECT_ROOT / config["log_path"]
    report_path = PROJECT_ROOT / config["report_path"]

    if trainer_output_dir.exists() and not args.force:
        raise FileExistsError(
            f"Trainer output already exists: {trainer_output_dir}. Use --force to overwrite."
        )

    training_kwargs: dict = {
        "output_dir": str(trainer_output_dir),
        "num_train_epochs": train_epochs,
        "per_device_train_batch_size": config["train_batch_size"],
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "warmup_ratio": config["warmup_ratio"],
        "lr_scheduler_type": config.get("lr_scheduler", "linear"),
        "optim": config.get("optim", "adamw_torch"),
        "fp16": use_fp16,
        "eval_steps": config["eval_steps"],
        "logging_steps": config["logging_steps"],
        "save_steps": config["save_steps"],
        "save_total_limit": 2,
        "load_best_model_at_end": config.get("load_best_model_at_end", True),
        "metric_for_best_model": config.get("metric_for_best_model", "qwk"),
        "greater_is_better": config.get("greater_is_better", True),
        "seed": config["seed"],
        "report_to": "none",
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }
    if config.get("gradient_checkpointing", False):
        training_kwargs["gradient_checkpointing"] = True
    if config.get("max_steps") is not None:
        training_kwargs["max_steps"] = int(config["max_steps"])
        # Avoid load_best_model_at_end when training ends before any save checkpoint.
        if config.get("smoke_test"):
            training_kwargs["load_best_model_at_end"] = False
            training_kwargs["save_strategy"] = "steps"
            training_kwargs["save_steps"] = max(1, int(config.get("save_steps", 2)))

    import inspect

    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if use_bf16 and "bf16" not in ta_params:
        raise RuntimeError(
            "Config requests bf16 but TrainingArguments has no bf16 parameter. "
            "Upgrade transformers for RTX 5090 runs."
        )
    if "bf16" in ta_params:
        training_kwargs["bf16"] = use_bf16
    if "eval_strategy" in ta_params:
        training_kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in ta_params:
        training_kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("TrainingArguments has neither eval_strategy nor evaluation_strategy")

    training_args = TrainingArguments(**training_kwargs)
    print(
        f"transformers={transformers.__version__}; "
        f"eval arg={'eval_strategy' if 'eval_strategy' in ta_params else 'evaluation_strategy'}"
    )

    trainer_params = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs: dict = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "compute_metrics": build_compute_metrics(label_min, label_max),
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("Starting Kaggle-aligned Trainer...")
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    save_kaggle_checkpoint(
        trainer.model,
        tokenizer,
        best_checkpoint_dir,
        extra={
            "train_result": train_result.metrics,
            "eval_metrics": eval_metrics,
            "config": config,
            "label_offset": label_offset,
            "pipeline": "kaggle_trainer",
        },
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_out = {
        "train": train_result.metrics,
        "eval": eval_metrics,
        "label_offset": label_offset,
        "label_space": [label_min, label_max],
        "score_space": [config.get("score_min", 1), config.get("score_max", 6)],
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    write_report(report_path, config, eval_metrics, checkpoint_saved=True)

    print("\nKaggle-aligned training complete.")
    print(f"Best checkpoint: {best_checkpoint_dir}")
    print(f"Metrics log: {log_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
