"""
Validation predictions for Kaggle-aligned Trainer pipeline.

Raw model outputs live in 0-5 label space; exported rubric scores add label_offset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import add_config_argument, resolve_config_path
from src.kaggle_dataset import KaggleEssayDataset
from src.kaggle_metrics import (
    apply_thresholds_label_space,
    labels_to_scores,
    rounded_predictions_label_space,
)
from src.kaggle_model_utils import load_kaggle_checkpoint


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def predict_raw_scores(model, dataloader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    raw_preds: list[float] = []
    true_labels: list[float] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.squeeze(-1).cpu().numpy()
            raw_preds.extend(preds.tolist())
            true_labels.extend(labels.numpy().tolist())

    return np.array(raw_preds), np.array(true_labels)


def run_raw_mode(config: dict, force: bool) -> None:
    output_path = PROJECT_ROOT / config["predictions_raw_path"]
    if output_path.exists() and not force:
        print(f"Skip: {output_path} already exists. Use --force to overwrite.")
        return

    valid_path = PROJECT_ROOT / config["valid_csv_path"]
    checkpoint_dir = PROJECT_ROOT / config["checkpoint_dir"] / "fold0_best"
    label_offset = config.get("label_offset", 1)
    label_min = config.get("label_min", 0)
    label_max = config.get("label_max", 5)

    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}. Run train_fold0_kaggle_trainer.py first."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle-aligned prediction requires CUDA.")

    device = torch.device("cuda")
    print(f"[raw mode] Using device: {device}")

    model, tokenizer = load_kaggle_checkpoint(checkpoint_dir)
    model.to(device)

    valid_df = pd.read_csv(valid_path)
    dataset = KaggleEssayDataset(
        valid_df,
        tokenizer,
        max_length=config["max_length"],
        label_offset=label_offset,
    )
    eval_batch_size = config.get("valid_batch_size", config.get("eval_batch_size", 2))
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    raw_pred, true_labels = predict_raw_scores(model, loader, device)
    true_score = valid_df["score"].values.astype(int)
    rounded_label = rounded_predictions_label_space(raw_pred, label_min, label_max)
    rounded_score = labels_to_scores(rounded_label, label_offset)

    result = pd.DataFrame(
        {
            "essay_id": valid_df["essay_id"].values,
            "full_text": valid_df["full_text"].values,
            "true_score": true_score,
            "true_label": true_labels.astype(int),
            "raw_pred": raw_pred,
            "rounded_label": rounded_label,
            "rounded_pred": rounded_score,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[raw mode] Saved {len(result)} rows -> {output_path}")


def run_final_mode(config: dict, force: bool) -> None:
    output_path = PROJECT_ROOT / config["predictions_path"]
    if output_path.exists() and not force:
        print(f"Skip: {output_path} already exists. Use --force to overwrite.")
        return

    valid_path = PROJECT_ROOT / config["valid_csv_path"]
    checkpoint_dir = PROJECT_ROOT / config["checkpoint_dir"] / "fold0_best"
    thresholds_path = PROJECT_ROOT / config["thresholds_path"]
    label_offset = config.get("label_offset", 1)
    label_min = config.get("label_min", 0)
    label_max = config.get("label_max", 5)

    if not valid_path.exists():
        raise FileNotFoundError(f"Validation data not found: {valid_path}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_dir}. Run train_fold0_kaggle_trainer.py first."
        )
    if not thresholds_path.exists():
        raise FileNotFoundError(
            f"Thresholds not found: {thresholds_path}\n"
            "Run: python src/optimize_thresholds_kaggle.py"
        )

    with open(thresholds_path, encoding="utf-8") as f:
        thresh_data = json.load(f)
    thresholds = thresh_data["thresholds_label_space"]

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle-aligned prediction requires CUDA.")

    device = torch.device("cuda")
    print(f"[final mode] Using device: {device}")
    print(f"[final mode] Thresholds (0-5 space): {thresholds}")

    model, tokenizer = load_kaggle_checkpoint(checkpoint_dir)
    model.to(device)

    valid_df = pd.read_csv(valid_path)
    dataset = KaggleEssayDataset(
        valid_df,
        tokenizer,
        max_length=config["max_length"],
        label_offset=label_offset,
    )
    eval_batch_size = config.get("valid_batch_size", config.get("eval_batch_size", 2))
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    raw_pred, true_labels = predict_raw_scores(model, loader, device)
    true_score = valid_df["score"].values.astype(int)
    rounded_label = rounded_predictions_label_space(raw_pred, label_min, label_max)
    rounded_score = labels_to_scores(rounded_label, label_offset)
    thresholded_label = apply_thresholds_label_space(raw_pred, thresholds)
    thresholded_score = labels_to_scores(thresholded_label, label_offset)

    result = pd.DataFrame(
        {
            "essay_id": valid_df["essay_id"].values,
            "full_text": valid_df["full_text"].values,
            "true_score": true_score,
            "true_label": true_labels.astype(int),
            "raw_pred": raw_pred,
            "rounded_label": rounded_label,
            "rounded_pred": rounded_score,
            "thresholded_label": thresholded_label,
            "thresholded_pred": thresholded_score,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"[final mode] Saved {len(result)} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kaggle-aligned validation predictions."
    )
    add_config_argument(parser)
    parser.add_argument(
        "--mode",
        choices=["raw", "final"],
        required=True,
        help="raw: raw+rounded; final: includes thresholded_pred in 1-6 space",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prediction file if it already exists.",
    )
    args = parser.parse_args()

    config_path = resolve_config_path(args.config, PROJECT_ROOT)
    config = load_config(config_path)

    if args.mode == "raw":
        run_raw_mode(config, args.force)
    else:
        run_final_mode(config, args.force)


if __name__ == "__main__":
    main()
