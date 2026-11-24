"""ASAP 2.0 Longformer B0: clean essay-level regression, isolated from DeBERTa outputs."""
from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paer.rhi_experiment_utils import save_json, sha256


def read_fold(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [(r["essay_id"], r["full_text"], int(r["score"])) for r in csv.DictReader(f)]
    if not rows or any(not i or not t.strip() or s not in range(1, 7) for i, t, s in rows):
        raise ValueError(f"Invalid ASAP 2.0 fold: {path}")
    if len({i for i, _, _ in rows}) != len(rows):
        raise ValueError(f"Duplicate essay IDs: {path}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="allenai/longformer-base-4096",
                   help="Hub ID or downloaded Longformer directory")
    p.add_argument("--online", action="store_true", help="Allow loading pretrained files from the Hub")
    p.add_argument("--smoke", action="store_true", help="Two updates, separate directory; not a B0 for experiments")
    p.add_argument("--dry-run", action="store_true", help="Validate local folds only; no model download")
    args = p.parse_args()
    train, valid = [ROOT / f"data/{name}_fold0.csv" for name in ("train", "valid")]
    tr, va = read_fold(train), read_fold(valid)
    if {i for i, _, _ in tr} & {i for i, _, _ in va} or (
        {" ".join(t.split()) for _, t, _ in tr} & {" ".join(t.split()) for _, t, _ in va}
    ):
        raise ValueError("Train/development overlap")
    out = ROOT / "outputs" / (("smoke_" if args.smoke else "") + "longformer_asap2_b0_seed42")
    print(f"[PLAN] {len(tr)} train / {len(va)} dev; max_length=1024; seed=42; output={out}", flush=True)
    if args.dry_run:
        return
    if out.exists():
        raise FileExistsError(f"Refusing to replace an existing run: {out}")
    import numpy as np
    import torch
    from sklearn.metrics import cohen_kappa_score
    from transformers import (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run on remote C with CUDA and bf16")
    set_seed(42)
    kw = dict(local_files_only=not args.online)
    cfg = AutoConfig.from_pretrained(args.model, **kw)
    if cfg.model_type != "longformer":
        raise ValueError(f"Expected longformer, got {cfg.model_type}")
    cfg.num_labels = 1
    cfg.problem_type = "regression"
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, **kw)
    if not tok.is_fast:
        raise ValueError("A fast tokenizer with offsets is required for subsequent PAER supervision")
    tok.padding_side = "right"
    net = AutoModelForSequenceClassification.from_pretrained(args.model, config=cfg, **kw)
    def collate(batch):
        enc = tok([row[1] for row in batch], padding=True, truncation=True,
                  max_length=1024, return_tensors="pt")
        enc["labels"] = torch.tensor([row[2]-1 for row in batch], dtype=torch.float32)
        return enc
    def metrics(pred):
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        y, f = pred.label_ids.reshape(-1), raw.reshape(-1)
        return dict(qwk=float(cohen_kappa_score(y, np.rint(f).clip(0, 5), weights="quadratic")),
                    mae=float(np.abs(y-f).mean()), rmse=float(np.sqrt(((y-f)**2).mean())))
    strategy = ("eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters
                else "evaluation_strategy")
    period = 1 if args.smoke else 72
    # Effective clean batch=16; small physical batch for the remote 24 GiB GPU.
    opts = dict(output_dir=str(out/"intermediate"), learning_rate=2e-5, weight_decay=.01,
                lr_scheduler_type="linear", num_train_epochs=3, per_device_train_batch_size=2,
                gradient_accumulation_steps=1 if args.smoke else 8, per_device_eval_batch_size=2,
                bf16=True, remove_unused_columns=False, dataloader_num_workers=0,
                save_strategy="steps", eval_steps=period, save_steps=period, logging_steps=period,
                save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="qwk",
                greater_is_better=True, seed=42, report_to=[], **{strategy:"steps"})
    if args.smoke:
        opts["max_steps"] = 2
        tr, va = tr[:16], va[:32]
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/"run_config.json", dict(model=args.model, train_sha256=sha256(train),
              valid_sha256=sha256(valid), train_n=len(tr), dev_n=len(va), max_length=1024,
              score_offset=1, smoke=args.smoke, training=opts))
    trainer = Trainer(model=net, args=TrainingArguments(**opts), train_dataset=tr,
                      eval_dataset=va, data_collator=collate, compute_metrics=metrics)
    trainer.train()
    measures = trainer.evaluate()
    trainer.save_model(str(out/"best"))
    tok.save_pretrained(out/"best")
    save_json(out/"best_clean_metrics.json", dict(metrics=measures, best_source=trainer.state.best_model_checkpoint))
    print(f"[SAVED] {out / 'best'}", flush=True)


if __name__ == "__main__":
    main()
