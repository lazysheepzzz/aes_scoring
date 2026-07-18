#!/bin/bash
# run_aes_adv_v4.sh — AES HotFlip adversarial training v4
# Stronger defense: weight=2.0, margin=0.1, epochs=5
# Goal: QWK close to undefended (~0.854), ASR < 87.78%

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache

VICTIM=/root/autodl-tmp/victim/fold0_best
TRAIN=/root/autodl-tmp/data/train_fold0.csv
VALID=/root/autodl-tmp/data/valid_fold0.csv
OUT=/root/autodl-tmp/aes_adv_v4

mkdir -p $OUT

source /root/miniconda3/etc/profile.d/conda.sh
conda activate aes

echo "[ENV] python: $(python --version)"
echo "[ENV] torch: $(python -c 'import torch; print(torch.__version__)')"

echo "[V4] weight=2.0, margin=0.1, epochs=5"
echo "[RUN] Starting adversarial training..."

python /root/autodl-tmp/robust_text_scoring/text_scoring_adv_training/training/aes_trainer.py \
    --checkpoint_path "$VICTIM" \
    --train_csv "$TRAIN" \
    --valid_csv "$VALID" \
    --output_dir "$OUT" \
    --num_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --hotflip_weight 2.0 \
    --hotflip_margin 0.1 \
    --hotflip_fraction 1.0 \
    --eval_every 200 \
    --save_every 1000

echo "[DONE] v4 training complete. Checkpoint: $OUT/final"
