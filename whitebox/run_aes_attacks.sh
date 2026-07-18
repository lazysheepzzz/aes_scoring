#!/bin/bash
# run_aes_attacks.sh — launch AES attack evaluation on the server
# Usage: bash run_aes_attacks.sh [attack_name]

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache

VICTIM=/root/autodl-tmp/victim/fold0_best
DATA=/root/autodl-tmp/data/valid_fold0.csv
OUT=/root/autodl-tmp/aes_results
ATTACK=${1:-all}
N=${2:-200}

mkdir -p $OUT

source /root/miniconda3/etc/profile.d/conda.sh
conda activate aes

echo "[ENV] python: $(python --version)"
echo "[ENV] torch: $(python -c 'import torch; print(torch.__version__)')"
echo "[ENV] transformers: $(python -c 'import transformers; print(transformers.__version__)')"

echo "[RUN] Starting attack=$ATTACK, n_essays=$N"

python /root/autodl-tmp/robust_text_scoring/text_scoring_adv_training/evaluation/aes/run_attacks.py \
    --victim "$VICTIM" \
    --data "$DATA" \
    --attack "$ATTACK" \
    --n-essays $N \
    --out "$OUT" \
    --device cuda \
    --dtype float32 \
    --batch-size 16

echo "[DONE] Results in $OUT"
