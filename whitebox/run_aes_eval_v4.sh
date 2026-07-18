#!/bin/bash
# run_aes_eval_v4.sh — evaluate v4 defended model: QWK + HotFlip ASR

set -e

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache

CHECKPOINT=/root/autodl-tmp/aes_adv_v4/final
VALID=/root/autodl-tmp/data/valid_fold0.csv
OUT=/root/autodl-tmp/aes_v4_run

mkdir -p $OUT

source /root/miniconda3/etc/profile.d/conda.sh
conda activate aes

echo "=== Step 1: QWK on clean valid set ==="
python -c "
import sys, torch, json, numpy as np
sys.path.insert(0, '/root/autodl-tmp/robust_text_scoring')
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
import pandas as pd
from torch.utils.data import DataLoader
from text_scoring_adv_training.training.aes_trainer import AESCollator, KaggleEssayDataset
from sklearn.metrics import cohen_kappa_score

scorer = AESScorer('$CHECKPOINT', device='cuda', dtype=torch.float32)
tokenizer = scorer.tokenizer

ds = KaggleEssayDataset('$VALID', tokenizer, max_length=1024)
collator = AESCollator(tokenizer, max_length=1024)
loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collator)

preds, labels = [], []
for batch in loader:
    input_ids = batch['input_ids'].cuda()
    att = batch['attention_mask'].cuda()
    logits = scorer.model(input_ids=input_ids, attention_mask=att).logits.squeeze(-1)
    preds.extend(logits.cpu().numpy())
    labels.extend(batch['labels'].numpy())

preds = np.array(preds)
labels = np.array(labels)

# Round to integer bins [0,5]
y_true = np.clip(np.round(labels).astype(int), 0, 5)
y_pred = np.clip(np.round(preds).astype(int), 0, 5)

qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
mae = float(np.mean(np.abs(preds - labels)))
print(f'Clean QWK={qwk:.4f} MAE={mae:.4f}')

result = {'qwk': round(qwk, 4), 'mae': round(mae, 4), 'n': len(labels)}
with open('$OUT/clean_qwk.json', 'w') as f:
    json.dump(result, f)
print(f'Saved: $OUT/clean_qwk.json')
"

echo ""
echo "=== Step 2: HotFlip ASR on valid set ==="
python /root/autodl-tmp/robust_text_scoring/text_scoring_adv_training/evaluation/aes/run_attacks.py \
    --victim "$CHECKPOINT" \
    --data "$VALID" \
    --attack hotflip \
    --n-essays 1154 \
    --out "$OUT" \
    --device cuda \
    --dtype float32 \
    --batch-size 16

echo "[DONE] Results in $OUT"
