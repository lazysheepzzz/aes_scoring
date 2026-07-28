#!/usr/bin/env python3
"""
HotFlip Adversarial Training Data Generator — TRAIN full 16153 essays

对 train_fold0.csv 全量运行 HotFlip，生成对抗训练数据。
数据格式：JSONL（流式写入，避免全量放内存）
每行：{"essay_id": "...", "original_text": "...", "perturbed_text": "...", "original_score": 0.0, "perturbed_score": 0.0, "delta": 0.0, "steps": 0, "ok": false}

输入：/root/autodl-tmp/data/train_fold0.csv
输出：
  服务器：/root/autodl-tmp/aes_final_run/hotflip_train_adv_data.jsonl
  本地  ：D:/here/robust_text_scoring-main/data/hotflip_train_adv_data.jsonl
"""
import sys, json, time, os, random, shutil
sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
import numpy as np
import pandas as pd
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack

OUT_DIR = "/root/autodl-tmp/aes_final_run"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

df_train = pd.read_csv("/root/autodl-tmp/data/train_fold0.csv")
N_ESSAYS = len(df_train)  # 动态读取，不硬编码
del df_train
THRESHOLD = 0.1
BATCH_LOG = 500    # 每500篇打一次进度

def split_sentences(text: str):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]

print("Loading scorer...", flush=True)
scorer = AESScorer("/root/autodl-tmp/victim/fold0_best", device="cuda", dtype=torch.float32)
print("Scorer loaded.", flush=True)
sys.stdout.flush()

_ = scorer.score_single("warmup sentence for CUDA context initialization.")
torch.cuda.synchronize()
print("GPU warmup done.", flush=True)
sys.stdout.flush()

# 数据流式读取，不全量加载
csv_path = "/root/autodl-tmp/data/train_fold0.csv"
print(f"Streaming from {csv_path} ({N_ESSAYS} essays)...", flush=True)

atk = HotFlipAttack(
    scorer,
    n_steps=30,
    beam_size=1,
    n_sample_pos=8,
    top_k_per_pos=2,
    max_candidates_per_step=16,
    threshold=THRESHOLD,
    max_token_edit_rate=0.1,
)

# 流式写入 JSONL
server_out = os.path.join(OUT_DIR, f"hotflip_train_{N_ESSAYS}_adv_data.jsonl")
tmp_out = f"/tmp/hotflip_train_{N_ESSAYS}_adv_data.jsonl"

n_ok = 0
t0 = time.time()

with open(tmp_out, "w", encoding="utf-8") as f_out:
    # 分块读取，避免 pandas 全量加载 42MB CSV
    for chunk in pd.read_csv(csv_path, chunksize=500):
        for row_idx, row in chunk.iterrows():
            idx = int(row_idx)
            text = str(row["full_text"]) if "full_text" in row else str(row["text"])
            essay_id = str(row.get("essay_id", f"idx_{idx}"))

            try:
                orig_s = scorer.score_single(text)
                pert_text, history = atk.attack(text)
                pert_s = scorer.score_single(pert_text)
                delta = pert_s - orig_s
                ok = delta >= THRESHOLD
                steps = len(history)
                orig_sents = len(split_sentences(text))
                pert_sents = len(split_sentences(pert_text))
                n_inserted = pert_sents - orig_sents
            except Exception as ex:
                print(f"[WARN] idx={idx} failed: {ex}", flush=True)
                orig_s = 0.0; pert_s = 0.0; delta = 0.0; ok = False
                steps = 0; pert_text = text; n_inserted = 0

            if ok:
                n_ok += 1

            record = {
                "essay_id": essay_id,
                "original_text": text,
                "perturbed_text": pert_text,
                "original_score": round(orig_s, 6),
                "perturbed_score": round(pert_s, 6),
                "delta": round(delta, 6),
                "steps": steps,
                "ok": ok,
                "text_changed": pert_text != text,
                "n_inserted": n_inserted,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            elapsed = time.time() - t0
            if (idx + 1) % BATCH_LOG == 0:
                asr = n_ok / (idx + 1)
                print(f"[{idx+1}/{N_ESSAYS}] ASR={asr:.4f} ({elapsed/60:.1f}min) | "
                      f"orig={orig_s:.3f} pert={pert_s:.3f} delta={delta:+.3f} steps={steps}", flush=True)

shutil.copy(tmp_out, server_out)
elapsed = time.time() - t0
n_changed = 0
with open(server_out, "r", encoding="utf-8") as f:
    for line in f:
        if json.loads(line).get("text_changed"):
            n_changed += 1

print(f"\n=== HotFlip Adversarial Training Data ===", flush=True)
print(f"Total: {N_ESSAYS} essays, {n_ok} successful (ASR={n_ok/N_ESSAYS:.4f})", flush=True)
print(f"Text changed: {n_changed}", flush=True)
print(f"Elapsed: {elapsed/60:.1f}min", flush=True)
print(f"Saved: {server_out}", flush=True)
