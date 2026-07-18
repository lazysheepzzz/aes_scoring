#!/usr/bin/env python3
"""
HotFlip attack on defended DeBERTa (adversarial training checkpoint).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack
import pandas as pd

CHECKPOINT = "/root/autodl-tmp/aes_adv_v4/final"
OUT_DIR = "/root/autodl-tmp/aes_v4_final_run"
RESULT_FILE = os.path.join(OUT_DIR, "hotflip_defended_result.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "hotflip_defended_progress.json")

df = pd.read_csv("/root/autodl-tmp/data/valid_fold0.csv")
text_col = "full_text" if "full_text" in df.columns else "text"
essays = list(zip(df[text_col].tolist(), df["score"].tolist()))
N_ESSAYS = len(essays)
THRESHOLD = 0.1

print(f"Loaded {N_ESSAYS} essays from valid_fold0", flush=True)
print(f"Using checkpoint: {CHECKPOINT}", flush=True)

print("Loading scorer...", flush=True)
scorer = AESScorer(CHECKPOINT, device="cuda", dtype=torch.float32)
print("Scorer loaded.", flush=True)
sys.stdout.flush()

_ = scorer.score_single("warmup")
torch.cuda.synchronize()
print("GPU warmup done.", flush=True)
sys.stdout.flush()

atk = HotFlipAttack(
    scorer,
    n_steps=30,
    beam_size=1,
    n_sample_pos=8,
    top_k_per_pos=2,
    max_candidates_per_step=16,
    threshold=THRESHOLD,
)

n_ok = 0
details = []
t0 = time.time()

for idx, (text, _) in enumerate(essays):
    orig_s = scorer.score_single(text)
    pert_text, history = atk.attack(text)
    pert_s = scorer.score_single(pert_text)
    ok = pert_s - orig_s >= THRESHOLD
    if ok:
        n_ok += 1
    delta = pert_s - orig_s
    steps = len(history)
    details.append({"idx": idx, "orig": orig_s, "pert": pert_s, "delta": delta, "ok": ok, "steps": steps})

    elapsed = time.time() - t0
    print(f"[{idx+1}/{N_ESSAYS}] ASR={n_ok/(idx+1):.4f} ({elapsed/60:.1f}min) | orig={orig_s:.3f} pert={pert_s:.3f} delta={delta:+.3f} steps={steps}", flush=True)

    if (idx + 1) % 50 == 0:
        progress = {
            "idx": idx + 1, "n": N_ESSAYS, "n_ok": n_ok,
            "asr": round(n_ok / (idx + 1), 4),
            "elapsed_min": round(elapsed / 60, 1)
        }
        with open("/tmp/hotflip_defended_progress.json", "w") as f:
            json.dump(progress, f)
        shutil.copy("/tmp/hotflip_defended_progress.json", PROGRESS_FILE)

asr = n_ok / N_ESSAYS
elapsed = time.time() - t0
print(f"\n=== HotFlip on DEFENDED model: ASR={asr:.4f} ({n_ok}/{N_ESSAYS}) in {elapsed/60:.1f}min ===", flush=True)

avg_delta = sum(d["delta"] for d in details) / len(details)
result = {
    "asr": asr, "n_ok": n_ok, "n": N_ESSAYS,
    "elapsed_min": round(elapsed / 60, 1),
    "avg_delta": round(avg_delta, 4),
    "params": {"n_steps": 30, "beam_size": 1, "max_candidates": 16, "threshold": THRESHOLD},
    "details": details
}
with open("/tmp/hotflip_defended_result.json", "w") as f:
    json.dump(result, f)
shutil.copy("/tmp/hotflip_defended_result.json", RESULT_FILE)
print(f"Saved: {RESULT_FILE}", flush=True)
