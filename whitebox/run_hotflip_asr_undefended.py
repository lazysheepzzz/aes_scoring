#!/usr/bin/env python3
"""
HotFlip Attack — VALID full (1154 essays, from correct fold0 split)

统一搜索策略：beam=1, 16候选/步, 30步, delta>=0.1
"""
import sys, json, time, os, random, shutil
sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
import numpy as np
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.attacks.hotflip import HotFlipAttack
import pandas as pd

OUT_DIR = "/root/autodl-tmp/aes_final_run"
RESULT_FILE = os.path.join(OUT_DIR, "hotflip_asr_undefended.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "hotflip_asr_undefended_progress.json")
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("/root/autodl-tmp/data/valid_fold0.csv")
text_col = "full_text" if "full_text" in df.columns else "text"
essays = list(zip(df[text_col].tolist(), df["score"].tolist()))
N_ESSAYS = len(essays)
THRESHOLD = 0.1

print(f"Loaded {N_ESSAYS} essays from valid_fold0", flush=True)

print("Loading scorer...", flush=True)
scorer = AESScorer("/root/autodl-tmp/victim/fold0_best", device="cuda", dtype=torch.float32)
print("Scorer loaded.", flush=True)
sys.stdout.flush()

_ = scorer.score_single("warmup sentence for CUDA context initialization.")
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
    max_token_edit_rate=0.1,
)

n_ok = 0
details = []
t0 = time.time()

for idx, (text, true_score) in enumerate(essays):
    orig_s = scorer.score_single(text)
    pert_text, history = atk.attack(text)
    pert_s = scorer.score_single(pert_text)
    ok = pert_s - orig_s >= THRESHOLD
    if ok:
        n_ok += 1
    delta = pert_s - orig_s
    steps = len(history)
    details.append({
        "idx": idx,
        "true_score": float(true_score),
        "orig": orig_s,
        "pert": pert_s,
        "delta": delta,
        "ok": ok,
        "steps": steps,
        "original_text": text,
        "perturbed_text": pert_text,
        "history": history,
    })

    elapsed = time.time() - t0
    print(f"[{idx+1}/{N_ESSAYS}] ASR={n_ok/(idx+1):.4f} ({elapsed/60:.1f}min) | orig={orig_s:.3f} pert={pert_s:.3f} delta={delta:+.3f} steps={steps}", flush=True)

    if (idx + 1) % 50 == 0:
        progress = {
            "idx": idx + 1, "n": N_ESSAYS, "n_ok": n_ok,
            "asr": round(n_ok / (idx + 1), 4),
            "elapsed_min": round(elapsed / 60, 1)
        }
        with open("/tmp/hotflip_asr_undefended_progress.json", "w", encoding="utf-8") as f:
            json.dump(progress, f)
        shutil.copy("/tmp/hotflip_asr_undefended_progress.json", PROGRESS_FILE)

asr = n_ok / N_ESSAYS
elapsed = time.time() - t0
print(f"\n=== HotFlip Valid {N_ESSAYS} ASR: {asr:.4f} ({n_ok}/{N_ESSAYS}) in {elapsed/60:.1f}min ===", flush=True)

avg_delta = sum(d["delta"] for d in details) / len(details)
result = {
    "asr": asr, "n_ok": n_ok, "n": N_ESSAYS,
    "elapsed_min": round(elapsed / 60, 1),
    "avg_delta": round(avg_delta, 4),
    "params": {
        "n_steps": 30,
        "beam_size": 1,
        "n_sample_pos": 8,
        "top_k_per_pos": 2,
        "max_candidates": 16,
        "threshold": THRESHOLD,
        "max_token_edit_rate": 0.1,
        "seed": SEED,
    },
    "details": details
}
with open("/tmp/hotflip_asr_undefended.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
shutil.copy("/tmp/hotflip_asr_undefended.json", RESULT_FILE)
print(f"Saved: {RESULT_FILE}", flush=True)
