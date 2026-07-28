#!/usr/bin/env python3
"""
Rudimentary Attack — FULL 1154 essays (统一搜索策略 + 阈值0.1)
beam=1, 每步候选16, 迭代上限30, delta>=0.1才停
"""
import sys, json, time, os, shutil
sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.aes.attacks.rudimentary import RudimentaryAttack
import pandas as pd

OUT_DIR = "/root/autodl-tmp/aes_final_run"
RESULT_FILE = os.path.join(OUT_DIR, "rudimentary_unified_thresh_result.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "rudimentary_unified_thresh_progress.json")

N_ESSAYS = 1154
N_STEPS = 30
N_CANDIDATES = 16
THRESHOLD = 0.1

print("Loading scorer...", flush=True)
scorer = AESScorer("/root/autodl-tmp/victim/fold0_best", device="cuda", dtype=torch.float32)
print("Scorer loaded.", flush=True)
sys.stdout.flush()

_ = scorer.score_single("warmup sentence for CUDA context initialization.")
torch.cuda.synchronize()
print("GPU warmup done.", flush=True)
sys.stdout.flush()

df = pd.read_csv("/root/autodl-tmp/data/valid_fold0.csv")
text_col = "full_text" if "full_text" in df.columns else "text"
essays = list(zip(df[text_col].tolist()[:N_ESSAYS], df["score"].tolist()[:N_ESSAYS]))
print(f"Loaded {len(essays)} essays", flush=True)

def _score_batch(scorer, candidates, batch_size=32):
    """Batch score multiple candidate texts."""
    if not candidates:
        return []
    device = scorer.device
    tokenizer = scorer.tokenizer
    model = scorer.model

    batch_ids = []
    for c in candidates:
        inputs = tokenizer(
            c,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=False,
        )
        batch_ids.append(inputs["input_ids"].squeeze(0))

    scores = []
    for start in range(0, len(batch_ids), batch_size):
        end = min(start + batch_size, len(batch_ids))
        ids_batch = batch_ids[start:end]
        max_len = max(ids.size(0) for ids in ids_batch)
        padded = torch.nn.utils.rnn.pad_sequence(
            [torch.cat([ids, torch.zeros(max_len - ids.size(0), dtype=ids.dtype)]) for ids in ids_batch],
            batch_first=True, padding_value=tokenizer.pad_token_id or 0
        ).to(device)
        mask = (padded != tokenizer.pad_token_id).long().to(device)

        with torch.no_grad():
            logits = model(input_ids=padded, attention_mask=mask).logits
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            scores.extend(logits.tolist())

    return scores


def iterative_rudimentary(scorer, text, n_steps=N_STEPS, n_candidates=N_CANDIDATES):
    best_text = text
    original_score = scorer.score_single(text)
    best_score = original_score
    history = [(best_score, best_text)]

    for step in range(n_steps):
        atk = RudimentaryAttack(scorer, n_variants=n_candidates)
        candidates = atk.attack(best_text)
        if not candidates:
            break

        cand_scores = _score_batch(scorer, candidates)
        scored = sorted(zip(cand_scores, candidates), reverse=True, key=lambda x: x[0])
        top_score, top_cand = scored[0]

        if top_score > best_score:
            best_score = top_score
            best_text = top_cand
            history.append((best_score, best_text))

        if best_score - original_score >= THRESHOLD:
            break

    return best_text, history

n_ok = 0
details = []
t0 = time.time()

for idx, (text, _) in enumerate(essays):
    orig_s = scorer.score_single(text)
    pert_text, hist = iterative_rudimentary(scorer, text)
    pert_s = scorer.score_single(pert_text)
    ok = pert_s - orig_s >= THRESHOLD
    if ok:
        n_ok += 1
    delta = pert_s - orig_s
    steps = len(hist) - 1
    details.append({"idx": idx, "orig": orig_s, "pert": pert_s, "delta": delta, "ok": ok, "steps": steps})

    elapsed = time.time() - t0
    print(f"[{idx+1}/{N_ESSAYS}] ASR={n_ok/(idx+1):.4f} ({elapsed/60:.1f}min) | orig={orig_s:.3f} pert={pert_s:.3f} delta={delta:+.3f} steps={steps}", flush=True)

    if (idx + 1) % 50 == 0:
        progress = {
            "idx": idx + 1, "n": N_ESSAYS, "n_ok": n_ok,
            "asr": round(n_ok / (idx + 1), 4),
            "elapsed_min": round(elapsed / 60, 1)
        }
        with open("/tmp/rudimentary_unified_thresh_progress.json", "w") as f:
            json.dump(progress, f)
        shutil.copy("/tmp/rudimentary_unified_thresh_progress.json", PROGRESS_FILE)

asr = n_ok / N_ESSAYS
elapsed = time.time() - t0
print(f"\n=== ASR: {asr:.4f} ({n_ok}/{N_ESSAYS}) in {elapsed/60:.1f}min ===", flush=True)

avg_delta = sum(d["delta"] for d in details) / len(details)
result = {
    "asr": asr, "n_ok": n_ok, "n": N_ESSAYS,
    "elapsed_min": round(elapsed/60, 1),
    "avg_delta": round(avg_delta, 4),
    "params": {"n_steps": N_STEPS, "n_candidates": N_CANDIDATES, "threshold": THRESHOLD},
    "details": details
}
with open("/tmp/rudimentary_unified_thresh_progress.json", "w") as f:
    json.dump(result, f)
shutil.copy("/tmp/rudimentary_unified_thresh_progress.json", RESULT_FILE)
print(f"Saved: {RESULT_FILE}", flush=True)
