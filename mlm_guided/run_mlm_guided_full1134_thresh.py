#!/usr/bin/env python3
"""
MLM-Guided Attack — FULL 1134 essays (统一搜索策略 + 阈值0.1)
参考论文：ModernBERT-large, top_k=16/位置, 单token逐次掩码, n_sample_pos=8
搜索策略：beam=1, 每步候选16, 迭代上限30, delta>=0.1才停
"""
# 必须在 import transformers 之前设置 HF 镜像
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys, json, time, shutil, random
sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
import pandas as pd
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer
from text_scoring_adv_training.evaluation.robustness_tests.common.mlm import (
    load_mlm,
    build_replacement_map,
)

OUT_DIR = "/root/autodl-tmp/aes_final_run"
RESULT_FILE = os.path.join(OUT_DIR, "mlm_guided_unified_thresh_result.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "mlm_guided_unified_thresh_progress.json")

N_ESSAYS = 1134
N_STEPS = 30
N_SAMPLE_POS = 8
TOP_K_PER_POS = 2
MAX_CANDIDATES_PER_STEP = 16
THRESHOLD = 0.1
MLM_VOCAB = 50368  # ModernBERT-large vocab size
MLM_MODEL_NAME = "answerdotai/ModernBERT-large"
MLM_BATCH_SIZE = 256

print("Loading AES scorer...", flush=True)
scorer = AESScorer("/root/autodl-tmp/victim/fold0_best", device="cuda", dtype=torch.float32)
print("AES scorer loaded.", flush=True)
sys.stdout.flush()

_ = scorer.score_single("warmup sentence for CUDA context initialization.")
torch.cuda.synchronize()
print("GPU warmup done.", flush=True)
sys.stdout.flush()

print(f"Loading MLM model ({MLM_MODEL_NAME})...", flush=True)
mlm_tok, mlm_model, mask_id, mlm_specials = load_mlm(MLM_MODEL_NAME, device="cuda", dtype=torch.float32)
print("MLM model loaded.", flush=True)
sys.stdout.flush()

df = pd.read_csv("/root/autodl-tmp/data/valid_fold0.csv")
text_col = "full_text" if "full_text" in df.columns else "text"
essays = list(zip(df[text_col].tolist()[:N_ESSAYS], df["score"].tolist()[:N_ESSAYS]))
print(f"Loaded {len(essays)} essays", flush=True)


def _score_batch_candidates(scorer, ids_list, batch_size=32):
    """Batch score multiple candidate sequences (list of tensors), with left-padding to match scorer."""
    if not ids_list:
        return []
    device = scorer.device
    tokenizer = scorer.tokenizer
    model = scorer.model
    scores = []
    pad_id = tokenizer.pad_token_id or 0

    for start in range(0, len(ids_list), batch_size):
        end = min(start + batch_size, len(ids_list))
        batch = ids_list[start:end]
        max_len = max(ids.size(0) for ids in batch)
        # Left-padding to match scorer's padding_side='left'
        padded_list = []
        for ids in batch:
            pad_len = max_len - ids.size(0)
            if pad_len > 0:
                padded_ids = torch.cat([torch.full((pad_len,), pad_id, dtype=ids.dtype, device=ids.device), ids])
            else:
                padded_ids = ids
            padded_list.append(padded_ids)
        padded = torch.stack(padded_list).to(device)
        mask = (padded != pad_id).long().to(device)
        try:
            with torch.no_grad():
                logits = model(input_ids=padded, attention_mask=mask).logits
                if logits.ndim > 1:
                    logits = logits.squeeze(-1)
                scores.extend(logits.tolist())
        except RuntimeError as e:
            # Fallback: score one by one
            for ids in padded_list:
                try:
                    with torch.no_grad():
                        out = model(input_ids=ids.unsqueeze(0), attention_mask=(ids != pad_id).long().unsqueeze(0))
                        s = out.logits.squeeze(-1).item()
                        scores.append(s)
                except Exception:
                    scores.append(0.0)
    return scores


def iterative_mlm_guided(scorer, text, mlm_tok, mlm_model, mask_id, mlm_specials):
    """
    MLM-guided attack with unified greedy search.
    Each step:
        1. Sample n_sample_pos positions
        2. For each, get top_k MLM candidates
        3. Score all candidates with scorer
        4. Pick best if it improves score
    """
    inputs = scorer.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
        padding=False,
    )
    input_ids = inputs["input_ids"].squeeze(0)
    attention_mask = inputs["attention_mask"].squeeze(0)
    original_score = float(
        scorer.model(
            input_ids=input_ids.unsqueeze(0).to(scorer.device),
            attention_mask=attention_mask.unsqueeze(0).to(scorer.device),
        ).logits.squeeze(-1).item()
    )

    best_ids = input_ids.clone()
    best_score = original_score
    history = []

    for step in range(N_STEPS):
        # Sample positions (non-special tokens)
        ids_list = best_ids.tolist()
        specials = set(scorer.tokenizer.all_special_ids)
        cand_pos = [i for i, tid in enumerate(ids_list) if tid not in specials]
        if not cand_pos:
            break

        if len(cand_pos) <= N_SAMPLE_POS:
            sample_pos = cand_pos
        else:
            sample_pos = random.sample(cand_pos, N_SAMPLE_POS)

        # Only use positions where the current token is within MLM vocab range
        # Only use positions where the current token is within MLM vocab range
        # (ModernBERT can't process tokens with IDs >= 50368)
        valid_pos = [p for p in sample_pos if int(best_ids[p]) < MLM_VOCAB]
        if not valid_pos:
            break

        # Replace out-of-range token IDs with pad_id before passing to MLM
        # (MLM only sees valid tokens; only masked positions get replaced)
        mlm_ids = best_ids.clone()
        pad_id = scorer.tokenizer.pad_token_id or 0
        mlm_ids[mlm_ids >= MLM_VOCAB] = pad_id

        # Build replacement map: position -> [candidate_ids]
        repl_map = build_replacement_map(
            mlm_ids.to(mlm_model.device),
            mask_id,
            mlm_specials,
            mlm_model,
            batch_size=MLM_BATCH_SIZE,
            top_k=TOP_K_PER_POS,
            prob_min=0.0,  # 论文无额外过滤，接受全部top_k
            positions=valid_pos,
        )

        if not repl_map:
            break

        # Build candidate sequences
        # build_replacement_map 内部已过滤 old_id 和 specials，此处不再重复过滤
        candidate_ids_list = []
        for pos, cand_ids in repl_map.items():
            for cid in cand_ids:
                new_ids = best_ids.clone()
                new_ids[pos] = cid
                candidate_ids_list.append(new_ids.cpu())

        if not candidate_ids_list:
            break

        # Limit to MAX_CANDIDATES_PER_STEP
        if len(candidate_ids_list) > MAX_CANDIDATES_PER_STEP:
            candidate_ids_list = random.sample(candidate_ids_list, MAX_CANDIDATES_PER_STEP)

        # Score all candidates
        scores = _score_batch_candidates(scorer, candidate_ids_list)

        # Pair and find best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        top_score = scores[best_idx]
        top_ids = candidate_ids_list[best_idx]

        if top_score > best_score:
            best_score = top_score
            best_ids = top_ids.clone().to(scorer.device)
            history.append({"step": step, "score": top_score, "gain": top_score - original_score})

        if best_score - original_score >= THRESHOLD:
            break

    best_text = scorer.tokenizer.decode(best_ids, skip_special_tokens=True)
    return best_text, history


n_ok = 0
details = []
t0 = time.time()

for idx, (text, _) in enumerate(essays):
    try:
        orig_s = scorer.score_single(text)
        pert_text, hist = iterative_mlm_guided(scorer, text, mlm_tok, mlm_model, mask_id, mlm_specials)
        pert_s = scorer.score_single(pert_text)
        ok = pert_s - orig_s >= THRESHOLD
        if ok:
            n_ok += 1
        delta = pert_s - orig_s
        steps = len(hist)
    except Exception as ex:
        print(f"[WARN] idx={idx} failed: {ex}", flush=True)
        orig_s = 0.0; pert_s = 0.0; delta = 0.0; ok = False; steps = 0
    details.append({"idx": idx, "orig": orig_s, "pert": pert_s, "delta": delta, "ok": ok, "steps": steps})

    elapsed = time.time() - t0
    print(f"[{idx+1}/{N_ESSAYS}] ASR={n_ok/(idx+1):.4f} ({elapsed/60:.1f}min) | orig={orig_s:.3f} pert={pert_s:.3f} delta={delta:+.3f} steps={steps}", flush=True)

    if (idx + 1) % 50 == 0:
        progress = {
            "idx": idx + 1, "n": N_ESSAYS, "n_ok": n_ok,
            "asr": round(n_ok / (idx + 1), 4),
            "elapsed_min": round(elapsed / 60, 1)
        }
        with open("/tmp/mlm_guided_unified_thresh_progress.json", "w") as f:
            json.dump(progress, f)
        shutil.copy("/tmp/mlm_guided_unified_thresh_progress.json", PROGRESS_FILE)

asr = n_ok / N_ESSAYS
elapsed = time.time() - t0
print(f"\n=== ASR: {asr:.4f} ({n_ok}/{N_ESSAYS}) in {elapsed/60:.1f}min ===", flush=True)

avg_delta = sum(d["delta"] for d in details) / len(details)
result = {
    "asr": asr, "n_ok": n_ok, "n": N_ESSAYS,
    "elapsed_min": round(elapsed / 60, 1),
    "avg_delta": round(avg_delta, 4),
    "params": {
        "n_steps": N_STEPS, "beam_size": 1, "max_candidates": MAX_CANDIDATES_PER_STEP,
        "threshold": THRESHOLD, "n_sample_pos": N_SAMPLE_POS, "top_k_per_pos": TOP_K_PER_POS,
        "mlm_model": MLM_MODEL_NAME,
    },
    "details": details
}
with open("/tmp/mlm_guided_unified_thresh_progress.json", "w") as f:
    json.dump(result, f)
shutil.copy("/tmp/mlm_guided_unified_thresh_progress.json", RESULT_FILE)
print(f"Saved: {RESULT_FILE}", flush=True)
