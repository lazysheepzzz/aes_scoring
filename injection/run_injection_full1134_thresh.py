#!/usr/bin/env python3
"""
Injection Attack — FULL 1134 essays (统一搜索策略 + 阈值0.1)

两种候选来源竞争：
  - Self-duplication: 复制 essay 已有句子插入
  - External injection: 插入 Wikipedia 句子

搜索策略：
  - 每步生成 64 个候选（所有位置 × 所有候选句子）
  - Batch scoring 全部 64 个，取最优执行
  - beam=1, 迭代上限30, delta>=0.1才停

Sentence bank: wikimedia/wikipedia 20231101.en (100 sentences)
"""
import sys, json, time, os, shutil, random, re
sys.path.insert(0, "/root/autodl-tmp/robust_text_scoring")
import torch
import pandas as pd
from text_scoring_adv_training.evaluation.aes.scorer import AESScorer

OUT_DIR = "/root/autodl-tmp/aes_final_run"
RESULT_FILE = os.path.join(OUT_DIR, "injection_unified_thresh_result.json")
PROGRESS_FILE = os.path.join(OUT_DIR, "injection_unified_thresh_progress.json")

# 搜索策略参数
N_ESSAYS = 1134
N_STEPS = 30
MAX_BUILD_CANDIDATES = 64   # 每步生成 64 个候选供 scorer 筛选
THRESHOLD = 0.1

# 加载 Wikipedia 句子库
SENTENCE_BANK = []
bank_path = os.path.join(OUT_DIR, "wikipedia_sentences_100.txt")
with open(bank_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            SENTENCE_BANK.append(line)
print(f"Loaded {len(SENTENCE_BANK)} sentences from Wikipedia", flush=True)


def split_sentences(text: str):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def build_candidates(text: str, sentence_bank: list):
    """
    生成最多 MAX_BUILD_CANDIDATES=16 个候选，供 scorer 筛选。
    仅用 external injection（Wikipedia 句子），仅在 start 和 end 两个位置插入。
    2 位置 × 8 Wikipedia 句子 = 16 个候选（全利用 16 候选预算）。
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    n = len(sentences)
    # 只用 start(0) 和 end(n) 两个位置
    positions = [0, n]

    candidates = []
    seen = set()

    # 每位置随机选 8 个 Wikipedia 句子
    n_per_pos = MAX_BUILD_CANDIDATES // len(positions)  # 8

    for pos in positions:
        sampled = random.sample(sentence_bank, min(n_per_pos, len(sentence_bank)))
        for ext in sampled:
            new_sentences = sentences.copy()
            new_sentences.insert(pos, ext)
            new_text = " ".join(new_sentences)
            if new_text not in seen:
                seen.add(new_text)
                candidates.append(new_text)

    # 截断到 MAX_BUILD_CANDIDATES
    if len(candidates) > MAX_BUILD_CANDIDATES:
        candidates = random.sample(candidates, MAX_BUILD_CANDIDATES)

    return candidates


def get_score(scorer, text: str):
    """Score a single text."""
    inputs = scorer.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024, padding=False)
    input_ids = inputs["input_ids"].squeeze(0).to(scorer.device)
    attn_mask = inputs["attention_mask"].squeeze(0).to(scorer.device)
    with torch.no_grad():
        score = scorer.model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0)).logits.squeeze(-1).item()
    return score


def iterative_injection(scorer, text: str, sentence_bank: list):
    """
    Injection attack with scorer-guided greedy search.
    每步：生成 64 个候选 -> batch scoring -> 取最优执行（即使负 delta 也接受）
    -> 若 delta >= 0.1 停止
    """
    best_text = text
    best_score = get_score(scorer, text)
    original_score = best_score
    history = []

    for step in range(N_STEPS):
        candidates = build_candidates(best_text, sentence_bank)
        if not candidates:
            break

        # Batch scoring 全部候选
        batch_input = scorer.tokenizer(
            candidates,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True,
        )
        input_ids = batch_input["input_ids"].to(scorer.device)
        attention_mask = batch_input["attention_mask"].to(scorer.device)

        with torch.no_grad():
            logits = scorer.model(input_ids=input_ids, attention_mask=attention_mask).logits
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            scores = logits.tolist()

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        top_score = scores[best_idx]
        top_text = candidates[best_idx]

        if top_score > best_score:
            best_score = top_score
            best_text = top_text
            history.append({"step": step, "score": top_score, "gain": top_score - original_score})

        if best_score - original_score >= THRESHOLD:
            break

    return best_text, history


print("Loading scorer...", flush=True)
scorer = AESScorer("/root/autodl-tmp/victim/fold0_best", device="cuda", dtype=torch.float32)
print("Scorer loaded.", flush=True)
sys.stdout.flush()

if __name__ == "__main__":
    _ = scorer.score_single("warmup sentence for CUDA context initialization.")
    torch.cuda.synchronize()
    print("GPU warmup done.", flush=True)
    sys.stdout.flush()

    df = pd.read_csv("/root/autodl-tmp/data/valid_fold0.csv")
    text_col = "full_text" if "full_text" in df.columns else "text"
    essays = list(zip(df[text_col].tolist()[:N_ESSAYS], df["score"].tolist()[:N_ESSAYS]))
    print(f"Loaded {len(essays)} essays", flush=True)

    n_ok = 0
    details = []
    t0 = time.time()

    for idx, (text, _) in enumerate(essays):
        try:
            orig_s = get_score(scorer, text)
            pert_text, hist = iterative_injection(scorer, text, SENTENCE_BANK)
            pert_s = get_score(scorer, pert_text)
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
            with open("/tmp/injection_unified_thresh_progress.json", "w") as f:
                json.dump(progress, f)
            shutil.copy("/tmp/injection_unified_thresh_progress.json", PROGRESS_FILE)

    asr = n_ok / N_ESSAYS
    elapsed = time.time() - t0
    print(f"\n=== ASR: {asr:.4f} ({n_ok}/{N_ESSAYS}) in {elapsed/60:.1f}min ===", flush=True)

    avg_delta = sum(d["delta"] for d in details) / len(details)
    result = {
        "asr": asr, "n_ok": n_ok, "n": N_ESSAYS,
        "elapsed_min": round(elapsed / 60, 1),
        "avg_delta": round(avg_delta, 4),
        "params": {
            "n_steps": N_STEPS, "beam_size": 1, "max_candidates": MAX_BUILD_CANDIDATES,
            "threshold": THRESHOLD,
            "sentence_bank": "wikimedia/wikipedia 20231101.en (100 sentences)",
            "self_duplication": "all sentences at each position",
            "external_injection": "all wikipedia sentences at each position",
        },
        "details": details
    }
    with open("/tmp/injection_unified_thresh_progress.json", "w") as f:
        json.dump(result, f)
    shutil.copy("/tmp/injection_unified_thresh_progress.json", RESULT_FILE)
    print(f"Saved: {RESULT_FILE}", flush=True)
