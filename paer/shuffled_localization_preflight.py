"""Check the real tokenized supervision on CPU before training the control."""
from __future__ import annotations

import random
from collections import defaultdict


def check_shuffle(payload, *, states_per_family=32):
    import torch
    from transformers import AutoTokenizer
    from paer.aes_rh_trainer import CounterfactualTraceDataset, PairedEssayTrainingDataset, RHTraceCollator, RHTrainingConfig

    config = RHTrainingConfig(**payload)
    traces = CounterfactualTraceDataset(config.trace_jsonl, max_records=config.max_trace_records,
                                        allowed_attacks=config.training_attacks)
    dataset = PairedEssayTrainingDataset(config.train_csv, traces, label_offset=config.label_offset,
                                          max_samples=config.max_train_samples)
    # Enumerate only states the training schedule can actually expose.
    by_attack = defaultdict(list)
    for row, states in sorted(dataset.traces_by_row.items()):
        for epoch, state in enumerate(states[:config.num_epochs]):
            by_attack[state["attack"]].append((row, epoch))
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path, local_files_only=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0
    common = (tokenizer, config.max_length, config.label_offset, config.attribution_gain_scale)
    original = RHTraceCollator(*common)
    shuffled = RHTraceCollator(*common, shuffled_localization_seed=config.shuffled_localization_seed)
    rng = random.Random(config.shuffled_localization_seed)
    stats = {}
    key = "cumulative_uplift_targets"
    for attack, candidates in sorted(by_attack.items()):
        sample = rng.sample(candidates, min(states_per_family, len(candidates)))
        count = {"eligible_states": len(candidates), "checked_states": len(sample), "active_states": 0,
                 "moved_states": 0, "positive_tokens": 0, "overlap_tokens": 0}
        for start in range(0, len(sample), 8):
            batch = []
            for row, epoch in sample[start:start + 8]:
                dataset.set_epoch(epoch)
                batch.append(dataset[row])
            before, after = original(batch), shuffled(batch)
            for name, value in before.items():
                if name == key:
                    continue
                same = torch.equal(value, after[name]) if isinstance(value, torch.Tensor) else value == after[name]
                if not same:
                    raise ValueError(f"Shuffle changed non-localization input: {name}")
            a, b = before[key], after[key]
            if not torch.equal(a.sort(dim=1).values, b.sort(dim=1).values):
                raise ValueError("Shuffle changed per-state target counts or strengths")
            if ((b > 0) & (before["localization_mask"] == 0)).any():
                raise ValueError("Shuffle supervised a masked or special token")
            count["active_states"] += int((a > 0).any(dim=1).sum())
            count["moved_states"] += int((a != b).any(dim=1).sum())
            count["positive_tokens"] += int((a > 0).sum())
            count["overlap_tokens"] += int(((a > 0) & (b > 0)).sum())
        stats[attack] = count
    if not sum(c["moved_states"] for c in stats.values()):
        raise ValueError("Shuffle moved no supervised positions on the sampled training states")
    return {"status": "PASS", "scope": "CPU tokenization of sampled scheduled states; no model forward",
            "sampling_seed": config.shuffled_localization_seed, "by_attack": stats}
