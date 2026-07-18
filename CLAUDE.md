# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements adversarial training and robustness evaluation for text scoring models (retrievers, rerankers, reward models), based on the paper "Unifying Adversarial Robustness and Training Across Text Scoring Models".

## Common Commands

### Installation
```bash
pip install -e .
```

### Training Models

Training uses DeepSpeed. Config files are in `configs/{reranker,retriever,reward}/`:

```bash
# Reranker (e.g., Qwen3-0.6B)
deepspeed text_scoring_adv_training/cli/train_reranker.py \
  --train_file=path/to/train.csv \
  --dev_file=path/to/dev.csv \
  --output_dir=path/to/output \
  --deepspeed configs/reranker/qwen3_0.6b_deepspeed_config.json

# Retriever (e.g., BERT-base)
deepspeed text_scoring_adv_training/cli/train_retriever.py \
  --train_file=path/to/train.csv \
  --dev_file=path/to/dev.csv \
  --output_dir=path/to/output \
  --deepspeed configs/retriever/e5_retriever_deepspeed_config.json

# Reward model (e.g., Llama-3.2-3B)
deepspeed text_scoring_adv_training/cli/train_reward.py \
  --train_file=path/to/train.csv \
  --dev_file=path/to/dev.csv \
  --output_dir=path/to/output \
  --deepspeed configs/reward/llama3b_deepspeed_config.json
```

### Adversarial Training Flags

Each trainer supports `--use_<attack>` flags to incorporate adversarial examples during training:
- `--use_paraphrased` — paraphrased text variants
- `--use_injected` — injected sentences
- `--use_rudimentary_manipulations` — character/word-level edits (typos, swaps, deletions)
- `--use_hotflip_swaps` — gradient-guided token replacement
- `--use_pgd` — projected gradient descent

Each has an associated `--<attack>_weight` to control its contribution to the training loss.

### Evaluation

Robustness tests are in `text_scoring_adv_training/evaluation/{retriever,reranker,reward}/robustness_tests/`:
```bash
python -m text_scoring_adv_training.evaluation.retriever.run_reranker ...
```

## Code Architecture

### Main Package: `text_scoring_adv_training/`

```
text_scoring_adv_training/
├── cli/                  # Training entrypoints (train_reranker.py, train_retriever.py, train_reward.py)
├── data/
│   ├── datasets.py       # RankingDataset for ranking/retrieval tasks
│   └── collators.py      # Data collators with built-in adversarial manipulation (rudimentary edits)
├── models/
│   ├── pointwise_scorer.py   # Base scoring model wrapper
│   └── embedding_model.py    # Embedding-based models
├── training/
│   ├── reranker_trainer.py   # Cross-encoder reranker training with DeepSpeed
│   ├── retriever_trainer.py  # Dense retriever training
│   ├── reward_trainer.py     # Reward model training with PGD/HotFlip support
│   └── losses.py             # hinge_loss, softmax_nll_loss, mse_loss
├── evaluation/
│   ├── aes/              # Automated Essay Scoring robustness evaluation
│   │   ├── attacks/      # Four attack types: hotflip, mlm_guided, rudimentary, injection
│   │   ├── evaluate.py   # Unified evaluation framework
│   │   └── scorer.py     # Victim model wrapper
│   ├── robustness_tests/ # Shared attack primitives
│   │   └── common/       # hotflip.py, mlm.py, rudimentary_edits.py, injections.py, beam.py
│   ├── retriever/        # Retriever-specific robustness tests
│   ├── reranker/         # Reranker-specific robustness tests
│   └── reward/           # Reward model-specific robustness tests
└── utils/
    ├── seed.py           # Reproducibility utilities
    └── rudimentary_injections_generator.py
```

### Standalone Attack Scripts (root-level directories)

These are standalone evaluation scripts that run full benchmark suites:
- `rudimentary/run_rudimentary_full1134_thresh.py` — character/word-level edit attacks
- `mlm_guided/run_mlm_guided_full1134_thresh.py` — MLM-guided synonym replacement
- `whitebox/run_hotflip_full1134_thresh.py` — gradient-guided token replacement
- `injection/run_injection_full1134_thresh.py` — sentence injection attacks

They read from `*_unified_thresh_result.json` and `wikipedia_sentences_100.txt`.

### Adversarial Training Config

`RerankerConfig` / `RetrieverConfig` / `RewardConfig` dataclasses manage all training hyperparameters including adversarial training weights and PGD settings (`eps_max`, `pgd_steps`, `pgd_lr_factor`, `eps_warmup_steps`).

### Key Design Patterns

- **DeepSpeed ZeRO**: Training uses DeepSpeed with zero redundancy for memory efficiency; config JSONs per model type
- **Epsilon Scheduler**: `_EpsilonScheduler` ramps up perturbation budget during training (`eps_warmup_steps`)
- **White-box attacks**: HotFlip uses victim model gradients to find optimal token replacements; implemented in `evaluation/robustness_tests/common/hotflip.py`
- **Black-box attacks**: MLM-guided uses WordNet candidates scored by the victim; rudimentary uses rule-based character/word edits
