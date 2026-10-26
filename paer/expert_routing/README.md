# Low-rank experts + PAER: independent RHI trial

This opt-in extension preserves the historical PAER models, traces and results.
It tests whether low-rank encoder specialization complements PAER's directional
token-evidence aggregation. Effectiveness is **not established by smoke tests**.

## What is being compared

| Entry point in `paer/` | Shared encoder + experts | PAER scoring / objectives |
| --- | --- | --- |
| `run_aes_low_rank_expert_training.py` | yes | no |
| `run_aes_paer_expert_training.py` | yes | yes, original v3 |

Both start from the same B0 and reuse the same frozen RHI training pool. Defaults:
3 experts, rank 8, last 2 encoder layers, V and attention-output projections.
The shared backbone remains trainable; this is **not frozen-backbone LoRA**.
The 3 experts are not hard-wired to the 3 attack families. Dense soft gates mix
local token features and the padding-masked essay mean with equal weights.
Inference needs only the essay, not its attack identity or edit history.

The architecture is inspired by [RoME (Kim et al., ECCV 2026)](https://arxiv.org/abs/2607.06109)
and its [official MIT-licensed implementation](https://github.com/wkim97/RoME).
This directory is an independently implemented **text adaptation, not a full RoME
reproduction**. Differences include V/O placement (DeBERTa reuses Q for positional
embeddings), linear text gates, and the prototype supervision below. Do not name
this trial an official RoME baseline in the paper.

## Routing supervision and fairness boundaries

Both new models receive the same auxiliary router supervision. Visible edited
token gates are averaged **per essay**, then compared against detached moving
family prototypes. R and H have separate prototypes; both Injection subtypes
share one prototype. This does not assign quality labels to edits.

Detached prototypes handle sparse attacked examples without resampling batches,
retaining encoder graphs across microbatches, or running a separate backbone per
expert. Before a second family is observed the prototype loss is zero. Edits
outside the model's visible window receive no invented routing label.
`--routing-scope all` permits a later all-content-token comparison; it is not the
default and has not been evaluated.

The original shuffle, rotating one trace per parent per epoch, batch 4, accumulation
8, max length 1024, bf16, backbone LR 1e-5, and 3 epochs are retained for full runs.
Expert/router and PAER-head LRs are 1e-4. The auxiliary router weight is 0.1.
MLM is excluded from training and selection. No quality-judge calls, new essay
generation or new attacks are performed by these launchers. Against historical
PAER this changes both capacity and router supervision; do not attribute any
future improvement solely to low rank without the appropriate controls.

## First run: remote C only, smoke tests sequentially

After A pushes and C pulls, run in `E:\xjj\aes_scoring` under `xjj_aes`:

```powershell
python .\paer\run_aes_low_rank_expert_training.py --smoke --dry-run
python .\paer\run_aes_paer_expert_training.py --smoke --dry-run

python .\paer\run_aes_low_rank_expert_training.py --smoke
python .\paer\run_aes_paer_expert_training.py --smoke
```

Dry runs only verify CPU-side source/data/config hashes; they create no files and
do not load models. Smoke runs use 160 **existing** train essays, a 32-essay
validation prefix and one epoch. Their attacked-parent subset explicitly covers
R:H:I-external:I-self-dup in 2:2:1:1 proportions; remaining rows are clean-only.
Full runs do not alter dataset coverage/order to achieve this smoke coverage.
Smoke QWK is not a paper result.

Outputs are separate:

```text
outputs/smoke_aes_low_rank_expert_rhi_seed42/
outputs/smoke_aes_paer_expert_rhi_seed42/
```

Inspect both runs before deciding on full training:

```powershell
$runs = @(
  '.\outputs\smoke_aes_low_rank_expert_rhi_seed42',
  '.\outputs\smoke_aes_paer_expert_rhi_seed42'
)
foreach ($run in $runs) {
  Write-Host "`n=== $run ==="
  Get-Content "$run\final_clean_metrics.json"
  Get-Content "$run\expert_resource_metrics.json"
  Get-Content "$run\training_diagnostics.jsonl" -Tail 3
  Test-Path "$run\best\model.safetensors"
  Test-Path "$run\best\paer_experts.pt"
}
```

Check finite losses, observations for all three prototype families, nonzero
router loss after bootstrap, and CUDA peak allocated/reserved memory. Expert
specialization and robustness still require full training and adaptive evaluation.
The RTX 3090 has 24576 MiB; actual memory is **not measured locally**. Run one
process at a time. Do not enable gradient checkpointing in this context-based v1.
If smoke memory is too high, adjust **both** variants equally before full runs;
record any batch/accumulation change rather than silently changing one model.

## Preservation and evaluation

Every launch requires a NEW output directory, including for retries; existing
directories are rejected even if empty. No training resume is implemented.
Use an explicit `--output-dir ..._run02` if a prior smoke launch already created
its output. Removing `--smoke` later uses full data and distinct `aes_*` outputs;
do not start full training until the smoke resource/coverage checks are reviewed.

Checkpoints contain normal HF backbone weights, `paer_config.json`, and
`paer_experts.pt`. Optional PAER heads are inside that sidecar. Loading only
`model.safetensors` with a generic HF loader omits the experts: use `AESScorer`,
which now dispatches on the new explicit model type. Existing RHI evaluation
tools can use it with NEW selection/evaluation directories and the uniform
30-step protocol. Historical bound outputs must not be re-bound or overwritten.

Engineering checks use tiny randomly initialized real DeBERTa models and temporary
test fixtures, not fabricated experimental essays:

```powershell
python -m unittest discover -b -s tests -p test_aes_expert_routing.py
```

Local verification: 20 extension checks and 18 original PAER checks passed on
CPU, including relative attention, embedding gradients/HotFlip, tiny training,
checkpoint reload, both variants, and fail-closed input/output guards. The full
test discovery also encountered two unrelated import errors because this local
checkout lacks the ignored `paper_results` package; those tests were not repaired
or counted as passing. No RTX 3090 training or robustness result is claimed yet.
