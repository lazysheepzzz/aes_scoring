# PAER-v3 experiment protocol

This file freezes the PAER-RH-v3 protocol before its formal seed-42 run. Existing
B0, C0, attack-specific defenses, Mixed-AT-RH, PAER-v1, and PAER-v2 artifacts
remain historical results and must not be overwritten.

## Attack-family taxonomy

Rudimentary, HotFlip, and Injection are the three peer attack--defense
families. Each has an undefended attack evaluation, a corresponding
attack-specific adversarial-training defense, and cross-defense robustness
evaluation. Neither implementation order nor seen/unseen status changes their
equal role. Injection External and Self-Duplication are additionally reported
as the two components of the Injection-family result.

MLM-guided has a different experimental role. It is an attack-only transfer
evaluation and has no separate MLM defense in the source paper or in this
protocol. It may appear as an additional attack column when reporting model
robustness, but it must not be described as a fourth peer attack--defense
family.

“Seen” and “unseen” describe only a particular model's training exposure. For
the formal PAER-RH-v3 method, Rudimentary and HotFlip are seen, while Injection
and MLM-guided are unseen. Injection nevertheless remains a peer
attack--defense family in the main evaluation matrix; its PAER-RH-v3 result
measures cross-family transfer. PAER-RHI-v3 is only an optional future
three-family extension, not a replacement for or prerequisite of the completed
PAER-RH-v3 experiment. If that extension is run, its required data-matched
baseline is Mixed-AT-RHI.

## Research claim under test

PAER-v3 tests whether direction-aware token evidence aggregation contributes
robustness beyond ordinary mixed adversarial training.  The model decomposes
each token's signed evidence into positive and negative parts and applies risk
routing only to the positive part.  Thus, suspicious score-inflating evidence
can be suppressed without hiding legitimate negative evidence such as grammar
damage or nonsensical substitutions.

## Fixed comparison controls

- Initialization: `deberta_checkpoints/fold0_best` (B0).
- Clean training and validation splits: `train_fold0.csv` and
  `valid_fold0.csv`.
- Adversarial training source: the existing seed-42 RH trace JSONL only.
- Seen attacks: Rudimentary and HotFlip, with the same trace rows used by
  Mixed-AT-RH.
- Training budget: 3 epochs, physical batch 4, gradient accumulation 8,
  effective batch 32, maximum length 1024, seed 42, checkpoints every 200
  optimizer steps.
- Common checkpoint cap: `gstep1400`.
- Clean-performance eligibility: the existing selector's C0 QWK minus 0.02.
- Among eligible checkpoints: minimize the equal-weight Rudimentary/HotFlip
  subset ASR under the existing fixed subset and attack settings.
- Selection inference: float32, batch size 4 on the 24 GB RTX 3090.

Architecture-specific v3 heads use their separately recorded head learning
rate and losses.  This is part of the proposed method, not an assertion that
v3 and Mixed-AT have identical parameterization.  All data, backbone,
training-budget, evaluation, and selection controls above remain aligned.
For a cumulative trace state, localization uses all token differences between
the original essay and that state (not only the final accepted edit), matching
the cumulative score-inflation target.

## Leakage boundary

MLM-guided and Injection-family examples are excluded from PAER-RH-v3 training,
validation, loss tuning, and checkpoint selection.  Their existing results
may be reported as historical context, but the v3 checkpoint must be frozen
after RH-only selection before either v3 evaluation is run.

This exclusion is a property of PAER-RH-v3, not a claim that Injection is
secondary. An optional RHI extension may make Injection a seen training and
selection family under the same status as Rudimentary and HotFlip, but that is
a distinct additional experiment. MLM-guided remains attack-only transfer
evaluation; no MLM specialized defense is introduced.

## Required reports; no invented success threshold

The paper will report, rather than silently gate on, all of the following:

1. Clean QWK, MAE, and RMSE.
2. Rudimentary and HotFlip ASR, average score inflation, band ASR, original
   QWK, and adversarial QWK.
3. Equal-weight RH macro ASR for PAER-v3 and Mixed-AT-RH.
4. Fixed-set route-on/route-off correction lift, mean-delta reduction, and ASR
   difference.  This diagnoses module contribution but does not replace an
   adaptive attack.
5. Injection-family and MLM-guided transfer results after freezing v3.

No unrequested “15 percentage-point ASR reduction” or similar post-hoc pass
criterion is part of this protocol.  A positive route-off result is necessary
to claim that the routing module itself is active; if it is absent, the honest
conclusion is that any robustness primarily comes from backbone adversarial
training.

## Resource constraint

The formal run targets one RTX 3090 with 24,576 MiB.  The trainer processes and
backpropagates the clean branch before constructing the adversarial graph, so
both full DeBERTa graphs are not retained simultaneously.  Do not raise the
physical training batch above 4 without a new smoke test.  Reduce the physical
batch and increase gradient accumulation proportionally if memory headroom is
insufficient; record that change while preserving effective batch size 32.
