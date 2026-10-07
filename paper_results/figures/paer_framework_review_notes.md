# Framework figure review

Status: VISUAL DRAFT ONLY. Do not use the current raster as a final journal figure.

The built-in image generator produced a useful composition but failed to make all
connectors match the equations. Further generation was blocked by usage limits.
`paer_rhi_framework_review_draft.png` is the second rendering, retained for layout
review, not as a validated computation graph. The earlier raster is also a draft.

Required corrections:
- Remove purple arrows that appear to connect risk, evidence and attention heads.
  Token-level proxy supervision targets risk and attention, not ground-truth
  signed evidence. It can indirectly affect the shared encoder through gradients.
- Give positive evidence its own clear input to the positive aggregate.
- Connect the offline pool to training inputs and both trajectory-supervision losses.
- Ensure no risk signal reaches the negative aggregate.
- Ensure predicted scores enter losses, not the clean-label source.

The equivalent score expression is
`s(x) = g(x) + sum_i alpha_i (1-lambda*r_i) max(e_i,0)
                - sum_i alpha_i max(-e_i,0)`.
`C(x) = lambda sum_i alpha_i r_i max(e_i,0)` is an equivalent diagnostic,
not an additional subtraction from the already gated score.

Suggested caption:
Overview of the trajectory-supervised PAER framework. (a) Rudimentary, HotFlip,
and Injection searches against a frozen B0 scorer construct an offline training
pool with edited texts, edit positions, and score gains. (b) A shared DeBERTa
encoder predicts a global score, token attention, signed evidence, and risk
weights. Directional aggregation gates positive token evidence while leaving
the negative term outside the risk gate. (c) Clean regression, adversarial
non-inflation, token-level proxy supervision, and correction objectives jointly
train the scorer. Inference requires only the essay. MLM-guided attacks are used
only for held-out attack evaluation, not training or checkpoint selection.

The 3-step/16-candidate search settings refer to offline RHI training trajectories,
not the 30-step full attack evaluation protocol. Proxy gain signals are not human
quality annotations. PAER is a provisional method name.
