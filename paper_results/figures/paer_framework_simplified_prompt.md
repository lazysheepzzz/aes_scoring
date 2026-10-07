Generate an English academic framework diagram for a PAER automatic essay scoring research paper. White background, navy strokes, pastel teal and lavender accents, crisp professional typography, landscape 16:9 high resolution. Not a poster, no photos or shadows. Prioritize correctness and uncluttered labeled boxes over complex wire diagrams. Use precisely THREE panels.

Left panel (a) 24% width heading "Offline trajectory construction". Box "Training essays (x, y)" -> three compact equal-peer labels "Rudimentary", "HotFlip", "Injection", Injection sublabel "External / Self-duplication" -> box "Greedy search" subtitle "beam = 1; 3 steps; ≤16 candidates/step"; side small "Frozen B0 scorer" -> search. Search -> "RHI trajectory pool" subtitle "Edited texts, edit positions, score gains". Bottom footnote "Gain-based proxy labels, not human quality judgments". Search is visually subordinate. Pool has solid arrow into input of center model panel, label "Clean / adversarial texts". Also purple dashed arrow down into training panel.

Right upper panel (b) 72% width heading "PAER: direction-aware evidence routing". SINGLE CLEAN HORIZONTAL PIPELINE with five modules and EXACTLY FORWARD arrows between them:
"Essay input" -> "Shared DeBERTa encoder" -> a SINGLE tall composite block "Prediction heads" -> a SINGLE tall composite block "Directional aggregation" -> "Essay score".
Inside Prediction heads block FOUR separate labeled rows of text, not internal arrows:
"Global score: g(x)"
"Attention: αᵢ = softmax(aᵢ)"
"Signed evidence: eᵢ = Eₘₐₓ tanh(zᵢ)"
"Risk: rᵢ = sigmoid(uᵢ)"
Inside Directional aggregation block EXACTLY TWO stacked colored rectangles:
teal rectangle "Risk-gated positive evidence" with formula "P(x) = Σᵢ αᵢ(1 − λrᵢ) max(eᵢ, 0)"
orange rectangle "Negative evidence bypasses risk gate" with formula "N(x) = Σᵢ αᵢ max(−eᵢ, 0)".
Prediction-heads composite connects to Aggregation composite using ONE single arrow labelled "αᵢ, eᵢ, rᵢ" entering outside boundary between colored rectangles, NOT separate arrows into individual negative/positive sub-boxes. ONE global-score bypass line from top of Prediction heads composite to Essay score top, labelled "g(x)".
Aggregation composite connects to Essay score via ONE arrow labelled "P(x), N(x)".
Essay score box formula "s(x) = g(x) + P(x) − N(x)".
Below these five modules inside panel b, one small readable callout: "Equivalent correction: C(x) = λΣᵢ αᵢrᵢ max(eᵢ, 0)" and note "Sums over valid content tokens". This is diagnostic text only, NOT an additional subtractor.

Full-width lower panel (c), 24% image height: "Training-only objectives". Three compact sibling boxes:
"Scoring" / "Clean-label regression" / "Adversarial non-inflation";
"Token-level trajectory supervision" / "Risk localization" / "Edited-token attention alignment";
"Correction learning" / "Gain calibration and correction lift" / "Clean regularization".
A single purple dashed bus from trajectory pool feeds second and third boxes. A single purple dashed arrow from second box UP ends at bottom boundary of PREDICTION HEADS composite and is labelled "Supervise αᵢ and rᵢ". A single purple dashed arrow from third box UP ends at the correction formula CALLOUT, labelled "Positive cumulative gain target". A clean labels small source "y" points into first objective. A thin line from Essay score follows outer right margin then lower margin to first objective labelled "Predicted scores". No line connects labels to model inference. No direct supervision arrow to signed evidence e_i.
Bottom slim note outside panels: "Inference: essay only → PAER → score" and "MLM-guided: held-out attack evaluation only". Legend solid navy = forward computation; dashed purple = training supervision.
All math and labels precise. No performance metrics, no PAER version numbers, no journal logo. No fake human quality judgments. No complicated internal head wiring. The equations explicitly specify positive-only gating. Both R/H/I trajectory data and clean labels exist only during training.
