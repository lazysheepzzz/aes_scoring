# PAER framework figure generation prompt

Generated with the built-in image generation tool. The English method figure is a draft for manuscript use; PAER is a working name. Equations reflect modeling_paer_v3.py. Offline search settings reflect the current RHI training pool, not the 30-step evaluation budget.

Create a polished, technically precise ENGLISH scientific method framework figure for an AES paper intended for IEEE Transactions on Learning Technologies. This is a journal architecture diagram, NOT an infographic poster, NOT a cover, NOT slides. Landscape high-resolution raster, preferably 3840 x 2160. White background, crisp thin vector-like outlines, no gradients, no shadows, no 3D, no cartoon brains or robots. Legible professional sans-serif labels and mathematical italic notation. Restrained navy blue model boxes, muted purple supervision lines, teal positive route, rust-orange negative route. Clear panel labels (a), (b), (c). PAER is a provisional model name, no v3, seed, results, performance claims, or journal logos. The MODEL is the dominant visual subject; search is a small secondary component.

EXACT ARCHITECTURE AND COMPOSITION:
Upper ~72% height: small left panel (a), large right panel (b).
Lower ~22% height: shallow panel (c) training objectives. Bottom slim inference legend.
Panel (a), about 25% width: heading "(a) Offline trajectory construction". Top small box "Training essays (x, y)" -> three equally-sized sibling boxes "Rudimentary", "HotFlip", "Injection". Under Injection inside same box small text "External / Self-duplication", no unequal emphasis among the three families. The three converge into ONE compact box "Greedy search (beam = 1)" with a small second line "3 steps; up to 16 candidates / step". A small side box "Frozen B0 scorer" has a bidirectional scoring loop to search, no connection representing B0 updating from losses. Search -> small stack of sheets "RHI trajectory pool" with subtitle "Edited text, edit positions, score gains". Small footnote "Proxy supervision, not human quality labels". NO MLM inside this panel.

Panel (b), about 70% width, dominant pale-blue outlined area, heading "(b) PAER: direction-aware evidence routing".
At left of (b) a small input box "Essay text" with second line "x or x′ during training" -> tall shared box "DeBERTa encoder". Indicate one shared model, NOT clean and attack separate models. Encoder branches into FOUR clearly drawn paths:
1. upper bypass "Global scoring head" -> a line marked "g(x)" continuing above other modules directly to final SUM node on far right.
2. "Attention head" -> "αᵢ = softmax(aᵢ)"
3. "Signed evidence head" -> "eᵢ = Eₘₐₓ tanh(zᵢ)" -> split into two short branches "eᵢ⁺ = max(eᵢ, 0)" (teal) and "eᵢ⁻ = max(−eᵢ, 0)" (orange).
4. "Risk head" -> "rᵢ = sigmoid(uᵢ)" -> feeds ONLY the positive gate.
Use positions to avoid crossing: global bypass top; attention row next; evidence row center; risk row lower feeding upwards to positive gate; negative route on a separate clearly parallel lane.
Two aggregate boxes to right of heads:
teal "Risk-gated positive evidence" formula "P(x) = Σᵢ αᵢ(1 − λrᵢ)eᵢ⁺"
orange "Unfiltered negative evidence" formula "N(x) = Σᵢ αᵢeᵢ⁻".
Attention feeds BOTH aggregate boxes. Positive evidence feeds positive aggregate, negative evidence feeds negative aggregate. Risk connects ONLY to positive aggregate, never negative. Label negative lane "Bypasses risk gate".
Global bypass, P(x), and N(x) connect to a final sum node, with "+", "+", "−" at respective incoming edges. Output box "Essay score" formula "s(x) = g(x) + P(x) − N(x)".
Within bottom edge of panel b a small mathematical callout, secondary but legible: "Correction: C(x) = λΣᵢ αᵢrᵢeᵢ⁺". This is an equivalent diagnostic, NOT an extra module subtracting correction a second time. DO NOT draw a second correction subtraction.
No assertion that these learned signed contributions represent verified teacher rubric criteria.

Panel (c), lower horizontal thin purple-tinted area heading "(c) Training-only supervision".
Three aligned small modules with short text:
"Scoring objectives" / "Clean-label regression; non-inflation on adversarial pairs"
"Token-level trajectory supervision" / "Risk localization + edited-token attention alignment"
"Correction objectives" / "Gain calibration + correction lift; clean regularization".
Thin dashed purple arrows from trajectory pool to token-level module and correction module, and dashed arrows from those modules back to relevant model heads or correction callout. Clean labels y feed scoring objectives; predicted scores feed them as well. Use clean orthogonal connectors in whitespace; do not spaghetti many loss arrows. Short note under correction objectives "Target: clipped positive cumulative gain". Short note near token supervision "Edit mask weighted by cumulative gain".
Bottom outside all panels:
solid-line legend "Forward computation"; dashed-purple-line legend "Training supervision".
Compact right-side note: "Inference: essay only → PAER → score" and below "MLM-guided: held-out attack evaluation only".
Inference must not need labels, traces, original paired essay, search, or B0.
Absolutely accurate spelling and clear arrowheads; adequate whitespace; text readable when scaled to a two-column paper figure. Use mathematics exactly as specified, sums over valid content tokens (small note if space). Keep searches visually subordinate, emphasize routing and supervision, not the entire experimental pipeline.
