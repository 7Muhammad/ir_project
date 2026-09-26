# Experiment 11: encoder per-head causal patching -- summary

Of 144 head-slots, 18 exceed the combined_effect > 0.02 threshold on the canonical attack (n=100). Top late-layer (9-11) heads: L10H0, L10H2, L11H3, L11H11, L9H6.
Robustness across the full 105-attack sweep (layers 9-11): L11H3: 104/105; L11H4: 104/105; L10H6: 103/105; L10H2: 102/105; L10H4: 100/105.

**Step 3 additivity:**
Prediction (stated before computing): sub-additive, matching Experiment 6's template-position finding (0.44x of whole-span effect).
Flagged heads (full-sweep peak combined_effect > 0.02, layers 9-11): 10 of 12 -> [0, 2, 3, 4, 6, 7, 8, 9, 10, 11]
Sum of flagged heads' individual peak combined_effect (canonical, n=100): 0.8496
Whole-span 'both' (query+document) peak combined_effect at layers 9-11 (Experiment 6, encoder_self_attn): 0.4840
Ratio: 1.755x -- super-additive (prediction DOES NOT hold, unexpected)
Consequence for Phase 4: effects compose close to additively, so a search algorithm may be unnecessary -- top-k heads by individual effect likely suffice for Phase 4, which can be scoped down.

**Step 5 attack-token-as-hub:**
HUB HYPOTHESIS HOLDS: flagged heads' mean elevation = +0.1419 vs. unflagged heads' +0.1103 (16 flagged / 20 unflagged (layer,head) cells at layers [9, 10, 11]).

**Step 4 overlap:** 6 of 48 (layer, head) cells at layers 8-11 carry both elevated document->query attention mass and a causal effect above threshold -- see outputs/layer9_crossref/classification.csv for the full per-head breakdown.
