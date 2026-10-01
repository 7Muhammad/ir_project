| metric | best |d| (head) | AUROC in-sample (best head, oriented) | CV best-head AUROC | top-5 heads | layer with max mean|d| | paired Δ at top-5 heads, as fraction of genuine gap (rel / nonrel) | max |ρ(Δmetric, Δscore)| any head (rel / nonrel) |
|---|---|---|---|---|---|---|---|
| cos | +1.32 (L11H2) | 0.809 | 0.809 | L11H2,L11H3,L9H6,L11H11,L10H6 | 11 | +0.05 / +0.05 | 0.43 / 0.47 |
| ccos | +1.07 (L11H7) | 0.779 | 0.76 | L11H7,L11H2,L11H9,L11H3,L11H11 | 11 | +0.05 / +0.04 | 0.45 / 0.42 |
| dot | +1.21 (L11H2) | 0.823 | 0.823 | L11H2,L11H11,L11H3,L9H6,L11H9 | 11 | +0.05 / +0.06 | 0.45 / 0.48 |
| qnorm | -1.24 (L11H2) | 0.799 | 0.799 | L11H2,L11H3,L11H11,L10H0,L11H6 | 11 | +0.05 / +0.05 | 0.50 / 0.67 |
| dnorm | -0.86 (L11H8) | 0.743 | 0.656 | L11H8,L11H9,L11H7,L11H4,L11H10 | 11 | +0.10 / +0.16 | 0.44 / 0.44 |
| ms_mean | +0.97 (L11H11) | 0.792 | 0.625 | L11H11,L9H6,L11H0,L11H10,L11H2 | 11 | +0.07 / +0.10 | 0.43 / 0.38 |
| ms_median | -0.85 (L9H6) | 0.757 | 0.585 | L9H6,L11H11,L11H0,L11H2,L11H10 | 11 | +0.07 / +0.08 | 0.36 / 0.37 |
| ms_top3 | -0.87 (L9H6) | 0.756 | 0.756 | L9H6,L11H2,L11H0,L11H11,L10H0 | 11 | +0.02 / +0.07 | 0.39 / 0.37 |