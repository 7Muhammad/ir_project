# Experiment 6 extension: template-token sink/signal analysis -- summary

- **Query** (SIGNAL, classified relay): peak combined_effect (layers 9-11, canonical attack) = +0.0720, exceeds threshold on 91/105 attacks in the full sweep.
- **:_after_query** (SIGNAL, classified relay): peak combined_effect (layers 9-11, canonical attack) = +0.0531, exceeds threshold on 105/105 attacks in the full sweep.
- **Document** (SIGNAL, classified relay): peak combined_effect (layers 9-11, canonical attack) = +0.0405, exceeds threshold on 77/105 attacks in the full sweep.
- **:_after_document** (SIGNAL, classified relay): peak combined_effect (layers 9-11, canonical attack) = +0.0352, exceeds threshold on 89/105 attacks in the full sweep.
- **Relevant** (sink/uninvolved, classified sink): peak combined_effect (layers 9-11, canonical attack) = +0.0022, exceeds threshold on 7/105 attacks in the full sweep.
- **:_after_relevant** (sink/uninvolved, classified sink): peak combined_effect (layers 9-11, canonical attack) = +0.0062, exceeds threshold on 5/105 attacks in the full sweep.
- **</s>** (sink/uninvolved, classified sink): peak combined_effect (layers 9-11, canonical attack) = +0.0022, exceeds threshold on 0/105 attacks in the full sweep.

Sum of the 7 template positions' individual peak combined_effect (layers 9-11) = 0.2114; exp6's existing whole-span 'both' (query+document) peak combined_effect at the same layers = 0.4840 (ratio 0.44x) -- diverges from additive (positions interact, or a modelling non-linearity dominates).

**Conclusion**: Query, :_after_query, Document, :_after_document carry non-trivial causal signal when patched in isolation at late encoder layers -- the sink hypothesis does NOT hold for these positions, so exp6's §4.4 query-only/document-only effect sizes (which freeze all 7 template positions at their control value) are underestimates for whatever fraction of the pathway runs through these positions, and the report needs a caveat quantifying this.
