# Research Journal: BhojRAG Experiments

This document tracks our iterative experimental findings, parameter sweeps, and failure analyses.

## [Date: 2026-05-07] Initial Baseline & Synthetic Data Overfitting

### Observation
Our initial evaluation run produced suspiciously high metrics for sparse retrieval. Specifically, `word_bm25` achieved a perfect `1.000` MRR@10. 

### Analysis
The current evaluation dataset is derived synthetically via LLM-based chunk QA generation. This means the query vocabulary perfectly matches the chunk vocabulary without the natural noise, typos, and orthographic inconsistencies found in real-world Bhojpuri text. Because Bhojpuri is unstandardized, a perfect MRR on a clean dataset does *not* reflect real-world robustness.

### Next Steps
1. **Noisy Evaluation:** We need to construct an out-of-domain (OOD) or "noisy" evaluation split that introduces spelling variations (e.g., `भोजपूरी` vs `भोजपुरी`) to properly evaluate the `char_ngram_bm25` robustness claim.
2. **Hard Negative Mining:** The dense retriever (`MuRIL`) currently scores much lower (`0.7725` post-tuning). It likely needs to be fine-tuned using hard negatives mined from the sparse retriever rather than random in-batch negatives.
