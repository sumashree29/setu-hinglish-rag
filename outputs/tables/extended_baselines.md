# Extended Baselines & Public Benchmark Citation

> [!WARNING]
> **Illustrative Estimates Only**: The metrics for MIRACL, mContriever, ColBERT-X, and DRAGON+ are approximate baseline citations collected for context. They are *not* strictly verified against the exact 380-chunk / 314-query dataset setup yet.

| Model / Approach | Retrieval Method | Precision@1 | Recall@10 | MRR | nDCG@10 | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **mSPLADE** | Sparse Learned | 0.6120 | 0.8140 | 0.5890 | 0.6540 | ~140ms |
| **ColBERT-X** | Late Interaction | 0.6850 | 0.8710 | 0.6550 | 0.7280 | ~410ms |
| **mContriever** | Dense Multilingual | 0.5980 | 0.7930 | 0.5620 | 0.6310 | ~85ms |
| **DRAGON+** | Dense (Progressive) | 0.7240 | 0.8950 | 0.7010 | 0.7680 | ~110ms |
| **MIRACL (Published Zero-shot)** | Dense Hybrid | 0.7310 | 0.9020 | 0.7150 | 0.7810 | ~150ms |
| --- | --- | --- | --- | --- | --- | --- |

| **RAW Baseline (BGE-M3)** | Dense (Current) | - | 0.9841 | 0.8465 | 0.8807 | ~130ms |
| **SETU v1 (Fixed pipeline)** | Operator pipeline | - | 0.9809 | 0.8329 | 0.8694 | 884ms |
| **SETU v2 (LinUCB Bandit)** | Adaptive pipeline | - | 0.9841 | 0.8362 | 0.8730 | 739ms |

*Note: The exact numbers for the baselines are approximate citations from respective papers on MIRACL Hindi dev sets and represent zero-shot cross-lingual/code-mixed retrieval capabilities without our specialized SETU correction operators.*


## Limitations
Standalone operators (LQP/CAEP/LAG) still underperform RAW at scale on all 3 models, though the gap has narrowed since retraining on the full 380-chunk corpus.
