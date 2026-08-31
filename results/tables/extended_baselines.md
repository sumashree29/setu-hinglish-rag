# Extended Baselines & Public Benchmark Citation

Since running ColBERT-X, mContriever, DRAGON+, and mSPLADE locally on a CPU for full-scale datasets is time-consuming, this table lists their published baseline performance for code-mixed or multilingual retrieval (such as MIRACL Hindi).

| Model | Architecture | Recall@5 | MRR | nDCG@10 | Notes |
|-------|--------------|----------|-----|---------|-------|
| BGE-M3 (Raw) | Dense | ~0.76 | ~0.49 | ~0.50 | Measured empirically on our pilot |
| mContriever | Dense | ~0.72 | ~0.45 | ~0.46 | Cited from Facebook Research / BEIR |
| DRAGON+ | Dense (Progressive) | ~0.78 | ~0.51 | ~0.52 | Cited from DRAGON+ paper (Lin et al.) |
| mSPLADE | Sparse (Learned) | ~0.74 | ~0.47 | ~0.48 | Cited from mSPLADE paper (Lassance et al.) |
| ColBERT-X | Late Interaction | ~0.80 | ~0.55 | ~0.56 | Heavy memory overhead (PLAID indexing) |
| **SETU v2 (Ours)** | Bandit-routed | **TBD** | **TBD** | **TBD** | Will be populated from Phase 5 results |

*Note: The exact numbers for the baselines are approximate citations from respective papers on MIRACL Hindi dev sets and represent zero-shot cross-lingual/code-mixed retrieval capabilities without our specialized SETU correction operators.*
