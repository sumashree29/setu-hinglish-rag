# FINAL EVIDENCE SCOPE

This document freezes the exact experimental scope for the IEEE evaluation of SETU-Hinglish-RAG. Any claims extending beyond these boundaries are unsupported by this repository.

1. **Dataset Size**: 314 canonical queries (75 pilot/manually verified, 239 auto-generated synthetic).
2. **Corpus Size**: 380 document chunks.
3. **Relevance Mapping**: Strict single-positive per query.
4. **Models Evaluated**: 
    - BGE-M3 (Primary backbone for SETU controller)
    - mE5-large (Baseline + Operator ablation)
    - Indic-SBERT (Baseline + Operator ablation)
    - mContriever (Baseline only)
    - BM25 (Baseline only)
5. **Operators Evaluated**: LQP (Query Projection), CAEP (Entity Preservation), LAG (Translation).
6. **Controller Definition**: SETU v1 (fixed heuristic order), SETU v2 (LinUCB contextual bandit with 5-fold OOF training).
7. **Statistical Tests**: 
    - Wilcoxon signed-rank test for paired retrieval metrics (MRR/nDCG).
    - Spearman rank correlation ($\rho$) for monotonic associations (e.g., Confidence vs MRR).
    - Chi-square test of independence for categorical distributions (e.g., Controller Action vs CMI Band).
    - Holm-Bonferroni correction applied across all concurrent hypothesis tests to control Family-Wise Error Rate (FWER).
8. **Latency Protocol**: Generic host machine CPU inference environment, measuring end-to-end execution per query excluding model warm-up and I/O.
9. **Canonical Source of Truth**: All empirical claims must derive strictly from the JSON structures housed in `results/canonical/`, which are generated deterministically by the `scripts/*_final.py` analytic suite based on raw execution logs.

*Note: Phase 13 (IndicLID linguistic validation) and Phase 15 (External Black-Box Baselines) were formally scoped out of this revision.*
