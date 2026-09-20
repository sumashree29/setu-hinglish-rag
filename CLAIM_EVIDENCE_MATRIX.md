# Claim-Evidence Matrix

This document maps every primary claim in the paper to its exact computational proof, source script, and canonical result file. It serves as the final, immutable bridge between the prose and the codebase.

| Hypothesis / Claim | Paper Prose (Draft) | Canonical Proof Source | Generating Script | Statistical Test & Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **H1: CMI vs Baseline Quality** | "Retrieval quality of the baseline multilingual encoder significantly degrades as query Code-Mixing Index (CMI) increases." | `results/tables/statistical_significance_H1_H10_scaled.json` | `scripts/run_statistical_tests_h1_h10_scaled.py` | Spearman rank correlation (rho). **Verdict: Supported.** |
| **H4: SETU v1 vs RAW Baseline** | "SETU v1 (fixed-order pipeline) significantly outperforms the raw BGE-M3 baseline, recovering performance drops associated with CMI." | `results/tables/statistical_significance_H1_H10_scaled.json` | `scripts/run_statistical_tests_h1_h10_scaled.py` | Paired Wilcoxon Signed-Rank. **Verdict: Supported.** |
| **H6: SETU v2 vs SETU v1** | "The dynamic controller (SETU v2) provides comparable retrieval quality to the fixed pipeline while attempting to minimize computational overhead." | `results/tables/statistical_significance_H1_H10_scaled.json` | `scripts/run_statistical_tests_h1_h10_scaled.py` | Paired Wilcoxon Signed-Rank. **Verdict: Not Supported (Equivalence).** |
| **H8: Step Reduction (Efficiency)** | "SETU v2 achieves its retrieval outcomes using significantly fewer operator steps on average compared to the fixed v1 pipeline." | `results/tables/statistical_significance_H1_H10_scaled.json` | `scripts/run_statistical_tests_h1_h10_scaled.py` | Paired Wilcoxon Signed-Rank. **Verdict: Supported.** |
| **H10: Confidence Gating** | "The margin-based proxy confidence signal effectively predicts final MRR, enabling early-stopping decisions." | `results/tables/statistical_significance_H1_H10_scaled.json` | `scripts/run_statistical_tests_h1_h10_scaled.py` | Spearman rank correlation. **Verdict: Supported.** |
| **Overcorrection on Strong Baselines** | "When applied to already-correct queries, SETU's operators degrade MRR significantly, highlighting a limitation of aggressive query rewriting." | `results/tables/overcorrection_final.json` | `scripts/statistical_correction_phase12.py` | Paired Wilcoxon (Holm-corrected). **Verdict: Confirmed Degradation.** |
| **Controller Adaptivity** | "The learned LinUCB controller predominantly converges to a static early-termination policy (LQP -> STOP) rather than dynamically routing based on CMI context." | `results/tables/controller_behavior_final.json` | `scripts/analyze_controller.py` | Chi-square test of independence. **Verdict: Confirmed Static.** |

### Data Provenance
- All canonical JSON files reside in `results/tables/`.
- The dataset comprises 314 finalized queries, tested in a strict 5-fold out-of-fold cross-validation setup to guarantee no data leakage between controller training and inference.
- Raw traces and per-query execution metrics are housed in `results/logs/setu_v2_per_query_v3.json` and `results/logs/per_query_metrics_v2.json`.

*Note: Any claim not listed here (such as external proprietary baselines or linguistic evaluation) was formally excluded from the experimental scope.*
