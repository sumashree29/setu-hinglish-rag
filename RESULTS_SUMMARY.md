# Results Summary

## Dataset & Evaluation Scope
- **Total Queries Evaluated:** 314
- **Chunks in Search Corpus:** 380
- **Misspelled-Entity Queries:** 15

## Baseline Performance (BGE-M3)
The dense retrieval baseline already solves the vast majority of queries natively:
- **MRR:** 0.8526
- **nDCG@10:** 0.8858
- **Recall@5:** 0.9554
- **Recall@10 / Hit@10:** 0.9873

## SETU v1 vs SETU v2 vs RAW
- **RAW MRR:** 0.8526 (0.8858 nDCG@10)
- **SETU v1 MRR:** 0.8480 (0.8801 nDCG@10)
- **SETU v2 MRR:** 0.8494 (0.8833 nDCG@10)

## Statistical Analysis Highlights
- **H1 (CMI Degradation):** Not supported (Spearman rho=0.0908, corrected p=0.4330).
- **H2 (Indic-SBERT robustness):** Not supported (BGE-M3 significantly outperforms Indic-SBERT).
- **H4 (SETU v1 > RAW):** Not supported. SETU v1 yields slightly lower MRR, and the difference is not statistically significant for improvement.
- **H6 (SETU v2 > SETU v1):** Not supported. No statistically significant difference was detected between SETU v1 and SETU v2 under the evaluated paired test. This is not an equivalence claim.
- **H8 (SETU v2 requires fewer steps):** Supported. SETU v2 takes significantly fewer operator actions (mean 1.697) than SETU v1 (fixed 3.0). Reduction = 43.4%.
- **H10 (Confidence correlates with success):** The previous margin-confidence correlation result is excluded because the underlying confidence-margin series is not present in the canonical evidence.

## Controller Behavior
The LinUCB controller largely functions as an early-termination policy with limited sequence diversity:
- **Explicit Stops:** 14 (4.46%)
- **Forced Stops (repeat action):** 300 (95.54%)
No statistically significant association between the initial action and CMI band was detected in the evaluated dataset.

## Overcorrection Analysis
- **Already Correct Queries:** For BGE-M3, the stored corrected results show:
  - LAG: significant degradation on the already-correct subset after Holm correction.
  - LQP: not significant after Holm correction.
  - CAEP: not significant after Holm correction.
  Across auxiliary encoders, significant overcorrection also occurs for specific combinations, so describe it as an operator/model-specific risk, not a universal property of every operator.
- **Incorrect Queries:** No tested operator produced significant improvement on the BGE-M3 baseline-failed subset after Holm correction.

## Latency
- **Hardware:** Windows 10, Intel Core 12-core CPU, 16GB RAM. Local CPU run.
- **RAW:** Mean 128.78 ms
- **SETU v1:** Mean 757.38 ms
- **SETU v2:** Mean 383.89 ms
