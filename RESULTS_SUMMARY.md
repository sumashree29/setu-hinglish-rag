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
- **H6 (SETU v2 > SETU v1):** Not supported. No statistically significant difference was detected between SETU v1 and SETU v2 under the evaluated test (corrected p=0.7423). Note: We do not claim strict equality as a TOST non-inferiority margin was not pre-specified.
- **H8 (SETU v2 requires fewer steps):** Supported. SETU v2 takes significantly fewer steps (mean 1.70) than SETU v1 (fixed 4.0).
- **H10 (Confidence correlates with success):** Supported (Spearman rho=0.2985, p=0.0000).

## Controller Behavior
The LinUCB controller largely functions as an early-termination policy with limited sequence diversity:
- **Explicit Stops:** 14 (4.46%)
- **Forced Stops (repeat action):** 300 (95.54%)
No statistically significant association between the initial action and CMI band was detected in the evaluated dataset.

## Overcorrection Analysis
- **Already Correct Queries:** All tested operators significantly degraded queries that the RAW baseline already answered correctly.
- **Incorrect Queries:** No statistically significant improvement was detected for any operator on queries that the RAW baseline failed on, after applying the Holm-Bonferroni correction.

## Latency
- **Hardware:** Windows 10, Intel Core 12-core CPU, 16GB RAM. Local CPU run.
- **RAW:** Mean 128.78 ms
- **SETU v1:** Mean 757.38 ms
- **SETU v2:** Mean 447.16 ms
