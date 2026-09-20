# Claim Evidence Matrix

Every major paper claim is mapped to the final canonical evidence, with its statistical verdict.

| Claim ID | Claim Text | Evidence Source | Test / Metric | Verdict | Limitations / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C1** | CMI degrades retrieval performance. | `ieee_ready_tables.json` (H1) | Spearman correlation | **NOT SUPPORTED** | rho=0.0908, p=0.433. No significant monotonic relationship detected across the 314 queries. |
| **C2** | Indic-tuned encoders handle code-mixed data better than general multilingual encoders. | `ieee_ready_tables.json` (H2) | Paired Wilcoxon | **NOT SUPPORTED** | BGE-M3 significantly outperforms Indic-SBERT. |
| **C3** | The SETU pipeline improves overall retrieval accuracy over RAW baseline. | `ieee_ready_tables.json` (H4) | Paired Wilcoxon | **NOT SUPPORTED** | Mean MRR dropped slightly. No significant positive effect. |
| **C4** | SETU v2 (LinUCB) outperforms SETU v1 (fixed-order). | `ieee_ready_tables.json` (H6) | Paired Wilcoxon | **NOT SUPPORTED** | p=0.7423. No statistically significant difference detected. Equivalence is not proven. |
| **C5** | SETU operators fix queries that the baseline gets wrong. | `ieee_ready_tables.json` (Table 6) | Paired Wilcoxon (Incorrect Subset) | **NOT SUPPORTED** | After Holm-Bonferroni correction, no operator provides statistically significant lift. |
| **C6** | SETU operators damage queries that the baseline gets right. | `ieee_ready_tables.json` (Table 6) | Paired Wilcoxon (Correct Subset) | **SUPPORTED** | Multiple operators (LQP, CAEP, LAG) significantly degrade already-correct queries. |
| **C7** | The SETU v2 controller sequence choices are independent of CMI context. | `ieee_ready_tables.json` | Chi-Square Independence | **DESCRIPTIVE ONLY** | p > 0.05. We can state no significant association was detected. We cannot claim complete independence. |
| **C8** | SETU v2 maintains accuracy with significantly fewer computational steps than v1. | `ieee_ready_tables.json` (H8) | Paired Wilcoxon | **SUPPORTED** | Mean steps: 1.70 (v2) vs 4.0 (v1). |
| **C9** | Margin-based confidence proxy correlates with retrieval success. | `ieee_ready_tables.json` (H10) | Spearman correlation | **SUPPORTED** | rho=0.2985, p < 0.05. Establishes correlation, not causality. |
| **C10** | SETU incurs latency overhead over raw retrieval. | `ieee_ready_tables.json` (Table 7) | Latency means | **DESCRIPTIVE ONLY** | Measured on a local CPU constraint. RAW=128ms, v2=447ms. |
