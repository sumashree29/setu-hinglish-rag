# SETU-Hinglish-RAG: Final Results Summary

This document summarizes the canonical, leakage-free results of the SETU evaluation following the Phase 0-12 remediation audit.

## 1. Zero-Shot Baselines (Phase 9)
Evaluating on the 314 canonical Hinglish queries without any operators demonstrates that modern multilingual dense models already solve the vast majority of cases:
- **BGE-M3**: MRR = 0.8526, Hit@10 = 0.9873
- **mE5-large**: MRR = 0.8767, Hit@10 = 0.9873
- **Indic-SBERT**: MRR = 0.5903, Hit@10 = 0.8280
- **mContriever**: MRR = 0.7285, Hit@10 = 0.9204

*Finding*: Baseline BGE-M3 and mE5-large perform exceptionally well natively, leaving a very small headroom (e.g., 20-25% of queries) for any corrective operator to improve.

## 2. Operator Ablation & Over-Correction (Phase 10 & 11)
When corrective operators (LQP, CAEP, LAG) are applied independently:
- **Aggregated Washout**: The operators yield flat or negative aggregate MRR deltas when evaluated over the full 314 queries, failing to outperform the raw baselines (particularly on BGE-M3 and mE5-large).
- **Over-Correction Mechanism**: Conditional analysis reveals that all operators exert a statistically significant *degradation* (negative MRR delta) on queries that the raw retriever had already mapped correctly.
- **Lack of Failure Lift**: Conversely, on queries where the raw retriever failed (MRR < 1), the operators provide *no statistically robust improvement* (Phase 12 multiple-testing correction confirmed that an initial observed lift for BGE-M3+LAG was a false positive, Holm p=0.398).

*Finding*: Operators are structurally predisposed to disrupt natively strong representations while offering no significant corrective power on actual failures.

## 3. Controller Adaptivity (Phase 4)
The LinUCB controller was hypothesized to dynamically sequence operators based on a 7-dimensional context vector (including CMI, entropy, and confidence).
- **Policy Collapse**: The controller operates almost exclusively as a static early-termination policy. In >95% of queries, it takes exactly one action before triggering a forced stop.
- **Context Independence**: A Chi-square test confirms that the initial action chosen by the controller is statistically independent of the query's complexity/CMI band (Holm p=1.0).
- **LAG Internal Collapse**: The LAG classifier itself collapses to predicting a single strategy (`light_normalize`) for 97.1% of queries due to severe class imbalance.

*Finding*: SETU functions as a fixed-policy correction system rather than a context-adaptive one. 

## 4. Final Verdict
The original hypothesis—that code-mixed specific operators adaptively orchestrated by a controller would outperform raw multilingual retrieval—is **rejected**. Following strict isolation of train/test data (fixing Phase 1/2 leakage) and multiple-testing correction (Phase 12), SETU demonstrates no robust empirical benefit over zero-shot BGE-M3 or mE5-large, and actively degrades correct retrievals.
