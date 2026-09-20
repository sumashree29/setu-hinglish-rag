# SETU-Hinglish-RAG Claim Contract

This document dictates the absolute boundaries of what can and cannot be claimed in the IEEE manuscript, based on the rigorous remediation audit and Holm-Bonferroni statistical corrections (Phases 0-12).

## 1. SAFE TO CLAIM
- **Baseline Dominance**: Modern multilingual dense zero-shot models (BGE-M3 MRR 0.8526, mE5-large MRR 0.8767) natively solve up to 80% of the Hinglish queries in this domain, leaving extremely limited headroom for correction.
- **The Over-Correction Effect (Degradation)**: Applying explicit operators (LAG, CAEP, LQP) causes a statistically significant degradation on natively correct retrievals. The degradation is robust against Holm-Bonferroni multiple testing correction across models (e.g., Indic-SBERT CAEP $p_{holm}=0.007$, BGE-M3 LAG $p_{holm}=0.022$).
- **Lack of Controller Context-Sensitivity**: The LinUCB controller operates as a fixed early-termination policy, rather than a context-sensitive sequencer. The controller's initial action choice is statistically independent of the query's complexity state (Chi-square test, $p_{holm}=1.0$).
- **Confidence Signal Inviability**: The raw embedding confidence signal exhibits near-zero variance across the dataset (313/314 queries <0.2), making it an unviable feature for dynamic gating in this environment.

## 2. CONDITIONALLY SAFE
- **Operator Efficacy on Weak Retrievers**: It is safe to frame SETU's operators as "detrimental to strong retrievals, and yielding non-significant lift on weak retrievals." However, you **must always cite the Holm-corrected p-values ($p_{holm} > 0.39$)**, and never cite the uncorrected raw p-values alone when discussing operator performance on failure cases.

## 3. DO NOT CLAIM
- **DO NOT CLAIM** that SETU improves retrieval performance overall compared to strong baselines.
- **DO NOT CLAIM** that the LinUCB controller learns an adaptive or dynamic routing policy based on context.
- **DO NOT CLAIM** that operators provide significant or robust corrective lift on queries where the baseline fails. The apparent lift on BGE-M3+LAG was proven to be a multiple-testing false positive.
- **DO NOT CLAIM** any generalization of these results beyond the highly narrow 380-chunk single-domain corpus evaluated here.
- **DO NOT CLAIM** comparison against any external black-box or API-based baselines, as Phase 15 was explicitly scoped out of the evaluation.
