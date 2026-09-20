# Claim Contract

This document strictly defines what claims the authors are permitted to make in the IEEE paper, based *only* on the final statistically corrected canonical evidence.

## Safe Claims (Supported by Evidence)
- **Baseline Strength:** Dense embedding models (e.g., BGE-M3, mE5-large) are robust zero-shot baselines for Hinglish code-mixed retrieval, solving a large majority (~77-80%) of the evaluated queries natively without modification.
- **Latency Overheads:** The dynamic controller pipeline incurs a significant latency overhead compared to the raw baseline.
- **Controller Efficiency (H8):** The contextual bandit controller (SETU v2) significantly reduces the number of inference steps (mean 1.70) compared to a fixed-order pipeline (mean 4.0), while producing comparable retrieval outcomes.
- **Confidence Correlation (H10):** The margin-based confidence proxy positively correlates with actual retrieval success (MRR) in the evaluated dataset.
- **Overcorrection Risk:** Operator-based query modification pipelines risk overcorrection, significantly degrading queries that the underlying dense retriever already resolves correctly.

## Conditional Claims (Require Precise Scientific Wording)
- **Controller Adaptivity:** Instead of claiming "the controller's sequence choices are fully detached from context," you must state: "No statistically significant association was detected between the initial action and the CMI band in the evaluated dataset. The controller largely functioned as an early-termination policy rather than a context-sensitive sequencer."
- **Performance Parity (H6):** Instead of claiming "SETU v1 and SETU v2 perform exactly the same," you must state: "No statistically significant difference was detected between SETU v1 and SETU v2 under the evaluated test." (Proving true equivalence requires a predefined non-inferiority margin).

## Prohibited Claims (Do NOT Write These)
- **Universal Superiority:** Do not claim SETU outperforms raw dense retrieval on this dataset. It does not.
- **Causality from Correlation:** Do not claim that confidence gating *causes* better retrieval.
- **Improvement on Failure Cases:** Do not claim that operators successfully correct baseline failures. After Holm-Bonferroni correction, no operator showed statistically significant improvement on natively failed queries.
- **Generalization:** Do not claim universal generalization of these results to other datasets, domains, or completely distinct multilingual setups beyond this 314-query Hinglish evaluation.
