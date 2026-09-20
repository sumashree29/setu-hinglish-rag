# Paper Notes & Narrative Constraints

This file records the specific narrative boundaries for drafting the IEEE paper based on the final evaluation results. 

## 1. Do Not Claim Universality
The results strictly evaluate a 314-query Hinglish corpus against 380 document chunks. BGE-M3 performs extremely well as a zero-shot baseline (MRR ~0.85). Any claim that code-mixed retrieval *universally requires* dynamic operator pipelines is unsupported by this data and must be avoided.

## 2. Controller Adaptivity (H6)
**Original flawed claim:** "The contextual bandit provides highly adaptive, dynamic routing that improves over fixed sequences."
**Corrected claim:** "Under the evaluated dataset, the LinUCB controller largely converged to a low-step early-termination policy rather than a context-sensitive sequencer. No statistically significant difference in retrieval outcomes was detected between the fixed pipeline and the dynamic controller (p=0.7423), though the controller achieved comparable outcomes in significantly fewer steps."

## 3. The Overcorrection Problem (C6)
**Narrative focus:** The paper must highlight overcorrection as a primary finding. Applying heuristic or model-based operators to queries that are already successfully retrieved by a robust dense baseline (like BGE-M3) significantly degrades performance. Future code-mixed pipelines must prioritize highly conservative gating (e.g., confidence thresholds) to protect strong zero-shot baseline performance.

## 4. Confidence Gating (H10)
**Narrative focus:** The significant correlation between margin-based confidence and empirical retrieval success (rho=0.2985) is the most promising avenue for avoiding overcorrection. The paper should propose this as a diagnostic tool, but explicitly refrain from claiming that the current SETU controller fully exploits this signal.

## 5. Statistical Rigor
Every p-value cited in the paper must be the Holm-Bonferroni corrected value. Any discussion of "improvement on baseline failure cases" must explicitly state that the results were statistically insignificant after correcting for multiple comparisons.

## 6. Language and Tone
Use conservative IEEE-style scientific language. 
- Avoid "proves", use "demonstrates" or "suggests".
- Avoid "equivalent", use "no statistically significant difference was detected".
- Avoid "failed", use "did not demonstrate statistically significant improvement".
