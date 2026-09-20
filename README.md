# SETU-Hinglish-RAG

## Research Question
Can explicit, code-mixed specific operators (such as translation, entity preservation, and query projection) orchestrated by an adaptive reinforcement learning controller outperform strong zero-shot dense multilingual retrieval models (like BGE-M3 and mE5-large) on Hinglish text?

## Method
SETU introduces an architecture consisting of three discrete operators (LAG, CAEP, LQP) aimed at mitigating the lexical variance and domain-specific challenges of Hinglish. A LinUCB contextual bandit controller was designed to adaptively sequence these operators based on the query's complexity (measured by CMI and entropy) and the initial retrieval confidence.

We conducted a rigorous, strictly isolated 5-fold Out-Of-Fold (OOF) cross-validation of the full controller pipeline, evaluating against 314 canonical Hinglish queries mapped to 380 domain chunks.

## The Negative Result
Our empirical audit conclusively **rejects** the hypothesis that the SETU architecture outperforms raw zero-shot dense representations. 
- **Baseline Dominance**: Modern multilingual models (BGE-M3, mE5-large) solve ~80% of the Hinglish queries natively, leaving minimal headroom for explicit correction.
- **Over-Correction Degradation**: We systematically demonstrate that explicitly transforming queries (via translation or projection) statically *degrades* the performance of already-correct retrievals. Statistical analysis across 24 hypothesis tests confirms that the operators offer no robust corrective lift on failure cases.
- **Controller Policy Collapse**: Due to the narrow distribution of contextual features (99.7% of queries possess initial embedding confidences <0.2), the LinUCB controller collapses into a static early-termination policy entirely independent of the query context.

We conclude that scaling strong dense multilingual representations is fundamentally more robust than developing brittle, context-adaptive operator pipelines for code-mixed retrieval.

## Explicit Limitations
1. **Domain Scale**: The evaluation corpus consists of only 380 chunks. This biases results heavily toward trivial lexical matches and restricts the generalizability of the findings to large-scale open-domain retrieval.
2. **Controller Scope**: The full SETU LinUCB controller was evaluated exclusively using the BGE-M3 backbone. Ablations for Indic-SBERT and mE5-large were conducted in isolation.
3. **Out-of-Scope Phases**: Deeper linguistic validation of the CMI heuristics against external tools (IndicLID) and comparisons against complex black-box/external API baselines were explicitly designated as out-of-scope for this revision.

## Reproducibility
For the exact command sequence required to fully reproduce the canonical metrics, statistical tests, and tables, please see [REPRODUCE_FINAL.md](REPRODUCE_FINAL.md).
