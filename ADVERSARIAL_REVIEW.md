# Adversarial Review Personas

### 1. The Information Retrieval Expert
**Concern**: "You claim your operators are designed for code-mixed retrieval, but your baseline models (BGE-M3, mE5-large) are already solving 80% of these queries zero-shot. Isn't this whole operator pipeline just an over-engineered solution to a problem that modern dense retrievers have already solved on their own?"
**Answer**: Yes, that is exactly the core finding of our paper. We rigorously demonstrate that scaling zero-shot dense representations is far more robust than explicit operator pipelines, and that deploying such pipelines on top of strong models leads to statistically significant degradation via over-correction.

### 2. The Machine Learning / RL Expert
**Concern**: "You trained a LinUCB bandit to route queries adaptively, but with 99.7% of queries having <0.2 initial confidence and only 380 total chunks in the environment, the context space is too narrow for any bandit to learn a meaningful dynamic policy. I bet it just collapsed into a static early-termination strategy."
**Answer**: Your intuition is entirely correct, and our Phase 4 evaluation proves it. We ran a Chi-square test on the controller's initial action choice against context complexity, finding perfect statistical independence ($p_{holm}=1.0$), confirming the controller functions as a static early-stopping policy rather than an adaptive sequencer.

### 3. The Reproducibility & Methodology Reviewer
**Concern**: "In RAG pipelines with multiple operators and sequential logic, it is incredibly easy for test data to leak into the operator training phase or for multiple ablations to produce false positives. How can I trust that your findings aren't artifacts of data leakage or p-hacking?"
**Answer**: We conduct a strict 5-fold Out-Of-Fold (OOF) cross-validation where all trajectories, controller weights, and local operator models are trained exclusively on out-of-fold data. Furthermore, we apply Holm-Bonferroni correction across all 24 hypothesis tests, which successfully caught and neutralized a false positive lift signal in our own pipeline.
