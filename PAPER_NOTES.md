# Paper Notes & Narrative Reframing

## Core Narrative Shift
The original draft of the paper pitched SETU as a state-of-the-art, context-adaptive architecture that solves code-mixed Hinglish retrieval by dynamically sequencing specialized operators.
**The new narrative** must pitch the paper as a rigorous, methodology-focused empirical study demonstrating the hidden pitfalls of RAG pipelines on code-mixed data. 

## Key Thematic Pillars for the Paper

1. **The Over-Correction Phenomenon (The "Hurt" Factor)**
   - We introduce the concept of "over-correction" in retrieval pipelines: applying explicit transformations (like translation or lexical substitution) to queries often severely degrades the dense embeddings of natively strong models (BGE-M3, mE5-large).
   - Our ablation studies prove that operators hurt already-correct queries far more frequently than they rescue failed queries.

2. **The Illusion of Adaptivity in RL Controllers**
   - We dissect the failure of the LinUCB controller. We demonstrate how extreme feature skew (e.g., 99.7% of queries having <0.2 confidence) forces bandit algorithms to collapse into static early-termination policies.
   - We highlight the danger of claiming "dynamic routing" without analyzing sequence diversity and feature independence (Chi-square test p=1.0).

3. **Methodological Rigor in RAG Evaluation**
   - We expose how subtle data leakages (e.g., matching by query text instead of query ID, or globally fitting classifiers before cross-validation) can artificially inflate the apparent success of corrective operators.
   - We emphasize the necessity of strict 5-fold Out-Of-Fold (OOF) cross-validation and multiple-testing corrections (Holm-Bonferroni) to prevent false positives in RAG ablations.

## Recommended Structure Updates
- **Introduction**: Shift from "We built a better system" to "We systematically evaluated the paradigm of pipeline-based correction vs. zero-shot dense representations."
- **Methodology**: Present the SETU architecture, but immediately follow with the rigorous OOF evaluation protocol designed to stress-test it.
- **Results**: Lead with the baseline performance (Table I) to establish the high zero-shot ceiling. Follow with the operator ablation (Table II) and the critical over-correction conditional analysis (Table IV). 
- **Discussion/Conclusion**: Advise the community to lean on scaling raw multilingual models (like mE5-large) rather than building complex, brittle operator pipelines for code-mixed text.
