# Final Evidence Scope

This document specifies the exact dataset boundaries, query subsets, and limits of the empirical evaluation presented in the final IEEE publication.

## 1. Evaluation Boundaries
- **Corpus (Chunks):** The search index contains exactly 380 document chunks spanning Hinglish conversational domains.
- **Query Set:** The final benchmark evaluates exactly 314 queries.
- **Misspelled Subset:** A pre-specified subset of 15 queries (IDs: Q61-Q75) evaluates entity-correction resilience.

## 2. Models Evaluated
- **Baseline (Primary):** `BAAI/bge-m3`
- **Baselines (Auxiliary):** `intfloat/multilingual-e5-large`, `l3cube-pune/indic-sentence-bert-nli`
- **Operators:** LAG (translation/normalization), CAEP (entity preservation), LQP (latent query projection).
- **Controller:** SETU v2 (LinUCB-based contextual bandit), SETU v1 (fixed 3-operator sequence).

## 3. Evidence Status
- **Reused:** All model inference, operator evaluations, embedding generations, and LinUCB trajectory formations have been reused from the existing stored outputs.
- **Regenerated:** Final aggregate tables, P-values, Holm-Bonferroni corrections, and figures have been strictly regenerated from the verified underlying JSON files.

## 4. Claims In Scope
- Comparative zero-shot MRR and nDCG@10 of BGE-M3 vs SETU pipelines on this specific 314-query corpus.
- The step-count efficiency of the LinUCB controller vs a fixed-order execution.
- The previous margin-confidence correlation result is excluded because the underlying confidence-margin series is not present in the canonical evidence.
- The statistical overcorrection effect on natively successful baseline queries.

## 5. Claims Out of Scope
- **Universal Generalization:** Results do not prove SETU works or fails on massive-scale corpora (e.g., millions of documents).
- **Causality of Performance:** CMI correlation and Confidence correlations are evaluated strictly as predictive proxies, not causal mechanisms.
- **Equivalence:** No statistically significant difference was detected between SETU v1 and SETU v2 under the evaluated paired test. This is not an equivalence claim.
