# Final Evidence Scope

This document specifies the exact dataset boundaries, query subsets, and limits of the empirical evaluation presented in the IEEE manuscript.

## 1. Evaluation Boundaries
- **Primary Corpus (Chunks):** 380 document chunks across **18 source categories** (17 RBI/PM-KISAN topics + curated pilot subset).
- **Primary Query Set:** 314 Hinglish queries (75 pilot queries + 239 paraphrased variants; partitioned via 5-fold `GroupKFold` over connected query text and gold chunk components).
- **Misspelled Subset:** Pre-specified 15-query subset (`Q61-Q75`) testing entity-correction robustness.
- **External Generalization Set:** 500 duplicate-free passages and 500 Hinglish queries from `aaditya/databricks-dolly-15k-Hinglish-Codemix` (completely disjoint from RBI/PM-KISAN and `PHINC`).

## 2. Models Evaluated
- **Primary Dense Baselines:** `BAAI/bge-m3`, `intfloat/multilingual-e5-large`
- **Auxiliary Baselines:** `l3cube-pune/indic-sentence-similarity-sbert`, `nthakur/mcontriever-base-msmarco`, `BM25`
- **Operators:** `LQP` (Ridge projection trained on 500 `PHINC` pairs), `CAEP` (entity preservation gate), `LAG` (5-fold OOF LightGBM strategy predictor).
- **Routers / Controllers:** `SETU_v1` (fixed pipeline), `SETU_v2` (GroupKFold LinUCB contextual bandit), `pi_threshold` (score-margin gating), `pi*` (empirical oracle).

## 3. Evidence Status
- **Executed End-to-End:** All operator checkpoints, trajectories, grouped 5-fold OOF comparisons, selective-strategy benchmarks, latency runs, statistical tests, and external Dolly-Hinglish runs were freshly executed and persisted.
- **Unannotated Human Review Batches:** `data/review/review_batch_{1,2,3}.json` were left unreviewed rather than synthesized; ground-truth relevance derives from curated source-chunk alignments.

## 4. Claims Out of Scope
- **Downstream Answer Generation (H3):** Evaluation stops at first-stage passage retrieval.
- **Frontier LLM Rewriters:** Only lightweight operators (`LQP`, `CAEP`, `LAG`) compatible with local CPU latency constraints were evaluated.
- **Equivalence Claims:** No statistically significant overall difference was detected between `SETU_v1` and `SETU_v2` under paired Wilcoxon (`p_corr = 0.8334`); no TOST equivalence test is claimed.
- **Causality:** Confidence margin (`rho = 0.5030`) is validated as a predictive correlation proxy, not a causal intervention.
