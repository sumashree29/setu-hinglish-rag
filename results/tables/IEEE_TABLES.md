# IEEE Paper Tables (Canonically Generated)

## Table I: Baseline Retriever Performance (Full 314-Query / 380-Chunk Corpus)

| Model | Architecture | MRR | NDCG@10 | Hit@1 | Recall@10 |
|---|---|---|---|---|---|
| **bge_m3** | Dense Multi-Lingual (XLM-RoBERTa) | 0.8526 | 0.8858 | 0.7739 | 0.9873 |
| **indic_sbert** | Dense Indic-Tuned (MuRIL) | 0.5903 | 0.6476 | 0.4713 | 0.8280 |
| **me5_large** | Dense Multi-Lingual (mE5) | 0.8767 | 0.9039 | 0.8089 | 0.9873 |
| **mcontriever** | Dense Multi-Lingual (Contriever) | 0.7285 | 0.7749 | 0.6242 | 0.9204 |
| **bm25** | Sparse Lexical (BM25) | 0.6490 | 0.6987 | 0.5382 | 0.8567 |


## Table II: Operator Ablation (MRR on Full 314 Queries)

| Embedding Model | RAW Baseline | +LQP Alone | +CAEP Alone | +LAG Alone |
|---|---|---|---|---|
| **bge_m3** | 0.8526 | 0.8407 | 0.8386 | 0.8387 |
| **indic_sbert** | 0.5903 | 0.5868 | 0.5654 | 0.5695 |
| **me5_large** | 0.8767 | 0.8664 | 0.8441 | 0.8705 |


## Table III: Full SETU Pipeline vs Baselines (BGE-M3 Backbone)

| Configuration | Description | MRR | NDCG@10 | Recall@10 | Mean Actions | Latency (ms) |
|---|---|---|---|---|---|---|
| **RAW** | Zero-shot dense baseline (no modification) | 0.8526 | 0.8858 | 0.9873 | - | 189.5 |
| **SETU_v1** | Fixed pipeline (LAG -> CAEP -> LQP -> CARF) | 0.8492 | 0.8810 | 0.9809 | 3.000 | 1186.8 |
| **SETU_v2** | Adaptive contextual bandit controller (LinUCB) | 0.8550 | 0.8875 | 0.9873 | 1.373 | 485.1 |


## Table IV: Over-Correction Diagnosis (Holm-Bonferroni Corrected p-values)

| Model | Operator / System | ΔMRR (Initially Correct) | p_corr | ΔMRR (Initially Failed) | p_corr |
|---|---|---|---|---|---|
| **bge_m3** | LQP | -0.0195  | 0.0702 | +0.0143  | 1.0000 |
| **bge_m3** | CAEP | -0.0231  | 0.1056 | +0.0172  | 1.0000 |
| **bge_m3** | LAG | -0.0310 ** | 0.0220 | +0.0446  | 0.3300 |
| **bge_m3** | SETU_v1 | -0.0106  | 0.6560 | +0.0213  | 1.0000 |
| **bge_m3** | SETU_v2 | -0.0075  | 0.9225 | +0.0364 ** | 0.0494 |
| **indic_sbert** | LQP | -0.0203  | 0.2145 | +0.0114  | 1.0000 |
| **indic_sbert** | CAEP | -0.0754 ** | 0.0046 | +0.0200  | 1.0000 |
| **indic_sbert** | LAG | -0.0757 ** | 0.0046 | +0.0282  | 1.0000 |
| **me5_large** | LQP | -0.0118  | 0.2145 | -0.0038  | 1.0000 |
| **me5_large** | CAEP | -0.0378 ** | 0.0063 | -0.0106  | 0.4609 |
| **me5_large** | LAG | -0.0098  | 0.3289 | +0.0094  | 1.0000 |

