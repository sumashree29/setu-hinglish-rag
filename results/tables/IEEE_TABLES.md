# IEEE Paper Tables

## Table I: Baseline Retriever Performance

| Model | MRR | NDCG@10 | Hit@10 |
|---|---|---|---|
| bge_m3 | 0.0000 | 0.0000 | 0.0000 |
| indic_sbert | 0.0000 | 0.0000 | 0.0000 |
| me5_large | 0.0000 | 0.0000 | 0.0000 |
| mcontriever | 0.0000 | 0.0000 | 0.0000 |
| bm25 | 0.0000 | 0.0000 | 0.0000 |


## Table II: Operator Ablation (MRR)

| Embedding Model | RAW | +LQP | +CAEP | +LAG |
|---|---|---|---|---|
| bge_m3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| indic_sbert | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| me5_large | 0.0000 | 0.0000 | 0.0000 | 0.0000 |


## Table III: Full SETU Pipeline vs Baselines

| System | MRR | NDCG@10 | Hit@10 |
|---|---|---|---|
| full_dataset | 0.0000 | 0.0000 | 0.0000 |
| misspelled_subset | 0.0000 | 0.0000 | 0.0000 |
*(Could not load SETU comparison: 'int' object has no attribute 'get')*

## Table IV: Over-Correction Diagnosis (Holm-corrected p-values)

| Model | Operator | ΔMRR (Correct) | p (Correct) | ΔMRR (Incorrect) | p (Incorrect) |
|---|---|---|---|---|---|
| bge_m3 | LQP | -0.0195 | 0.0740 | 0.0143 | 1.0000 |
| bge_m3 | CAEP | -0.0238 | 0.1252 | 0.0195 | 1.0000 |
| bge_m3 | LAG | -0.0310 | 0.0222 | 0.0446 | 0.3981 |
| bge_m3 | SETU_v1 | -0.0127 | 0.5117 | 0.0230 | 1.0000 |
| bge_m3 | SETU_v2 | -0.0089 | 0.7054 | 0.0162 | 0.3981 |
| indic_sbert | LQP | -0.0101 | 0.9159 | -0.0039 | 1.0000 |
| indic_sbert | CAEP | -0.0692 | 0.0073 | 0.0182 | 1.0000 |
| indic_sbert | LAG | -0.0740 | 0.0056 | 0.0335 | 0.9159 |
| me5_large | LQP | -0.0118 | 0.2432 | -0.0038 | 1.0000 |
| me5_large | CAEP | -0.0261 | 0.0222 | -0.0029 | 1.0000 |
| me5_large | LAG | -0.0098 | 0.3981 | 0.0094 | 1.0000 |

