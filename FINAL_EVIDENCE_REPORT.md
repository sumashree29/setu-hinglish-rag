# Final Evidence Report

This report summarizes the completed end-to-end empirical evaluation and statistical verification of the SETU Hinglish RAG repository.

## 1. Evaluation Scope
- **Primary In-Domain Benchmark:** 314 Hinglish queries across 380 RBI/PM-KISAN chunks spanning **18 categories**. Includes 15 misspelled-entity queries (`Q61-Q75`).
- **External Validation Benchmark:** 500 unseen duplicate-free Romanized Hinglish QA pairs from `aaditya/databricks-dolly-15k-Hinglish-Codemix` (zero overlap with RBI or PHINC).
- **Models Evaluated:** `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, `l3cube-pune/indic-sentence-similarity-sbert`, `nthakur/mcontriever-base-msmarco`, and `BM25`.

## 2. Full Pipeline Rerun & Leakage-Free Grouping
All operator checkpoints, trajectories, OOF evaluations, latency benchmarks, and statistical suites were cleanly executed end-to-end:
- **Grouped 5-Fold `GroupKFold` CV:** Connected components over `(normalized_query_text, gold_chunk_ids)` ensure the 4 duplicate-text query pairs (`Q088/Q089`, `Q148/Q185`, `Q151/Q186`, `Q222/Q223`) and shared gold-chunk items never straddle train and held-out folds.
- **Independent External Operator Training:** LQP Ridge projections are trained strictly on 500 parallel pairs from `LingoIITGN/PHINC`; CAEP gates are trained strictly on corpus entities + synthetic perturbations.

## 3. Authoritative Evidence Files
- `results/tables/ieee_ready_tables.json` (Tables 1–9 regenerated cleanly via `scripts/reproduce_canonical_analysis.py`)
- `results/tables/selective_correction_strategies.json`
- `results/tables/external_dolly_validation.json`
- `results/tables/overcorrection_final.json`
- `results/tables/controller_behavior_final.json`
- `results/tables/statistical_significance_H1_H10_scaled.json`
- `results/tables/latency_final.json` and `results/canonical/{latency.json, per_query_metrics.json, setu_trajectories.json}`

## 4. Major Supported Findings
1. **Zero-Shot Resilience vs. Indic-Tuning:** `mE5-large` (0.8767 MRR) and `BGE-M3` (0.8526 MRR) resolve 77.4–80.9% of RBI queries and 83.2–88.6% of Dolly-Hinglish queries at rank 1 without rewriting, significantly outperforming `Indic-SBERT` (`p_corr = 3.50e-17`).
2. **Asymmetric Overcorrection Hazard:** Standalone operators significantly degrade already-correct queries across both domains (e.g., `BGE-M3+LAG -0.0310`, `p_corr=0.0220`; `mE5+CAEP -0.0378`, `p_corr=0.0063`; Dolly `CAEP/LAG` `p_corr <= 0.0022`). No standalone operator delivers Holm-significant lift on failed RBI queries (`p_corr >= 0.3300`).
3. **Selective Gating Recovery & Step Efficiency (`SETU_v2`):** Under leak-free GroupKFold CV, `SETU_v2` avoids significant overcorrection (`-0.0075`, `p_corr=0.9225`), recovers initially failed queries (`+0.0364`, `p_corr=0.0494`), reduces actions by 54.25% (`3.000 -> 1.373`, `p_corr=1.79e-55`), and reduces latency by 59.1% (`1186.8 ms -> 485.1 ms`).
4. **Confidence Validity & Oracle Bound:** Top-1/top-2 margin correlates strongly with MRR (`Spearman rho = 0.5030`, `p_corr = 1.07e-20`). Oracle `pi*` hits `0.8842` (`+0.0316`, `p_corr=5e-6`), leaving 89.5% of queries unmodified (`RAW`).
5. **External Zero-Shot Transfer (`Dolly-15k`):** On unseen open-domain Hinglish QA, `mE5-large + LQP` recovers failed queries (`+0.0934, p_corr=0.0008`), lifting overall MRR (`0.8695 -> 0.8812`), while `CAEP`/`LAG` degrade working queries across every model.

## 5. Limitations
- Human expert annotation batches (`data/review/review_batch_{1,2,3}.json`) remain unreviewed on disk.
- Controller sequence diversity remains constrained (`LQP->LAG->STOP`: 188, `STOP`: 71, `LQP->STOP`: 55; 75.8% forced repeat stops due to missing action masking).
- Heuristic regex CMI shows low construct agreement with IndicLID (`kappa = 0.0868`), and downstream LLM answer generation (H3) was out of scope.
- Latency measurements reflect local CPU execution (`i7-13620H`, `torch==2.10.0+cpu`, 3 uncached repeats).

## 6. Reproducibility & Readiness
- Unit & leakage tests: `pytest tests/ -q` (**33/33 passed**)
- Canonical tables & gate: `scripts/reproduce_canonical_analysis.py` & `scripts/final_readiness_gate.py` (**PASS, `IEEE_READY = TRUE`**)
