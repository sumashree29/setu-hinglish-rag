# Final Evidence Report

This document confirms the final IEEE-readiness of the SETU repository. The primary objective is establishing rigorous, reproducible, and defensible scientific evidence for the evaluation.

## 1. Evaluation Scope
- **Dataset:** 314 benchmark code-mixed (Hinglish) queries.
- **Search Corpus:** 380 document chunks.
- **Models Evaluated:** BGE-M3 (Primary Baseline), Indic-SBERT, mE5-large.
- **Subsets:** Includes 15 explicitly misspelled-entity queries for robustness checks.

## 2. Authoritative Evidence Source
The canonical machine-readable evidence layer has been established from existing experimental results without unnecessary model re-execution.
- **Source Files:** Aggregate tables (`setu_v1_v2_comparison_scaled.json`, `overcorrection_final.json`, `statistical_significance_H1_H10_scaled.json`) serve as the verified canonical data sources reflecting the raw per-query underlying evaluations.
- **Downstream Generation:** All IEEE-ready tables and figures are regenerated from this canonical evidence layer using `scripts/reproduce_canonical_analysis.py`.

## 3. Reused vs Rerun Experiments
- **Reused:** Baseline retrievals (BGE-M3, Indic-SBERT, mE5-large), SETU v1 (fixed-order) pipeline executions, SETU v2 (LinUCB) trajectories, overcorrection evaluations, and latency tests were entirely reused from existing valid json outputs.
- **Rerun:** No expensive model reruns were necessary. Downstream statistical reporting files were regenerated purely from the pre-existing authoritative JSON outputs.

## 4. Leakage / OOF Protocol Checks
The evaluation employs a strict 5-fold Out-Of-Fold (OOF) protocol:
- Trajectories are partitioned strictly by `query_id`.
- The `LinUCBController` and `lag_model` are trained dynamically per fold on 4 partitions and evaluated blindly on the 1 held-out partition.
- External operators (CAEP, LQP) were trained on fully independent parallel/synthetic corpora. There is no query leakage into the evaluation phase.

## 5. Statistical Methodology
- **Tests Used:** Paired Wilcoxon signed-rank tests for comparative accuracy (MRR) and Spearman correlations for monotonic associations (CMI, Confidence).
- **Multiple Testing Correction:** Holm-Bonferroni correction was applied to control the Family-Wise Error Rate (FWER) across the hypothesis tests and overcorrection splits.
- **Equivalence:** No statistically significant difference was detected between SETU v1 and SETU v2 under the evaluated paired test. This is not an equivalence claim.

## 6. Major Supported Findings
- SETU v2 effectively operates as an early-termination policy, achieving comparable retrieval results to a fixed-order pipeline (SETU v1) but with significantly fewer computational steps (1.697 operator actions vs 3.0 operator actions).
- The previous margin-confidence correlation result is excluded because the underlying confidence-margin series is not present in the canonical evidence.
- Existing dense retrievers are highly capable zero-shot baselines, resolving the majority of Hinglish queries natively.

## 7. Unsupported Hypotheses (Negative Results)
- The SETU operator pipeline does not provide a statistically significant overall lift over the raw dense retriever on this dataset.
- The operators significantly risk overcorrection, degrading previously correct baseline queries.
- Corrective lift on baseline failure cases was statistically insignificant after multiple-testing correction.

## 8. Limitations
- **Corpus Scale:** The evaluation corpus is constrained to 380 chunks, raising the potential for metrics being dominated by lexical overlaps.
- **Latency Benchmarks:** The latency figures reflect local CPU-bottlenecked measurements, not large-scale deployment metrics.
- **Controller Adaptivity:** The controller demonstrated limited sequence diversity and did not dynamically sequence operations based on CMI context.

## 9. Reproducibility
Reviewers and users can exactly reproduce the final paper's numeric claims without retraining any models by executing:
```bash
python scripts/reproduce_canonical_analysis.py
python scripts/final_readiness_gate.py
```

## 10. Final IEEE-Readiness Status
- **Consistency Check:** PASS
- **Readiness Gate:** PASS
- **IEEE_READY:** TRUE
