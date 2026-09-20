# SETU: Hinglish RAG

SETU (Self-Evolving Tool-Use) is a framework intended to improve retrieval-augmented generation for code-mixed languages like Hinglish. It employs an operator pipeline (LAG, CAEP, LQP) sequenced by a contextual bandit controller (LinUCB) to dynamically correct user queries.

This repository contains the IEEE publication-ready artifacts, code, and verified evaluation results.

## Final Evidence and Evaluation Scope
The evaluation consists of **314 benchmark queries** executed against a search corpus of **380 chunks**. 

In the evaluated corpus, **SETU did not demonstrate statistically significant retrieval improvement over the raw BGE-M3 baseline.** The dense retrieval baseline correctly answered 77% of queries out-of-the-box (MRR = 0.8526). Overcorrection analysis showed that SETU operators significantly degraded already-correct baseline retrievals, while no operator achieved statistically significant improvement on natively failed queries after Holm-Bonferroni correction.

The controller analysis demonstrated limited sequence diversity, largely functioning as an early-termination policy.

All reported numbers are strictly drawn from canonical per-query evidence.

## Reproducing the Analysis
Expensive model executions are NOT required to reproduce the paper's tables and statistical results. We have verified and persisted the underlying per-query evidence.

To automatically regenerate all tables, figures, and multiple-testing corrections from the canonical data:
```bash
python scripts/reproduce_canonical_analysis.py
```
This produces `results/tables/ieee_ready_tables.json`.

To run the final readiness gate and consistency checker:
```bash
python scripts/final_readiness_gate.py
```

## Documentation Map
* `RESULTS_SUMMARY.md`: High-level numeric results.
* `CLAIM_CONTRACT.md`: Explicit definitions of safe, conditional, and prohibited claims for the IEEE paper.
* `CLAIM_EVIDENCE_MATRIX.md`: Mapping of hypotheses to exact evidence and verdicts.
* `FINAL_EVIDENCE_REPORT.md`: Comprehensive breakdown of evaluation scope and methodology.
* `FINAL_EVIDENCE_SCOPE.md`: Dataset details and pipeline limits.
* `MASTER_CHECKLIST.md`: Task tracking for publication readiness.
