# SETU: Hinglish RAG

SETU (Self-Evolving Tool-Use) is a framework for studying selective query correction in code-mixed Romanized Hinglish Retrieval-Augmented Generation (`LQP`, `CAEP`, `LAG`, contextual bandit `SETU_v2`, and confidence gating).

This repository contains the IEEE manuscript sources (`ieee_submission/`), leakage-free evaluation pipeline, unit/leakage tests, and reproducible canonical artifacts.

## Final Evidence and Evaluation Scope
- **Primary Benchmark:** **314** Hinglish queries across **380** RBI/PM-KISAN chunks spanning **18** source categories (`GroupKFold` 5-fold OOF CV grouped by normalized query text and shared gold chunks).
- **External Validation Benchmark:** **500** duplicate-free Romanized Hinglish QA pairs from `aaditya/databricks-dolly-15k-Hinglish-Codemix`.
- **Key Findings:**
  - Multilingual dense retrievers resolve **77.4–80.9%** of RBI queries (`BGE-M3` MRR = **0.8526**, `mE5-large` MRR = **0.8767**) and **83.2–88.6%** of Dolly-Hinglish queries (`BGE-M3` MRR = **0.9121**) natively at rank 1.
  - Standalone operators inflict statistically significant **overcorrection** on initially correct queries across both datasets (Holm $p_{\text{corr}} \le 0.0220$).
  - Under leak-free GroupKFold CV, `SETU_v2` (**0.8550** MRR) avoids significant overcorrection (`-0.0075`, $p_{\text{corr}} = 0.9225$), achieves significant failed-query lift (`+0.0364`, $p_{\text{corr}} = 0.0494$), and cuts operator actions by **54.25%** (`3.00 -> 1.37`) and CPU latency by **59.1%** (`1186.8 ms -> 485.1 ms`).
  - Score-margin confidence strongly tracks MRR ($\rho = 0.5030, p_{\text{corr}} = 1.07 \times 10^{-20}$), and the empirical oracle (`0.8842` MRR) favors leaving **89.5%** of queries unmodified (`RAW`).

## Reproducing the Canonical Analysis & Readiness Gate
To regenerate `results/tables/ieee_ready_tables.json` (Tables I–IX), formatted IEEE Markdown/LaTeX tables, figures, and run the final consistency/test suite:
```bash
pytest tests/ -q
python scripts/reproduce_canonical_analysis.py
python scripts/generate_ieee_tables.py
python scripts/generate_figures.py
python scripts/final_readiness_gate.py
```

## Documentation Map
- `RESULTS_SUMMARY.md`: Complete canonical results summary across primary and external arms.
- `CLAIM_CONTRACT.md` & `CLAIM_EVIDENCE_MATRIX.md`: Verified claims, verdicts, and statistical bounds.
- `FINAL_EVIDENCE_REPORT.md` & `FINAL_EVIDENCE_SCOPE.md`: Evaluation protocol, limitations, and scope.
- `REPRODUCE_FINAL.md` & `ENVIRONMENT.md`: Full end-to-end execution guide and environment pin details.
- `MASTER_CHECKLIST.md`: Publication verification checklist.
