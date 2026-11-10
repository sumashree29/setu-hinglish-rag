# IEEE Submission Readiness Report

## Status: READY

### Summary of Hardening & Execution Audit
The SETU-Hinglish-RAG codebase, experiments, tables, figures, manuscript (`ieee_submission/main.tex`), bibliography (`ieee_submission/references.bib`), and documentation have undergone a full end-to-end verification and clean re-execution across all 5 publication-hardening phases:

1. **Leak-Free Grouped Cross-Validation (`GroupKFold`):** Fixed duplicate-text leakage (`4` duplicate pairs plus shared gold-chunk components) across `compare_setu_v1_v2_scaled.py`, `train_and_ablate_operators_v2.py`, `diagnose_overcorrection_v3.py`, `run_statistical_tests_h1_h10_scaled.py`, `error_analysis_v2.py`, `measure_latency.py`, `check_lag_strategies.py`, and `evaluate_selective_strategies.py` (enforced by `tests/test_leakage.py`).
2. **Bug Fixes Across Metrics & Controllers:**
   - Fixed `build_dataset_stats.py` category counting (`18` categories instead of `1`).
   - Fixed `final_pass_script.py` confidence trace lookup (`confidence_trace[0]`), resolving `NaN` in Phase 14 (`Spearman rho = 0.5030`, `p_corr = 1.07e-20`).
   - Fixed `analyze_controller_behavior.py` to separate inference trajectories (`alpha=0.0`) and guard zero-marginal Chi-square columns (`24.20%` explicit stops, `75.80%` repeat-action stops, raw Chi-square `p = 0.0043` / `p_corr = 0.0731`).
   - Updated `tests/test_setu_bandit.py` (`query_id` & temp log isolation) -> **33/33 `pytest` tests passing**.
3. **Selective Correction & External Generalization:**
   - Added `scripts/evaluate_selective_strategies.py` (`selective_correction_strategies.json`): Oracle `pi*` hits `0.8842` MRR (`+0.0316`, `p_corr = 5e-6`), preferring raw `RAW` on `89.49%` (`281/314`) of queries.
   - Added `scripts/evaluate_external_dolly.py` (`external_dolly_validation.json`): Evaluated `500` unseen `Dolly-15k-Hinglish` QA pairs across `BGE-M3`, `mE5-large`, and `Indic-SBERT`, replicating significant CAEP/LAG overcorrection out-of-domain (`p_corr <= 0.0022`) and demonstrating zero-shot `mE5-large + LQP` recovery on failed queries (`+0.0934`, `p_corr = 0.0008`).
4. **Bibliography & Manuscript Sync:** Verified `references.bib` entries against primary sources (removed `"and others"` placeholders, corrected 6 citation metadata entries), regenerated `ieee_ready_tables.json` (Tables I–IX), `IEEE_TABLES.md`, `ieee_tables.tex`, and PNG figures, and synchronized all numbers across `ieee_submission/main.tex` including an explicit Limitations subsection.
5. **Automated Consistency & Readiness Gate:** `scripts/check_paper_consistency.py`, `scripts/check_consistency.py`, and `scripts/final_readiness_gate.py` -> **PASS (`IEEE_READY = TRUE`)**.

### Remaining Scope Boundaries (Documented Explicitly)
- `Phase 13 (CMI Validation)`: Heuristic regex CMI vs. IndicLID (`kappa = 0.0868`) documented as a construct limitation.
- Unannotated human review files (`data/review/review_batch_{1,2,3}.json`) retained unfilled without fabrication and disclosed in Limitations.
