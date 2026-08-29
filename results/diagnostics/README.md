# SETU Diagnostics Directory

This directory contains diagnostic artifacts, investigations, and audit traces generated during development and verification passes.

## Schema Templates
- `TEMPLATE_model_collapse_investigation.json`: Schema for diagnosing model weight collapses, parallel pair matrix norms, and hyperparameter checks.
- `TEMPLATE_operator_trace.json`: Schema for logging detailed per-query intermediate operator steps (LQP projections, CAEP gate predictions, LAG substrategies) and diagnostic flags.
- `TEMPLATE_stratified_breakdown.json`: Schema for stratified operator performance (e.g. per-substrategy evaluation).
- `TEMPLATE_ranking_diff_counts.json`: Schema for comparing ranking divergence across pipeline versions.

## Diagnostic Artifacts (2026-08-29 Session)
- `lqp_collapse_investigation_20260829.json`: Root-cause diagnosis of LQP matrix collapse (PHINC column name mismatch triggering dummy constant vectors), before/after matrix norms ($6.75 \times 10^{-9} \rightarrow 6.371$), and 5-query spot checks.
- `q73_caep_lqp_trace_20260829.json`: Per-query traces for queries Q73, Q61, Q21, Q18, Q58, Q63, Q71, including gate features, probabilities, and ranking changes. Includes flag documenting token matching behavior for `pm-kisaan`.
- `lag_substrategy_breakdown_20260829.json`: 3-way evaluation across `light_normalize` (30 queries), `dual_variant` (30 queries), and `full_rewrite` (15 queries).
- `v1_v2_ranking_diffs_20260829.json`: Detailed divergence counts (14/75 pre-fix, 32/75 post-fix) with complete lists of differing query IDs.
