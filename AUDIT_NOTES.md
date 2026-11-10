# Audit Notes - Updated Post Leak-Free Verification

Historical Phase 0 audit notes updated with final grouped cross-validation fixes and post-remediation measurements.

## 1. Operators Summary
- **LQP (`setu/operators/lqp.py`):** Ridge regression matrix `W` trained on `500` PHINC Hinglish–English parallel pairs; applied with CMI blend `W_eff(q) = (1 - alpha) * I + alpha * W`.
- **CAEP (`setu/operators/caep.py`):** Logistic regression gate over `[fuzzy_score, embedding_cosine, entity_frequency]` substituting entity tokens against corpus entities.
- **LAG (`setu/operators/lag.py`):** Per-fold OOF LightGBM strategy predictor (`[cmi, lid_entropy, entity_density]`) over `light_normalize`, `dual_variant`, `full_rewrite`.
- **CARF (`setu/fusion/carf.py`):** CMI-weighted rank fusion combining raw and corrected rankings.

## 2. SETU v1 vs. SETU v2 Execution
- **SETU v1:** Fixed pipeline (`LAG -> CAEP -> LQP -> CARF`), `3.00` non-`STOP` operator steps.
- **SETU v2:** `LinUCBController(context_dim=7)` over `[cmi, lid_entropy, confidence, step, tried_LAG, tried_CAEP, tried_LQP]` and actions `["LAG", "CAEP", "LQP", "STOP"]`, `train=False` (`alpha=0.0`) at evaluation, mean `1.373` non-`STOP` actions (`54.25%` reduction, `p_corr = 1.79e-55`).

## 3. Leakage Remediation & Duplicate Query Grouping
- **Original Bug:** Prior split matched only exact `query` text and failed to group `4` duplicate-text ID pairs (`Q088/Q089`, `Q148/Q185`, `Q151/Q186`, `Q222/Q223`) or shared gold chunks across folds.
- **Fix Verified:** `build_grouped_folds` (`scripts/compare_setu_v1_v2_scaled.py`) applies `GroupKFold` over union-find connected components of `(normalized_text, relevant_doc_ids)`, verified across all scripts and `tests/test_leakage.py`.

## 4. Controller Sequence & Confidence Bugs Fixed
- **Controller Behavior (`scripts/analyze_controller_behavior.py`):** Separated inference traces from exploration trajectories; verified `LQP->LAG->STOP: 188`, `STOP: 71`, `LQP->STOP: 55` (`24.20%` explicit stops, `75.80%` `repeat_action_forced`).
- **Confidence Correlation (`scripts/final_pass_script.py`):** Fixed `confidence_trace[0]` extraction (eliminating prior `NaN`), establishing `Spearman rho = 0.5030` (`p_corr = 1.07e-20`).

## 5. Claim Boundaries
- Never claim context-adaptive sequencing without acknowledging the `3` dominant inference sequences and `75.80%` repeat-action guards.
- Never claim equivalence between `SETU_v1` and `SETU_v2` without TOST testing; state "no statistically significant difference detected overall."
