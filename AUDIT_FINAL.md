# AUDIT FINAL - Phase 0 through External Validation

## 1. CURRENT ARCHITECTURE
Query Feature Extraction (`cmi`, `lid_entropy`, `entity_density`) -> Initial Dense Retrieval (`BGE-M3`) -> Adaptive Controller (`LinUCB`) over `["LAG", "CAEP", "LQP", "STOP"]` -> `CARF` Rank Fusion.

## 2. CURRENT DATA FLOW
1. Queries load from `data/processed/queries_v3_final.json` (`314` queries) over `data/processed/corpus_chunks_v2.jsonl` (`380` chunks across `18` categories).
2. Exploration trajectories (`alpha=1.0`) log to `data/logs/trajectories_v3.jsonl` via `scripts/generate_trajectories_scaled.py`.
3. Leak-free 5-fold `GroupKFold` evaluation runs via `scripts/compare_setu_v1_v2_scaled.py`, outputting `results/tables/setu_v1_v2_comparison_scaled.json`, `results/logs/setu_v2_per_query_v3.json`, `results/canonical/setu_trajectories.json`, and `results/canonical/per_query_results.jsonl`.

## 3. TRAIN/EVAL FLOW & GROUPED LEAKAGE PREVENTION
- Evaluation uses 5-fold `GroupKFold` over union-find connected components of `(normalized_query_text, relevant_doc_ids)`, preventing the `4` duplicate-text query pairs (`Q088/Q089`, `Q148/Q185`, `Q151/Q186`, `Q222/Q223`) and shared gold chunks from bridging train and held-out splits (enforced in `tests/test_leakage.py`).
- Inside each fold, `LinUCBController(alpha=0.0)` and `fit_lag_v2` train strictly on out-of-fold rows (`train_ids.isdisjoint(test_ids)`).
- **CAEP** trains synthetically on corpus entities + hard negatives; **LQP** trains strictly externally on `500` parallel `LingoIITGN/PHINC` pairs.

## 4. LATENCY & DEFINITION OF A SETU STEP
- **Step:** One non-`STOP` operator action (`len([o for o in ops if o != "STOP"])`).
- **Latency (`scripts/measure_latency.py`):** `N=3` uncached CPU repeats across all `314` queries (`RAW` `189.48 ms`, `SETU_v1` `1186.80 ms`, `SETU_v2` `485.10 ms`).

## 5. RANDOMNESS & SEEDING
All scripts call `setu.config.set_seed(42)`, fixing `random`, `numpy`, and `torch`.

## 6. KNOWN LIMITATIONS
- Local CPU latency measurements (`i7-13620H`, `torch==2.10.0+cpu`).
- Controller terminates via `repeat_action_forced` on `238/314` (`75.80%`) inference episodes (`76/314 = 24.20%` explicit `STOP`), clustering into 3 sequences (`LQP->LAG->STOP`: 188, `STOP`: 71, `LQP->STOP`: 55) with `313/314` initial margins `< 0.20`.
- Heuristic regex CMI vs. `IndicLID` agreement is `kappa = 0.0868` (`results/tables/cmi_validity.txt`).
- Human review batches `data/review/review_batch_{1,2,3}.json` remain unannotated (`unreviewed`).

## 7. CANCELLED / OUT-OF-SCOPE PHASES
- **Phase 13 (CMI Validation)**: Out of scope beyond documenting the existing `kappa = 0.0868` construct audit; replacing CMI with neural LID benchmarks is left for future work.
- **External Validation (Added Phase 3)**: Completed cleanly on `500` unseen QA pairs from `aaditya/databricks-dolly-15k-Hinglish-Codemix` (`results/tables/external_dolly_validation.json`).

---

## Phase 4–14 Summary (Updated Post GroupKFold Re-Run)
- **Controller Sequences (`controller_behavior_final.json`):** `LQP->LAG->STOP` (`188`), `STOP` (`71`), `LQP->STOP` (`55`); explicit stops `24.20%` (`76/314`), forced repeat-action stops `75.80%` (`238/314`). Raw Chi-square of first action vs. CMI band is `13.18 (p = 0.0043)`, which does not survive 17-test Holm correction (`p_corr = 0.0731`).
- **Overcorrection (`overcorrection_final.json`):** Standalone operators significantly degrade already-correct queries (`BGE-M3+LAG -0.0310, p_corr=0.0220`; `Indic-SBERT+CAEP/LAG -0.0754/-0.0757, p_corr=0.0046`; `mE5+CAEP -0.0378, p_corr=0.0063`) while failing to significantly lift failed queries (`p_corr >= 0.3300`). GroupKFold `SETU_v2` avoids correct-query harm (`-0.0075, p_corr=0.9225`) and significantly recovers failed RBI queries (`+0.0364, p_corr=0.0494`).
- **Confidence Validity (`Phase 14`, Fixed Bug):** Reading `confidence_trace[0]` yields `Spearman rho = 0.5030`, raw `p = 1.53e-21`, Holm `p_corr = 1.07e-20`.
- **Selective Strategies (`selective_correction_strategies.json`):** Empirical Oracle `pi*` hits `0.8842` MRR (`+0.0316`, `p_corr = 5e-6`), keeping `RAW` untouched on `281/314` (`89.49%`) queries.
- **External Dolly Validation (`external_dolly_validation.json`):** Replicates significant CAEP/LAG overcorrection across all 3 backbones (`p_corr <= 0.0022`) and confirms zero-shot `mE5-large + LQP` recovery on failed external queries (`+0.0934, p_corr=0.0008`).
- **Test Status (`Phase 5`):** `pytest tests/ -q` passes (`33/33`, including `tests/test_leakage.py` grouped fold verification and `tests/test_setu_bandit.py`).
