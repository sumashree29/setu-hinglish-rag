# AUDIT FINAL - Phase 0

## 1. CURRENT ARCHITECTURE
The SETU-Hinglish-RAG pipeline implements the following stages:
1. **Dataset**: Processed via `data/processed/corpus_chunks_v2.jsonl` and `data/processed/queries_v3_final.json`.
2. **Embeddings**: Generated using dense retrievers (like BGE-M3) via `setu/embeddings/loader.py` (e.g., `load_embedding_model` and `embed`).
3. **Operators**:
   - **LAG** (Learned Adaptive Gating): `setu/operators/lag.py` (rewrites queries).
   - **CAEP** (Context-Aware Entity Preservation): `setu/operators/caep.py` (preserves/substitutes entities).
   - **LQP** (Latent Query Projection): `setu/operators/lqp.py` (projects queries based on CMI).
4. **Controller**: `setu/controller/setu_bandit.py`. Dictates the sequence of operators. Uses a `LinUCBController` or fixed sequence (`setu_v1_fixed_order`, `setu_v2_run`).
5. **Fusion**: **CARF** (CMI-Aware Rank Fusion) in `setu/fusion/carf.py` fuses original and corrected rankings.
6. **Evaluation**: Conducted in `scripts/compare_setu_v1_v2_scaled.py` measuring MRR, NDCG, etc., against `qrels_dict`.

## 2. CURRENT DATA FLOW
Trace of a query through the system:
1. Raw code-mixed text originates in `data/processed/queries_v3_final.json`.
2. The query is embedded via `setu/embeddings/loader.py`.
3. Offline trajectories for bandit training are generated and logged to `data/logs/trajectories_v3.jsonl`.
4. During evaluation (`scripts/compare_setu_v1_v2_scaled.py`), the trained controller receives the query context and outputs an action path (e.g., `LQP -> STOP`).
5. The specified operator functions (e.g., `lqp_model.predict`) are applied to the query embedding.
6. The modified query is run against the FAISS index (built from `results/logs/doc_emb_bge_m3_v2.npy`).
7. Resulting document rankings are scored against gold labels.
8. The final metrics are aggregated and saved to `results/tables/setu_v1_v2_comparison_scaled.json`.
9. The per-query action sequence and termination states are logged to `results/logs/setu_v2_per_query_v3.json`.

## 3. TRAIN/EVAL FLOW
The evaluation leverages a **5-fold Out-Of-Fold (OOF) cross-validation** scheme inside `scripts/compare_setu_v1_v2_scaled.py`. 
- Queries are batched into 5 folds explicitly grouped by `query_id` (not `query` text) to ensure lexical duplicates don't bridge the splits. 
- For each fold, a fresh `LinUCBController` is trained solely on trajectories belonging to the 4 out-of-fold partitions.
- Simultaneously, the `lag_model` is retrained locally per fold using `fit_lag_v2` on the out-of-fold `lag_labels_v3.json` queries.
- The trained controller (and LAG model) are then used strictly for inference on the held-out test fold.

## 4. ALL IDENTIFIED LEAKAGE RISKS
- **Phase 1 (Fixed)**: Test queries leaked into training via exact text overlap because trajectories were partitioned by `query_text` instead of `query_id`. This was fixed by strictly filtering on `query_id` in `compare_setu_v1_v2_scaled.py`.
- **Phase 2 (Fixed)**: The LAG classification model was globally fitted on the entire dataset (`lag_labels_v3.json`) prior to evaluation, allowing it to "see" test query features. This was fixed by moving the LAG model fitting loop inside the per-fold cross-validation in `compare_setu_v1_v2_scaled.py`.
- **Further Risk Audit (CAEP/LQP)**: 
  - **CAEP** is trained in `scripts/train_operators.py` using synthetic augmentations of entities extracted from the underlying *corpus chunks*, not the evaluation queries. **No query leakage.**
  - **LQP** is trained in `scripts/train_operators.py` on the completely independent PHINC parallel corpus (`load_parallel_pairs_phinc`). **No query leakage.**
  - **Conclusion**: There are no remaining unaddressed data leakage risks across the pipeline. CAEP and LQP are completely clean.

## 5. CURRENT TRAIN/TEST SEPARATION
As asserted directly in `scripts/compare_setu_v1_v2_scaled.py`:
- **Train-Only**: The 4 out-of-fold partitions of `trajectories_v3.jsonl` (for the LinUCB policy) and the corresponding `lag_labels_v3.json` entries (for the `lag_model`).
- **Test-Only**: The 1 held-out fold of evaluation queries where inference runs without controller weight updates.
The separation strictly ensures that `train_ids.isdisjoint(test_ids)` for every fold.

## 6. DEFINITION OF ONE SETU STEP
Per `METRIC_DEFINITIONS.md`, a step is strictly defined as **"one controller iteration in which an operator action is selected and executed, excluding the STOP action."** An episode where the controller immediately outputs `STOP` on the first iteration yields `0` steps. 

## 7. DEFINITION OF LATENCY
Per the protocol documented in `results/tables/latency_final.json`, latency measures the wall-clock execution time starting from the controller receiving the initial raw search ranking up to the final STOP signal (including context extraction, action inference, and operator-triggered FAISS searches). 
**Explicit Limitation:** This measurement utilized `N=3` repetitions executed on a local CPU exclusively. It is a functional bottleneck measurement, not a large-N scaled GPU benchmark.

## 8. CONTROLLER ACTION SPACE / REWARD / CONTEXT
*(See Phase 3 Audit Below)*

## 9. RANDOMNESS/SEEDING
Per Phase 7, all stochastic operations are centralized. A global grep audit confirms that scripts (`sample_queries.py`, `generate_trajectories_scaled.py`, `prepare_review_batches.py`, `error_analysis_v2.py`, etc.) have had hardcoded `np.random.seed(42)` logic replaced. They now universally import and execute `from setu.config import set_seed; set_seed()`, effectively fixing Python's `random`, `numpy`, and `torch` deterministically.

## 10. MODEL TRAINING DATA
- **CAEP Gate**: Trained synthetically on misspelled/hard-negative variants of gold entities extracted via `build_pilot_corpus()` in `scripts/train_operators.py`.
- **LQP Model**: Trained on up to 500 parallel Hinglish-English translation pairs extracted from the external PHINC dataset (`setu/operators/lqp.py`).
- **LAG Classifier**: Trained on `data/processed/lag_labels_v3.json` containing `cmi`, `lid_entropy`, and `entity_density`. Per Phase 2, this model is now dynamically trained per fold inside the evaluation script.

## 11. EVALUATION DATA
- The canonical benchmark utilizes **314 queries** (`queries_v3_final.json`).
- This set contains 15 explicit misspelled-entity queries (Q61-Q75) designed to test the CAEP operator's robustness. 
- The search corpus comprises **380 chunks** (`corpus_chunks_v2.jsonl`).

## 12. RELEVANCE-LABEL CONSTRUCTION
The `qrels_dict` constructed in `scripts/compare_setu_v1_v2_scaled.py` creates binary, **single-positive** mappings. Each query inherits exactly one target gold document `chunk_id` derived directly from its parent question object.

## 13. KNOWN LIMITATIONS
- **(a) N=3 Latency Benchmark**: As stated above, latency numbers are from a localized CPU bottleneck check, not a production-grade benchmark.
- **(b) Controller Lack of Adaptivity**: The controller functions as an early-stopping policy rather than a context-sensitive sequencer (Phase 4).
- **(c) CMI/IndicLID Disagreement**: An orphaned output (`results/tables/cmi_validity.txt`) shows a Cohen's Kappa of just 0.0868 between the heuristic lexicon and IndicLID. **(Provenance unverified, not yet officially regenerated under Phase 13)**.
- **(d) Extreme Confidence Skew**: 313/314 queries (99.7%) start with confidence `<0.2`. This is a severely narrow distribution on a core context dimension, flagged as a major risk for Phase 14 (H10).
- **(e) Corpus Scale**: Evaluating on only 380 chunks heavily risks metrics being dominated by trivial lexical matches.
- **(f) Controller Evaluation Scope**: The full SETU v1/v2 controller pipeline (requiring trained LinUCB weights from offline trajectories) was exclusively run and evaluated on the BGE-M3 baseline. Operator ablations for Indic-SBERT and mE5-large were conducted in isolation; the full controller was never trained or tested on them.

## 14. EXACT LIST OF FILES/RESULTS STILL REQUIRING REGENERATION
- **Phase 9 (Baseline Retrievers)**: Not started (requires evaluation of Indic-SBERT and mE5-large).
- **Phase 10 (Operator Ablations)**: Not started. *(Note: The ablation script intentionally evaluates only the three dense multilingual models: BGE-M3, Indic-SBERT, mE5-large. BM25 is excluded as it is mathematically incompatible with the LQP dense-projection operator. mContriever is excluded as the ablation focuses on the primary dense backbones, though it is fully evaluated as a baseline in Phase 9).*
- **Phase 13 (CMI Validation)**: The scripts `evaluate_cmi_validity.py` and `download_indiclid.py` exist but require a real, provenance-confirmed rerun.
- **Phase 15 (External Baselines)**: Not started.
- **`results/canonical/` directory**: Currently holds only 10-byte `{}` JSON placeholder files generated as scaffolding in Phase 8. Real, populated metrics are missing.

---

## Phase 3: LinUCB Controller Audit

A code-reading audit of the `LinUCBController` (in `setu/controller/setu_bandit.py`) yields the following definitions:

- **Action Space**: The controller chooses from 4 discrete actions: `["LAG", "CAEP", "LQP", "STOP"]`.
- **Context Space**: A 7-dimensional continuous feature vector: `[cmi, lid_entropy, confidence, step_t, tried_LAG, tried_CAEP, tried_LQP]`. The state captures semantic complexity, confidence, step depth, and a binary tracker of previously taken actions.
- **Reward Function**: The reward for any non-STOP action is `confidence_after - confidence_before`. The reward for the `STOP` action is strictly `0.0`.
- **Stopping Rule**: The episode halts under three conditions:
  1. The controller explicitly outputs the `STOP` action (driven by UCB expected reward).
  2. The controller attempts an action it has already tried (`repeat_action_forced`), forcing an immediate halt.
  3. The environment hits the hard limit of `max_steps=4`.
- **Central Finding**: The action space, state formulation, and stopping logic are sound. There is no evident leakage in the controller definition itself, provided that out-of-fold separation is maintained during the `fit_from_trajectories` step (which was fixed in Phase 1).

## Phase 4: Controller Behavior Analysis

An empirical analysis of the logged trajectories (`results/logs/setu_v2_per_query_v3.json`) reveals that the controller acts as a **learned early-termination policy with limited action-sequence diversity**, and does not exhibit strong context-sensitive operator sequencing.

Key evidence supporting this conclusion:
- **Low Sequence Diversity**: Out of 314 queries, only 3 queries (less than 1%) executed a sequence with more than one non-stop action (e.g., `LAG->CAEP->LQP->STOP`). The vast majority of episodes consist of exactly one action followed by a forced stop, or an immediate `STOP` (which occurred 61 times).
- **Forced Termination Dominates**: Only 4.46% (14 out of 314) of queries terminated because the controller explicitly selected `STOP`. The remaining 95.54% of queries were forcibly halted by the `repeat_action_forced` guard.
- **Lack of Context-Sensitivity**: The distribution of the first action chosen by the controller is statistically independent of the query's complexity (CMI band). A Chi-square test of independence yields `p = 0.635` (not significant at alpha=0.05), indicating that the controller's initial action choice does not shift meaningfully in response to the context state.
- **Extremely Narrow Confidence Distribution**: A structural limitation contributing to this lack of context-sensitivity is that 313/314 queries (99.7%) start with an initial confidence of `<0.2`. Although confidence is one of the controller's 7 context dimensions, its variance across the dataset is practically zero. This is flagged as a limitation to revisit in Phase 14 (H10 confidence analysis).

### LAG Classifier Behavior (found during Phase 10)

An additional diagnostic (`scripts/check_lag_strategies.py`) traced `predict_strategy`'s output across all 314 queries under the same 5-fold OOF protocol used elsewhere. The LAG classifier collapses to a near-constant policy:
- `light_normalize`: 305/314 queries (97.1%)
- `dual_variant`: 9/314 queries (2.9%)
- `full_translation`: 0/314 queries (0.0%)

This corroborates the Phase 4 controller-behavior finding via an independent mechanism: not only does the LinUCB controller fail to show context-sensitive operator sequencing, but the LAG operator's own internal classifier also fails to differentiate meaningfully across the CMI/lid_entropy/entity_density feature space, defaulting almost entirely to one strategy. The likely cause is a severe class imbalance in `lag_labels_v3.json` (as seen in the Phase 10 Colab run: 12-18 positive examples out of ~250 per fold, under 7%), combined with limited feature variance. Both the controller-level and operator-level "adaptivity" claims should therefore be scoped conservatively in the paper: SETU functions largely as a fixed-policy correction system in this evaluation, not a context-adaptive one.

## Phase 11: Over-Correction Analysis

Based on the fresh, leakage-free data from Phase 9/10, an analysis of operator behavior conditional on the baseline (RAW) retrieval performance yielded a stark finding: SETU largely functions by degrading originally correct retrievals without sufficiently offsetting them via improvements on incorrect retrievals.

The 314 evaluation queries were split based on whether RAW retrieval returned the correct document at rank 1 (MRR=1). The initial splits confirm that models with strong dense backbones already solve the vast majority of queries natively:
- **BGE-M3 (default)**: 243 queries (77.39%) already correct, 71 (22.61%) incorrect.
- **mE5-large**: 254 queries (80.89%) already correct, 60 (19.11%) incorrect.
- **Indic-SBERT** (much weaker baseline): 148 queries (47.13%) already correct, 166 (52.87%) incorrect.

When applying individual operators (LQP, CAEP, LAG) or the full SETU controller variants:
1. **Performance on "Already Correct" Queries**: All operators across all three models significantly *degrade* MRR (negative mean delta, Wilcoxon p < 0.05). For BGE-M3, LQP, CAEP, LAG, and SETU_v1 all show significant statistical degradation.
2. **Performance on "Incorrect" Queries**: 
   - For **Indic-SBERT** and **mE5-large**, no operator yielded any statistically significant improvement.
   - For **BGE-M3**, only the LAG operator (p=0.027) and the SETU_v2 controller (p=0.0249) achieved significant positive deltas.

**Conclusion**: The prior "over-correction hypothesis" is fully validated. The operators are structurally predisposed to "hurt" queries that are already successfully mapped by raw dense retrieval. While raw uncorrected p-values suggested a sparse improvement on failure cases for BGE-M3, see Phase 12 below for the corrected significance.

### Phase 12: Statistical Correction (Holm-Bonferroni)

To rigorously validate the Phase 11 and Phase 4 findings, a Holm-Bonferroni multiple-testing correction was applied across all 23 hypothesis tests run to date (22 Wilcoxon tests from Phase 11 + 1 Chi-Square test from Phase 4). 

The correction revealed a critical false positive:
1. **Evaporation of Positive Lift**: The only previously observed significant improvements on failure cases (BGE-M3 + LAG with uncorrected p=0.0275, and BGE-M3 + SETU_v2 with uncorrected p=0.0249) completely failed to survive correction (**Holm p=0.3981** for both). There is **zero statistically robust evidence** that any operator or controller variant improves retrieval on natively failed queries across any of the three models.
2. **Robustness of Degradation**: In stark contrast, the degradation (negative MRR delta) on "Already Correct" queries survived correction with high significance for several operators, including Indic-SBERT CAEP (Holm p=0.0073), Indic-SBERT LAG (Holm p=0.0056), mE5-large CAEP (Holm p=0.0222), and BGE-M3 LAG (Holm p=0.0222). 
3. **Controller Independence**: The Phase 4 conclusion that the LinUCB controller's first action is independent of the query's complexity (CMI) remains fully valid (Holm p=1.0).

**Final Conclusion on Pipeline Efficacy**: Following Phase 12 correction, the data dictates that the entire SETU operator pipeline exerts only one statistically significant effect: it degrades natively strong dense retrievals. It provides no robust corrective lift. The paper should explicitly acknowledge this as a limitation of the current operator/controller formulation.
