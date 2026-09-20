# AUDIT FINAL - Phase 0

## Current Architecture
The SETU-Hinglish-RAG architecture consists of a retriever (BGE-M3, Indic-SBERT, or mE5-large) and a set of corrective operators designed to mitigate retrieval degradation on code-mixed Hinglish queries:
- **LQP (Latent Query Projection):** Projects query embeddings via Ridge regression conditional on CMI score.
- **CAEP (Context-Aware Entity Preservation):** Substitutes entity mentions based on a confidence gate (fuzzy matching, cosine similarity, entity frequency).
- **LAG (Learned Adaptive Gating):** Rewrites queries using LightGBM/Logistic Regression strategies (`light_normalize`, `dual_variant`, `full_rewrite`) based on CMI, entropy, and entity density.
- **CARF (CMI-Aware Rank Fusion):** Fuses original and operator-corrected rankings.
- **Controller (SETU):** 
  - **v1**: Fixed pipeline applying all operators unconditionally (LAG -> CAEP -> LQP -> CARF).
  - **v2**: A contextual bandit (LinUCB or Epsilon-Greedy) that adaptively selects the operator sequence and halts based on a confidence signal.

## Current Data Flow
1. Canonical dataset (`queries_v3_final.json`, `corpus_chunks_v2.jsonl`) loaded.
2. Code-mixed queries and documents are embedded using a dense retriever.
3. Offline exploration trajectories are generated via `generate_trajectories_scaled.py`.
4. Trajectories train the LinUCB controller.
5. In evaluation (`compare_setu_v1_v2_scaled.py`), queries undergo OOF cross-validation, applying the trained controller to dictate operator paths and generate final rankings.

## Training/Evaluation Flow
The controller uses a 5-fold Out-Of-Fold (OOF) cross-validation scheme. Folds are chunked sequentially by `query_id`. The controller is pre-trained on the offline trajectories of out-of-fold queries and then evaluates on the held-out fold.

## Identified Leakage Risks (Phase 1 Target)
- **OOF Query Leakage:** The evaluation script (`scripts/compare_setu_v1_v2_scaled.py`) and trajectory generator (`scripts/generate_trajectories_scaled.py`) log and filter trajectories by `query` text, not `query_id`. Since some different `query_id`s share identical `query_text`, test queries leak into the training fold.
- **LAG Label Leakage:** LAG labels were purportedly derived from trajectory optimization on the evaluation queries rather than a strictly held-out training set.

## Current Train/Test Separation
- Split by 5-fold OOF on `query_id`.
- Flawed due to text-based filtering (see Leakage Risks).

## Definition of One SETU Step
- **v1:** Hardcoded as 4.00 steps by definition in downstream scripts. Never dynamically measured.
- **v2:** Explicitly measured as the number of executed actions before termination, i.e., `len([o for o in ops if o != "STOP"])`.
*Ambiguity to resolve:* We will standardize the definition of a step to be "one controller iteration in which an operator action is selected and executed" (excluding STOP) across all scripts.

## Definition of Latency
Latency measures wall-clock time (`time.perf_counter()`).
- **Outside the timing window:** Model loading, embedding query, initial FAISS search.
- **Inside the timing window:** 
  - **v1:** Execution of LAG, CAEP, LQP, CARF, and any internal FAISS searches.
  - **v2:** Bandit context construction, action prediction, operator applications, internal FAISS searches, up to the STOP signal.

## Controller Action Space
- `LAG`, `CAEP`, `LQP`, `STOP`

## Controller Reward
- Reward = `confidence_after - confidence_before` (margin between top retrieval scores). 
- Reward for `STOP` is 0.0. No penalty is applied for step count or latency.

## Controller Context
7 dimensions:
1. `cmi_score`: [0, 1]
2. `lid_entropy`: float
3. `confidence`: float (margin)
4. `step`: integer count of loop iterations
5. `tried_LAG`: 0.0 or 1.0
6. `tried_CAEP`: 0.0 or 1.0
7. `tried_LQP`: 0.0 or 1.0

## Randomness/Seeding
- `EpsilonGreedyController` uses unseeded `random` module calls.
- `generate_trajectories_scaled.py` uses `np.random.seed(42)`.
- `LinUCBController` is deterministic given fixed data.
- Need to ensure global seed across all evaluation scripts.

## Model Training Data
- LQP, CAEP, LAG models are trained on real corpus/PHINC/pilot labels. LAG labels are generated in-sample.

## Evaluation Data
- **Queries:** 314 total queries (original pilot + misspelled subset + generated).
- **Corpus:** 380 corpus chunks from the RBI banking FAQ domain.

## Relevance-Label Construction
- Binary, un-pooled, single-positive labels.
- Queries inherit exactly one gold chunk from their source question's parent chunk.

## Known Limitations
1. **Corpus Scale:** 380 chunks limit generalizability. High recall numbers may be driven by lexical overlap.
2. **Hand-rolled LID Tagger:** CMI is calculated via a heuristic lexicon rather than IndicLID, affecting construct validity.
3. **LAG In-sample Labeling:** Labels derived from evaluation queries instead of a dedicated train split.
4. **CMI Band Skew:** 75.5% of queries fall into the "High" CMI band, limiting statistical power for correlation tests.

## Files/Results Requiring Regeneration
- `data/logs/trajectories_v3.jsonl`
- `results/tables/setu_v1_v2_comparison_scaled.json`
- `results/logs/setu_v2_per_query_v3.json`
- `results/tables/statistical_significance_H1_H10_scaled.json`
- `results/tables/controller_behavior_final.json`
- `results/figures/controller_behavior.png`
- `results/tables/latency_final.json`
- `results/tables/baseline_retrieval_final.json`
- `results/tables/overcorrection_final.json`
- `results/figures/overcorrection.png`

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
