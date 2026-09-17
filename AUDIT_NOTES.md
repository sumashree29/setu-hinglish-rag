# Audit Notes - Phase 0

## 1. What LQP does (input, transform, output)
**Citation:** `setu/operators/lqp.py:10-11`, `41-48`, `51-61`
- **Input**: Hinglish-to-English parallel embedding pairs (e.g. from PHINC dataset).
- **Transform**: Fits a Ridge regression matrix `W` minimizing `||X┬╖W - Y||^2 + lambda*||W||^2`. During inference, it builds a CMI-conditional blend matrix `W_eff(q) = (1 - alpha) * I + alpha * W`, where `alpha = min(1, CMI(q) / CMI_max)`.
- **Output**: The projected query embedding `projected_embedding = query_embedding @ W_eff`.

## 2. What CAEP does
**Citation:** `setu/operators/caep.py:5-7`, `124-130`, `133-179`
CAEP (Context-Aware Entity Preservation) decides whether to preserve or substitute candidate entity mentions. It uses a logistic regression confidence gate over the features `[fuzzy_score, embedding_cosine, entity_frequency]`. If the gate predicts `1`, it substitutes the word with the best-matching known entity; otherwise, it keeps the original word.

## 3. What LAG does, including every sub-strategy in apply_lag()
**Citation:** `setu/operators/lag.py:77-90`, `93-140`
LAG (Learned Adaptive Gating) predicts the correction strategy for a query using a LogisticRegression or LightGBM model based on features `[cmi, lid_entropy, entity_density]`.
- **light_normalize**: Lowcases, strips extra whitespace, and normalizes punctuation. Does not modify entities or word order.
- **dual_variant**: Returns `[query, entity_aware_variant]`, where `entity_aware_variant` applies `light_normalize` and replaces matched entities exactly.
- **full_rewrite**: Applies `light_normalize` and then defers to `apply_caep` for entity substitution.

## 4. What CARF does and where it is invoked
**Citation:** `setu/fusion/carf.py:30-54`, `90-111` | `setu/controller/setu_bandit.py:123`, `408`
CARF (CMI-Aware Rank Fusion) fuses raw and corrected document rankings. 
- `carf_v1` applies a hand-specified weighting to Reciprocal Rank Fusion based on `alpha = min(1.0, max(0.0, cmi_score / cmi_max))`.
- `carf_v2` learns the interpolation weights `w_cmi` and `w_entropy` via Ridge regression.
It is invoked at the very end of `setu_v1_fixed_order` and outside the main loop in `setu_v2_run` to generate the `fused_ranking`.

## 5. SETU v1's exact sequence and termination
**Citation:** `setu/controller/setu_bandit.py:93-124`
SETU v1 applies a strictly fixed sequence of operators: `LAG` -> `CAEP` -> `LQP` -> `CARF`. It terminates after CARF has produced the final ranking. No adaptivity or early stopping is present.

## 6. SETU v2's exact loop, including every break path
**Citation:** `setu/controller/setu_bandit.py:338-348`
SETU v2 runs a contextual bandit controller in a loop bounded by `max_steps` (default 4). At each step, it selects an action and updates the context. It breaks/terminates under three conditions:
1. The action selected is `"STOP"`.
2. The action selected has already been executed (`action in tried and tried[action] == 1.0`).
3. The loop reaches `max_steps`.

## 7. The exact 7 context features, their ranges, and whether they are normalised
**Citation:** `setu/controller/setu_bandit.py:249-256`, `339-342`
The 7 context features are:
1. `cmi_score`: [0, 1] (or occasionally [0, 100], depending on `cmi()` internal state).
2. `entropy_score`: `lid_entropy` output (float).
3. `confidence`: Margin between top retrieval scores (float).
4. `step`: Integer count of current loop iteration [0, 3].
5. `tried["LAG"]`: Float [0.0, 1.0].
6. `tried["CAEP"]`: Float [0.0, 1.0].
7. `tried["LQP"]`: Float [0.0, 1.0].
**None** of these features are explicitly normalized prior to being passed into `select_action`.

## 8. The exact action space and how STOP is reached (all three ways)
**Citation:** `setu/controller/setu_bandit.py:21`, `344`
The action space is `["LAG", "CAEP", "LQP", "STOP"]`.
STOP is reached if:
1. The controller explicitly predicts `"STOP"`.
2. The controller attempts to pick an action that was already `tried`.
3. The `max_steps` boundary (4 steps) is hit.

## 9. The exact reward definition, its scale, and what it does not include
**Citation:** `setu/controller/setu_bandit.py:40`, `393`
The reward is defined strictly as `confidence_after - confidence_before` (the difference in the retrieval confidence margin). 
It **does not include** any computational penalty, latency cost, or intervention cost for applying an operator.

## 10. The definition of one "step" for v1 and for v2, separately
**Citation:** `scripts/compare_setu_v1_v2_scaled.py:157`, `172-173`
- **v1**: 4.00 steps assumed by construction. It is never actually measured or calculated in `compare_setu_v1_v2_scaled.py`.
- **v2**: Measured explicitly as `len([o for o in ops if o != "STOP"])`.

## 11. How relevance labels were produced for each of the 75 pilot and 239 generated queries
**Citation:** `data/processed/queries_v3_final.json` (as per findings)
Relevance labels were automatically constructed during dataset generation: each query inherits exactly one gold chunk from its `source_question`'s parent chunk. Labels are purely binary, un-pooled, and single-positive. For the 239 generated queries, they retain `"review_status": "auto_generated_v3_paraphrased"` and lack human verification.

## 12. Whether controller training and evaluation data overlap, query-by-query
**Citation:** `scripts/compare_setu_v1_v2_scaled.py:113-114`
Yes, there is overlap. The 5-fold Out-Of-Fold (OOF) cross-validation splits on `query_id`, but filters the training trajectories by matching `query_text` (`r.get("query") not in test_query_texts`). Because 4 queries share identical text but have different `query_id`s, they leak from the training set into the held-out evaluation fold. Furthermore, `trajectories.jsonl` contains the old 75 pilot queries, missing 238 queries entirely.

## 13. Exactly what is inside and outside each time.perf_counter() window
**Citation:** `scripts/compare_setu_v1_v2_scaled.py:95-101`, `147-156`
- **Outside**: Loading of embedding and Faiss models, query embedding generation (`embed_fn`), and the raw Faiss search (`faiss_search_fn`).
- **Inside (v1)**: The `setu_v1_fixed_order` execution, which includes running LAG, CAEP, LQP, and CARF, and performing internal Faiss searches for rewritten queries.
- **Inside (v2)**: The `setu_v2_run` execution, which includes bandit context construction, action predictions, operator applications, and internal Faiss searches.

## 14. Every place a random number is drawn and whether it is seeded
**Citation:** `setu/controller/setu_bandit.py:160, 161, 171, 185`, `scripts/compare_setu_v1_v2_scaled.py`, `scripts/run_statistical_tests_h1_h10_scaled.py:13`
- `EpsilonGreedyController` uses `random.random()` and `random.choice(self.actions)` during action selection. This is **unseeded**.
- `scripts/compare_setu_v1_v2_scaled.py` uses `EpsilonGreedyController` (if switched to) without setting a seed anywhere in the script.
- `scripts/run_statistical_tests_h1_h10_scaled.py` sets `np.random.seed(42)` at the beginning of the script.

## 16. Task 4-fix: True Stop Reason Distribution and Sequence Collapse
- **stop_reason distribution**: `repeat_action_forced: 302 (96.2%)`, `explicit_stop: 12 (3.8%)`, `max_steps_reached: 0`.
- **Action-sequence diversity** is essentially zero: every one of the 314 queries falls into exactly one of three fixed patterns — `LQP -> LAG -> STOP` (199 queries, all via forced repeat), `LQP -> STOP` (103 queries, all via forced repeat), `STOP` immediately (12 queries, genuine argmax choice).
- **Conclusion**: "At inference (alpha=0, greedy-linear), the policy almost always attempts LQP first, sometimes also LAG, and terminates via forced repeat-action in 96.2% of cases rather than by the controller selecting STOP as the highest-value action. The step and latency reduction versus SETU v1 is real and measured, but it is a consequence of the greedy policy converging to a near-static 1–2-operator ordering, not of context-sensitive per-query decision-making. Action-sequence diversity across the 314-query evaluation set is limited to three fixed patterns."

## CLAIM CONTRACT UPDATE
- **Forbidden**: "the controller adaptively decides per query" / "accurately cuts off unproductive operators" / any phrasing that implies the 314 queries received meaningfully different treatment when the evidence shows 3 fixed macro-patterns.
- **Permitted**: "reduces mean steps and latency relative to a fixed-order pipeline, though the underlying policy shows limited per-query behavioral diversity (see AUDIT_NOTES.md §16)."
