# Results Summary

## Dataset & Evaluation Scope
- **Primary Corpus:** 380 chunks across **18 categories** (17 official RBI/PM-KISAN FAQ categories + pilot subset).
- **Primary Query Set:** 314 Romanized Hinglish queries (75 pilot + 239 paraphrased; 15 misspelled-entity queries `Q61-Q75`).
- **Cross-Validation Protocol:** 5-fold `GroupKFold` partitioned by connected components of normalized query text and relevant chunk IDs (`4` duplicate-text pairs grouped together).
- **External Validation Set:** 500 duplicate-free Hinglish QA pairs from `aaditya/databricks-dolly-15k-Hinglish-Codemix` (zero overlap with RBI or PHINC).

## Baseline Performance (314 Queries, 380 Chunks)
- **`mE5-large` (`intfloat/multilingual-e5-large`):** MRR **0.8767** | nDCG@10 **0.9039** | Hit@1 **80.89%** | Recall@10 **0.9873**
- **`BGE-M3` (`BAAI/bge-m3`):** MRR **0.8526** | nDCG@10 **0.8858** | Recall@5 **0.9554** | Recall@10 **0.9873** (77.39% rank-1)
- **`mContriever`:** MRR **0.7285** | nDCG@10 **0.7749** | Hit@1 **62.42%** | Recall@10 **0.9204**
- **`BM25`:** MRR **0.6490** | nDCG@10 **0.6987** | Hit@1 **53.82%** | Recall@10 **0.8567**
- **`Indic-SBERT` (`l3cube-pune/indic-sentence-similarity-sbert`):** MRR **0.5903** | nDCG@10 **0.6476** | Hit@1 **47.13%** | Recall@10 **0.8280**

## SETU v1 vs SETU v2 vs RAW (BGE-M3, 5-Fold GroupKFold)
- **RAW (`pi_never`):** MRR **0.8526** | nDCG@10 **0.8858** | Recall@10 **0.9873** | Latency **189.48 ms**
- **SETU v1 (`pi_always`):** MRR **0.8492** | nDCG@10 **0.8810** | Recall@10 **0.9809** | Actions **3.000** | Latency **1186.80 ms**
- **SETU v2 (`pi_learned`):** MRR **0.8550** | nDCG@10 **0.8875** | Recall@10 **0.9873** | Actions **1.373** | Latency **485.10 ms**
- **Empirical Oracle (`pi*`):** MRR **0.8842** (+0.0316 vs RAW, Holm $p = 5 \times 10^{-6}$; leaves **281/314 = 89.49%** queries untouched as RAW).
- **Threshold Gating (`tau=0.05` + LAG, 175/314 triggered):** MRR **0.8387** (-0.0139 vs RAW, Holm $p = 0.8972$).

## Statistical Analysis Highlights (Holm-Bonferroni Corrected)
- **H1 (CMI Degradation):** Not supported (Spearman rho = 0.0712, raw $p = 0.2084$, Holm $p_{\text{corr}} = 0.8334$).
- **H2 (Indic-SBERT robustness):** Not supported (BGE-M3 significantly outperforms Indic-SBERT, Holm $p_{\text{corr}} = 3.50 \times 10^{-17}$).
- **H4 (SETU v1 > RAW):** Not supported ($p = 0.6373$, Holm $p_{\text{corr}} = 1.0000$).
- **H6 (SETU v2 > SETU v1 overall):** Not supported ($p = 0.2482$, Holm $p_{\text{corr}} = 0.8334$). No statistically significant difference was detected between SETU v1 and SETU v2 overall; this is not an equivalence claim.
- **H8 (SETU v2 requires fewer steps):** Supported. SETU v2 takes significantly fewer non-STOP operator actions (**1.373** mean vs **3.000** for SETU v1; **54.25%** reduction, raw $p = 2.24 \times 10^{-56}$, Holm $p_{\text{corr}} = 1.79 \times 10^{-55}$).
- **H10 (Confidence correlates with success):** Supported. Score margin (`confidence_trace[0]`) significantly correlates with MRR (**Spearman rho = 0.5030**, raw $p = 1.53 \times 10^{-21}$, Holm $p_{\text{corr}} = 1.07 \times 10^{-20}$).

## Controller Behavior (314 Queries, Inference $\alpha = 0$)
- **Action Sequences Observed:** `LQP->LAG->STOP`: **188** (59.9%), `STOP`: **71** (22.6%), `LQP->STOP`: **55** (17.5%).
- **Explicit Stops:** **76** (24.20%) | **Forced Repeat-Action Stops:** **238** (75.80%).
- **First-Action vs CMI Band:** Raw $\chi^2 = 13.18, p = 0.0043$ (Holm $p_{\text{corr}} = 0.0731$ across 17 tests; H9 step count vs CMI $\rho = 0.0053, p = 0.9255$).

## Overcorrection Analysis (Primary RBI Corpus & External Dolly-Hinglish)
- **Primary Corpus (`BGE-M3`, 243 correct / 71 failed):**
  - Standalone `LAG`: degrades correct queries by `-0.0310` (Holm $p_{\text{corr}} = 0.0220$); failed subset `+0.0446` (Holm $p_{\text{corr}} = 0.3300$, `0.33`).
  - Standalone `LQP` and `CAEP`: failed subset Holm $p_{\text{corr}} = 1.0000$.
  - Auxiliary encoders: `Indic-SBERT + CAEP` (`-0.0754`, $p_{\text{corr}} = 0.0046$), `Indic-SBERT + LAG` (`-0.0757`, $p_{\text{corr}} = 0.0046$), `mE5-large + CAEP` (`-0.0378`, $p_{\text{corr}} = 0.0063$).
  - **`SETU v2` (GroupKFold OOF):** preserves already-correct queries (`-0.0075`, $p_{\text{corr}} = 0.9225$) while delivering significant recovery on initially failed queries (`+0.0364`, Holm $p_{\text{corr}} = 0.0494$).
- **External Dolly-Hinglish (500 Queries):**
  - `BGE-M3`: RAW MRR **0.9121** (88.6% rank-1); `CAEP` (`-0.0240`, $p_{\text{corr}} = 0.0022$) and `LAG` (`-0.0259`, $p_{\text{corr}} = 0.0017$) significantly degrade initially correct queries.
  - `mE5-large`: RAW MRR **0.8695** (83.2% rank-1); zero-shot `LQP` lifts overall MRR to **0.8812** via failed-query recovery (`+0.0934`, $p_{\text{corr}} = 0.0008$) with non-significant impact on correct queries (`-0.0048`, $p_{\text{corr}} = 0.4095$), while `CAEP` (`-0.0469`, $p_{\text{corr}} < 0.0001$) and `LAG` (`-0.0576`, $p_{\text{corr}} < 0.0001$) overcorrect working queries.
  - `Indic-SBERT`: RAW MRR **0.7921** (72.4% rank-1); `LQP`, `CAEP`, and `LAG` all significantly degrade correct queries ($p_{\text{corr}} \le 0.0145$).

## Latency (Local CPU, 3 Repeats $\times$ 314 Queries)
- **Hardware:** Windows 11, 13th Gen Intel Core i7-13620H (10 physical / 16 logical cores), 15.64 GB RAM, CPU PyTorch `2.10.0+cpu`.
- **RAW:** Mean **189.48 ms** (median 168.71 ms, p95 321.55 ms)
- **SETU v1:** Mean **1186.80 ms** (median 1088.84 ms, p95 1953.96 ms)
- **SETU v2:** Mean **485.10 ms** (median 495.25 ms, p95 945.12 ms — 59.1% latency reduction vs. SETU v1)
