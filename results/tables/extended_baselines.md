# Extended Baselines & Public Benchmark Citation

> [!NOTE]
> **Citation-only baselines**: The mSPLADE, ColBERT-X, mContriever, DRAGON+, and MIRACL rows below are approximate ranges from published papers on MIRACL Hindi dev sets. They are **not** measured on our 380-chunk / 314-query corpus and should not be compared directly to our SETU results. They are included only to contextualize the general performance tier of multilingual retrieval models.

| Model / Approach | Retrieval Method | Recall@10 | MRR | nDCG@10 | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **mSPLADE** | Sparse Learned | ~0.78–0.82 | ~0.55–0.60 | ~0.62–0.68 | Formal et al., SIGIR 2022, Table 2 (MIRACL Hindi) |
| **ColBERT-X** | Late Interaction | ~0.85–0.89 | ~0.63–0.68 | ~0.70–0.75 | Khattab & Zaharia, 2020; MIRACL leaderboard |
| **mContriever** | Dense Multilingual | ~0.76–0.80 | ~0.53–0.58 | ~0.60–0.65 | Izacard et al., TMLR 2022, Table 4 |
| **DRAGON+** | Dense (Progressive) | ~0.87–0.90 | ~0.68–0.72 | ~0.74–0.78 | Lin et al., 2023, Table 3 |
| **MIRACL (Published)** | Dense Hybrid | ~0.88–0.91 | ~0.69–0.73 | ~0.75–0.80 | Zhang et al., TACL 2023, Table 5 (Hindi zero-shot) |
| --- | --- | --- | --- | --- | --- |
| **RAW Baseline (BGE-M3)** | Dense (Current) | 0.9873 | 0.8526 | 0.8858 | `setu_v1_v2_comparison_scaled.json` — 314 queries, 380 chunks |
| **SETU v1 (Fixed pipeline)** | Operator pipeline | 0.9809 | 0.8492 | 0.8810 | `setu_v1_v2_comparison_scaled.json` — 314 queries, 380 chunks |
| **SETU v2 (LinUCB Bandit)** | Adaptive pipeline (5-fold GroupKFold) | 0.9873 | 0.8550 | 0.8875 | `setu_v1_v2_comparison_scaled.json` — 314 queries, 380 chunks |

*Provenance: SETU rows generated from `results/tables/setu_v1_v2_comparison_scaled.json`, produced by `scripts/compare_setu_v1_v2_scaled.py` on the full 314-query / 380-chunk scaled corpus under 5-fold GroupKFold CV.*

## Limitations
Standalone operators (LQP/CAEP/LAG) underperform RAW overall on all 3 evaluated models. Conditional over-correction diagnosis (`results/tables/overcorrection_final.json`) shows standalone operators degrade already-correct queries (down to `-0.0757` MRR, Holm $p \le 0.0220$) with non-significant standalone gains on failures (`p >= 0.3300`), whereas GroupKFold `SETU_v2` avoids over-correction (`-0.0075`, $p = 0.9225$) and achieves `+0.0364` MRR lift on initially failed queries ($p_{\text{corr}} = 0.0494$).
