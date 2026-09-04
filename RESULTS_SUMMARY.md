# RESULTS SUMMARY — SETU Hypothesis Testing (H1–H10)

All results from the pilot/domain-scale evaluation: **314 queries, 380 corpus chunks, BGE-M3 primary model**.
Source: `results/tables/statistical_significance_H1_H10_scaled.json`

## Hypothesis Verdicts

| # | Hypothesis | Test | Statistic | p-value | Verdict | Plain-English Meaning |
|---|-----------|------|-----------|---------|---------|----------------------|
| H1 | Retrieval quality decreases as CMI increases | Spearman (CMI vs MRR) | ρ=0.091 | 0.108 | **Not supported** | No significant monotonic relationship between code-mixing intensity and retrieval degradation. Band imbalance (low=14, medium=28, high=237, very_high=35) limits statistical power. |
| H2 | Indic-tuned encoders degrade less than general multilingual encoders | Paired Wilcoxon (Indic-SBERT vs BGE-M3) | r_rb=-0.718 | 1.03e-16 | **⚠️ Significant in OPPOSITE direction** | Indic-SBERT (MRR=0.604) performs dramatically *worse* than BGE-M3 (MRR=0.847), not better. The Indic-tuned model's smaller vocabulary and training data make it inferior to the general multilingual model on this corpus. |
| H3 | Retrieval degradation predicts answer-quality degradation | — | — | — | **Insufficient data** | Deferred to Phase 6 (downstream LLM generation evaluation not yet implemented). |
| H4 | SETU-processed queries achieve higher Recall/MRR/nDCG than raw | Paired Wilcoxon (RAW vs SETU v1 MRR) | r_rb=-0.157 | 0.484 | **Not supported** | SETU v1 (fixed pipeline) does not significantly improve over RAW. Mean MRR diff = -0.005 (slight decrease). |
| H5 | SETU's recovery exceeds generic baselines (interpolation, Rewrite-Retrieve-Read) | — | — | — | **Insufficient data** | External comparison baselines not implemented. |
| H6 | SETU v2 outperforms SETU v1 in retrieval quality | Paired Wilcoxon (v1 vs v2 MRR) | r_rb=0.268 | 0.232 | **Not supported (equivalent)** | v1 MRR=0.8475 vs v2 MRR=0.8505 — statistically indistinguishable. Quality is matched. |
| H7 | LQP alone recovers CMI-driven degradation | Paired Wilcoxon (RAW vs LQP MRR) | r_rb=-0.417 | 0.153 (adj) | **Not supported (failed Holm correction)** | LQP hurts aggregate retrieval (MRR 0.853→0.841), though not statistically significantly after multiple-testing correction. The projection over-corrects the 76% of queries where RAW is already correct (see Over-correction Diagnosis below). |
| H8 | SETU v2 matches quality with fewer steps than v1 | Paired Wilcoxon (step counts) | r_rb=-1.000 | 5.28e-59 | **✅ Supported** | v2 uses mean 1.20 steps vs v1's 4.00 steps (2.80 fewer, p≈0). The LinUCB controller learned to STOP immediately for ~80% of queries while maintaining matched retrieval quality. |
| H9 | v2 step count correlates positively with CMI | Spearman (CMI vs steps) | ρ=-0.086 | 0.127 | **Not supported** | No significant correlation — the controller does not allocate more steps to higher-CMI queries as hypothesized. |
| H10 | Confidence proxy (score margin) correlates with retrieval success | Spearman (margin vs MRR) | ρ=-0.009 | 0.872 | **Not supported** | Score margin has zero predictive power for retrieval correctness at this scale. |

## Key Supplementary Finding: Over-correction Diagnosis

Source: `results/tables/overcorrection_diagnosis.json`

The single most important finding outside the formal hypotheses:

| Group | n | LQP Δ MRR | CAEP Δ MRR | LAG Δ MRR |
|-------|---|-----------|------------|-----------|
| RAW already correct (MRR=1.0) | 238 (76%) | **-0.160** | **-0.158** | **-0.155** |
| RAW got it wrong (MRR<1.0) | 76 (24%) | **+0.478** | **+0.465** | **+0.479** |

**Interpretation**: All three operators provide a massive MRR boost (+0.47) when the base model fails, but they *actively harm* retrieval (-0.16) when applied to queries the base model already handles correctly. Since 76% of queries fall in the "already correct" group, the net aggregate effect is negative — explaining why H4 and H7 fail.

This directly justifies the adaptive controller (SETU v2/H8): the operators ARE useful, but only when the base model is struggling. The controller learned exactly this — to STOP immediately on easy queries and only invoke operators selectively.

## Model Comparison (BGE-M3 vs Indic-SBERT vs mE5-large)

Source: `results/tables/scaled_corpus_retrieval_v3.json`

| Model | MRR | nDCG@10 | Recall@10 |
|-------|-----|---------|-----------|
| **BGE-M3** | 0.8526 | 0.8858 | 0.9873 |
| **mE5-large** | 0.8767 | 0.9039 | 0.9873 |
| **Indic-SBERT** | 0.5903 | 0.6476 | 0.8280 |

mE5-large is the strongest model overall. Indic-SBERT substantially underperforms both general multilingual models, contradicting H2's premise.

## Additional Limitations

In addition to the CMI band distribution noted below, the following methodological limitations must be considered:
1. **Pilot-Scale Corpus**: The evaluation corpus size (380 chunks) is pilot-scale. High recall numbers may be partially driven by lexical overlap confounds, limiting the generalizability of the H1 null finding.
2. **Hand-rolled LID Tagger**: CMI scores rely on a placeholder lexicon tagger instead of the target IndicLID model, threatening construct validity and potentially making CMI scores noisy.
3. **LAG In-sample Labeling**: The LAG operator's training labels were derived from trajectory optimization on the evaluation queries themselves, rather than a strict hold-out fold.
4. **Missing MIRACL Benchmark**: Public benchmark evaluation (MIRACL/Aksharantar) was not completed.

## CMI Band Distribution (Limitation)

| Band | CMI Range | n Queries | % of Total |
|------|-----------|-----------|------------|
| Low | 0.00–0.15 | 14 | 4.5% |
| Medium | 0.15–0.35 | 28 | 8.9% |
| High | 0.35–0.55 | 237 | 75.5% |
| Very High | 0.55–1.00 | 35 | 11.1% |

The severe skew toward the "high" band limits statistical power for H1 (degradation hypothesis) and H9 (step-CMI correlation). The 239 auto-generated queries cluster in the 0.35–0.55 CMI range because the code-mixed query generation strategy naturally produces queries in this band.
