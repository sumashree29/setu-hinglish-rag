# Master Checklist — Resume Point

If work stops, the first unchecked row tells you exactly where to resume.
Tick boxes as you go. File to fill in is listed so there's no ambiguity about where the work lives.

| # | Task | Phase | Owner | File(s) | Status |
|---|------|-------|-------|---------|--------|
| 1 | Environment installed by all 3 members | 1 | All | `requirements.txt` | ☑ |
| 2 | 3 embedding models downloaded | 1 | R1 | `setu/embeddings/loader.py`, `scripts/download_models.py` | ☑ |
| 3 | IndicLID + HinGE + PHINC downloaded | 1 | R1 | `setu/diagnosis/lid_entropy.py`, `setu/operators/lqp.py` | ☑ |
| 4 | Pilot corpus built (50-100 queries) | 1 | R1 (+team) | `setu/diagnosis/corpus.py`, `scripts/build_pilot_corpus.py` | ☑ (20 chunks, 75 queries) |
| 5 | CMI(q) / LID-entropy(q) implemented | 1 | R1 | `setu/diagnosis/cmi.py`, `setu/diagnosis/lid_entropy.py` | ☑ |
| 6 | Raw-query diagnosis curve produced (H1/H2) | 1 | R1 | `scripts/run_diagnosis_curve.py` | ☑ |
| 7 | LQP trained | 2 | R1 | `setu/operators/lqp.py` | ☑ |
| 8 | CAEP trained | 2 | R2 | `setu/operators/caep.py` | ☑ |
| 9 | LAG trained (300-500 labeled queries) | 2 | R2 | `setu/operators/lag.py` | ☑ (Relabeled using empirical trajectory optimization, 314 queries) |
| 10 | CARF v1 + v2 implemented | 3 | R2 | `setu/fusion/carf.py` | ☑ |
| 11 | Confidence-correlation study done (H10) | 3 | R3 | `setu/evaluation/metrics.py`, `setu/evaluation/stats.py` | ☑ (ρ=0.503, p≈0.0 — H10 supported; score margin positively correlates with retrieval success) |
| 12 | Calibration model fit (if needed) | 3 | R3 | (add to `setu/evaluation/`) | ☑ (margin/entropy validated in H10 study) |
| 13 | Operator trajectories logged | 4 | R3 | `setu/controller/setu_bandit.py` | ☑ |
| 14 | Bandit controller (SETU v2) trained | 4 | R3 | `setu/controller/setu_bandit.py` | ☑ (LinUCB implemented & verified) |
| 15 | SETU v1 vs v2 comparison table done (H6/H8/H9) | 4 | R3 | `scripts/compare_setu_v1_v2.py`, `results/tables/` | ☑ |
| 16 | Corpus/queries scaled to pilot/domain-scale (380/3,000–5,000 target); not full-scale | 5 | R1 | `scripts/build_pilot_corpus.py` (scaled) | ☑ |
| 17 | Public benchmark arm (MIRACL+Aksharantar) run | 5 | R3 | `results/tables/extended_baselines.md` | ☑ (Not run locally — citation-only per plan's explicit fallback option (§7.3). Local run attempted twice, found methodologically invalid (corpus contained only ground-truth documents, no distractors) and not worth further engineering time given lowest priority.) |
| 18 | Extended baselines table compiled | 5 | R3 | `results/tables/extended_baselines.md` | ☑ |
| 19 | Statistical tests run for H1-H10 | 5 | R3 | `results/tables/statistical_significance_H1_H10_scaled.json` | ☑ (H8,H10 supported; H1,H4,H6,H7,H9 not supported; H2 significant in opposite direction; H3,H5 insufficient data — see RESULTS_SUMMARY.md) |
| 20 | All figures/tables assembled | 6 | All | `outputs/figures/`, `outputs/tables/` | ☐ |
| 21 | Paper drafted in IEEE format | 6 | All | (Overleaf, outside repo) | ☐ |
| 22 | (Stretch) Online policy adaptation | 6 | R3 | `setu/controller/setu_bandit.py` | ☐ |
