# Master Checklist — Resume Point

If work stops, the first unchecked row tells you exactly where to resume.
Tick boxes as you go. File to fill in is listed so there's no ambiguity about where the work lives.

| # | Task | Phase | Owner | File(s) | Status |
|---|------|-------|-------|---------|--------|
| 1 | Environment installed by all 3 members | 1 | All | `requirements.txt` | ☐ |
| 2 | 3 embedding models downloaded | 1 | R1 | `setu/embeddings/loader.py`, `scripts/download_models.py` | ☐ |
| 3 | IndicLID + HinGE + PHINC downloaded | 1 | R1 | `setu/diagnosis/lid_entropy.py`, `setu/operators/lqp.py` | ☐ |
| 4 | Pilot corpus built (50-100 queries) | 1 | R1 (+team) | `setu/diagnosis/corpus.py`, `scripts/build_pilot_corpus.py` | ☐ |
| 5 | CMI(q) / LID-entropy(q) implemented | 1 | R1 | `setu/diagnosis/cmi.py`, `setu/diagnosis/lid_entropy.py` | ☐ |
| 6 | Raw-query diagnosis curve produced (H1/H2) | 1 | R1 | `scripts/run_diagnosis_curve.py` | ☐ |
| 7 | LQP trained | 2 | R1 | `setu/operators/lqp.py` | ☐ |
| 8 | CAEP trained | 2 | R2 | `setu/operators/caep.py` | ☐ |
| 9 | LAG trained (300-500 labeled queries) | 2 | R2 | `setu/operators/lag.py` | 🟡 rewrite executor done, real 300-500 labeled training data still pending (Phase 5) |
| 10 | CARF v1 + v2 implemented | 3 | R2 | `setu/fusion/carf.py` | ☐ |
| 11 | Confidence-correlation study done (H10) | 3 | R3 | `setu/evaluation/metrics.py`, `setu/evaluation/stats.py` | ☐ |
| 12 | Calibration model fit (if needed) | 3 | R3 | (add to `setu/evaluation/`) | ☐ |
| 13 | Operator trajectories logged | 4 | R3 | `setu/controller/setu_bandit.py` | ☐ |
| 14 | Bandit controller (SETU v2) trained | 4 | R3 | `setu/controller/setu_bandit.py` | ☐ |
| 15 | SETU v1 vs v2 comparison table done (H6/H8/H9) | 4 | R3 | `notebooks/phase4_controller.ipynb` | ☐ |
| 16 | Corpus/queries scaled to full target | 5 | R1 | `scripts/build_pilot_corpus.py` (scaled) | ☐ |
| 17 | Public benchmark arm (MIRACL+Aksharantar) run | 5 | R3 | `notebooks/phase5_evaluation.ipynb` | ☐ |
| 18 | Extended baselines table compiled | 5 | R3 | `notebooks/phase5_evaluation.ipynb` | ☐ |
| 19 | Statistical tests run for H1-H10 | 5 | R3 | `setu/evaluation/stats.py` | ☐ |
| 20 | All figures/tables assembled | 6 | All | `outputs/figures/`, `outputs/tables/` | ☐ |
| 21 | Paper drafted in IEEE format | 6 | All | (Overleaf, outside repo) | ☐ |
| 22 | (Stretch) Online policy adaptation | 6 | R3 | `setu/controller/setu_bandit.py` | ☐ |

