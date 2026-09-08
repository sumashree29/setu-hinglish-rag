# SETU — Hinglish RAG Retrieval Degradation & Adaptive Correction

IEEE-target research project. Diagnoses how Hinglish/code-mixed queries degrade RAG
retrieval quality, and corrects it with a learned controller (SETU) that sequences
three operators (LQP, CAEP, LAG) and fuses ranks (CARF).

## Team roles (see MASTER_CHECKLIST.md for the live task list)
- **R1 — Retrieval & Diagnosis Lead**: `setu/diagnosis/`, `setu/embeddings/`, LQP, Phase 1 + corpus scaling
- **R2 — Operators & Fusion Lead**: CAEP, LAG, CARF, confidence calibration
- **R3 — Controller & Evaluation Lead**: SETU bandit controller, stats, benchmark arm, paper assembly

## Setup
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
No GPU needed anywhere in this project — everything is CPU-only.

## How this repo is organized
```
setu/
  diagnosis/     CMI(q), LID-entropy(q), pilot corpus builder      -> R1, Phase 1
  embeddings/    loads BGE-M3 / Indic-SBERT / multilingual-e5      -> R1, Phase 1
  operators/     lqp.py (R1), caep.py (R2), lag.py (R2)            -> Phase 2
  fusion/        carf.py                                          -> R2, Phase 3
  controller/    setu_bandit.py (SETU v1 baseline + v2 learned)    -> R3, Phase 4
  evaluation/    metrics.py (Recall/MRR/nDCG), stats.py (Wilcoxon) -> R1 seeds it, R3 owns it

scripts/         one script per pipeline stage — the thing you actually run
notebooks/       exploratory work per phase; graduate stable code into setu/ once it's solid
data/            raw/ (downloaded corpora), processed/ (built pilot corpus), embeddings/ (cached vectors), logs/ (operator trajectories)
outputs/         figures/ and tables/ for the paper — one subfolder per phase is fine
tests/           one test file per module; write these as you fill in the TODOs
```

## Workflow for filling in a module
1. Open your file (e.g. `setu/diagnosis/cmi.py`)
2. Read the docstring — it says what the function must take in/return, and links back to
   the exact section of `SETU_Implementation_Plan.pdf` that specifies it
3. Implement it, replacing the `raise NotImplementedError(...)`
4. Add/run the matching test in `tests/`
5. Tick the row off in `MASTER_CHECKLIST.md`

## Where to run things
Everything here is plain Python — works identically in Jupyter, Colab, or an IDE.
Recommended: prototype in `notebooks/`, then once a function is stable, move its final
version into the matching `setu/` module so the rest of the pipeline can import it.

## Reproducibility
For the bandit controller evaluation, we use a rigorous 5-fold out-of-fold cross validation.
The splits are fully deterministic, based on sequential chunking of the sorted query IDs in `queries_v3_final.json`, eliminating random seed variance. All initializations of the LinUCB context weights start deterministically at zero, ensuring the evaluations in `test_phase4_integration.py` and `run_statistical_tests_h1_h10_scaled.py` are strictly reproducible.
