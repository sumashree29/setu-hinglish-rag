# Reproduction Guide (Final Evaluation)

This guide documents the exact sequence of commands required to reproduce the canonical evaluation and statistical outputs from a clean clone of the repository.

> [!WARNING]
> This exact sequence was derived from the command history of the final remediation audit, but has not been executed end-to-end in a truly clean environment. A final clean-environment execution is recommended before paper submission.

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Operator Ablations and Baselines
These scripts must be run to generate the raw retrieval metrics and baseline benchmarks across all embedding models.
```bash
python scripts/evaluate_scaled_retrieval_v3.py
python scripts/train_and_ablate_operators_v2.py
```
*Expected Output*: Generation of `scaled_corpus_retrieval_v3.json` and `scaled_operator_ablation_v3.json` in the `results/tables/` directory.

### 3. Generate Controller Sequences (SETU v1 / v2)
This script runs the 5-fold OOF cross-validation, dynamically fitting the LAG classifier and LinUCB controller on out-of-fold partitions, and evaluates on the test fold.
```bash
python scripts/compare_setu_v1_v2_scaled.py
```
*Expected Output*: Generation of `setu_v1_v2_comparison_scaled.json` and the query-level trajectory logs in `results/logs/setu_v2_per_query_v3.json`.

### 4. Run Statistical Battery and Diagnosis
These scripts apply the conditional analysis, compute multiple-testing corrections, and check for controller independence.
```bash
python scripts/diagnose_bandit.py
python scripts/diagnose_overcorrection_v3.py
python scripts/final_pass_script.py
```
*Expected Output*: `overcorrection_final.json` with 24 Holm-corrected p-values, `controller_behavior_final.json`, and the `baseline_comparison.png` figure.

### 5. Generate IEEE Tables
Generates the markdown/LaTeX tables directly from the canonical JSON outputs.
```bash
python scripts/generate_ieee_tables.py
```
*Expected Output*: `IEEE_TABLES.md` populated with the canonical data.
