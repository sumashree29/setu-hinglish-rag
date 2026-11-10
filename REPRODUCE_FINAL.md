# Reproduction Guide (Final Verified Pipeline)

This guide documents the exact command sequence executed to reproduce all checkpoints, trajectories, primary/external results, tables, figures, tests, and readiness gates from a clean Python 3.11 environment.

### 1. Environment Setup (Windows 11 / Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 -r requirements.txt pytest psutil
.\.venv\Scripts\python.exe -m pip install pyarrow==19.0.1 faiss-cpu==1.10.0 llvmlite==0.44.0 numba==0.61.2 cbor2==5.6.5
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

### 2. Operator Training & Trajectory Generation
```powershell
.\.venv\Scripts\python.exe scripts/train_all_operator_models.py
.\.venv\Scripts\python.exe scripts/generate_trajectories_scaled.py
.\.venv\Scripts\python.exe scripts/run_diagnosis_curve_v2.py
Copy-Item results\logs\per_query_metrics_v2.json results\canonical\per_query_metrics.json -Force
```

### 3. Primary 314-Query Grouped 5-Fold OOF Evaluations
```powershell
.\.venv\Scripts\python.exe scripts/compare_setu_v1_v2_scaled.py
.\.venv\Scripts\python.exe scripts/analyze_controller_behavior.py
.\.venv\Scripts\python.exe scripts/analyze_controller_final.py
.\.venv\Scripts\python.exe scripts/analyze_confidence_final.py
.\.venv\Scripts\python.exe scripts/analyze_steps_final.py
.\.venv\Scripts\python.exe scripts/run_final_statistics.py
.\.venv\Scripts\python.exe scripts/build_dataset_stats.py
.\.venv\Scripts\python.exe scripts/train_and_ablate_operators_v2.py
.\.venv\Scripts\python.exe scripts/diagnose_overcorrection_v3.py
.\.venv\Scripts\python.exe scripts/evaluate_selective_strategies.py
.\.venv\Scripts\python.exe scripts/measure_latency.py
.\.venv\Scripts\python.exe scripts/run_statistical_tests_h1_h10_scaled.py
.\.venv\Scripts\python.exe scripts/error_analysis_v2.py
.\.venv\Scripts\python.exe scripts/final_pass_script.py
```

### 4. External Validation Arm (500 Dolly-Hinglish QA Pairs)
```powershell
.\.venv\Scripts\python.exe -u scripts/evaluate_external_dolly.py
```

### 5. Canonical Tables, Figures & Final Readiness Gate
```powershell
.\.venv\Scripts\python.exe scripts/reproduce_canonical_analysis.py
.\.venv\Scripts\python.exe scripts/generate_ieee_tables.py
.\.venv\Scripts\python.exe scripts/generate_figures.py
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe scripts/check_paper_consistency.py
.\.venv\Scripts\python.exe scripts/check_consistency.py
.\.venv\Scripts\python.exe scripts/final_readiness_gate.py
```
